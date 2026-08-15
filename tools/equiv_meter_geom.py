"""Équivalence de la GÉOMÉTRIE des VU-mètres entre le MOTEUR (script.py, Python) et l'APERÇU du
composer (multiview.js). Exit 0 = les deux dessinent au même endroit.

Pourquoi : l'aperçu du composer étalait ses barres sur toute la largeur du bloc, sans colonne de
graduations, sans largeur de barre réelle, sans mode de largeur ni alignement — « la largeur des
peak meters et l'alignement ne sont pas représentatifs de ce qui est diffusé » (testeur, 2026-08-11).
La correction (0.111.0) porte `_meter_grad` / `_meter_layout` / `_meter_fit_dims` en JS. DEUX COPIES
de la même géométrie : sans ce banc, la prochaine correction faite d'un seul côté re-fabriquerait
l'écart en silence — et un aperçu qui ment est pire qu'une absence d'aperçu.

Méthode (celle de equiv_meters.py) : rendre le script du plugin, en extraire par AST les seules
constantes et fonctions de géométrie, puis comparer à la sortie de `mwMeterGeom` (node) sur toute
la grille canaux × largeur × échelle × mode de largeur × alignement.

    ./venv/bin/python plugins/multiview/tools/equiv_meter_geom.py
"""
import ast, json, os, subprocess, sys

sys.path.insert(0, "/opt/bobistudio")
from app import plugins

ICI = os.path.dirname(os.path.abspath(__file__))
JS = os.path.join(ICI, "..", "multiview.js")

CONSTANTES = {"METER_BAR_W", "METER_GAP", "METER_TICK_W_DBFS", "METER_TICK_W_PPM",
              "METER_TICK_W_MARKS", "METER_TICK_W"}
FONCTIONS = {"_meter_grad", "_meter_layout", "_meter_fit_dims"}


def moteur_ns():
    """Constantes + fonctions de géométrie extraites du script RENDU (jamais recopiées ici :
    c'est le moteur qui fait foi, y compris ses seuils)."""
    src = plugins.render_script("multiview", {"shm_out": "geomtest"}, "geomtest")
    tree = ast.parse(src)
    bouts = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) in CONSTANTES for t in node.targets):
            bouts.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.FunctionDef) and node.name in FONCTIONS:
            bouts.append(ast.get_source_segment(src, node))
    ns = {}
    exec("\n".join(bouts), ns)
    manquant = (CONSTANTES | FONCTIONS) - set(ns)
    if manquant:
        sys.exit("introuvables dans le script du moteur : %s" % ", ".join(sorted(manquant)))
    return ns


def cas_moteur(ns):
    out = []
    for n in (1, 2, 6, 8, 16):
        for rw in (20, 40, 90, 200, 600, 1200):
            for scale in ("dbfs", "ppm"):
                for wm in ("auto", "fit"):
                    for al in ("left", "center", "right"):
                        grad, tick = ns["_meter_grad"]({}, scale, rw)
                        if wm == "fit":
                            tw, bar, gap, mw = ns["_meter_fit_dims"](n, rw, tick)
                            mx = 0
                        else:
                            tw, bar, gap = tick, ns["METER_BAR_W"], ns["METER_GAP"]
                            mw = ns["_meter_layout"](n, tw, bar, gap)
                            mx = ((rw - mw) // 2 if al == "center"
                                  else (rw - mw if al == "right" else 0))
                        out.append({"n": n, "rw": rw, "scale": scale, "width_mode": wm,
                                    "align": al, "grad": grad, "tick": tw, "bar": bar,
                                    "gap": gap, "mw": mw, "mx": mx})
    return out


PILOTE_JS = r"""
const fs = require('fs'), vm = require('vm');
const sb = {console}; sb.globalThis = sb; vm.createContext(sb);
// argv : [0]=node, [1]=ce pilote, [2]=multiview.js, [3]=cas du moteur
const src = fs.readFileSync(process.argv[2], 'utf8');
// On n'exécute QUE le bloc de géométrie (pas de DOM à stubber).
const i0 = src.indexOf('const MW_METER = {'), i1 = src.indexOf('function drawBlocksLayer');
if (i0 < 0 || i1 < 0) { console.log(JSON.stringify({erreur: 'bloc de géométrie introuvable'})); process.exit(0); }
vm.runInContext(src.slice(i0, i1), sb);
const mwMeterGeom = vm.runInContext('mwMeterGeom', sb);
const cas = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
console.log(JSON.stringify(cas.map(c => {
    const g = mwMeterGeom({channels: c.n, w: c.rw, h: 200, scale: c.scale,
                           width_mode: c.width_mode, align: c.align, x: 0, y: 0});
    return {grad: g.grad, tick: g.tick, bar: g.bar, gap: g.gap, mw: g.mw, mx: g.mx};
})));
"""


def main():
    ns = moteur_ns()
    cas = cas_moteur(ns)
    tmp_cas = os.path.join(ICI, "_geom_cas.json")
    tmp_js = os.path.join(ICI, "_geom_pilote.js")
    try:
        open(tmp_cas, "w").write(json.dumps(cas))
        open(tmp_js, "w").write(PILOTE_JS)
        r = subprocess.run(["node", tmp_js, JS, tmp_cas], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("node a échoué : %s" % (r.stderr.strip() or r.returncode))
        vus = json.loads(r.stdout)
    finally:
        for f in (tmp_cas, tmp_js):
            if os.path.exists(f):
                os.remove(f)
    if isinstance(vus, dict):
        sys.exit(vus.get("erreur"))
    ecarts = 0
    for c, v in zip(cas, vus):
        diff = [k for k in ("grad", "tick", "bar", "gap", "mw", "mx") if c[k] != v[k]]
        if diff:
            ecarts += 1
            if ecarts <= 8:
                print("ÉCART %s → %s" % ({k: c[k] for k in ("n", "rw", "scale", "width_mode", "align")},
                                         ", ".join("%s moteur=%s aperçu=%s" % (k, c[k], v[k]) for k in diff)))
    if ecarts:
        sys.exit("%d / %d cas divergent" % (ecarts, len(cas)))
    print("aperçu IDENTIQUE au moteur sur %d cas "
          "(canaux × largeur × échelle × mode de largeur × alignement)" % len(cas))


if __name__ == "__main__":
    main()
