"""Re-preuve d'équivalence pixel du lot PIL (multiview 0.31.0) : chemin TUILE des VU-mètres
(_meter_tile_gpu avec xp=numpy) ≡ chemin PIL historique (_draw_meter + rgba_to_yuv), max|Δ|=0
sur Y/U/V/alpha. Méthode de la validation 0.20.0 : rendre le script du plugin, en extraire par
AST les défs nécessaires (exec stmt par stmt, garde-fous contre serveurs/threads/boucles),
puis comparer les deux chemins sur une grille de configs. Exit 0 = tout identique."""
import ast, sys, itertools
import numpy as np

sys.path.insert(0, "/opt/bobistudio")
sys.path.insert(0, "/opt/bobistudio/script_templates")   # import bobimxl (gate _MVK du script)
from app import plugins

NEEDED = {"_meter_layout", "_meter_scale_params", "_draw_meter", "_draw_meter_static",
          "_meter_static_xp", "_rgba_to_yuv_xp", "_meter_comp_rect", "_meter_tile_gpu",
          "rgba_to_yuv", "_make_font", "_font", "_as_bool"}
BANNED = ("HTTPServer", "serve_forever", ".start()", "Instance(", "Writer(", "Reader(",
          "socket.", "while True")


def load_ns(cfg):
    src = plugins.render_script("multiview", cfg, "equivtest")
    tree = ast.parse(src)
    ns = {"__name__": "mv_equiv"}
    for node in tree.body:
        if isinstance(node, (ast.While, ast.If)) and any(
                b in ast.unparse(node) for b in BANNED):
            continue
        seg = ast.unparse(node)
        if any(b in seg for b in BANNED):
            continue
        try:
            exec(compile(ast.Module([node], []), "<mv>", "exec"), ns)
        except Exception:
            pass
        if NEEDED <= set(ns):
            # tout ce qu'il faut est défini — on n'exec pas la suite (boucle principale etc.)
            if "METER_TICK_W" in ns and "_MET_GREEN" in ns and "_NP_DT" in ns:
                break
    missing = NEEDED - set(ns)
    if missing:
        raise SystemExit("défs manquantes après extraction : %s" % missing)
    return ns


def compare(ns):
    Image = ns["Image"]
    _CW, _CH = ns["_CW"], ns["_CH"]
    OUT_W, OUT_H = ns["OUT_WIDTH"], ns["OUT_HEIGHT"]
    fails = 0
    rng = np.random.default_rng(0)
    for (n, scale, op) in itertools.product((2, 4, 8), ("dbfs", "ppm"), (60, 100)):
        # géométrie représentative + bbox chroma-alignée = même calcul que _meter_tiles_at
        mx, my = 13, 7
        lay = ns["_meter_layout"](n)
        mw, mh = lay if isinstance(lay, tuple) else (150, 220)
        peaks = list(rng.uniform(-60, 0, n))
        holds = [min(0.0, p + rng.uniform(0, 6)) for p in peaks]
        bx0 = max(0, mx); by0 = max(0, my)
        bx1 = min(OUT_W, mx + mw + 1); by1 = min(OUT_H, my + mh + 1)
        bx0 -= bx0 % _CW; by0 -= by0 % _CH
        if (bx1 - bx0) % _CW: bx1 = min(OUT_W, bx1 + (_CW - (bx1 - bx0) % _CW))
        if (by1 - by0) % _CH: by1 = min(OUT_H, by1 + (_CH - (by1 - by0) % _CH))
        W, H, rmx, rmy = bx1 - bx0, by1 - by0, mx - bx0, my - by0
        # chemin PIL historique
        tile = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ns["_draw_meter"](tile, rmx, rmy, mw, mh, n, peaks, holds, scale, op, 0)
        ref = ns["rgba_to_yuv"](tile)
        # chemin tuile (xp=np)
        got = ns["_meter_tile_gpu"](W, H, rmx, rmy, mw, mh, n, peaks, holds, scale, op, 0)
        deltas = [int(np.abs(r.astype(np.int64) - g.astype(np.int64)).max()) if r.size else 0
                  for r, g in zip(ref, got)]
        ok = all(d == 0 for d in deltas)
        fails += 0 if ok else 1
        print("%s n=%d scale=%s op=%d%% maxΔ(Y,U,V,a,a2)=%s" %
              ("OK  " if ok else "FAIL", n, scale, op, deltas))
    return fails


total = 0
for label, cfg in (("8 bits 420", {"flux_config": [], "force_cpu": True}),
                   ("10 bits 422", {"flux_config": [], "force_cpu": True,
                                    "bit_depth": 10, "chroma": "422"})):
    print("== profil", label)
    total += compare(load_ns(cfg))
print("Échecs :", total)
sys.exit(1 if total else 0)
