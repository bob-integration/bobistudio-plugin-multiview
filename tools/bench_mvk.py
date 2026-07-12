#!/usr/bin/env python3
"""Banc avant/après du kernel compose fusionné libbobi_mvk (script_templates/mvcompose.c).

Rejoue une trame représentative « mur 3×3 » (canvas Y 1080×1920 + U/V 540×960, chroma 420) :
  - 9 sources 1080p → tuiles 640×360 par downscale nearest 3× ENTIER ;
  - 1 source NON-entière 1372×770 → tuile 640×360 (indices float-troncature), qui remplace
    la 9ᵉ tuile dans une 2ᵉ passe de mesure ;
  - chrome : blend_pre sur bbox 1080×1920 entière (opérandes inv_a/src_a pré-calculés) ;
  - 12 tuiles VU/horloges : blend sur bboxes 200×48 (Y) + chroma.

Deux implémentations COMPLÈTES de la trame :
  1. REF numpy : exactement les formules du multiview — placement par fancy-indexing/vue
     stridée, blend ((dst·(255−α)+src·α)//255), blend_pre ((dst·inv_a+src_a)//255) ;
  2. MVK : les wrappers réels de bobimxl (mvk_place_into / mvk_blend_into /
     mvk_blend_pre_into) sur un canvas séparé.

Après CHAQUE trame MVK : np.array_equal sur Y/U/V vs la référence — le banc ÉCHOUE (exit 1)
si un octet diffère. Mesure ~100 itérations (warmup 3), ms/trame ventilées
(place / chrome blend_pre / tuiles blend) + total, chemin MVK mesuré à threads=1 PUIS
threads=4 (bobimxl.mvk_set_threads). --deep = uint16 (10/12 bits). --json <path> pour dumper.

Autonome : numpy + les wrappers bobimxl (la lib C se charge via env BOBI_MVK_LIB, sinon
/usr/local/lib comme dans les images runtime). Lancement type (contrôleur) :
  BOBI_MVK_LIB=/path/libbobi_mvk.so /opt/bobistudio/venv/bin/python \\
      plugins/multiview/tools/bench_mvk.py --deep --json /tmp/mvk.json
"""
import argparse, json, statistics, sys, time
import numpy as np

# Wrappers réels de la lib (mêmes chemins de chargement que dans les conteneurs)
sys.path.insert(0, "/opt/bobistudio/script_templates")
import bobimxl

# ─── Géométrie du mur 3×3 (mêmes conventions que le multiview : chroma 420, _CW=_CH=2) ───
W, H = 1920, 1080                     # canvas Y
CW, CH = 2, 2                         # sous-échantillonnage chroma 420
WC, HC = W // CW, H // CH             # canvas U/V : 960×540
TW, TH = 640, 360                     # tuile par source
NE_W, NE_H = 1372, 770                # source NON-entière (ratio ni 2× ni 3×)

# Layout 3×3 : (source, vy, vx) — 9 fenêtres 640×360
LAYOUT = [(i, (i // 3) * TH, (i % 3) * TW) for i in range(9)]

# 12 tuiles VU/horloges : bboxes 200×48 (Y)
VU_W, VU_H = 200, 48
VU_POS = [(40, 100), (1000, 100), (1600, 100),
          (40, 400), (1000, 400), (1600, 400),
          (40, 700), (1000, 700), (1600, 700),
          (40, 950), (1000, 950), (1600, 950)]


def _maxval(dt):
    """Borne haute plausible du contenu : 8 bits pleins, ou 10 bits en profond (uint16)."""
    return 256 if dt == np.uint8 else 1024


def make_inputs(dt, seed=0):
    """Toutes les entrées de la trame, seedées rng(0) : sources, chrome, VU, indices nearest.

    Renvoie un dict — TOUT est pré-calculé ici (les deux implémentations consomment les MÊMES
    tableaux et les MÊMES indices ; seule l'exécution du compose est mesurée)."""
    rng = np.random.default_rng(seed)
    acc = np.uint16 if dt == np.uint8 else np.uint32     # _ACC du multiview
    mx = _maxval(dt)

    # 9 sources 1080p : plans Y 1080×1920, U/V 540×960
    srcs = [(rng.integers(0, mx, (H, W), dtype=dt),
             rng.integers(0, mx, (HC, WC), dtype=dt),
             rng.integers(0, mx, (HC, WC), dtype=dt)) for _ in range(9)]

    # Source NON-entière : Y 770×1372, chroma 385×686
    src_ne = (rng.integers(0, mx, (NE_H, NE_W), dtype=dt),
              rng.integers(0, mx, (NE_H // CH, NE_W // CW), dtype=dt),
              rng.integers(0, mx, (NE_H // CH, NE_W // CW), dtype=dt))

    # Indices nearest — calculés par l'appelant (contrat mvk_place_into : le C n'impose
    # aucune formule ; ceux-ci sont EXACTEMENT ceux consommés par les deux implémentations).
    ri3 = (np.arange(TH) * 3).astype(np.int32)                          # downscale 3× entier
    ri3c = (np.arange(TH // CH) * 3).astype(np.int32)                   # chroma (180 lignes)
    # Float-troncature de la source non-entière (formule resize_plane du multiview)
    ri_ne = (np.arange(TH) * NE_H / TH).astype(np.int32)
    ci_ne = (np.arange(TW) * NE_W / TW).astype(np.int32)
    ri_ne_c = (np.arange(TH // CH) * (NE_H // CH) / (TH // CH)).astype(np.int32)
    ci_ne_c = (np.arange(TW // CW) * (NE_W // CW) / (TW // CW)).astype(np.int32)

    # Chrome : alpha uint8 plausible (habillage translucide) → opérandes pré-calculés
    # inv_a = 255−α, src_a = src·α, dtype _ACC (uint16 en 8 bits, uint32 en profond).
    a = rng.integers(0, 96, (H, W), dtype=np.uint8)
    s_y = rng.integers(0, mx, (H, W), dtype=dt)
    ac = np.ascontiguousarray(a[::CH, ::CW])
    s_u = rng.integers(0, mx, (HC, WC), dtype=dt)
    s_v = rng.integers(0, mx, (HC, WC), dtype=dt)
    chrome = {
        "inv_y": (255 - a).astype(acc), "sa_y": s_y.astype(acc) * a,
        "inv_c": (255 - ac).astype(acc),
        "sa_u": s_u.astype(acc) * ac, "sa_v": s_v.astype(acc) * ac,
    }

    # 12 tuiles VU : src/alpha aléatoires, alpha chroma CONTIGU (contrat _mvk_ok2d)
    vu = []
    for (vx, vy) in VU_POS:
        oy = rng.integers(0, mx, (VU_H, VU_W), dtype=dt)
        ou = rng.integers(0, mx, (VU_H // CH, VU_W // CW), dtype=dt)
        ov = rng.integers(0, mx, (VU_H // CH, VU_W // CW), dtype=dt)
        oa = rng.integers(0, 256, (VU_H, VU_W), dtype=np.uint8)
        oa2 = np.ascontiguousarray(oa[::CH, ::CW])
        vu.append((vx, vy, oy, ou, ov, oa, oa2))

    return {"dt": dt, "acc": acc, "srcs": srcs, "src_ne": src_ne, "chrome": chrome,
            "vu": vu, "ri3": ri3, "ri3c": ri3c,
            "ri_ne": ri_ne, "ci_ne": ci_ne, "ri_ne_c": ri_ne_c, "ci_ne_c": ci_ne_c}


def _canvas(dt):
    return (np.zeros((H, W), dt), np.zeros((HC, WC), dt), np.zeros((HC, WC), dt))


# ─── Implémentation 1 : REF numpy (formules exactes du multiview) ────────────────────────────
def render_numpy(inp, out, non_integer):
    """Trame complète en numpy pur. Renvoie les temps par phase (s) : place / chrome / vu."""
    dt, acc = inp["dt"], inp["acc"]
    cy, cu, cv = out
    t = {}

    # ── Placement : fancy-indexing (source non-entière) / vue stridée (ratio entier) ──
    t0 = time.perf_counter()
    for (si, vy, vx) in LAYOUT:
        if non_integer and si == 8:
            sy, su, sv = inp["src_ne"]
            cy[vy:vy + TH, vx:vx + TW] = sy[inp["ri_ne"]][:, inp["ci_ne"]]
            cu[vy // CH:(vy + TH) // CH, vx // CW:(vx + TW) // CW] = \
                su[inp["ri_ne_c"]][:, inp["ci_ne_c"]]
            cv[vy // CH:(vy + TH) // CH, vx // CW:(vx + TW) // CW] = \
                sv[inp["ri_ne_c"]][:, inp["ci_ne_c"]]
            continue
        sy, su, sv = inp["srcs"][si]
        cy[vy:vy + TH, vx:vx + TW] = sy[::3, ::3]           # downscale 3× entier (vue stridée)
        cu[vy // CH:(vy + TH) // CH, vx // CW:(vx + TW) // CW] = su[::3, ::3]
        cv[vy // CH:(vy + TH) // CH, vx // CW:(vx + TW) // CW] = sv[::3, ::3]
    t["place"] = time.perf_counter() - t0

    # ── Chrome : blend_pre plein écran ((dst·inv_a + src_a) // 255) ──
    t0 = time.perf_counter()
    ch = inp["chrome"]
    cy[...] = ((cy.astype(acc) * ch["inv_y"] + ch["sa_y"]) // 255).astype(dt)
    cu[...] = ((cu.astype(acc) * ch["inv_c"] + ch["sa_u"]) // 255).astype(dt)
    cv[...] = ((cv.astype(acc) * ch["inv_c"] + ch["sa_v"]) // 255).astype(dt)
    t["chrome"] = time.perf_counter() - t0

    # ── Tuiles VU : blend ((dst·(255−α) + src·α) // 255) sur chaque bbox ──
    t0 = time.perf_counter()
    for (vx, vy, oy, ou, ov, oa, oa2) in inp["vu"]:
        dv = cy[vy:vy + VU_H, vx:vx + VU_W]
        dv[...] = ((dv.astype(acc) * (255 - oa.astype(acc))
                    + oy.astype(acc) * oa.astype(acc)) // 255).astype(dt)
        cx0, cy0 = vx // CW, vy // CH
        for (plane, src) in ((cu, ou), (cv, ov)):
            dv = plane[cy0:cy0 + VU_H // CH, cx0:cx0 + VU_W // CW]
            dv[...] = ((dv.astype(acc) * (255 - oa2.astype(acc))
                        + src.astype(acc) * oa2.astype(acc)) // 255).astype(dt)
    t["vu"] = time.perf_counter() - t0
    return t


# ─── Implémentation 2 : MVK (wrappers bobimxl, kernels C fusionnés) ──────────────────────────
def render_mvk(inp, out, non_integer):
    """Trame complète via les wrappers mvk_*. Toute passe refusée (False) est une ERREUR du
    banc (les entrées sont taillées pour le contrat) → exception, pas de repli silencieux."""
    cy, cu, cv = out
    t = {}

    def _ok(ret, what):
        if not ret:
            raise RuntimeError(f"wrapper mvk refusé : {what} (dtype/forme/contiguïté ?)")

    # ── Placement : mvk_place_into (col_step régulier ou gather col_idx) ──
    t0 = time.perf_counter()
    for (si, vy, vx) in LAYOUT:
        if non_integer and si == 8:
            sy, su, sv = inp["src_ne"]
            _ok(bobimxl.mvk_place_into(cy[vy:vy + TH, vx:vx + TW], sy,
                                       inp["ri_ne"], col_idx=inp["ci_ne"]), "place Y non-ent")
            _ok(bobimxl.mvk_place_into(
                cu[vy // CH:(vy + TH) // CH, vx // CW:(vx + TW) // CW], su,
                inp["ri_ne_c"], col_idx=inp["ci_ne_c"]), "place U non-ent")
            _ok(bobimxl.mvk_place_into(
                cv[vy // CH:(vy + TH) // CH, vx // CW:(vx + TW) // CW], sv,
                inp["ri_ne_c"], col_idx=inp["ci_ne_c"]), "place V non-ent")
            continue
        sy, su, sv = inp["srcs"][si]
        _ok(bobimxl.mvk_place_into(cy[vy:vy + TH, vx:vx + TW], sy,
                                   inp["ri3"], col0=0, col_step=3), "place Y 3x")
        _ok(bobimxl.mvk_place_into(
            cu[vy // CH:(vy + TH) // CH, vx // CW:(vx + TW) // CW], su,
            inp["ri3c"], col0=0, col_step=3), "place U 3x")
        _ok(bobimxl.mvk_place_into(
            cv[vy // CH:(vy + TH) // CH, vx // CW:(vx + TW) // CW], sv,
            inp["ri3c"], col0=0, col_step=3), "place V 3x")
    t["place"] = time.perf_counter() - t0

    # ── Chrome : mvk_blend_pre_into plein écran ──
    t0 = time.perf_counter()
    ch = inp["chrome"]
    _ok(bobimxl.mvk_blend_pre_into(cy, ch["inv_y"], ch["sa_y"]), "blend_pre Y")
    _ok(bobimxl.mvk_blend_pre_into(cu, ch["inv_c"], ch["sa_u"]), "blend_pre U")
    _ok(bobimxl.mvk_blend_pre_into(cv, ch["inv_c"], ch["sa_v"]), "blend_pre V")
    t["chrome"] = time.perf_counter() - t0

    # ── Tuiles VU : mvk_blend_into sur chaque bbox ──
    t0 = time.perf_counter()
    for (vx, vy, oy, ou, ov, oa, oa2) in inp["vu"]:
        _ok(bobimxl.mvk_blend_into(cy[vy:vy + VU_H, vx:vx + VU_W], oy, oa), "blend Y VU")
        cx0, cy0 = vx // CW, vy // CH
        _ok(bobimxl.mvk_blend_into(cu[cy0:cy0 + VU_H // CH, cx0:cx0 + VU_W // CW], ou, oa2),
            "blend U VU")
        _ok(bobimxl.mvk_blend_into(cv[cy0:cy0 + VU_H // CH, cx0:cx0 + VU_W // CW], ov, oa2),
            "blend V VU")
    t["vu"] = time.perf_counter() - t0
    return t


# ─── Mesure ───────────────────────────────────────────────────────────────────────────────────
def _stats(phase_times):
    """Agrège une liste de dicts de phases → ms moyennes par phase + total (mean/p50)."""
    out = {}
    totals = [sum(d.values()) for d in phase_times]
    for ph in ("place", "chrome", "vu"):
        out[ph] = statistics.fmean(d[ph] for d in phase_times) * 1000
    s = sorted(totals)
    out["total"] = statistics.fmean(totals) * 1000
    out["total_p50"] = s[len(s) // 2] * 1000
    return out


def _fmt(name, st):
    return (f"{name:16s} place {st['place']:6.2f}  chrome {st['chrome']:6.2f}  "
            f"vu {st['vu']:6.2f}  | total {st['total']:7.2f} ms (p50 {st['total_p50']:7.2f})")


def bench_pass(inp, non_integer, frames, warmup, ref_out):
    """Une passe de mesure (entière ou non-entière) : réf numpy 1×, MVK à threads=1 puis 4.
    Chaque trame MVK est comparée octet à octet à `ref_out` → exit 1 au moindre écart."""
    label = "NON-entière (9ᵉ tuile 1372×770)" if non_integer else "entière (9× downscale 3×)"
    print(f"\n── Passe {label} ──")
    res = {}

    # Référence numpy (une seule série de mesures)
    times = []
    out = _canvas(inp["dt"])
    for i in range(warmup + frames):
        t = render_numpy(inp, out, non_integer)
        if i >= warmup:
            times.append(t)
    for a, b in zip(ref_out, out):
        a[...] = b                                     # sortie de référence de cette passe
    res["numpy"] = _stats(times)
    print(_fmt("numpy (réf)", res["numpy"]))

    # MVK, threads=1 puis threads=4 — équivalence octet vérifiée à CHAQUE trame
    for nthreads in (1, 4):
        bobimxl.mvk_set_threads(nthreads)
        times = []
        out = _canvas(inp["dt"])
        for i in range(warmup + frames):
            t = render_mvk(inp, out, non_integer)
            for pl, a, b in zip("YUV", ref_out, out):
                if not np.array_equal(a, b):
                    d = int((a != b).sum())
                    print(f"❌ ÉCART OCTET plan {pl} (mvk threads={nthreads}, "
                          f"trame {i}) : {d} éléments diffèrent")
                    sys.exit(1)
            if i >= warmup:
                times.append(t)
        key = f"mvk_t{nthreads}"
        res[key] = _stats(times)
        sp = res["numpy"]["total"] / res[key]["total"]
        print(_fmt(f"mvk threads={nthreads}", res[key]) + f"  → speedup {sp:.1f}×")
        res[key]["speedup"] = sp
    return res


def main():
    ap = argparse.ArgumentParser(description="Banc numpy vs libbobi_mvk (mur 3×3)")
    ap.add_argument("--deep", action="store_true", help="uint16 (10/12 bits) au lieu de uint8")
    ap.add_argument("--frames", type=int, default=100, help="trames mesurées (défaut 100)")
    ap.add_argument("--warmup", type=int, default=3, help="trames de warmup (défaut 3)")
    ap.add_argument("--json", default=None, help="dump des résultats (JSON)")
    args = ap.parse_args()

    dt = np.uint16 if args.deep else np.uint8
    print(f"Banc MVK mur 3×3 — dtype {dt.__name__}, {args.frames} trames (warmup {args.warmup})")
    if not bobimxl.mvk_available():
        print("libbobi_mvk INTROUVABLE (env BOBI_MVK_LIB ? /usr/local/lib ?) — abandon")
        return 2

    inp = make_inputs(dt)
    ref_out = _canvas(dt)
    res = {"dtype": dt.__name__, "frames": args.frames,
           "integer": bench_pass(inp, False, args.frames, args.warmup, ref_out),
           "non_integer": bench_pass(inp, True, args.frames, args.warmup, ref_out)}
    print("\n✓ Équivalence octet OK sur toutes les trames (Y/U/V, threads 1 et 4)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"→ {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
