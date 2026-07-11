#!/usr/bin/env python3
"""Banc MICRO-BATCH GPU du multiview (TISSU_SLICE_GPU.md §b variante 2, post-verdict GO-MEGA-135).

Rejoue le rendu multiview COMPLET (4 entrées 1080p synthétiques → mur 1080p 2×2, chrome statique
pré-calculé plein écran + 4 tuiles VU dynamiques) en 3 modes :

  1. cpu_slice   — référence SÉMANTIQUE : compose + publication bande par bande (36 l), numpy ;
  2. gpu_whole   — pipeline GPU actuel : 1 H2D groupé épinglé, compose pleine trame VRAM,
                   1 D2H épinglé, publication d'un bloc ;
  3. gpu_batch   — micro-batch implémenté dans script.py 0.27.0 : gather hôte au grain 36 l,
                   H2D groupé PAR LOT de `--grp` bandes, kernels par lot, D2H par lot RECOUVERT
                   (stream dédié non-bloquant : D2H du lot j pendant le compose du lot j+1),
                   commits simulés au grain 36 l (copies pinned→sortie par bande).

Mesures : temps de trame et latence de 1ʳᵉ bande SORTIE (36 l posées dans le buffer de sortie),
p50/p90 sur --frames trames (T4 PARTAGÉE : pauses intercalées, rapporter p90). Vérifie aussi
l'ÉQUIVALENCE OCTET des 3 modes (mêmes entrées → même sortie, tolérance 0).

Autonome : cupy + numpy seulement, aucun accès MXL/orchestrateur. Lancement type (nœud GPU) :
  docker run --rm --gpus all --entrypoint python3 -v /tmp:/b bobi-compute-gpu:0.1 \
      /b/bench_gpu_batch.py --json /b/batch_result.json
"""
import argparse, json, statistics, time
import numpy as np
try:
    import cupy as cp
except Exception as e:                       # pragma: no cover
    cp = None
    _CP_ERR = e

# ─── Géométrie (mêmes conventions que script.py : 4:2:2 8 bits, _CW=2, _CH=1) ───
W, H = 1920, 1080
CW, CH = 2, 1
SL = 36                                       # grain MXL (SLICE_LINES)
NB = H // SL                                  # 30 bandes
DT = np.uint8
ACC = np.uint16                               # accumulateur blend 8 bits (comme script.py)


# Blends fusionnés GPU (mêmes kernels que script.py 0.27.0 — 1 lancement par plan) ; repli numpy
# à l'arithmétique entière identique → équivalence octet vérifiée plus bas.
if cp is not None:
    _blend_k = cp.ElementwiseKernel(
        "T dst, T src, uint8 alpha", "T out",
        "out = (T)(((unsigned int)dst * (255u - (unsigned int)alpha) + "
        "(unsigned int)src * (unsigned int)alpha) / 255u)", "bobi_blend")
    _blend_pre_k = cp.ElementwiseKernel(
        "T dst, A inv_a, A src_a", "T out",
        "out = (T)(((unsigned int)dst * (unsigned int)inv_a + (unsigned int)src_a) / 255u)",
        "bobi_blend_pre")


def blend(xp, dst, src, a):
    """blend() du plugin — identique octet CPU/GPU (entier, //255)."""
    if xp is not np:
        return _blend_k(dst, src, a)
    return ((dst.astype(ACC) * (255 - a.astype(ACC)) + src.astype(ACC) * a.astype(ACC)) // 255).astype(DT)


def blend_pre(xp, dst, inv_a, src_a):
    """blend_pre() du plugin (opérandes pré-calculés inv_a=(255−α), src_a=src·α)."""
    if xp is not np:
        return _blend_pre_k(dst, inv_a, src_a)
    return ((dst.astype(ACC) * inv_a + src_a) // 255).astype(DT)


def make_inputs(n, seed=1234):
    """n entrées 1080p synthétiques (Y, U, V) — contenu pseudo-aléatoire déterministe."""
    rng = np.random.default_rng(seed)
    tiles = []
    for _ in range(n):
        y = rng.integers(0, 256, (H, W), dtype=DT)
        u = rng.integers(0, 256, (H // CH, W // CW), dtype=DT)
        v = rng.integers(0, 256, (H // CH, W // CW), dtype=DT)
        tiles.append((y, u, v))
    return tiles


def make_chrome(seed=99):
    """Chrome statique plein écran pré-calculé (inv_a / src·α par plan, comme _chrome_pre)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 96, (H, W), dtype=DT)            # alpha faible (habillage)
    sy = rng.integers(0, 256, (H, W), dtype=DT)
    ac = a[::CH, ::CW]
    su = rng.integers(0, 256, (H // CH, W // CW), dtype=DT)
    sv = rng.integers(0, 256, (H // CH, W // CW), dtype=DT)
    piY = (255 - a).astype(ACC); saY = (sy.astype(ACC) * a)
    piC = (255 - ac).astype(ACC)
    saU = (su.astype(ACC) * ac); saV = (sv.astype(ACC) * ac)
    return (0, 0, W, H, piY, saY, piC, saU, saV)


def make_vu_tiles(seed=7):
    """4 tuiles VU dynamiques (bbox + plans + alphas), une par fenêtre."""
    rng = np.random.default_rng(seed)
    tiles = []
    tw, th = 200, 120
    for (vx, vy) in ((40, 400), (1000, 400), (40, 940), (1000, 940)):
        oy = rng.integers(0, 256, (th, tw), dtype=DT)
        ou = rng.integers(0, 256, (th // CH, tw // CW), dtype=DT)
        ov = rng.integers(0, 256, (th // CH, tw // CW), dtype=DT)
        oa = rng.integers(0, 256, (th, tw), dtype=DT)
        oa2 = oa[::CH, ::CW]
        tiles.append((vx, vy, vx + tw, vy + th, oy, ou, ov, oa, oa2))
    return tiles


# Layout 2×2 : (tuile, vy, vh, vx, vw) — ratio ENTIER 2 → vues stridées (fast-path du plugin)
LAYOUT = [(0, 0, 540, 0, 960), (1, 0, 540, 960, 960),
          (2, 540, 540, 0, 960), (3, 540, 540, 960, 960)]


def _views(tiles):
    """Vues stridées nearest (fast-path ratio entier de _tile_views) par fenêtre."""
    out = []
    for (ti, vy, vh, vx, vw) in LAYOUT:
        sy, su, sv = tiles[ti]
        st_y, st_x = H // vh, W // vw
        out.append((sy[::st_y, ::st_x], su[::st_y, ::st_x], sv[::st_y, ::st_x], vy, vh, vx, vw))
    return out


def _blend_band(xp, cy, cu, cv, chrome, vu, b0, b1, ip=False):
    """Habillage borné aux lignes [b0, b1) — code IDENTIQUE au plugin (chrome puis VU).
    ip=True (mode micro-batch) : kernels fusionnés IN-PLACE (out=vue canvas, comme script.py
    0.27.0 — évite le kernel de recopie du setitem)."""
    def _bp(dv, ia, sa):
        if ip and xp is not np:
            _blend_pre_k(dv, ia, sa, dv)
        else:
            dv[...] = blend_pre(xp, dv, ia, sa)
    def _b(dv, s, aa):
        if ip and xp is not np:
            _blend_k(dv, s, aa, dv)
        else:
            dv[...] = blend(xp, dv, s, aa)
    bx0, by0, bx1, by1, piY, saY, piC, saU, saV = chrome
    a = max(by0, b0); b = min(by1, b1)
    if a < b:
        l0, l1 = a - by0, b - by0
        _bp(cy[a:b, bx0:bx1], piY[l0:l1], saY[l0:l1])
        ca0, cb0 = a // CH, b // CH
        lc0 = ca0 - by0 // CH; lc1 = lc0 + (cb0 - ca0)
        cx0, cx1 = bx0 // CW, bx1 // CW
        _bp(cu[ca0:cb0, cx0:cx1], piC[lc0:lc1], saU[lc0:lc1])
        _bp(cv[ca0:cb0, cx0:cx1], piC[lc0:lc1], saV[lc0:lc1])
    for (bx0, by0, bx1, by1, oy, ou, ovv, oa, oa2) in vu:
        a = max(by0, b0); b = min(by1, b1)
        if a >= b:
            continue
        l0, l1 = a - by0, b - by0
        _b(cy[a:b, bx0:bx1], oy[l0:l1], oa[l0:l1])
        ca0, cb0 = a // CH, b // CH
        lc0 = ca0 - by0 // CH; lc1 = lc0 + (cb0 - ca0)
        cx0, cx1 = bx0 // CW, bx1 // CW
        _b(cu[ca0:cb0, cx0:cx1], ou[lc0:lc1], oa2[lc0:lc1])
        _b(cv[ca0:cb0, cx0:cx1], ovv[lc0:lc1], oa2[lc0:lc1])


def _out_bufs():
    return (np.empty((H, W), DT), np.empty((H // CH, W // CW), DT), np.empty((H // CH, W // CW), DT))


# ─── Mode 1 : CPU slice (référence sémantique) ───────────────────────────────────────────────
def run_cpu_slice(views, chrome, vu, out):
    ov_y, ov_u, ov_v = out
    cy = np.zeros((H, W), DT); cu = np.zeros((H // CH, W // CW), DT); cv = np.zeros((H // CH, W // CW), DT)
    t0 = time.perf_counter(); t_first = None
    for k in range(NB):
        b0, b1 = k * SL, (k + 1) * SL
        for (ty, tu, tv, vy, vh, vx, vw) in views:
            a = max(vy, b0); b = min(vy + vh, b1)
            if a >= b:
                continue
            r0 = a - vy; r1 = r0 + (b - a)
            cy[a:b, vx:vx + vw] = ty[r0:r1]
            ca0, cb0 = a // CH, b // CH
            if cb0 > ca0:
                rc0 = r0 // CH
                cu[ca0:cb0, vx // CW:vx // CW + vw // CW] = tu[rc0:rc0 + (cb0 - ca0)]
                cv[ca0:cb0, vx // CW:vx // CW + vw // CW] = tv[rc0:rc0 + (cb0 - ca0)]
        _blend_band(np, cy, cu, cv, chrome, vu, b0, b1)
        ov_y[b0:b1] = cy[b0:b1]
        ov_u[b0 // CH:b1 // CH] = cu[b0 // CH:b1 // CH]
        ov_v[b0 // CH:b1 // CH] = cv[b0 // CH:b1 // CH]
        if t_first is None:
            t_first = time.perf_counter() - t0
    return time.perf_counter() - t0, t_first


# ─── Mode 2 : GPU whole-frame (pipeline actuel) ──────────────────────────────────────────────
class GpuWhole:
    def __init__(self, chrome, vu):
        n_in = 4 * (H * W + 2 * (H // CH) * (W // CW))
        self.hin = np.frombuffer(cp.cuda.alloc_pinned_memory(n_in), dtype=DT, count=n_in)
        self.din = cp.empty(n_in, dtype=DT)
        n_out = H * W + 2 * (H // CH) * (W // CW)
        self.hout = np.frombuffer(cp.cuda.alloc_pinned_memory(n_out), dtype=DT, count=n_out)
        self.chrome = tuple(chrome[:4]) + tuple(cp.asarray(p) for p in chrome[4:])
        self.vu = [t[:4] + tuple(cp.asarray(p) for p in t[4:]) for t in vu]
        self.cy = cp.zeros((H, W), DT)
        self.cu = cp.zeros((H // CH, W // CW), DT); self.cv = cp.zeros((H // CH, W // CW), DT)

    def run(self, tiles, out):
        ov_y, ov_u, ov_v = out
        t0 = time.perf_counter()
        off = 0; meta = []
        for (sy, su, sv) in tiles:                     # gather hôte → épinglé (comme _place_batch)
            ny, nu = sy.size, su.size
            self.hin[off:off + ny] = sy.ravel()
            self.hin[off + ny:off + ny + nu] = su.ravel()
            self.hin[off + ny + nu:off + ny + 2 * nu] = sv.ravel()
            meta.append(off); off += ny + 2 * nu
        self.din[:off].set(self.hin[:off])             # 1 seul H2D
        gt = []
        for o, (sy, su, sv) in zip(meta, tiles):
            gy = self.din[o:o + sy.size].reshape(sy.shape)
            gu = self.din[o + sy.size:o + sy.size + su.size].reshape(su.shape)
            gv = self.din[o + sy.size + su.size:o + sy.size + 2 * su.size].reshape(sv.shape)
            gt.append((gy, gu, gv))
        for (ti, vy, vh, vx, vw) in LAYOUT:            # place (vues stridées device)
            gy, gu, gv = gt[ti]
            self.cy[vy:vy + vh, vx:vx + vw] = gy[::H // vh, ::W // vw]
            self.cu[vy // CH:(vy + vh) // CH, vx // CW:(vx + vw) // CW] = gu[::H // vh, ::W // vw]
            self.cv[vy // CH:(vy + vh) // CH, vx // CW:(vx + vw) // CW] = gv[::H // vh, ::W // vw]
        _blend_band(cp, self.cy, self.cu, self.cv, self.chrome, self.vu, 0, H)
        of = cp.concatenate([self.cy.ravel(), self.cu.ravel(), self.cv.ravel()])
        of.get(out=self.hout)                          # 1 seul D2H épinglé (sync)
        n = H * W; m = (H // CH) * (W // CW)
        ov_y[...] = self.hout[:n].reshape(H, W)
        ov_u[...] = self.hout[n:n + m].reshape(H // CH, W // CW)
        ov_v[...] = self.hout[n + m:n + 2 * m].reshape(H // CH, W // CW)
        dt = time.perf_counter() - t0
        return dt, dt                                  # 1ʳᵉ bande = trame entière (whole-frame)


# ─── Mode 3 : GPU micro-batch (script.py 0.27.0) ─────────────────────────────────────────────
class GpuBatch:
    def __init__(self, chrome, vu, grp):
        self.grp = grp
        cap = 4 * grp * (SL * W + 2 * (SL // CH) * (W // CW))   # borne large (bandes × tuiles)
        self.hin = np.frombuffer(cp.cuda.alloc_pinned_memory(cap), dtype=DT, count=cap)
        self.din = cp.empty(cap, dtype=DT)

        def _mk(h, w):
            return np.frombuffer(cp.cuda.alloc_pinned_memory(h * w), dtype=DT, count=h * w).reshape(h, w)
        self.hb = [(_mk(grp * SL, W), _mk(grp * SL // CH, W // CW), _mk(grp * SL // CH, W // CW))
                   for _ in (0, 1)]
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.chrome = tuple(chrome[:4]) + tuple(cp.asarray(p) for p in chrome[4:])
        self.vu = [t[:4] + tuple(cp.asarray(p) for p in t[4:]) for t in vu]
        self.cy = cp.zeros((H, W), DT)
        self.cu = cp.zeros((H // CH, W // CW), DT); self.cv = cp.zeros((H // CH, W // CW), DT)

    def run(self, tiles, out):
        ov_y, ov_u, ov_v = out
        grp = self.grp
        pend = [None]                              # (event, ka, kb, bufs) — D2H en vol
        t0 = time.perf_counter(); t_first = [None]

        def flush():
            if pend[0] is None:
                return
            ev, ka, kb, (hy, hu, hv) = pend[0]; pend[0] = None
            ev.synchronize()
            for k in range(ka, kb):                # commits simulés au grain SL
                r = (k - ka) * SL; a = k * SL; b = a + SL
                ov_y[a:b] = hy[r:r + SL]
                ov_u[a // CH:b // CH] = hu[r // CH:r // CH + SL // CH]
                ov_v[a // CH:b // CH] = hv[r // CH:r // CH + SL // CH]
                if t_first[0] is None:
                    t_first[0] = time.perf_counter() - t0

        for k in range(NB):
            b1 = (k + 1) * SL
            # (les attentes get_slice par bande — protocole inchangé — seraient ici ; le banc
            # mesure le RENDU pur, entrées déjà complètes)
            if (k + 1) % grp != 0 and k + 1 < NB:
                continue                                   # lot incomplet : accumule
            B0 = (k - k % grp) * SL
            # gather COALESCÉ au lot, SLAB rangées pleine largeur (copies hôte contiguës par
            # rangée) + décimation COLONNE en VRAM (csx) — comme script.py 0.27.0
            stage = []
            for (ti, vy, vh, vx, vw) in LAYOUT:
                a = max(vy, B0); b = min(vy + vh, b1)
                if a >= b:
                    continue
                sy, su, sv = tiles[ti]
                sty, stx = H // vh, W // vw
                r0 = a - vy; r1 = r0 + (b - a)
                by_ = sy[::sty][r0:r1]                     # rangées décimées, colonnes PLEINES
                bu = bv = None
                ca0, cb0 = a // CH, b // CH
                if cb0 > ca0:
                    rc0 = r0 // CH
                    bu = su[::sty][rc0:rc0 + (cb0 - ca0)]; bv = sv[::sty][rc0:rc0 + (cb0 - ca0)]
                stage.append((by_, bu, bv, a, b, ca0, cb0, vx, vw, vx // CW, stx))
            # H2D groupé du LOT (staging épinglé, 1 memcpy) + placement VRAM (_gpu_place_band)
            off = 0; meta = []
            for (by_, bu_, bv_, a, b, ca0, cb0, vx, vw, cx0, csx) in stage:
                ny = by_.size
                self.hin[off:off + ny].reshape(by_.shape)[...] = by_
                nu = 0
                if bu_ is not None:
                    nu = bu_.size
                    self.hin[off + ny:off + ny + nu].reshape(bu_.shape)[...] = bu_
                    self.hin[off + ny + nu:off + ny + 2 * nu].reshape(bv_.shape)[...] = bv_
                meta.append((off, by_.shape, bu_.shape if bu_ is not None else None,
                             a, b, ca0, cb0, vx, vw, cx0, csx))
                off += ny + 2 * nu
            self.din[:off].set(self.hin[:off])
            for (o, shy, shu, a, b, ca0, cb0, vx, vw, cx0, csx) in meta:
                ny = shy[0] * shy[1]
                self.cy[a:b, vx:vx + vw] = self.din[o:o + ny].reshape(shy)[:, ::csx]
                if shu is not None:
                    nu = shu[0] * shu[1]
                    self.cu[ca0:cb0, cx0:cx0 + vw // CW] = self.din[o + ny:o + ny + nu].reshape(shu)[:, ::csx]
                    self.cv[ca0:cb0, cx0:cx0 + vw // CW] = self.din[o + ny + nu:o + ny + 2 * nu].reshape(shu)[:, ::csx]
            _blend_band(cp, self.cy, self.cu, self.cv, self.chrome, self.vu, B0, b1, ip=True)
            # D2H du lot RECOUVERT : event compose → stream dédié → memcpyAsync ; flush du lot j-1
            hy, hu, hv = self.hb[(k // grp) % 2]
            evc = cp.cuda.Event(block=False, disable_timing=True)
            evc.record()
            self.stream.wait_event(evc)
            flush()                                        # publie le lot précédent (D2H recouvert)
            rows = b1 - B0; wc = W // CW
            dth = cp.cuda.runtime.memcpyDeviceToHost
            cp.cuda.runtime.memcpyAsync(hy.ctypes.data, int(self.cy.data.ptr) + B0 * W,
                                        rows * W, dth, self.stream.ptr)
            cp.cuda.runtime.memcpyAsync(hu.ctypes.data, int(self.cu.data.ptr) + (B0 // CH) * wc,
                                        (rows // CH) * wc, dth, self.stream.ptr)
            cp.cuda.runtime.memcpyAsync(hv.ctypes.data, int(self.cv.data.ptr) + (B0 // CH) * wc,
                                        (rows // CH) * wc, dth, self.stream.ptr)
            evd = cp.cuda.Event(block=False, disable_timing=True)
            evd.record(self.stream)
            pend[0] = (evd, B0 // SL, k + 1, (hy, hu, hv))
        flush()                                            # dernier lot
        return time.perf_counter() - t0, t_first[0]


def _pcts(v):
    s = sorted(v)
    return {"p50": s[len(s) // 2] * 1000, "p90": s[int(len(s) * 0.9)] * 1000,
            "mean": statistics.fmean(v) * 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--grp", type=int, default=4, help="gpu_batch_bands (bandes de 36 l par lot)")
    ap.add_argument("--pause", type=float, default=0.015, help="pause entre trames (T4 partagée)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    tiles = make_inputs(4)
    chrome = make_chrome()
    vu = make_vu_tiles()
    views = _views(tiles)

    # ── Équivalence octet (mêmes entrées → même sortie, tolérance 0) ──
    out_cpu, out_whole, out_batch = _out_bufs(), _out_bufs(), _out_bufs()
    run_cpu_slice(views, chrome, vu, out_cpu)
    res = {"grp": args.grp, "frames": args.frames}
    if cp is None:
        print(f"cupy indisponible ({_CP_ERR}) — mode CPU seul")
        modes = {"cpu_slice": lambda o: run_cpu_slice(views, chrome, vu, o)}
        res["equal"] = None
    else:
        gw = GpuWhole(chrome, vu)
        gb = GpuBatch(chrome, vu, args.grp)
        gw.run(tiles, out_whole)
        gb.run(tiles, out_batch)
        eq_w = all(np.array_equal(a, b) for a, b in zip(out_cpu, out_whole))
        eq_b = all(np.array_equal(a, b) for a, b in zip(out_cpu, out_batch))
        res["equal_whole_vs_cpu"] = eq_w
        res["equal_batch_vs_cpu"] = eq_b
        print(f"équivalence octet : whole=={'OK' if eq_w else 'ÉCHEC'}  batch=={'OK' if eq_b else 'ÉCHEC'}")
        if not (eq_w and eq_b):
            for name, o in (("whole", out_whole), ("batch", out_batch)):
                for pl, a, b in zip("YUV", out_cpu, o):
                    d = int((a != b).sum())
                    if d:
                        print(f"  {name} plan {pl}: {d} octets diffèrent")
        modes = {"cpu_slice": lambda o: run_cpu_slice(views, chrome, vu, o),
                 "gpu_whole": lambda o: gw.run(tiles, o),
                 "gpu_batch": lambda o: gb.run(tiles, o)}

    # ── Chrono (T4 partagée : pauses intercalées, rapporter p90) ──
    for name, fn in modes.items():
        out = _out_bufs()
        ft, fb = [], []
        for i in range(args.warmup + args.frames):
            t, tf = fn(out)
            if i >= args.warmup:
                ft.append(t); fb.append(tf)
            time.sleep(args.pause)
        res[name] = {"frame_ms": _pcts(ft), "first_band_ms": _pcts(fb)}
        f, b = res[name]["frame_ms"], res[name]["first_band_ms"]
        print(f"{name:10s} trame p50 {f['p50']:6.2f} ms  p90 {f['p90']:6.2f} ms | "
              f"1ʳᵉ bande p50 {b['p50']:6.2f} ms  p90 {b['p90']:6.2f} ms")
        time.sleep(0.5)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"→ {args.json}")


if __name__ == "__main__":
    main()
