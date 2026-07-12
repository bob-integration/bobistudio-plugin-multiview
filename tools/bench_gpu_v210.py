#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 BOBI SAS, France
# Auteur : Cyril Mazouer, pour le compte de BOBI SAS

"""
bench_gpu_v210 — chiffrage GPU (T4) de la conversion v210 ↔ planar 4:2:2 (donnée interop MXL).

Contexte : le bus MXL est en planar maison (8 bits = Y+U+V octets, 4,15 Mo/trame 1080p) ;
l'interop multi-éditeurs pousserait vers v210 (10 bits packés, 5,53 Mo/trame). Références CPU
MESURÉES (Gen10/6240R, cf. mémoire planar-v210-cpu-measure) : unpack v210→planar ≈ 8,9 ms,
pack ≈ 9,0 ms, aller-retour ≈ 17,8 ms par trame 1080p. Ce banc mesure la même conversion sur
GPU (kernels cupy) + les transferts PCIe épinglés, en PLEINE TRAME et PAR LOT DE BANDES
(gpu_batch_bands=4 → 144 lignes, le grain GPU du multiview slice 0.27.0), puis synthétise le
coût TOTAL par étage GPU « bus v210 » vs « bus planar ».

v210 (SMPTE, 4:2:2 10 bits) : 6 pixels par groupe de 4 mots de 32 bits ; chaque mot porte
3 échantillons de 10 bits (bits 0-9 / 10-19 / 20-29, bits 30-31 nuls) :
    mot 0 : U0  | Y0<<10 | V0<<20
    mot 1 : Y1  | U1<<10 | Y2<<20
    mot 2 : V1  | Y3<<10 | U2<<20
    mot 3 : Y4  | V2<<10 | Y5<<20
Stride ligne = ceil(w/6)*16 octets, aligné 128 (1920 → 5120 octets, déjà aligné).
VALIDATION : pack(unpack10(x)) == x, bit-exact, sur un payload aléatoire (bits 30-31 à 0).

Usage (docker run autonome sur un nœud GPU, image bobi-compute-gpu) :
  docker run --rm --gpus all --entrypoint python3 \
    -v /opt/bobistudio/plugins/multiview/tools:/b bobi-compute-gpu:0.1 \
    /b/bench_gpu_v210.py --reps 300 [--json /b/v210_result.json]
"""

import argparse
import json
import time

import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    raise SystemExit("cupy requis (image bobi-compute-gpu)")

# ─── kernels v210 (1 thread = 1 groupe de 4 mots / 6 pixels) ─────────────────────────────

_UNPACK_SRC = r"""
extern "C" __global__
void v210_unpack_@SUF@(const unsigned int* __restrict__ src,
                         @T@* __restrict__ y, @T@* __restrict__ u, @T@* __restrict__ v,
                         int groups_per_line, int words_per_line, int lines)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)groups_per_line * lines;
    if (i >= total) return;
    int line = i / groups_per_line, g = i % groups_per_line;
    const unsigned int* w = src + (long)line * words_per_line + (long)g * 4;
    unsigned int w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    long yb = (long)line * (groups_per_line * 6) + (long)g * 6;
    long cb = (long)line * (groups_per_line * 3) + (long)g * 3;
    y[yb+0] = (@T@)(((w0 >> 10) & 0x3FF) @SH@);
    y[yb+1] = (@T@)(( w1        & 0x3FF) @SH@);
    y[yb+2] = (@T@)(((w1 >> 20) & 0x3FF) @SH@);
    y[yb+3] = (@T@)(((w2 >> 10) & 0x3FF) @SH@);
    y[yb+4] = (@T@)(( w3        & 0x3FF) @SH@);
    y[yb+5] = (@T@)(((w3 >> 20) & 0x3FF) @SH@);
    u[cb+0] = (@T@)(( w0        & 0x3FF) @SH@);
    u[cb+1] = (@T@)(((w1 >> 10) & 0x3FF) @SH@);
    u[cb+2] = (@T@)(((w2 >> 20) & 0x3FF) @SH@);
    v[cb+0] = (@T@)(((w0 >> 20) & 0x3FF) @SH@);
    v[cb+1] = (@T@)(( w2        & 0x3FF) @SH@);
    v[cb+2] = (@T@)(((w3 >> 10) & 0x3FF) @SH@);
}
"""

_PACK_SRC = r"""
extern "C" __global__
void v210_pack(const unsigned short* __restrict__ y, const unsigned short* __restrict__ u,
               const unsigned short* __restrict__ v, unsigned int* __restrict__ dst,
               int groups_per_line, int words_per_line, int lines)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)groups_per_line * lines;
    if (i >= total) return;
    int line = i / groups_per_line, g = i % groups_per_line;
    long yb = (long)line * (groups_per_line * 6) + (long)g * 6;
    long cb = (long)line * (groups_per_line * 3) + (long)g * 3;
    unsigned int* w = dst + (long)line * words_per_line + (long)g * 4;
    w[0] = (u[cb+0] & 0x3FFu) | ((unsigned int)(y[yb+0] & 0x3FFu) << 10) | ((unsigned int)(v[cb+0] & 0x3FFu) << 20);
    w[1] = (y[yb+1] & 0x3FFu) | ((unsigned int)(u[cb+1] & 0x3FFu) << 10) | ((unsigned int)(y[yb+2] & 0x3FFu) << 20);
    w[2] = (v[cb+1] & 0x3FFu) | ((unsigned int)(y[yb+3] & 0x3FFu) << 10) | ((unsigned int)(u[cb+2] & 0x3FFu) << 20);
    w[3] = (y[yb+4] & 0x3FFu) | ((unsigned int)(v[cb+2] & 0x3FFu) << 10) | ((unsigned int)(y[yb+5] & 0x3FFu) << 20);
}
"""

def _tpl(src, suf, t, sh):
    return src.replace("@SUF@", suf).replace("@T@", t).replace("@SH@", sh)


_k_unpack10 = cp.RawKernel(_tpl(_UNPACK_SRC, "10", "unsigned short", ""), "v210_unpack_10")
_k_unpack8 = cp.RawKernel(_tpl(_UNPACK_SRC, "8", "unsigned char", ">> 2"), "v210_unpack_8")
_k_pack = cp.RawKernel(_PACK_SRC, "v210_pack")

_BLOCK = 256


def _launch(kern, args, n_threads):
    grid = ((n_threads + _BLOCK - 1) // _BLOCK,)
    kern(grid, (_BLOCK,), args)


def v210_words_per_line(w):
    return ((w + 5) // 6) * 4          # 1920 → 1280 mots = 5120 octets (aligné 128)


def unpack10(src_words, w, lines, y, u, v):
    gpl = (w + 5) // 6
    _launch(_k_unpack10, (src_words, y, u, v,
                          np.int32(gpl), np.int32(v210_words_per_line(w)), np.int32(lines)),
            gpl * lines)


def unpack8(src_words, w, lines, y, u, v):
    gpl = (w + 5) // 6
    _launch(_k_unpack8, (src_words, y, u, v,
                         np.int32(gpl), np.int32(v210_words_per_line(w)), np.int32(lines)),
            gpl * lines)


def pack10(y, u, v, dst_words, w, lines):
    gpl = (w + 5) // 6
    _launch(_k_pack, (y, u, v, dst_words,
                      np.int32(gpl), np.int32(v210_words_per_line(w)), np.int32(lines)),
            gpl * lines)


# ─── mesures ──────────────────────────────────────────────────────────────────────────────

def _pcts(xs):
    a = np.sort(np.asarray(xs, dtype=np.float64))
    return float(np.percentile(a, 50)), float(np.percentile(a, 90))


def time_gpu(fn, reps, warmup=30):
    """Kernel pur : events CUDA (médiane/p90 ms)."""
    s, e = cp.cuda.Event(), cp.cuda.Event()
    for _ in range(warmup):
        fn()
    cp.cuda.runtime.deviceSynchronize()
    out = []
    for _ in range(reps):
        s.record()
        fn()
        e.record()
        e.synchronize()
        out.append(cp.cuda.get_elapsed_time(s, e))
    return _pcts(out)


def time_host(fn, reps, warmup=30):
    """Chemin mixte hôte (H2D/D2H épinglés) : perf_counter + sync."""
    for _ in range(warmup):
        fn()
    cp.cuda.runtime.deviceSynchronize()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        cp.cuda.runtime.deviceSynchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return _pcts(out)


def pinned(nbytes):
    mem = cp.cuda.alloc_pinned_memory(nbytes)
    return np.frombuffer(mem, dtype=np.uint8, count=nbytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--band-lines", type=int, default=144,
                    help="lot de bandes (gpu_batch_bands=4 × slice_lines=36)")
    ap.add_argument("--json", help="dump JSON des résultats")
    args = ap.parse_args()
    W, H, REPS, BL = args.width, args.height, args.reps, args.band_lines
    assert W % 6 == 0, "largeur multiple de 6 requise (v210 sans tail)"

    wpl = v210_words_per_line(W)
    v210_bytes = wpl * 4 * H
    planar8_bytes = W * H * 2                    # bus planar maison 8 bits (Y+U+V, 4:2:2)
    planar10_bytes = W * H * 3                   # hypothèse « planar 10 bits » ≈ 6,2 Mo (énoncé)
    n_lots = (H + BL - 1) // BL

    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print(f"GPU: {props['name'].decode()} — {W}x{H}, reps={REPS}, lot={BL} lignes ({n_lots} lots/trame)")
    print(f"tailles/trame : v210 {v210_bytes/1e6:.2f} Mo | planar8 {planar8_bytes/1e6:.2f} Mo "
          f"| planar10 {planar10_bytes/1e6:.2f} Mo")

    # ── VALIDATION aller-retour pack(unpack10(x)) == x (payload aléatoire, bits 30-31 nuls) ──
    rng = np.random.default_rng(1234)
    host_v210 = rng.integers(0, 1 << 30, size=wpl * H, dtype=np.uint32)
    d_src = cp.asarray(host_v210)
    d_y = cp.empty(W * H, dtype=cp.uint16)
    d_u = cp.empty(W * H // 2, dtype=cp.uint16)
    d_v = cp.empty(W * H // 2, dtype=cp.uint16)
    d_dst = cp.empty_like(d_src)
    unpack10(d_src, W, H, d_y, d_u, d_v)
    pack10(d_y, d_u, d_v, d_dst, W, H)
    ok = bool((d_src == d_dst).all())
    print(f"VALIDATION pack(unpack10(x)) == x : {'OK (bit-exact)' if ok else 'ÉCHEC'}")
    assert ok, "aller-retour v210 non bit-exact — kernel faux"
    # cohérence 8 bits : unpack8 == unpack10 >> 2
    d_y8 = cp.empty(W * H, dtype=cp.uint8)
    d_u8 = cp.empty(W * H // 2, dtype=cp.uint8)
    d_v8 = cp.empty(W * H // 2, dtype=cp.uint8)
    unpack8(d_src, W, H, d_y8, d_u8, d_v8)
    ok8 = bool((d_y8 == (d_y >> 2).astype(cp.uint8)).all()
               and (d_u8 == (d_u >> 2).astype(cp.uint8)).all()
               and (d_v8 == (d_v >> 2).astype(cp.uint8)).all())
    print(f"VALIDATION unpack8 == unpack10>>2 : {'OK' if ok8 else 'ÉCHEC'}")
    assert ok8

    res = {"gpu": props["name"].decode(), "width": W, "height": H, "reps": REPS,
           "band_lines": BL, "lots_per_frame": n_lots,
           "bytes": {"v210": v210_bytes, "planar8": planar8_bytes, "planar10": planar10_bytes}}

    # ── K1 : kernels pleine trame ──────────────────────────────────────────────────────────
    print("\n── kernels (events CUDA, ms) ──")
    for name, fn in [
        ("unpack v210→planar10 (trame)", lambda: unpack10(d_src, W, H, d_y, d_u, d_v)),
        ("unpack v210→planar8  (trame)", lambda: unpack8(d_src, W, H, d_y8, d_u8, d_v8)),
        ("pack   planar10→v210 (trame)", lambda: pack10(d_y, d_u, d_v, d_dst, W, H)),
    ]:
        p50, p90 = time_gpu(fn, REPS)
        res[name] = {"p50_ms": round(p50, 3), "p90_ms": round(p90, 3)}
        print(f"  {name}: p50 {p50:.3f}  p90 {p90:.3f}")

    # kernels PAR LOT de BL lignes (même trame, lancement par lot → ×n_lots par trame)
    def _per_lot(kern_fn):
        def run():
            for lot in range(n_lots):
                l0 = lot * BL
                nl = min(BL, H - l0)
                src = d_src[l0 * wpl:]
                y = d_y[l0 * W:]
                u = d_u[l0 * W // 2:]
                v = d_v[l0 * W // 2:]
                kern_fn(src, nl, y, u, v)
        return run

    p50, p90 = time_gpu(_per_lot(lambda s, nl, y, u, v: unpack10(s, W, nl, y, u, v)), REPS)
    res["unpack10 par lots (trame = n lots)"] = {"p50_ms": round(p50, 3), "p90_ms": round(p90, 3)}
    print(f"  unpack v210→planar10 par lots de {BL} l ({n_lots} lancements): p50 {p50:.3f}  p90 {p90:.3f}")

    def _pack_lots():
        for lot in range(n_lots):
            l0 = lot * BL
            nl = min(BL, H - l0)
            pack10(d_y[l0 * W:], d_u[l0 * W // 2:], d_v[l0 * W // 2:],
                   d_dst[l0 * wpl:], W, nl)

    p50, p90 = time_gpu(_pack_lots, REPS)
    res["pack10 par lots (trame = n lots)"] = {"p50_ms": round(p50, 3), "p90_ms": round(p90, 3)}
    print(f"  pack planar10→v210 par lots de {BL} l ({n_lots} lancements): p50 {p50:.3f}  p90 {p90:.3f}")

    # ── K2 : transferts épinglés (trame et par lot) ────────────────────────────────────────
    print("\n── transferts épinglés (perf_counter+sync, ms) ──")
    stream = cp.cuda.Stream(non_blocking=True)
    tr = {}
    for label, nbytes in [("v210", v210_bytes), ("planar8", planar8_bytes),
                          ("planar10", planar10_bytes)]:
        h = pinned(nbytes)
        h[:] = 7
        d = cp.empty(nbytes, dtype=cp.uint8)
        lot_b = nbytes // n_lots

        def h2d_full(d=d, h=h, n=nbytes):
            d.data.copy_from_host(h.ctypes.data, n)

        def d2h_full(d=d, h=h, n=nbytes):
            d.data.copy_to_host(h.ctypes.data, n)

        def h2d_lots(d=d, h=h, n=nbytes, lb=lot_b):
            for i in range(n_lots):
                off = i * lb
                sz = lb if i < n_lots - 1 else n - off
                cp.cuda.runtime.memcpyAsync(d.data.ptr + off, h.ctypes.data + off, sz,
                                            cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
            stream.synchronize()

        r = {}
        for k, fn in [("H2D trame", h2d_full), ("D2H trame", d2h_full), ("H2D par lots", h2d_lots)]:
            p50, p90 = time_host(fn, REPS)
            r[k] = {"p50_ms": round(p50, 3), "p90_ms": round(p90, 3)}
            print(f"  {label:9s} {k}: p50 {r[k]['p50_ms']:.3f}  p90 {r[k]['p90_ms']:.3f}")
        tr[label] = r
    res["transferts"] = tr

    # ── K3 : synthèse coût par étage GPU (p50, ms/trame) ───────────────────────────────────
    # « bus v210 »   : H2D v210 + unpack10 + [compose, connue par ailleurs] + pack10 + D2H v210
    # « bus planar » : H2D planar8 + [compose] + D2H planar8   (chemin actuel)
    g = lambda k: res[k]["p50_ms"]
    t = lambda l, k: tr[l][k]["p50_ms"]
    v210_stage = (t("v210", "H2D trame") + g("unpack v210→planar10 (trame)")
                  + g("pack   planar10→v210 (trame)") + t("v210", "D2H trame"))
    planar_stage = t("planar8", "H2D trame") + t("planar8", "D2H trame")
    res["synthese"] = {
        "surcout_etage_bus_v210_vs_planar8_ms": round(v210_stage - planar_stage, 3),
        "etage_v210_hors_compose_ms": round(v210_stage, 3),
        "etage_planar8_hors_compose_ms": round(planar_stage, 3),
        "cpu_reference_ms": {"unpack": 8.9, "pack": 9.0, "aller_retour": 17.8},
    }
    print("\n── synthèse (p50, ms/trame, hors compose — la compose est identique aux 2 bus) ──")
    print(f"  étage « bus v210 »   (H2D v210 + unpack + pack + D2H v210) : {v210_stage:.3f} ms")
    print(f"  étage « bus planar » (H2D planar8 + D2H planar8)           : {planar_stage:.3f} ms")
    print(f"  SURCOÛT v210 par étage GPU : {v210_stage - planar_stage:+.3f} ms "
          f"(CPU : +17,8 ms aller-retour)")
    ar_gpu = g("unpack v210→planar10 (trame)") + g("pack   planar10→v210 (trame)")
    print(f"  conversion seule (GPU) : unpack {g('unpack v210→planar10 (trame)'):.3f} + "
          f"pack {g('pack   planar10→v210 (trame)'):.3f} = {ar_gpu:.3f} ms "
          f"(CPU mesuré : 8,9 + 9,0 = 17,8 ms → ×{17.8/max(ar_gpu,1e-9):.0f})")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nJSON → {args.json}")


if __name__ == "__main__":
    main()
