#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 BOBI SAS, France
"""Banc GATE GO/NO-GO du slice GPU multiview (TISSU_SLICE_GPU.md §a) — dl360-2/T4.

Autonome : cupy + numpy seulement (aucun MXL/orchestrateur). Mesure, pour une trame
composite (défaut 1080p 4:2:2 8 bits, 4 tuiles d'entrée 1080p) :

  M1  H2D entrées : groupé pageable / groupé épinglé (référence) / bandé épinglé sync
      (36..540 lignes) / bandé épinglé 2 streams (plafond pipeline).
  M2  D2H sortie  : groupé épinglé (référence) / bandé épinglé sync / bandé recouvert
      (stream + commit décalé d'une bande, simulé).
  M3  compose par bande vs pleine trame (placement + blend chrome + blends VU) :
      surcoût de lancement kernels ×N bandes.
  M4  bout-en-bout simulé par taille de bande : gather hôte→épinglé, H2D, place, blends,
      D2H, écriture mmap simulée — temps trame TOTAL + latence 1ʳᵉ bande de sortie.

VERDICT (budget T=20 ms @50p, cf. TISSU_SLICE_GPU.md) :
  GO fin (36)     : surcoût M1+M2 bandé(36) ≤ +1,5 ms ET M4(36) trame ≤ 12 ms ET 1ʳᵉ ≤ 3 ms
  GO méga-bandes  : ∃ b ∈ {135,270,540} : surcoût ≤ +0,5 ms ET trame ≤ 12 ms ET 1ʳᵉ ≤ 8 ms
  NO-GO           : sinon (GPU reste whole-frame ; nœuds tissu GPU → force_cpu + slice CPU)

Usage :
  docker run --rm --gpus all -v .../multiview/tools:/b bobi-compute-gpu \
      python3 /b/bench_gpu_slice_gate.py --json /b/gate_result.json
"""
import argparse, json, statistics, time

import numpy as np
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception as e:
    raise SystemExit(f"cupy/GPU indisponible : {e} (lancer avec --gpus dans l'image GPU)")

P = argparse.ArgumentParser()
P.add_argument("--width", type=int, default=1920)
P.add_argument("--height", type=int, default=1080)
P.add_argument("--tiles", type=int, default=4, help="tuiles d'entrée (sources 1080p)")
P.add_argument("--fps", type=float, default=50.0)
P.add_argument("--reps", type=int, default=300)
P.add_argument("--warmup", type=int, default=50)
P.add_argument("--bands", type=int, nargs="+", default=[36, 72, 135, 270, 540])
P.add_argument("--json", type=str, default=None, help="chemin du résultat JSON")
A = P.parse_args()

W, H, NT = A.width, A.height, A.tiles
T_MS = 1000.0 / A.fps
CW, CH = 2, 1                     # 4:2:2 : chroma demi-largeur, pleine hauteur
DT = np.uint8
Y_IN, C_IN = (W, H), (W // CW, H)   # entrées = même format que la sortie (1080p)
FRAME_B = W * H + 2 * (W // CW) * H  # octets/trame 4:2:2 8b

dev = cp.cuda.Device(0)
name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8", "replace")
print(f"GPU: {name} — trame {W}x{H} 4:2:2 8b = {FRAME_B/1e6:.2f} Mo ; {NT} tuiles d'entrée ; "
      f"T={T_MS:.1f} ms ({A.fps} fps) ; reps={A.reps}")


def med_p90(samples_ms):
    s = sorted(samples_ms)
    return (statistics.median(s), s[int(len(s) * 0.9)])


def bench(fn, reps=None, warmup=None):
    """fn() = une itération complète (doit inclure sa propre synchro). → (med, p90) ms."""
    reps = reps or A.reps; warmup = warmup or A.warmup
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return med_p90(out)


def pinned(n):
    return np.frombuffer(cp.cuda.alloc_pinned_memory(n), dtype=DT, count=n)


R = {"gpu": name, "width": W, "height": H, "tiles": NT, "fps": A.fps, "m1": {}, "m2": {},
     "m3": {}, "m4": {}}

# ── données : NT tuiles source pleine résolution (hôte), planaire Y,U,V concaténé ─────────────
rng = np.random.default_rng(42)
srcs = [rng.integers(0, 255, FRAME_B, dtype=DT) for _ in range(NT)]
total_in = NT * FRAME_B
h_page = np.concatenate(srcs)                       # staging pageable (anti-référence)
h_pin = pinned(total_in); h_pin[:] = h_page         # staging épinglé
d_in = cp.empty(total_in, dtype=DT)

# ── M1 : H2D entrées ──────────────────────────────────────────────────────────────────────────
print("\nM1 — H2D entrées (%.1f Mo)" % (total_in / 1e6))

def m1_case(label, fn):
    med, p90 = bench(fn)
    R["m1"][label] = {"med": med, "p90": p90}
    print(f"  {label:34s} {med:7.3f} ms  (p90 {p90:.3f})")

m1_case("groupé pageable (anti-réf)", lambda: (d_in.set(h_page), dev.synchronize()))
m1_case("groupé épinglé 1x (RÉFÉRENCE)", lambda: (d_in.set(h_pin), dev.synchronize()))

# bandé : par bande de sortie, chaque tuile contribue ~band_lines lignes source (entrées 1:1
# avec la sortie ici = pire cas d'octets ; les vraies tuiles réduites transfèrent MOINS).
for bl in A.bands:
    nb = H // bl
    if H % bl:
        continue
    per_band = total_in // nb

    def h2d_banded(nb=nb, per=per_band):
        for k in range(nb):
            d_in[k * per:(k + 1) * per].set(h_pin[k * per:(k + 1) * per])
        dev.synchronize()
    m1_case(f"bandé épinglé sync {bl}l ({nb}x)", h2d_banded)

    s0, s1 = cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)

    def h2d_banded_2s(nb=nb, per=per_band, s0=s0, s1=s1):
        for k in range(nb):
            with (s0 if k % 2 == 0 else s1):
                d_in[k * per:(k + 1) * per].set(h_pin[k * per:(k + 1) * per])
        s0.synchronize(); s1.synchronize()
    m1_case(f"bandé épinglé 2 streams {bl}l", h2d_banded_2s)

# ── M2 : D2H sortie ───────────────────────────────────────────────────────────────────────────
print("\nM2 — D2H sortie (%.1f Mo)" % (FRAME_B / 1e6))
d_out = cp.asarray(rng.integers(0, 255, FRAME_B, dtype=DT))
ho_pin = pinned(FRAME_B)
shm_sim = np.empty(FRAME_B, dtype=DT)               # mmap simulé (pageable, comme le grain)

def m2_case(label, fn):
    med, p90 = bench(fn)
    R["m2"][label] = {"med": med, "p90": p90}
    print(f"  {label:34s} {med:7.3f} ms  (p90 {p90:.3f})")

m2_case("groupé épinglé 1x (RÉFÉRENCE)",
        lambda: (d_out.get(out=ho_pin), shm_sim.__setitem__(slice(None), ho_pin)))

for bl in A.bands:
    nb = H // bl
    if H % bl:
        continue
    per = FRAME_B // nb

    def d2h_banded(nb=nb, per=per):
        for k in range(nb):
            d_out[k * per:(k + 1) * per].get(out=ho_pin[k * per:(k + 1) * per])
            shm_sim[k * per:(k + 1) * per] = ho_pin[k * per:(k + 1) * per]   # + commit (noop)
    m2_case(f"bandé épinglé sync {bl}l ({nb}x)", d2h_banded)

    st = cp.cuda.Stream(non_blocking=True)
    evs = [cp.cuda.Event() for _ in range(nb)]
    _D2H = cp.cuda.runtime.memcpyDeviceToHost

    def d2h_overlap(nb=nb, per=per, st=st, evs=evs):
        # recouvrement profondeur 1 : D2H bande k async (memcpyAsync bas niveau : dispo sur
        # toutes les versions cupy, contrairement à get(blocking=False)), copie/commit de la
        # bande k-1 pendant ce temps — simule le commit décalé d'une bande (§c).
        for k in range(nb):
            cp.cuda.runtime.memcpyAsync(
                ho_pin[k * per:(k + 1) * per].ctypes.data,
                d_out[k * per:(k + 1) * per].data.ptr, per, _D2H, st.ptr)
            evs[k].record(st)
            if k > 0:
                evs[k - 1].synchronize()
                shm_sim[(k - 1) * per:k * per] = ho_pin[(k - 1) * per:k * per]
        evs[nb - 1].synchronize()
        shm_sim[(nb - 1) * per:] = ho_pin[(nb - 1) * per:]
    m2_case(f"bandé recouvert stream {bl}l", d2h_overlap)

# ── M3 : compose par bande vs pleine trame (lancements kernels) ──────────────────────────────
print("\nM3 — compose (place + blend chrome + blends VU) : pleine trame vs par bande")
cy = cp.zeros((H, W), dtype=DT)
cu = cp.full((H, W // CW), 128, dtype=DT); cv = cp.full((H, W // CW), 128, dtype=DT)
# chrome plein écran pré-calculé (inv_a, src_a) + 4 tuiles VU 160x400
inv_a = cp.asarray(rng.integers(0, 255, (H, W), dtype=np.uint16))
src_a = cp.asarray(rng.integers(0, 255, (H, W), dtype=np.uint16)) * 255
vu = [(rng.integers(0, W - 160), rng.integers(0, H - 400)) for _ in range(4)]
vuy = cp.asarray(rng.integers(0, 255, (400, 160), dtype=DT))
vua = cp.asarray(rng.integers(0, 255, (400, 160), dtype=np.uint32))
d_tiles = [cp.asarray(s[:W * H].reshape(H, W)) for s in srcs]   # tuiles déjà en VRAM
qh, qw = H // 2, W // 2
geo = [(0, 0), (0, qw), (qh, 0), (qh, qw)][:NT]


def blend_band(a, b):
    cy[a:b] = ((cy[a:b].astype(np.uint16) * inv_a[a:b] + src_a[a:b]) // 255).astype(DT)
    for (vx, vy) in vu:
        ya, yb = max(vy, a), min(vy + 400, b)
        if ya < yb:
            r0 = ya - vy; r1 = yb - vy
            reg = cy[ya:yb, vx:vx + 160].astype(np.uint32)
            cy[ya:yb, vx:vx + 160] = ((reg * (255 - vua[r0:r1]) + vuy[r0:r1] * vua[r0:r1]) // 255).astype(DT)


def place_band(a, b):
    for ti, (gy, gx) in enumerate(geo):
        ta, tb = max(gy, a), min(gy + qh, b)
        if ta < tb:
            cy[ta:tb, gx:gx + qw] = d_tiles[ti][(ta - gy) * 2:(tb - gy) * 2:2, ::2]


def m3_case(label, fn):
    med, p90 = bench(fn)
    R["m3"][label] = {"med": med, "p90": p90}
    print(f"  {label:34s} {med:7.3f} ms  (p90 {p90:.3f})")

m3_case("pleine trame (RÉFÉRENCE)", lambda: (place_band(0, H), blend_band(0, H), dev.synchronize()))
for bl in A.bands:
    if H % bl:
        continue

    def compose_banded(bl=bl):
        for k in range(H // bl):
            place_band(k * bl, (k + 1) * bl); blend_band(k * bl, (k + 1) * bl)
        dev.synchronize()
    m3_case(f"par bande {bl}l ({H//bl}x)", compose_banded)

# ── M4 : bout-en-bout simulé par taille de bande ─────────────────────────────────────────────
print("\nM4 — bout-en-bout par bande (gather→épinglé, H2D, place, blends, D2H, mmap) ; "
      "latence 1ʳᵉ bande sortie entre []")
in_pin = pinned(total_in)
d_band = cp.empty(total_in, dtype=DT)
srcs2d = [s[:W * H].reshape(H, W) for s in srcs]     # vues hôte (gather = copie vers épinglé)


def m4_case(bl):
    nb = H // bl
    firsts = []

    def frame():
        t0 = time.perf_counter()
        for k in range(nb):
            a, b = k * bl, (k + 1) * bl
            off = 0
            for ti, (gy, gx) in enumerate(geo):      # gather hôte → épinglé (rangées de la bande)
                ta, tb = max(gy, a), min(gy + qh, b)
                if ta < tb:
                    rows = srcs2d[ti][(ta - gy) * 2:(tb - gy) * 2]   # 2× lignes source (downscale 2:1)
                    in_pin[off:off + rows.size].reshape(rows.shape)[...] = rows
                    off += rows.size
            if off:
                d_band[:off].set(in_pin[:off])       # 1 H2D groupé par bande
            place_band(a, b); blend_band(a, b)       # compose VRAM
            per = FRAME_B // nb                      # D2H bande (Y + U + V) + mmap simulé
            ny, nc = (b - a) * W, (b - a) * (W // CW)
            cy[a:b].get(out=ho_pin[:ny].reshape(b - a, W))
            cu[a:b].get(out=ho_pin[ny:ny + nc].reshape(b - a, W // CW))
            cv[a:b].get(out=ho_pin[ny + nc:ny + 2 * nc].reshape(b - a, W // CW))
            shm_sim[k * per:k * per + ny + 2 * nc] = ho_pin[:ny + 2 * nc]
            if k == 0:
                firsts.append((time.perf_counter() - t0) * 1e3)
    med, p90 = bench(frame)
    fmed = statistics.median(firsts[-A.reps:])
    R["m4"][str(bl)] = {"med": med, "p90": p90, "first_band_ms": fmed}
    print(f"  bande {bl:4d}l ({nb:2d}x)              {med:7.3f} ms  (p90 {p90:.3f})  [1ʳᵉ {fmed:.3f} ms]")
    return med, fmed

# référence whole-frame simulée (mêmes étapes, 1 seule « bande » = trame)
wf_med, _ = m4_case(H)
for bl in A.bands:
    if H % bl == 0 and bl != H:
        m4_case(bl)

# ── VERDICT ───────────────────────────────────────────────────────────────────────────────────
ref_h2d = R["m1"]["groupé épinglé 1x (RÉFÉRENCE)"]["med"]
ref_d2h = R["m2"]["groupé épinglé 1x (RÉFÉRENCE)"]["med"]


def surcout(bl):
    h = R["m1"].get(f"bandé épinglé sync {bl}l ({H//bl}x)", {}).get("med")
    d = R["m2"].get(f"bandé épinglé sync {bl}l ({H//bl}x)", {}).get("med")
    if h is None or d is None:
        return None
    return (h - ref_h2d) + (d - ref_d2h)

verdict, why = "NO-GO", []
m4_36 = R["m4"].get("36")
sc36 = surcout(36)
if m4_36 and sc36 is not None and sc36 <= 1.5 and m4_36["med"] <= 12.0 and m4_36["first_band_ms"] <= 3.0:
    verdict = "GO-FIN-36"
    why.append(f"36l : surcoût transferts {sc36:+.2f} ms, trame {m4_36['med']:.2f} ms, "
               f"1ʳᵉ bande {m4_36['first_band_ms']:.2f} ms")
else:
    if m4_36 and sc36 is not None:
        why.append(f"36l REJETÉ : surcoût {sc36:+.2f} ms (≤1,5 ?), trame {m4_36['med']:.2f} ms "
                   f"(≤12 ?), 1ʳᵉ {m4_36['first_band_ms']:.2f} ms (≤3 ?)")
    for bl in (135, 270, 540):
        m4b = R["m4"].get(str(bl)); sc = surcout(bl)
        if m4b and sc is not None and sc <= 0.5 and m4b["med"] <= 12.0 and m4b["first_band_ms"] <= 8.0:
            verdict = f"GO-MEGA-{bl}"
            why.append(f"{bl}l : surcoût {sc:+.2f} ms, trame {m4b['med']:.2f} ms, "
                       f"1ʳᵉ bande {m4b['first_band_ms']:.2f} ms")
            break

R["verdict"] = verdict; R["why"] = why
R["reference"] = {"h2d_grouped_ms": ref_h2d, "d2h_grouped_ms": ref_d2h, "m4_wholeframe_ms": wf_med}
print(f"\n══ VERDICT : {verdict} ══")
for w in why:
    print("  " + w)
if verdict == "NO-GO":
    print("  → GPU reste whole-frame ; nœuds tissu GPU : force_cpu + slice CPU (0.25.x).")
elif verdict.startswith("GO-MEGA"):
    print("  → implémenter gpu_batch_bands (micro-batch §b var.2, TISSU_SLICE_GPU.md) ; "
          "puis banc réel avec gpu_slice=true.")
else:
    print("  → activer gpu_slice=true sur un mur de banc (slice_mode, cadence=flow) et "
          "comparer own/slice.* au même mur force_cpu.")
if A.json:
    with open(A.json, "w") as f:
        json.dump(R, f, indent=1, ensure_ascii=False)
    print(f"résultats JSON → {A.json}")
