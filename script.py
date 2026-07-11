# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 BOBI SAS, France
# Auteur : Cyril Mazouer, pour le compte de BOBI SAS
# Distribué sous licence GNU GPL v3 (ou ultérieure) ; voir le fichier LICENSE.

import mmap, socket, struct, time, numpy as np, threading, json, os, re, base64, io, signal
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image, ImageDraw, ImageFont
import bobimxl   # migration MXL Phase 1 : entrées vidéo+ANC via Reader, sortie via Writer
# (mmap/struct conservés : audio VU encore en legacy — différé jusqu'au producteur audio MXL).

# ─── Accélération GPU (cupy) auto-détectée ───────────────────────────────────
# `xp` = backend de calcul du HOT PATH compositing (lecture plans → resize → blend) : cupy si un
# GPU NVIDIA est visible (image bobi-compute-gpu + `docker run --gpus`), sinon numpy. MÊME script.py
# CPU/GPU (cf. chantier multiview-GPU). L'habillage PIL reste HÔTE/numpy (caché, optimisé en tuiles) ;
# seules ses tuiles YUV résultantes sont uploadées pour le blend. Repli numpy = OCTET-IDENTIQUE à
# avant (xp is np). Le gain GPU EXIGE le transfert épinglé+groupé (banc Phase 0 : sinon régression).
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()      # lève si aucun device (pas de --gpus / image CPU)
    xp = cp
    GPU = True
    try:
        _GPU_NAME = cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8", "replace")
    except Exception:
        _GPU_NAME = "GPU"
except Exception:
    xp = np
    GPU = False
    _GPU_NAME = None

def _to_xp(a):
    """numpy hôte → tableau backend (device si GPU, identité si CPU)."""
    return xp.asarray(a) if GPU else a

def _from_xp(a):
    """tableau backend → numpy hôte (download si GPU, identité si CPU)."""
    return cp.asnumpy(a) if GPU else a

# ─── Latence : Δ ts_out - ts_in_PiP (rolling avg par input) ────
class RollingMs:
    def __init__(self, n=30):
        self.d = deque(maxlen=n); self.last_ns = 0
    def push(self, ms_value):
        self.d.append(ms_value); self.last_ns = time.time_ns()
    def avg(self):
        if not self.d: return None
        if time.time_ns() - self.last_ns > 2_000_000_000: return None
        return round(sum(self.d) / len(self.d), 1)

lat_in = {{}}  # {{shm_name: RollingMs}} — TRANSIT par entrée (ts_read − ts_in producteur) = arrivée
own_lat = RollingMs()  # traitement PROPRE du nœud (ts_out − ts_cycle_start), exposé own_latency_ms
# Profiling du compositing (où vont les ms de own_latency) : entrées vidéo / habillage / sortie.
_t_inputs = RollingMs(); _t_overlays = RollingMs(); _t_output = RollingMs()
# Sous-ventilation de l'habillage (overlays) : rendu PIL meters/fg / conversion RGBA→YUV / blend.
_t_ov_render = RollingMs(); _t_ov_convert = RollingMs(); _t_ov_blend = RollingMs()

# ─── Config injectée (contrat plugin) ───────────────────────
CONFIG         = {config}
HOSTNAME       = "{hostname}"
PLUGIN_VERSION = "{plugin_version}"

# Filet anti-régression : `force_cpu` force le chemin numpy ÉPROUVÉ (xp=np) MÊME sur un nœud GPU,
# sans changer d'image. Réassigne les globals AVANT toute boucle de rendu → repli instantané identique
# au multiview 100% CPU si le chemin GPU posait souci. Absent/false → aucun effet (auto-détection).
if CONFIG.get("force_cpu"):
    xp = np
    GPU = False
    _GPU_NAME = None

FLUX_CONFIG   = CONFIG.get("flux_config") or []
SHM_OUT_NAME  = CONFIG.get("shm_out") or "mxl_mix"
SHM_OUT       = "/dev/shm/" + SHM_OUT_NAME   # conservé pour l'audio legacy / dérivations de noms
inst          = bobimxl.Instance()           # domaine MXL ($MXL_DOMAIN ou /dev/shm/mxl)
OUT_WIDTH     = int(CONFIG.get("out_width") or CONFIG.get("width") or 1280)
OUT_HEIGHT    = int(CONFIG.get("out_height") or CONFIG.get("height") or 720)
# Orientation : en portrait, on COMPOSE dans OUT_WIDTH×OUT_HEIGHT (canevas logique vertical, ex.
# 1080×1920) puis on tourne 90° le résultat → trame de sortie PAYSAGE (panneau physiquement incliné).
# Le transport/l'aval ne voient que les dims TOURNÉES (cf app.scripts.multiview_output_dims).
ORIENTATION   = (CONFIG.get("orientation") or "landscape").strip().lower()
_PORTRAIT     = ORIENTATION in ("portrait_cw", "portrait_ccw")
OUTPUT_W      = OUT_HEIGHT if _PORTRAIT else OUT_WIDTH    # largeur du flux émis (après rotation)
OUTPUT_H      = OUT_WIDTH  if _PORTRAIT else OUT_HEIGHT   # hauteur du flux émis (après rotation)
def _as_bool(v):
    # bool("False") == True : on parse explicitement les chaînes de CONFIG.
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)

BORDER_W      = int(CONFIG.get("border_w") or 0)        # bordure globale (px)
BORDER_COLOR  = CONFIG.get("border_color") or "#ffffff" # couleur de bordure globale
OVERLAY_BELOW = _as_bool(CONFIG.get("overlay_below"))   # bandeau sous l'image vs par-dessus
LABEL_SIZE    = int(CONFIG.get("label_size") or 14)     # taille du texte du label (px)
TSL_PORT      = int(CONFIG.get("tsl_port") or 0)        # port TCP TSL 5.0 local (mode Direct)
# Mode tally/UMD : "central" (push /tally_bulk par l'orchestrateur) | "direct" (serveur TSL local).
# Dérivation depuis l'ancien schéma tsl_port/tsl_remote si tsl_mode absent (compat sans migration).
TSL_MODE      = (CONFIG.get("tsl_mode")
                 or ("direct" if (TSL_PORT > 0 and not _as_bool(CONFIG.get("tsl_remote"))) else "central"))
SHOW_NO_SIGNAL = _as_bool(CONFIG.get("show_no_signal", True))  # placeholder « NO SIGNAL » quand la source manque
try:
    FREEZE_DETECT_S = max(0.0, float(CONFIG.get("freeze_detect_s", 2.0)))  # s sans avance du frame_index → badge FREEZE (0 = off)
except (TypeError, ValueError):
    FREEZE_DETECT_S = 2.0
SHOW_FORMAT   = _as_bool(CONFIG.get("show_format"))     # chip format déclaré par fenêtre (mode ingénierie)
SHOW_PROXY    = _as_bool(CONFIG.get("show_proxy"))      # badge proxy lu par tuile (mode ingénierie pyramide)

# Monitoring pyramide : dernier choix de proxy par tuile (lu par _refresh_lat_metrics → :8080).
_proxy_usage_latest = {{}}   # {{idx: {{src, read, cost, kind}}}}
_proxy_read_latest  = []     # noms de shm RÉELLEMENT lus (≠ chemins pleins) → détection orphelins

# Chroma uniforme du pipeline (entrées ET sortie ont le même layout ; défaut 4:2:2).
CHROMA = str(CONFIG.get("chroma") or "422")
# Profondeur shm 8/10/12 bits. Conversions paramétrées par _SCALE ; blend en accumulateur
# uint32 (10/12 bits débordent uint16) ; en 8 bits dtype uint8 → traitement byte-identique.
BIT_DEPTH = int(CONFIG.get("bit_depth") or 8)
_DEEP  = BIT_DEPTH >= 10
_BPS   = 2 if _DEEP else 1
_NP_DT = np.uint16 if _DEEP else np.uint8
_SCALE = 1 << (BIT_DEPTH - 8)
_MAXV  = (1 << BIT_DEPTH) - 1
_NEUTRAL = 1 << (BIT_DEPTH - 1)
_CW = {{"420": 2, "422": 2, "444": 1}}.get(CHROMA, 2)   # diviseur largeur chroma
_CH = {{"420": 2, "422": 1, "444": 1}}.get(CHROMA, 1)   # diviseur hauteur chroma
PIX_FMT = ({{"420": "yuv420p", "422": "yuv422p", "444": "yuv444p"}}.get(CHROMA, "yuv422p")
           + (("12le" if BIT_DEPTH >= 12 else "10le") if _DEEP else ""))
OUT_FRAME_SIZE = (OUTPUT_W * OUTPUT_H + 2 * (OUTPUT_W // _CW) * (OUTPUT_H // _CH)) * _BPS

def _rotate_out(cy, cu, cv):
    # Tourne le canevas portrait (OUT_WIDTH×OUT_HEIGHT logique) de 90° vers la trame paysage
    # (OUTPUT_W×OUTPUT_H). La chroma 4:2:x ne se tourne PAS directement (les axes de
    # sous-échantillonnage permutent) : on upsample en 4:4:4, on tourne, on re-sous-échantillonne
    # vers la chroma cible → formes correctes (Y OUTPUT_H×OUTPUT_W, U/V OUTPUT_H//_CH × OUTPUT_W//_CW).
    # np.rot90 : k=1 = sens anti-horaire (CCW), k=-1 = horaire (CW). Vue → ascontiguousarray.
    # _xp = backend du canvas (cupy si GPU) → rotation 100 % en VRAM (xp=np en CPU, inchangé).
    _xp = cp if (GPU and isinstance(cy, cp.ndarray)) else np
    k = 1 if ORIENTATION == "portrait_ccw" else -1
    ry = _xp.ascontiguousarray(_xp.rot90(cy, k))
    if _CW == 1 and _CH == 1:                    # 4:4:4 : rotation directe des 3 plans
        return ry, _xp.ascontiguousarray(_xp.rot90(cu, k)), _xp.ascontiguousarray(_xp.rot90(cv, k))
    uf = _xp.rot90(_xp.repeat(_xp.repeat(cu, _CH, axis=0), _CW, axis=1), k)   # → 4:4:4 paysage
    vf = _xp.rot90(_xp.repeat(_xp.repeat(cv, _CH, axis=0), _CW, axis=1), k)
    ru = _xp.ascontiguousarray(uf[::_CH, ::_CW])  # re-sous-échantillonne à la chroma cible
    rv = _xp.ascontiguousarray(vf[::_CH, ::_CW])
    return ry, ru, rv
HEADER_SIZE    = 64
RING_SIZE   = CONFIG.get("shm_video_ring", 10)
OUT_TOTAL      = HEADER_SIZE + (OUT_FRAME_SIZE * RING_SIZE)
def _rate_nd(v):
    """Cadence → (num, den) EXACT (fractionnaire NTSC = N*1000/1001 ; accepte \"30000/1001\")."""
    try:
        if isinstance(v, str) and "/" in v:
            a, b = v.split("/"); return int(a), int(b)
        f = float(v or 25)
    except Exception:
        return 25, 1
    n = round(f)
    if abs(f - n) < 0.01: return (n or 25), 1
    nominal = round(f * 1001.0 / 1000.0); return nominal * 1000, 1001
_FN, _FD = _rate_nd(CONFIG.get("fps") or 25)
FRAME_INTERVAL = _FD / _FN                # période exacte (cadence rationnelle, gère 29.97/59.94)
# Genlock broadcast : sortie du multiview (mur de monitoring) calée en PHASE sur la grille PTP
# (CLOCK_REALTIME, disciplinée phc2sys) → write_ts = instant de grille. off → cadence libre héritée.
_gl = CONFIG.get("genlock", True)
GENLOCK = _gl if isinstance(_gl, bool) else str(_gl).strip().lower() in ("1", "true", "yes", "on")
# CADENCE : "genlock" (défaut, attend la grille PTP) | "input" (DATA-DRIVEN : émet dès qu'une
# entrée a une nouvelle trame, propage le timestamp, n'attend PAS la grille). Le mode "input" est
# destiné aux nœuds INTERMÉDIAIRES d'un tissu de composition (shards/assembleur chaînés) : il évite
# d'ajouter une trame de latence par étage (la latence cumulée = Σ calcul du chemin critique, pas
# N×intervalle) et le slip sur grille (33 fps au lieu de 25 quand own≈30 ms).
CADENCE = str(CONFIG.get("cadence", "genlock")).strip().lower()
INPUT_LOCKED = (CADENCE == "input")
# ── MODE TRANCHE (chantier latence sous-trame, cf. patch mxl-planar-slices) ──────────────────
# slice_mode=true → composition BANDE PAR BANDE : chaque entrée est lue via get_slice (réveil au
# commit partiel du producteur — un 2110_io RX en SLICE_MODE committe ~toutes les 0,7 ms), et la
# SORTIE est publiée progressivement (open_grain 1×, commit validSlices=1..N) → l'étage aval
# (TX 2110 slice, autre multiview…) démarre sur la 1ʳᵉ bande sans attendre la trame. Convention
# validSlices (identique moteur) : k tranches ⇔ lignes [0, k·slice_lines) valides sur les 3 plans.
# Une entrée whole-frame (proxy pyramide, flux non tranché : totalSlices=1) dégrade proprement
# (attend son grain complet). Prérequis : CPU (pas GPU v1), paysage, hauteur divisible.
_slm = CONFIG.get("slice_mode", False)
SLICE_MODE  = _slm if isinstance(_slm, bool) else str(_slm).strip().lower() in ("1", "true", "yes", "on")
SLICE_LINES = int(CONFIG.get("slice_lines") or 36)
SLICE_ON = (SLICE_MODE and not GPU and not _PORTRAIT
            and SLICE_LINES > 0 and OUT_HEIGHT % SLICE_LINES == 0 and SLICE_LINES % _CH == 0)
if SLICE_MODE and not SLICE_ON:
    print(f"multiview: slice_mode demandé mais inéligible (gpu={{GPU}} portrait={{_PORTRAIT}} "
          f"h={{OUT_HEIGHT}}%{{SLICE_LINES}}) — whole-frame")
# ── CADENCE « flow » (tissu en tranches, TISSU_SLICE.md) ──────────────────────────────────────
# Data-flow ALIGNÉ SUR LA GRILLE : pas de barrière (le point fixe deadline/période de l'input-
# locked disparaît structurellement — le tick d'epoch EST le déclencheur, aligné à ~1,6 ms sur
# l'arrivée des grains). À chaque epoch TAI : la composition CIBLE l'index d'epoch fi_out —
# chaque tuile lit LE grain fi_out de SA source (même grille PTP → mêmes index) et le SUIT par
# bandes pendant son arrivée ; la sortie est écrite À CE MÊME INDEX → l'assembleur aval lit tous
# ses shards au même index k : alignement inter-étages PARFAIT (fini le get_latest désaligné).
# Source hors-grille (index libre) → suivie par sa tête ; en retard/morte → repli/backoff par
# tuile (0.24.1). Exige le mode tranche + genlock possible ; sinon dégrade en genlock whole-frame.
FLOW = (CADENCE == "flow") and SLICE_ON
if CADENCE == "flow" and not FLOW:
    print("multiview: cadence=flow exige le mode tranche éligible — repli genlock")
if FLOW:
    INPUT_LOCKED = False      # pacing = grille (branche genlock) ; la spécificité flow est dans
                              # le CIBLAGE d'index (collecte + open_grain(fi_out)), pas le pacing
    GENLOCK = True
def _grid_next(now_s, interval_s):
    import math as _m
    return (_m.floor(now_s / interval_s) + 1) * interval_s

TALLY_COLORS = {{
    # fill : rouge seulement quand le signal rouge est actif
    "red":   (220,  40,  40, 230),
    "green": (  0,   0,   0,   0),  # transparent : vert visible via bordure
    "amber": (220,  40,  40, 230),  # rouge + bordure verte
    "off":   (  0,   0,   0,   0),  # transparent
}}
TALLY_BORDER_COLORS = {{
    "off":   (255, 255, 255, 255),
    "red":   (255, 255, 255, 255),
    "green": (255, 255, 255, 255),
    "amber": (255, 255, 255, 255),
}}
TALLY_TEXT_BG = {{
    # fond coloré de la zone label quand le signal vert est actif
    "off":   None,
    "red":   None,
    "green": ( 30, 150,  60, 210),
    "amber": ( 30, 150,  60, 210),
}}
TALLY_TEXT_COLORS = {{
    "off":   (200, 200, 200, 255),
    "red":   (255,  90,  90, 255),   # rouge sur fond noir
    "green": (255, 255, 255, 255),   # blanc sur fond vert
    "amber": (255,  90,  90, 255),   # rouge sur fond vert
}}
# Tailles d'habillage (texte/bandeau/tally) calculées PAR FENÊTRE via _label_metrics()
# (plafonnées à la hauteur du PiP) — LABEL_SIZE n'est que la taille DEMANDÉE.
FRAME_STYLE = CONFIG.get("frame_style") or "none"  # none | classic(UMD) | tally_border | stylized(bezel) | viewfinder | flat
# Styles d'habillage v0.12 : réservent leurs marges AUTOUR de l'image (_frame_metrics).
_DRESS_STYLES = ("classic", "tally_border", "stylized", "viewfinder", "flat")

# Couleurs de bordure épaisse selon tally dominant (styles tally_border / stylized)
_TALLY_BORDER_RGBA = {{
    "off":   (  0,   0,   0,   0),
    "red":   (220,  40,  40, 255),
    "green": ( 40, 200,  80, 255),
    "amber": (220, 160,   0, 255),
}}
# Couleur du texte TSL selon dominante (styles tally_border / stylized)
_TALLY_TEXT_BY_DOMINANT = {{
    "off":   (200, 200, 200, 255),
    "red":   (255, 100, 100, 255),
    "green": (120, 255, 140, 255),
    "amber": (255, 200,  60, 255),
}}
# (fill, outline) des pastilles rondes par état (style stylized)
_PILL_COLORS = {{
    "off":   (( 50,  50,  50, 180), ( 90,  90,  90, 200)),
    "red":   ((220,  40,  40, 240), (255,  80,  80, 255)),
    "green": (( 40, 200,  80, 240), ( 80, 255, 120, 255)),
    "amber": ((220, 160,   0, 240), (255, 200,  40, 255)),
}}

# Habillages v0.12 : neutres « au repos » — TOUJOURS visibles (jamais transparents,
# contrairement à l'ancien tally_border invisible hors antenne).
_FRAME_NEUTRAL = ( 70,  70,  78, 255)   # cadre/bordure au repos
_FLAT_NEUTRAL  = ( 90,  90,  98, 255)   # soulignement flat au repos
# Teinte de fond (onglet tally_border) par dominante.
_BAR_TINTS = {{
    "off":   ( 50,  50,  56, 235),
    "red":   (120,  20,  20, 235),
    "green": ( 20,  90,  35, 235),
    "amber": (120,  80,   0, 235),
}}

# ─── Peak meters (audio) ─────────────────────────────────────
# Format shm audio cohérent avec receiver_nmos.py : header 64B + ring 100 chunks
# de 1152 bytes (1 ms à L24/48k/8ch).
A_SAMPLE_RATE       = 48000
A_CHANNELS_MAX      = 8
A_BIT_DEPTH         = 24
A_SAMPLES_PER_CHUNK = A_SAMPLE_RATE // 1000        # 48
A_CHUNK_SIZE        = A_SAMPLES_PER_CHUNK * A_CHANNELS_MAX * (A_BIT_DEPTH // 8)  # 1152
A_HEADER_SIZE       = 64
A_RING_SIZE   = CONFIG.get("shm_audio_ring", 100)
A_TOTAL_SIZE        = A_HEADER_SIZE + A_RING_SIZE * A_CHUNK_SIZE

METER_BAR_W          = 5
METER_GAP            = 1
METER_TICK_W         = 16
METER_MIN_DB         = -60.0
METER_DECAY_DB_PER_S = 20.0    # vitesse de chute du peak hold

# audio_states[flux_idx] = {{ar (bobimxl.AudioReader), name, peaks (np), holds (np), hold_ts (np)}}
audio_states = {{}}

def _derive_audio_name(video_path):
    """`mire1_0` (ou `/dev/shm/mire1_0`) → nom de flux audio `mire1_audio_0`. None si pas de _N final."""
    name = (video_path or "").removeprefix("/dev/shm/")
    m = re.match(r"(.+?)_(\d+)$", name)
    return f"{{m.group(1)}}_audio_{{m.group(2)}}" if m else None

def _open_audio_state(flux_idx, video_path):
    """Ouvre le FLUX MXL audio dérivé du flux vidéo (bobimxl.AudioReader). Renvoie state ou None."""
    nm = _derive_audio_name(video_path)
    if not nm:
        return None
    try:
        ar = bobimxl.AudioReader(inst, nm)   # lève si le flux n'existe pas encore
    except Exception:
        return None
    return {{"ar": ar, "name": nm, "peaks": None, "holds": None, "hold_ts": None}}

# États audio. SILENCE = flux FRAIS mais niveau au plancher (signal présent, muet). ABSENCE =
# producteur qui n'écrit plus (flux coupé) OU pas de flux audio du tout.
SILENCE_DB = METER_MIN_DB + 5.0    # niveau instantané sous ce seuil = considéré muet
SILENCE_HOLD_S = 5.0               # muet en continu pendant ce délai → statut "silence" (anti court-blanc)
ABSENCE_MS = 200.0                 # pas d'écriture producteur depuis ce délai = flux coupé

def _update_peaks(state, n_channels, now):
    """Lit le dernier bloc audio (float32 MXL) → (peaks, holds, statut) avec statut ∈ ok|silence|
    absence. Fraîcheur via last_write_time() (TAI) : flux COUPÉ → barres ramenées à MIN (jamais
    FIGÉES — sinon read_latest rejoue le dernier bloc à l'infini) + statut absence."""
    ar = state.get("ar")
    if ar is None:
        # Reader PERDU (state["ar"] mis à None par le chemin d'exception quand le flux a DISPARU un
        # instant — typiquement un producteur amont REDÉMARRÉ : restart du moteur 2110_io qui détruit
        # puis recrée tous ses flux). Sans retenter ICI, l'état reste figé ar=None À JAMAIS (le state
        # est un dict non-None → render_meters ne le recrée pas via _open_audio_state) → ABSENCE
        # permanente. On tente donc de ROUVRIR (avec garbage_collect pour se rattacher à la génération
        # vivante). Échec (flux toujours absent) → absence, on retentera au tour suivant.
        try: inst.garbage_collect()
        except Exception: pass
        try:
            ar = bobimxl.AudioReader(inst, state["name"]); state["ar"] = ar
            state["last_head"] = None; state["last_head_ts"] = now
        except Exception:
            state["ar"] = None
            return None, None, "absence"
    # Fraîcheur audio via l'AVANCÉE de l'index d'échantillon (head_index), PAS via last_write_time :
    # NI l'AudioWriter MXL (simu/tonalité, modèle « samples continus » mxlFlowWriterCommitSamples)
    # NI mtl_rx (RX réel) ne mettent à jour lastWriteTime (réservé aux flux à grains/vidéo) →
    # last_write_time() reste FIGÉ et ferait croire à une absence alors que le son est frais (VU
    # vides). On considère le flux COUPÉ si head_index ne bouge plus depuis ABSENCE_MS → MIN.
    # RECONNEXION : un producteur qui RECRÉE le flux sous le même nom (bascule simu→live mtl_rx,
    # redéploiement) laisse notre reader collé à l'orphelin MORT (head figé, AUCUNE exception). À
    # head figé > ABSENCE_MS on ROUVRE le reader pour se rattacher à la génération vivante (≠ index
    # → frais), symétrique de la reconnexion des readers vidéo. Re-arme le timer = retry borné.
    def _mn():
        mn = np.full(n_channels, METER_MIN_DB, dtype=np.float64)
        state["peaks"] = mn; state["holds"] = mn
        state["hold_ts"] = np.full(n_channels, now, dtype=np.float64)
        return mn, mn, "absence"
    try:
        head = int(ar.head_index())
    except Exception:
        head = -1
    last_head = state.get("last_head")
    if head >= 0 and head != last_head:
        state["last_head"] = head; state["last_head_ts"] = now        # avance → frais
    elif last_head is None:
        state["last_head"] = head; state["last_head_ts"] = now        # 1er passage
        if head < 0:
            return _mn()
    elif (now - float(state.get("last_head_ts") or now)) * 1000.0 > ABSENCE_MS:
        # Figé > seuil : tenter de se rattacher à la génération courante du flux. Le producteur a
        # pu DÉTRUIRE+RECRÉER le flux sous le même nom (toggle générateur, (dé)abonnement 2110-30) ;
        # notre instance MXL longue durée garde une référence à l'ancienne génération MORTE → un
        # simple ré-open s'y rattacherait encore (head figé). garbage_collect() purge les flows à
        # writer mort (comme le fait le PRODUCTEUR avant de recréer) → le ré-open voit la vivante.
        try: ar.close()
        except Exception: pass
        try: inst.garbage_collect()
        except Exception: pass
        try:
            ar = bobimxl.AudioReader(inst, state["name"]); state["ar"] = ar
            head = int(ar.head_index())
        except Exception:
            state["ar"] = None; head = -1
        state["last_head"] = head; state["last_head_ts"] = now
        if head < 0 or head == last_head:    # toujours mort (même orphelin / pas de flux) → coupé
            return _mn()
        # sinon : rattaché à une génération vivante → frais, on lit ci-dessous
    try:
        r = ar.read_latest(A_SAMPLES_PER_CHUNK)   # (≤48, channels) float32 normalisé [-1,1]
    except Exception:
        # flux invalidé (producteur recréé) → garbage_collect (purge l'ancienne génération) + rouvrir
        # sur la vivante. last_head=None pour ré-amorcer le suivi de fraîcheur au tour suivant.
        try: ar.close()
        except Exception: pass
        try: inst.garbage_collect()
        except Exception: pass
        try:
            state["ar"] = bobimxl.AudioReader(inst, state["name"]); state["last_head"] = None
        except Exception:
            state["ar"] = None
        return None, None, "absence"
    if r is None:
        return None, None, "absence"
    # float32 déjà normalisé → 0 dBFS = |1.0| (plus de dépack s24be / full_scale).
    peaks_lin = np.max(np.abs(r), axis=0).astype(np.float64)  # (channels,)
    peak_db = np.where(peaks_lin > 0, 20.0 * np.log10(peaks_lin), METER_MIN_DB - 1)
    peak_db = np.clip(peak_db, METER_MIN_DB, 0.0)
    peak_db = peak_db[:n_channels]
    # Holds avec decay
    holds = state.get("holds")
    hold_ts = state.get("hold_ts")
    if holds is None or len(holds) != n_channels:
        holds = np.full(n_channels, METER_MIN_DB, dtype=np.float64)
        hold_ts = np.full(n_channels, now, dtype=np.float64)
    for ch in range(n_channels):
        if peak_db[ch] >= holds[ch]:
            holds[ch] = peak_db[ch]
        else:
            elapsed = max(0.0, now - hold_ts[ch])
            holds[ch] = max(peak_db[ch], holds[ch] - METER_DECAY_DB_PER_S * elapsed)
        hold_ts[ch] = now
    state["peaks"] = peak_db
    state["holds"] = holds
    state["hold_ts"] = hold_ts
    # Flux frais mais MUET = signal présent au plancher. Temporisation DÉDIÉE (découplée de la
    # ballistique du hold) : « silence » seulement si le niveau INSTANTANÉ reste sous SILENCE_DB
    # de façon continue pendant ≥ SILENCE_HOLD_S → un court blanc (pause parole/musique) ne le
    # déclenche pas. Toute trame au-dessus du seuil ré-arme le compteur.
    if float(np.max(peak_db)) > SILENCE_DB:
        state["last_loud_ts"] = now
    status = "silence" if (now - float(state.get("last_loud_ts", now))) >= SILENCE_HOLD_S else "ok"
    return peak_db, holds, status

def _meter_layout(n_channels):
    """Renvoie (width, ...) pour un meter à N canaux. Sans bordure."""
    return METER_TICK_W + n_channels * METER_BAR_W + (n_channels - 1) * METER_GAP

def _draw_meter(img, mx, my, mw, mh, n_channels, peaks_db, holds_db, scale, opacity_pct):
    """Dessine un peak meter sur l'image RGBA. opacity_pct 10..100.
    Réserve 12 px en bas pour afficher le numéro de canal sous chaque barre."""
    d = ImageDraw.Draw(img, "RGBA")
    a_bg   = int(180 * opacity_pct / 100)
    a_bar  = int(220 * opacity_pct / 100)
    a_text = int(255 * opacity_pct / 100)
    a_hold = int(255 * opacity_pct / 100)
    # On réserve 12 px en bas pour le numéro de canal sous chaque barre.
    label_h = 12
    bars_mh = max(20, mh - label_h)
    # Mapping dB → fraction 0..1 selon échelle (pour le rendu des barres : dB en dBFS).
    # Pour les ticks, on a deux représentations :
    #   - dBFS : valeur affichée = valeur dB en dBFS (ex -18)
    #   - PPM  : valeur affichée = valeur en EBU dB (ex +6) = dBFS + 18
    if scale == "ppm":
        def to_frac(dbfs):
            ebu = dbfs + 18
            return max(0.0, min(1.0, (ebu + 12) / 24.0))
        green_top  = to_frac(-12)   # +6 EBU
        yellow_top = to_frac(-3)    # +15 EBU
        ticks = [(+12, "+12"), (+9, "+9"), (+6, "+6"), (+3, "+3"),
                 (0, "0"), (-3, "-3"), (-6, "-6"), (-9, "-9"), (-12, "-12")]
        # Convertit ticks EBU dB en dBFS pour calcul de position
        ticks_dbfs = [(ebu - 18, lbl) for ebu, lbl in ticks]
    else:
        def to_frac(dbfs):
            return max(0.0, min(1.0, (dbfs - METER_MIN_DB) / (-METER_MIN_DB)))
        green_top  = to_frac(-20.0)
        yellow_top = to_frac(-6.0)
        # Marques broadcast dBFS courantes
        ticks_dbfs = [(0, "0"), (-3, "-3"), (-6, "-6"), (-9, "-9"), (-12, "-12"),
                      (-18, "-18"), (-20, "-20"), (-30, "-30"), (-40, "-40"), (-50, "-50")]
    # Fond (toute la zone du meter, incluant la bande de labels canaux en bas)
    d.rectangle([mx, my, mx + mw, my + mh], fill=(0, 0, 0, a_bg))
    bars_bottom = my + bars_mh   # ligne du bas des barres
    # Graduations : petite ligne pour chaque tick + label si l'espace permet
    last_label_y = -10
    for tick_dbfs, lbl in ticks_dbfs:
        f = to_frac(tick_dbfs)
        y_tick = my + bars_mh - int(round(f * bars_mh))
        # Petite ligne 3px à droite de la zone tick (donc dans la zone des barres, devant)
        d.line([mx + METER_TICK_W - 4, y_tick, mx + METER_TICK_W - 1, y_tick],
               fill=(180, 180, 180, a_text))
        # Label seulement si suffisamment d'espace vertical avec le précédent
        if abs(y_tick - last_label_y) >= 9 and y_tick - 4 >= my and y_tick + 4 <= bars_bottom:
            d.text((mx + 1, y_tick - 4), lbl,
                   font=ImageFont.load_default(), fill=(220, 220, 220, a_text))
            last_label_y = y_tick
    # Barres + numéro de canal en bas
    bar_x0 = mx + METER_TICK_W
    green_top_px  = int(round(green_top  * bars_mh))
    yellow_top_px = int(round(yellow_top * bars_mh))
    for ch in range(n_channels):
        bx = bar_x0 + ch * (METER_BAR_W + METER_GAP)
        peak_h = int(round(to_frac(peaks_db[ch]) * bars_mh))
        hold_h = int(round(to_frac(holds_db[ch]) * bars_mh))
        # Green zone
        gh = min(peak_h, green_top_px)
        if gh > 0:
            d.rectangle([bx, bars_bottom - gh, bx + METER_BAR_W - 1, bars_bottom],
                        fill=(60, 200, 60, a_bar))
        # Yellow zone
        if peak_h > green_top_px:
            yh = min(peak_h, yellow_top_px) - green_top_px
            if yh > 0:
                d.rectangle([bx, bars_bottom - green_top_px - yh, bx + METER_BAR_W - 1, bars_bottom - green_top_px],
                            fill=(220, 180, 40, a_bar))
        # Red zone
        if peak_h > yellow_top_px:
            rh = peak_h - yellow_top_px
            if rh > 0:
                d.rectangle([bx, bars_bottom - yellow_top_px - rh, bx + METER_BAR_W - 1, bars_bottom - yellow_top_px],
                            fill=(230, 60, 60, a_bar))
        # Peak hold (ligne fine)
        if hold_h > 0:
            yh = bars_bottom - hold_h
            d.line([bx, yh, bx + METER_BAR_W - 1, yh], fill=(255, 255, 255, a_hold), width=1)
        # Numéro de canal sous la barre (centré sur la barre, ch+1 = 1-indexé)
        ch_label = str(ch + 1)
        # ImageFont.load_default() est très petit, label sur 1 caractère → ~5px wide
        lx = bx + (METER_BAR_W // 2) - 2
        ly = bars_bottom + 2
        d.text((lx, ly), ch_label,
               font=ImageFont.load_default(), fill=(220, 220, 220, a_text))

# ─── Peak meters : chemin GPU (cupy) ─────────────────────────────────────────
# Graduations/fond/labels/n° de canal = STATIQUES → rendus PIL UNE fois (image), convertis YUV et
# résidents backend (VRAM si GPU), cachés par config. Barres + peak-hold = DYNAMIQUES → composés par
# trame en RGBA sur le backend (xp), puis UNE conversion RGBA→YUV. Plus aucun PIL par trame.
# Le chemin CPU (GPU=False) n'utilise PAS ce code : render_meters garde la branche PIL d'origine VERBATIM.
_meter_static_xp_cache = {{}}   # key -> RGBA backend float32 (rendu PIL une fois, uploadé une fois)
_MET_GREEN = (60, 200, 60); _MET_YELLOW = (220, 180, 40); _MET_RED = (230, 60, 60); _MET_WHITE = (255, 255, 255)

def _meter_scale_params(scale):
    """(to_frac, green_top, yellow_top) selon l'échelle — réplique la logique de _draw_meter."""
    if scale == "ppm":
        def to_frac(dbfs):
            ebu = dbfs + 18
            return max(0.0, min(1.0, (ebu + 12) / 24.0))
        return to_frac, to_frac(-12), to_frac(-3)
    def to_frac(dbfs):
        return max(0.0, min(1.0, (dbfs - METER_MIN_DB) / (-METER_MIN_DB)))
    return to_frac, to_frac(-20.0), to_frac(-6.0)

def _draw_meter_static(img, mx, my, mw, mh, n_channels, scale, opacity_pct):
    """Partie STATIQUE du meter (fond + graduations + labels dB + n° de canal), SANS les barres —
    identique à _draw_meter hors boucle barres/hold. Rendu une seule fois (caché)."""
    d = ImageDraw.Draw(img, "RGBA")
    a_bg = int(180 * opacity_pct / 100); a_text = int(255 * opacity_pct / 100)
    bars_mh = max(20, mh - 12)
    to_frac, _gt, _yt = _meter_scale_params(scale)
    if scale == "ppm":
        ticks = [(+12, "+12"), (+9, "+9"), (+6, "+6"), (+3, "+3"),
                 (0, "0"), (-3, "-3"), (-6, "-6"), (-9, "-9"), (-12, "-12")]
        ticks_dbfs = [(ebu - 18, lbl) for ebu, lbl in ticks]
    else:
        ticks_dbfs = [(0, "0"), (-3, "-3"), (-6, "-6"), (-9, "-9"), (-12, "-12"),
                      (-18, "-18"), (-20, "-20"), (-30, "-30"), (-40, "-40"), (-50, "-50")]
    d.rectangle([mx, my, mx + mw, my + mh], fill=(0, 0, 0, a_bg))
    bars_bottom = my + bars_mh
    last_label_y = -10
    for tick_dbfs, lbl in ticks_dbfs:
        y_tick = my + bars_mh - int(round(to_frac(tick_dbfs) * bars_mh))
        d.line([mx + METER_TICK_W - 4, y_tick, mx + METER_TICK_W - 1, y_tick], fill=(180, 180, 180, a_text))
        if abs(y_tick - last_label_y) >= 9 and y_tick - 4 >= my and y_tick + 4 <= bars_bottom:
            d.text((mx + 1, y_tick - 4), lbl, font=ImageFont.load_default(), fill=(220, 220, 220, a_text))
            last_label_y = y_tick
    for ch in range(n_channels):
        bx = mx + METER_TICK_W + ch * (METER_BAR_W + METER_GAP)
        d.text((bx + (METER_BAR_W // 2) - 2, bars_bottom + 2), str(ch + 1),
               font=ImageFont.load_default(), fill=(220, 220, 220, a_text))

def _meter_static_xp(W, H, rmx, rmy, mw, mh, n, scale, opacity_pct):
    """RGBA statique (backend float32) cachée pour cette config de meter — l'idée 'graduations en VRAM'."""
    key = (W, H, rmx, rmy, mw, mh, n, scale, opacity_pct)
    arr = _meter_static_xp_cache.get(key)
    if arr is None:
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        _draw_meter_static(img, rmx, rmy, mw, mh, n, scale, opacity_pct)
        host = np.array(img).astype(np.float32)
        arr = xp.asarray(host) if GPU else host
        _meter_static_xp_cache[key] = arr
    return arr

def _rgba_to_yuv_xp(arr):
    """rgba_to_yuv pour un RGBA backend (xp) float32 — réplique rgba_to_yuv sans PIL (xp=np en CPU)."""
    r = arr[..., 0]; g = arr[..., 1]; b = arr[..., 2]; a = arr[..., 3]
    y = (( 0.299 * r + 0.587 * g + 0.114 * b      ) * _SCALE).clip(0, _MAXV).astype(_NP_DT)
    u = ((-0.169 * r - 0.331 * g + 0.500 * b + 128) * _SCALE).clip(0, _MAXV).astype(_NP_DT)
    v = (( 0.500 * r - 0.419 * g - 0.081 * b + 128) * _SCALE).clip(0, _MAXV).astype(_NP_DT)
    def _sub_avg(p):
        pp = p.astype(xp.uint32)
        if _CW == 2: pp = (pp[:, 0::2] + pp[:, 1::2] + 1) // 2
        if _CH == 2: pp = (pp[0::2, :] + pp[1::2, :] + 1) // 2
        return pp.astype(_NP_DT)
    def _sub_max(p):
        if _CW == 2: p = xp.maximum(p[:, 0::2], p[:, 1::2])
        if _CH == 2: p = xp.maximum(p[0::2, :], p[1::2, :])
        return p
    a8 = a.astype(_NP_DT)
    return y, _sub_avg(u), _sub_avg(v), a8, _sub_max(a8)

def _meter_comp_rect(tile, x0, y0, x1, y1, rgb, a):
    """PEINT (remplace) une couleur pleine (rgb, a 0..255) sur la région RGBA `tile` (xp) — réplique
    EXACTEMENT le comportement de PIL ImageDraw 'RGBA' (paint, pas un over-compositing : le pixel prend
    la couleur+alpha du fill). Bornée aux dimensions de la tuile."""
    Ht = tile.shape[0]; Wt = tile.shape[1]
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(Wt, x1); y1 = min(Ht, y1)
    if x1 <= x0 or y1 <= y0:
        return
    tile[y0:y1, x0:x1, 0] = rgb[0]
    tile[y0:y1, x0:x1, 1] = rgb[1]
    tile[y0:y1, x0:x1, 2] = rgb[2]
    tile[y0:y1, x0:x1, 3] = a

def _meter_tile_gpu(W, H, rmx, rmy, mw, mh, n, peaks_db, holds_db, scale, opacity_pct):
    """Tuile YUV d'un meter (chemin GPU) : statique caché copié + barres/hold composées en RGBA (xp)."""
    tile = _meter_static_xp(W, H, rmx, rmy, mw, mh, n, scale, opacity_pct).copy()
    a_bar = int(220 * opacity_pct / 100); a_hold = int(255 * opacity_pct / 100)
    bars_mh = max(20, mh - 12); bars_bottom = rmy + bars_mh
    to_frac, green_top, yellow_top = _meter_scale_params(scale)
    green_top_px = int(round(green_top * bars_mh)); yellow_top_px = int(round(yellow_top * bars_mh))
    for ch in range(n):
        bx = rmx + METER_TICK_W + ch * (METER_BAR_W + METER_GAP)
        peak_h = int(round(to_frac(peaks_db[ch]) * bars_mh))
        hold_h = int(round(to_frac(holds_db[ch]) * bars_mh))
        # NB : PIL d.rectangle est INCLUSIF sur (x1,y1) → bas +1 ici pour égaler les hauteurs.
        gh = min(peak_h, green_top_px)
        if gh > 0:
            _meter_comp_rect(tile, bx, bars_bottom - gh, bx + METER_BAR_W, bars_bottom + 1, _MET_GREEN, a_bar)
        if peak_h > green_top_px:
            yh = min(peak_h, yellow_top_px) - green_top_px
            if yh > 0:
                _meter_comp_rect(tile, bx, bars_bottom - green_top_px - yh, bx + METER_BAR_W, bars_bottom - green_top_px + 1, _MET_YELLOW, a_bar)
        if peak_h > yellow_top_px:
            rh = peak_h - yellow_top_px
            if rh > 0:
                _meter_comp_rect(tile, bx, bars_bottom - yellow_top_px - rh, bx + METER_BAR_W, bars_bottom - yellow_top_px + 1, _MET_RED, a_bar)
        if hold_h > 0:
            yh = bars_bottom - hold_h
            _meter_comp_rect(tile, bx, yh, bx + METER_BAR_W, yh + 1, _MET_WHITE, a_hold)
    return _rgba_to_yuv_xp(tile)

# état tally : {{"<idx>_L": "red"|"green"|"amber"|"off", "<idx>_R": ...}}
tally_state = {{}}
# texte dynamique reçu par TSL, indexé par flux idx (utilisé si label_source == 'protocol')
tsl_text = {{}}
# texte TSL indexé par index TSL brut (utilisé par les overlays texte sourcés TSL, hors fenêtres)
tsl_text_by_index = {{}}
# overlays texte en mode CENTRAL : id d'overlay → {{"text": str, "active": bool}} poussé par
# l'orchestrateur (résolu depuis une ligne du tableau /labels, pas un index TSL local).
overlay_central = {{}}
tally_dirty = threading.Event()
tally_dirty.set()
# Accumulateur par slot TSL : (tsl_index, 'rh'|'tt'|'lh') → [valeur, last_ts, smoothed_interval_or_None]
# Protocole delta : seuls les slots actifs sont envoyés en keepalive.
# TTL adaptatif : mesuré sur les arrivées successives du même slot (2.5× intervalle observé).
# CONTROL==0 complet = clear explicite immédiat.
_tsl_slots = {{}}
_tsl_combined = {{}}         # index → True si le contrôleur envoie des paquets combinés
TSL_SLOT_TTL_FACTOR = 2.5   # TTL = factor × intervalle keepalive mesuré
TSL_SLOT_TTL_MIN    = 0.05  # 50 ms plancher absolu
metrics = {{"fps": 0.0, "inputs_latency_ms": {{}}, "own_latency_ms": None,
           "gpu": GPU, "gpu_name": _GPU_NAME}}   # GPU = compositing accéléré cupy (sinon numpy CPU)

def _compute_proxy_needs():
    """Besoins de tailles par source câblée pour la pyramide, AVEC le nombre de tuiles qui
    réclament chaque taille : {{"<src-shm>": [[w, h, count], …]}}. Le compteur est essentiel :
    l'orchestrateur ne génère un proxy sur-mesure que si une (source, taille) est demandée
    ≥ seuil fois — une même source affichée dans N tuiles à la même taille pèse donc N (sinon,
    dédupliqué à 1, le seuil ≥2 ne se déclenchait jamais pour un multiview seul). Taille = vw×vh
    via _video_rect (après marges habillage) → source de vérité unique."""
    counts = {{}}   # shm → {{(w,h): count}}
    try:
        with state_lock:
            fc = list(FLUX_CONFIG)
    except Exception:
        return {{}}
    for cfg in fc:
        if cfg.get("hidden"):
            continue
        path = cfg.get("path") or ""
        shm = path[len("/dev/shm/"):] if path.startswith("/dev/shm/") else path
        if not shm:
            continue
        try:
            g = _video_rect(cfg)
            w = int(g.get("vw") or 0); h = int(g.get("vh") or 0)
        except Exception:
            continue
        if w < 2 or h < 2:
            continue
        d = counts.setdefault(shm, {{}})
        d[(w, h)] = d.get((w, h), 0) + 1
    return {{shm: [[w, h, c] for (w, h), c in d.items()] for shm, d in counts.items()}}

def _refresh_lat_metrics():
    out = {{}}
    for shm_name, rm in lat_in.items():
        out[shm_name] = rm.avg()
    metrics["inputs_latency_ms"] = out
    # Retard par entrée (mode input-locked) : nb d'images de décalage par shm (0 = synchrone).
    # Restreint aux shms réellement câblés (lat_in) pour ne pas exposer d'anciennes entrées.
    metrics["inputs_lag_frames"] = {{k: _lag_frames.get(k, 0) for k in out}}
    metrics["own_latency_ms"] = own_lat.avg()
    metrics["proxy_needs"] = _compute_proxy_needs()
    metrics["proxy_usage"] = dict(_proxy_usage_latest)   # idx → {{src,read,cost,kind}}
    metrics["proxy_read"]  = list(_proxy_read_latest)     # shm proxy réellement lus (orphan-detect)
    # Profiling du compositing : ventilation de own_latency (entrées / habillage / sortie).
    metrics["compose_breakdown_ms"] = {{"inputs": _t_inputs.avg(), "overlays": _t_overlays.avg(),
                                       "output": _t_output.avg(),
                                       "ov_render": _t_ov_render.avg(), "ov_convert": _t_ov_convert.avg(),
                                       "ov_blend": _t_ov_blend.avg()}}
# debug TSL : dernier paquet reçu (mis à jour par _handle_tsl_client)
tsl_debug = {{"last_raw_hex": None, "last_ver": None, "last_index": None,
              "last_control": None, "last_text": None, "last_error": None,
              "connections": 0, "slots": {{}}}}

# ─── HTTP : metrics + tally ──────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/tally":
            self._send_json(tally_state)
        elif self.path == "/tsl_debug":
            self._send_json({{**tsl_debug,
                              "tally_state": dict(tally_state),
                              "tsl_text": {{str(k): v for k, v in tsl_text.items()}}}})
        else:
            self._send_json(metrics)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length).decode() or "{{}}"
        try:
            data = json.loads(body)
        except Exception as e:
            self.send_response(400); self.end_headers()
            self.wfile.write(str(e).encode()); return

        if self.path == "/tally":
            try:
                idx   = int(data["flux_idx"])
                color = str(data.get("color", "off")).lower()
                if color not in TALLY_COLORS:
                    raise ValueError("color invalide")
                if "slot" in data:
                    # Forme par-lampe (service TSL) : une lampe L/R, une couleur.
                    slot = str(data["slot"]).upper()
                    if slot not in ("L", "R"):
                        raise ValueError("slot invalide")
                    tally_state[f"{{idx}}_{{slot}}"] = color
                else:
                    # Forme simple (action shotbox/macro, sans slot) : couleur DOMINANTE
                    # de la fenêtre — red (antenne), green (préparation), off (éteint).
                    tally_state[f"{{idx}}_L"] = color if color in ("red", "amber") else "off"
                    tally_state[f"{{idx}}_R"] = "green" if color in ("green", "amber") else "off"
                tally_dirty.set()
                self._send_json({{"status": "ok"}})
            except Exception as e:
                self.send_response(400); self.end_headers()
                self.wfile.write(str(e).encode())

        elif self.path == "/tally_bulk":
            try:
                updates = data.get("updates") or []
                changed = False
                for upd in updates:
                    idx   = int(upd["flux_idx"])
                    slot  = str(upd["slot"]).upper()
                    color = str(upd.get("color", "off")).lower()
                    if slot not in ("L", "R") or color not in TALLY_COLORS:
                        continue
                    tally_state[f"{{idx}}_{{slot}}"] = color
                    changed = True
                    # Texte label optionnel (label_col depuis l'orchestrateur)
                    text = upd.get("text")
                    if text is not None and slot == "L":
                        tsl_text[idx] = str(text)
                # Overlays texte central : id → texte + état actif (résolu côté orchestrateur)
                ov_changed = False
                for ov in (data.get("overlays") or []):
                    oid = str(ov.get("id") or "")
                    if not oid:
                        continue
                    overlay_central[oid] = {{"text": str(ov.get("text") or ""),
                                             "active": bool(ov.get("active"))}}
                    ov_changed = True
                if changed:
                    tally_dirty.set()
                if ov_changed:
                    overlay_dirty.set()
                self._send_json({{"status": "ok", "updates": len(updates)}})
            except Exception as e:
                self.send_response(400); self.end_headers()
                self.wfile.write(str(e).encode())

        else:
            self.send_response(404); self.end_headers()

    def _send_json(self, payload):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, *a): pass

threading.Thread(
    target=lambda: HTTPServer(("0.0.0.0", 8080), Handler).serve_forever(),
    daemon=True).start()

# ─── Helpers ─────────────────────────────────────────────────

def open_source(cfg):
    """Ouvre une source comme flux MXL (Reader). Le NOM = chemin sans le préfixe /dev/shm/.
    Format LU DU flow_def du producteur (source de vérité) → in_w/in_h/chroma/bit_depth."""
    try:
        name = cfg["path"].removeprefix("/dev/shm/")
        rd = bobimxl.Reader(inst, name)        # lève si le flux n'existe pas encore
        fmt = rd.format()
        if not fmt:
            rd.close(); return None
        # ENTRELACÉ NATIF : un grain = 1 CHAMP (½ hauteur ; in_h = hauteur de CHAMP via format()).
        # Le multiview sort TOUJOURS en PROGRESSIF (mur de monitoring) → on « bobe » : un champ scalé
        # à la hauteur de tuile (resize_plane) = désentrelacement par champ, suffisant pour un preview.
        in_w, in_h = fmt["width"], fmt["height"]
        frame_size = (in_w * in_h + 2 * (in_w // _CW) * (in_h // _CH)) * _BPS
        return {{"reader": rd, "in_w": in_w, "in_h": in_h, "frame_size": frame_size,
                 "interlaced": bool(fmt.get("interlaced"))}}
    except Exception:
        return None

def resize_plane(plane, target_h, target_w):
    from_h, from_w = plane.shape
    if target_h <= 0 or target_w <= 0:
        return plane[:1, :1]
    # Downscale à ratio ENTIER (grilles 2×2/3×3/4×4… → tuile = ½, ⅓, ¼ de la source) : slicing à
    # pas constant `plane[::sy, ::sx]` (une VUE, zéro copie) au lieu du gather np.ix_ (alloue+copie).
    # Résultat octet-identique au nearest-neighbor (arange*from/target = arange*s pour s entier).
    # Sinon (ratio non entier / upscale), repli sur le gather générique.
    if from_h % target_h == 0 and from_w % target_w == 0:
        return plane[::from_h // target_h, ::from_w // target_w]
    # Gather (ratio non entier/upscale) : indices via le MÊME backend que `plane` (xp) — un index
    # numpy sur un tableau cupy lèverait. xp=np en CPU → comportement inchangé (octet-identique).
    _xp = cp if (GPU and isinstance(plane, cp.ndarray)) else np
    row_idx = (_xp.arange(target_h) * from_h / target_h).astype(int)
    col_idx = (_xp.arange(target_w) * from_w / target_w).astype(int)
    return plane[_xp.ix_(row_idx, col_idx)]

def rgba_to_yuv(img):
    """Image PIL RGBA → (Y full-res, U sub 2x2, V sub 2x2, alpha full, alpha sub 2x2)."""
    arr = np.array(img)
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    a = arr[..., 3]
    y = (( 0.299 * r + 0.587 * g + 0.114 * b      ) * _SCALE).clip(0, _MAXV).astype(_NP_DT)
    u = ((-0.169 * r - 0.331 * g + 0.500 * b + 128) * _SCALE).clip(0, _MAXV).astype(_NP_DT)
    v = (( 0.500 * r - 0.419 * g - 0.081 * b + 128) * _SCALE).clip(0, _MAXV).astype(_NP_DT)
    # Sous-échantillonnage chroma selon le format du pipeline (_CW/_CH) : moyenne pour U/V,
    # max pour l'alpha. 4:2:0 → bloc 2×2 ; 4:2:2 → 1×2 (hauteur pleine) ; 4:4:4 → identité.
    def _sub_avg(p):
        pp = p.astype(np.uint32)
        if _CW == 2: pp = (pp[:, 0::2] + pp[:, 1::2] + 1) // 2
        if _CH == 2: pp = (pp[0::2, :] + pp[1::2, :] + 1) // 2
        return pp.astype(_NP_DT)
    def _sub_max(p):
        if _CW == 2: p = np.maximum(p[:, 0::2], p[:, 1::2])
        if _CH == 2: p = np.maximum(p[0::2, :], p[1::2, :])
        return p
    u2 = _sub_avg(u); v2 = _sub_avg(v); a2 = _sub_max(a)
    return y, u2, v2, a, a2

def blend(dst, src, alpha):
    """dst/src YUV (uint8/uint16) + alpha uint8 (0..255) → même dtype. Accumulateur uint32
    car en 10/12 bits dst*(255-a) déborde uint16. NB : `// 255` (une seule ufunc vectorisée) est
    PLUS rapide en numpy que l'astuce entière `(t+(t>>8)+1)>>8` (4 passes mémoire = memory-bound)."""
    a32 = alpha.astype(np.uint32)
    return ((dst.astype(np.uint32) * (255 - a32) + src.astype(np.uint32) * a32) // 255).astype(_NP_DT)

# Accumulateur du blend : en 8 bits la somme dst·(255−α)+src·α ≤ 255·255 = 65025 tient en uint16
# (op memory-bound → 2× moins d'octets qu'uint32) ; en 10/12 bits il faut uint32.
_ACC = np.uint32 if _DEEP else np.uint16

def blend_pre(dst, inv_a, src_a):
    """Blend RAPIDE quand src/α sont STATIQUES (habillage caché) : inv_a=(255−α) et src_a=src·α
    pré-calculés UNE fois (cf. _chrome_pre). Par trame il ne reste que dst·inv_a + src_a puis //255 —
    ~2× moins de passes mémoire que blend() (plus de (255−α), src·α, ni cast d'alpha). Résultat
    numériquement IDENTIQUE à blend() (vérifié 8 et 10 bits)."""
    return ((dst.astype(_ACC) * inv_a + src_a) // 255).astype(_NP_DT)

# ─── Placement groupé des tuiles vidéo (CPU direct / GPU upload épinglé groupé) ───────────────
_pin = {{"host": None, "dev": None, "cap": 0}}   # buffers persistants GPU (staging épinglé entrées + device)
_outpin = {{"buf": None}}                        # buffer hôte ÉPINGLÉ persistant pour le download sortie (GPU)

def _place_batch(cy, cu, cv, batch):
    """Resize + place les tuiles vidéo collectées dans le canvas (cy/cu/cv, backend xp).
    CPU : resize numpy direct (octet-identique à l'inline d'origine). GPU : UN seul upload H2D des
    plans collectés via un buffer hôte ÉPINGLÉ persistant (1 H2D/trame — sinon régression, banc
    Phase 0), puis resize+place en VRAM. Tuiles disjointes → l'ordre de placement n'importe pas."""
    if not batch:
        return
    if not GPU:
        for (sy, su, sv, vy, vh, vx, vw) in batch:
            cy[vy:vy+vh, vx:vx+vw] = resize_plane(sy, vh, vw)
            cu[vy//_CH:vy//_CH+vh//_CH, vx//_CW:vx//_CW+vw//_CW] = resize_plane(su, vh//_CH, vw//_CW)
            cv[vy//_CH:vy//_CH+vh//_CH, vx//_CW:vx//_CW+vw//_CW] = resize_plane(sv, vh//_CH, vw//_CW)
        return
    total = 0
    for it in batch:
        total += it[0].size + it[1].size + it[2].size
    if _pin["cap"] < total:                     # (ré)alloue le staging épinglé + device (croissance ×2)
        cap = max(total, (_pin["cap"] * 2) or total)
        _pin["host"] = np.frombuffer(cp.cuda.alloc_pinned_memory(cap * _BPS), dtype=_NP_DT, count=cap)
        _pin["dev"] = cp.empty(cap, dtype=_NP_DT)
        _pin["cap"] = cap
    host = _pin["host"]; dev = _pin["dev"]
    off = 0; meta = []
    for (sy, su, sv, vy, vh, vx, vw) in batch:   # concatène les plans dans le buffer hôte épinglé
        ny, nu = sy.size, su.size
        host[off:off+ny] = sy.ravel()
        host[off+ny:off+ny+nu] = su.ravel()
        host[off+ny+nu:off+ny+2*nu] = sv.ravel()
        meta.append((off, sy.shape, su.shape, sv.shape, vy, vh, vx, vw))
        off += ny + 2 * nu
    dev[:total].set(host[:total])                # UN seul H2D (épinglé → device)
    for (o, shy, shu, shv, vy, vh, vx, vw) in meta:
        ny = shy[0] * shy[1]; nu = shu[0] * shu[1]
        gy = dev[o:o+ny].reshape(shy)
        gu = dev[o+ny:o+ny+nu].reshape(shu)
        gv = dev[o+ny+nu:o+ny+2*nu].reshape(shv)
        cy[vy:vy+vh, vx:vx+vw] = resize_plane(gy, vh, vw)
        cu[vy//_CH:vy//_CH+vh//_CH, vx//_CW:vx//_CW+vw//_CW] = resize_plane(gu, vh//_CH, vw//_CW)
        cv[vy//_CH:vy//_CH+vh//_CH, vx//_CW:vx//_CW+vw//_CW] = resize_plane(gv, vh//_CH, vw//_CW)

# ─── Rendu d'overlay (PIL) ───────────────────────────────────

def _make_font(size):
    """Police d'overlay à la taille demandée. DejaVu Bold si présente (image avec
    fonts-dejavu-core), sinon repli sur la police par défaut Pillow — SCALABLE
    depuis Pillow 10.1 (load_default(size)) → le texte grossit même sans DejaVu."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.load_default(size)
        except Exception:
            return ImageFont.load_default()

_font_cache = {{}}
def _font(size):
    f = _font_cache.get(size)
    if f is None:
        f = _make_font(size); _font_cache[size] = f
    return f

def _label_metrics(cfg):
    """Tailles d'habillage PAR FENÊTRE, plafonnées à la hauteur du PiP : le texte ne
    dépasse jamais 30 % de la hauteur, le bandeau 40 % (45 % pour les barres hautes
    classic/stylized). Sans plafond, un label_size global énorme mangeait la fenêtre.
    `label_proportional` (par fenêtre) : le texte vaut LABEL_SIZE quand la fenêtre fait
    1/4 de l'image (h = OUT_HEIGHT/2), puis grossit/réduit linéairement avec la hauteur
    (les plafonds en aval bornent toujours contre le débordement).
    Formules miroir côté éditeur : multiview.js dessiner()."""
    h = max(2, int(cfg.get("h") or 0))
    if cfg.get("label_proportional"):
        eff = max(6, int(round(LABEL_SIZE * (2.0 * h / OUT_HEIGHT))))
    else:
        eff = max(6, min(LABEL_SIZE, int(h * 0.30)))
    bar = min(max(14, int(round(eff * 2))), max(8, int(h * 0.40)))
    eff = max(6, min(eff, bar - 4))
    cbar = min(max(bar, int(round(eff * 2.5))), max(8, int(h * 0.45)))   # barre classic
    grad = min(max(bar, int(round(eff * 2.2))), max(8, int(h * 0.45)))   # dégradé stylized
    return {{
        "size": eff, "font": _font(eff),
        "bar_h": bar, "cbar": cbar, "grad_h": grad,
        "tally": max(4, min(int(round(eff * 1.4)), bar - 2)),
        "pad": max(2, int(round(eff * 0.35))),
    }}

def _bar_reserved_h(cfg, m):
    """Hauteur réservée par le bandeau du style courant (pour le mode « sous l'image »)."""
    if FRAME_STYLE == "classic":
        return m["cbar"]
    if FRAME_STYLE == "stylized":
        return m["grad_h"]
    return m["bar_h"]

def _frame_metrics(cfg):
    """Marges réservées par l'habillage (frame_style) AUTOUR de l'image — l'image est
    réduite homothétiquement dans la zone restante (_video_rect), rien n'est recouvert.
    Épaisseur auto : t = 3 % du petit côté de la fenêtre, bornée 3..24 px.
    Renvoie {{t, ml, mt, mr, mb, band}} ; band = hauteur de la zone nom/tally du style.
    Fenêtre trop petite : marges dégradées (bande puis côtés) plutôt que d'écraser
    l'image. Formules miroir côté éditeur : multiview.js _frameMetrics()."""
    w = max(2, int(cfg.get("w") or 0)); h = max(2, int(cfg.get("h") or 0))
    m = _label_metrics(cfg)
    t = max(3, min(24, int(round(min(w, h) * 0.03))))
    show_label = bool(cfg.get("show_label"))
    bar_on = show_label or bool(cfg.get("show_tally"))
    band = m["bar_h"]
    ml = mt = mr = mb = 0
    if FRAME_STYLE == "stylized":        # Moniteur : bezel + menton nom/LED
        bez = max(4, int(round(t * 2.2)))
        ml = mr = mt = bez
        mb = bez + (band if bar_on else 0)
    elif FRAME_STYLE == "classic":       # UMD : cadre fin + boîtier sous l'image
        f = max(2, t // 2)
        ml = mr = mt = f
        mb = f + ((band + max(2, t // 2)) if bar_on else 0)
    elif FRAME_STYLE == "tally_border":  # cadre FIN (3 px) toujours visible + onglet nom AU-DESSUS
        b = 3
        ml = mr = mb = b
        mt = b + (band if show_label else 0)
    elif FRAME_STYLE == "viewfinder":    # équerres + chip nom
        ml = mr = mb = t
        mt = (band + 4) if show_label else t
    elif FRAME_STYLE == "flat":          # nom + soulignement pleine largeur
        mb = t + ((band + 2) if show_label else 0)
    if h - mt - mb < 16:
        mb = max(0, min(mb, h - 16 - mt))
        if h - mt - mb < 16:
            mt = max(0, h - 16 - mb)
    if w - ml - mr < 16:
        side = max(0, (w - 16) // 2)
        ml = min(ml, side); mr = min(mr, side)
    return {{"t": t, "ml": ml, "mt": mt, "mr": mr, "mb": mb, "band": band}}

def _umd_geom(cfg, m, fm):
    """Boîtier UMD (style classic) : rect (x0, y0, x1, y1) centré sous l'image,
    largeur fixe ~55 % de la cellule (le texte est ajusté dedans), hauteur = band.
    Les lampes tally G/D se posent de part et d'autre (render_dynamic)."""
    x, y, w, h = cfg["x"], cfg["y"], cfg["w"], cfg["h"]
    lamp_zone = m["tally"] + 2 * m["pad"] + 6
    bw = max(24, min(int(w * 0.55), w - 2 * lamp_zone))
    bx = x + (w - bw) // 2
    by = y + h - fm["band"]
    return bx, by, bx + bw - 1, y + h - 1

def _fit_text(d, text, m, max_w):
    """Police du label, réduite si le texte déborde de max_w (sinon il bave sur les
    fenêtres voisines : l'overlay est dessiné sur le canvas complet)."""
    font = m["font"]
    if not text or max_w <= 0:
        return font
    try:
        tw = d.textlength(text, font=font)
        if tw > max_w > 0:
            return _font(max(6, int(m["size"] * max_w / tw)))
    except Exception:
        pass
    return font

def _is_protocol_label(cfg):
    return cfg.get("show_label") and cfg.get("label_source") == "protocol"

def _window_tally_dominant(i):
    """Dominante des deux slots L/R d'une fenêtre → "red"|"green"|"amber"|"off".
    Rouge prioritaire ; rouge+vert simultanés → amber."""
    stL = tally_state.get(f"{{i}}_L", "off")
    stR = tally_state.get(f"{{i}}_R", "off")
    states = {{stL, stR}}
    if "amber" in states:
        return "amber"
    has_red   = "red" in states
    has_green = "green" in states
    if has_red and has_green:
        return "amber"
    if has_red:
        return "red"
    if has_green:
        return "green"
    return "off"

def _render_border_colored(d, x, y, w, h, color_rgba, thickness):
    """Rectangle plein de `thickness` px tout autour de (x,y,w,h)."""
    if color_rgba[3] == 0 or thickness <= 0:
        return
    for k in range(thickness):
        d.rectangle([x + k, y + k, x + w - 1 - k, y + h - 1 - k], outline=color_rgba)

def _video_rect(cfg):
    """Géométrie UNIQUE de la cellule — partagée par la boucle composite, la couche
    bordure et les meters (plus de copie divergente). Renvoie un dict :
      x/y/w/h    : cellule clampée au canvas (dimensions paires) ;
      vx/vy/vw/vh: rectangle de l'IMAGE vidéo — exclut la bande VU « hors image » et
                   le bandeau « sous l'image ». L'image est réduite HOMOTHÉTIQUEMENT
                   (ratio de la cellule conservé) dans la zone restante et centrée
                   (pillarbox/letterbox) : on ne déforme JAMAIS l'image, ni pour le
                   bandeau ni pour la bande VU ;
      m          : _label_metrics(cfg) (tailles d'habillage par-fenêtre)."""
    m = _label_metrics(cfg)
    x, y, w, h = cfg["x"], cfg["y"], cfg["w"], cfg["h"]
    x = max(0, min(x, OUT_WIDTH - 1)); y = max(0, min(y, OUT_HEIGHT - 1))
    w = max(2, min(w, OUT_WIDTH - x)); h = max(2, min(h, OUT_HEIGHT - y))
    w -= w % 2; h -= h % 2
    bar_on = bool(cfg.get("show_label") or cfg.get("show_tally"))
    fm = None
    if FRAME_STYLE in _DRESS_STYLES:
        # Habillage v0.12 : marges réservées sur les 4 côtés (nom/tally intégrés au
        # style) — le bandeau legacy OVERLAY_BELOW ne s'applique qu'au style 'none'.
        fm = _frame_metrics(cfg)
        ay = y + fm["mt"]
        avail_h = h - fm["mt"] - fm["mb"]
        ax0 = x + fm["ml"]
        avail_w0 = w - fm["ml"] - fm["mr"]
    else:
        # 'none' : hauteur amputée du bandeau (mode « sous l'image »).
        ay = y
        avail_h = h - _bar_reserved_h(cfg, m) if (OVERLAY_BELOW and bar_on) else h
        ax0 = x
        avail_w0 = w
    if avail_h < 2:
        avail_h = h; ay = y  # cellule trop petite — fallback overlay
    meter_n = int(cfg.get("meter_channels") or 0)
    moff = 0
    if meter_n > 0 and not cfg.get("meter_inside"):
        moff = _meter_layout(meter_n) + 4
        moff += moff % 2
    avail_w = max(2, avail_w0 - moff)
    ax = ax0 + (moff if cfg.get("meter_position") == "left" else 0)
    # Fit homothétique du ratio de la cellule (w×h) dans la zone, centré.
    scale = min(avail_w / w, avail_h / h)
    vw = max(2, int(w * scale)); vw -= vw % 2
    vh = max(2, int(h * scale)); vh -= vh % 2
    vx = ax + (avail_w - vw) // 2
    vx -= vx % 2   # alignement chroma (vx//_CW)
    vy = ay + (avail_h - vh) // 2
    vy -= vy % 2   # alignement chroma (vy//_CH en 4:2:0)
    return {{"x": x, "y": y, "w": w, "h": h,
             "vx": vx, "vy": vy, "vw": vw, "vh": vh,
             "ay": ay,         # haut de la ZONE disponible (pour les VU-mètres)
             "ah": avail_h,    # hauteur de la ZONE hors bandeau/marges (idem)
             "fm": fm,         # marges d'habillage (None pour 'none')
             "m": m}}

def _render_pill(d, cx, cy, r, fill, outline):
    """Pastille ronde centrée (cx, cy) de rayon r."""
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=2)

def _render_gradient_bar(img, x, y, w, h, rgb, a_top, a_bot):
    """Dégradé vertical alpha (a_top → a_bot) de couleur `rgb`, collé sur `img` RGBA."""
    if w <= 0 or h <= 0:
        return
    ramp = np.linspace(a_top, a_bot, h).astype(np.uint8)          # (h,)
    alpha = np.repeat(ramp[:, None], w, axis=1)                   # (h, w)
    tile = np.zeros((h, w, 4), dtype=np.uint8)
    tile[..., 0] = rgb[0]; tile[..., 1] = rgb[1]; tile[..., 2] = rgb[2]
    tile[..., 3] = alpha
    img.alpha_composite(Image.fromarray(tile, "RGBA"), (x, y))

def render_static():
    """Pré-rendu une fois : décor fixe (bordures + bandeau + labels statiques).
    Les labels 'protocol' et les éléments dépendant du tally → render_dynamic."""
    img = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for cfg in FLUX_CONFIG:
        if cfg.get("hidden"):
            continue
        x, y, w, h = cfg["x"], cfg["y"], cfg["w"], cfg["h"]
        m = _label_metrics(cfg)
        show_label = bool(cfg.get("show_label"))
        show_tally = bool(cfg.get("show_tally"))
        bar_on = show_label or show_tally
        static_label = show_label and not _is_protocol_label(cfg)
        name = cfg.get("name", "") or ""

        if FRAME_STYLE == "classic":
            # UMD broadcast : boîtier noir centré sous l'image. Lampes G/D et texte
            # protocole en dynamique ; cadre fin dans render_border.
            if show_label:
                fm = _frame_metrics(cfg)
                bx0, by0, bx1, by1 = _umd_geom(cfg, m, fm)
                d.rectangle([bx0, by0, bx1, by1], fill=(8, 8, 10, 255),
                            outline=(130, 130, 138, 255))
                if static_label:
                    d.text(((bx0 + bx1) // 2, (by0 + by1) // 2), name,
                           font=_fit_text(d, name, m, bx1 - bx0 - 2 * m["pad"]),
                           fill=(240, 240, 245, 255), anchor="mm")

        elif FRAME_STYLE == "tally_border":
            pass  # onglet nom : suit la teinte tally → render_dynamic

        elif FRAME_STYLE == "stylized":
            # Moniteur : nom « gravé » dans le menton du bezel (corps du bezel dans
            # render_border, LED tally en dynamique).
            if static_label:
                fm = _frame_metrics(cfg)
                cx = x + w // 2
                cy = y + h - fm["mb"] // 2 - 1
                font = _fit_text(d, name, m, w - 2 * fm["ml"] - 4 * m["pad"])
                d.text((cx, cy - 1), name, font=font, fill=(8, 8, 10, 255), anchor="mm")
                d.text((cx, cy), name, font=font, fill=(168, 168, 178, 255), anchor="mm")

        elif FRAME_STYLE == "viewfinder":
            pass  # chip nom : pastille d'état tally intégrée → render_dynamic

        elif FRAME_STYLE == "flat":
            # Nom au-dessus du soulignement, aligné sur le bord gauche de l'image
            # (le trait, coloré tally, est dans render_border).
            if static_label:
                g = _video_rect(cfg)
                fm = g["fm"]
                d.text((g["vx"] + 2, y + h - fm["t"] - 2 - fm["band"] // 2), name,
                       font=_fit_text(d, name, m, g["vw"] - 4),
                       fill=(235, 235, 240, 255), anchor="lm")

        else:  # "none" — comportement historique
            # Bordure globale → couche bordure (render_border), au-dessus du bandeau.
            if bar_on:
                bar_top = y + h - m["bar_h"]
                d.rectangle([x, bar_top, x + w, y + h], fill=(0, 0, 0, 180))
                if static_label:
                    text_l, text_r = x, x + w
                    if show_tally:
                        text_l += m["pad"] + m["tally"] + 4
                        text_r -= m["pad"] + m["tally"] + 4
                    d.text(((text_l + text_r) // 2, bar_top + m["bar_h"] // 2), name,
                           font=_fit_text(d, name, m, text_r - text_l),
                           fill=(255, 255, 255, 255), anchor="mm")
    return img   # RGBA — consolidation : converti une seule fois après alpha_composite

def render_dynamic():
    """Re-rendu à chaque changement Tally/TSL : éléments dépendant de l'état tally
    (bordures colorées, pastilles, lampes, labels protocole)."""
    img = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, cfg in enumerate(FLUX_CONFIG):
        if cfg.get("hidden"):
            continue
        show_label = bool(cfg.get("show_label"))
        show_tally = bool(cfg.get("show_tally"))
        if not (show_label or show_tally):
            continue
        x, y, w, h = cfg["x"], cfg["y"], cfg["w"], cfg["h"]
        m = _label_metrics(cfg)
        tally_sz, tally_pad = m["tally"], m["pad"]
        is_proto = _is_protocol_label(cfg)
        proto_txt = (tsl_text.get(i, "") or "") if is_proto else ""
        dom = _window_tally_dominant(i)

        if FRAME_STYLE == "classic":
            # UMD : lampes carrées G/D de part et d'autre du boîtier + texte protocole.
            fm = _frame_metrics(cfg)
            bx0, by0, bx1, by1 = _umd_geom(cfg, m, fm)
            cy = (by0 + by1) // 2
            if show_tally:
                ty = cy - tally_sz // 2
                for st, lx in ((tally_state.get(f"{{i}}_L", "off"),
                                bx0 - tally_pad - 6 - tally_sz),
                               (tally_state.get(f"{{i}}_R", "off"),
                                bx1 + tally_pad + 6)):
                    fill = _TALLY_BORDER_RGBA.get(st, (0, 0, 0, 0))
                    if fill[3] == 0:
                        fill = (40, 40, 44, 220)
                    d.rectangle([lx, ty, lx + tally_sz, ty + tally_sz],
                                fill=fill, outline=(205, 205, 212, 255))
            if is_proto and proto_txt and show_label:
                d.text(((bx0 + bx1) // 2, cy), proto_txt,
                       font=_fit_text(d, proto_txt, m, bx1 - bx0 - 2 * tally_pad),
                       fill=(240, 240, 245, 255), anchor="mm")

        elif FRAME_STYLE == "tally_border":
            # Onglet nom intégré sous le bord haut, teinté par la dominante tally
            # (la bordure épaisse est tracée par render_border).
            if show_label:
                g = _video_rect(cfg)
                fm = g["fm"]
                b = fm["ml"]
                txt = proto_txt if is_proto else (cfg.get("name", "") or "")
                # Onglet AU-DESSUS du cadre fin (dehors), aligné sur le bord externe du cadre.
                tab_y0 = g["vy"] - b - fm["band"]
                tab_y1 = g["vy"] - b - 1
                try:
                    tw = int(d.textlength(txt, font=m["font"])) if txt else 0
                except Exception:
                    tw = 0
                tab_w = max(24, min(g["vw"] + 2 * b, tw + 4 * tally_pad))
                tab_x0 = g["vx"] - b
                d.rectangle([tab_x0, tab_y0, tab_x0 + tab_w - 1, tab_y1],
                            fill=_BAR_TINTS.get(dom, _BAR_TINTS["off"]))
                if txt:
                    d.text((tab_x0 + 2 * tally_pad, (tab_y0 + tab_y1) // 2), txt,
                           font=_fit_text(d, txt, m, tab_w - 4 * tally_pad),
                           fill=(245, 245, 250, 255), anchor="lm")

        elif FRAME_STYLE == "stylized":
            # Moniteur : LED tally rondes G/D dans le menton + texte protocole gravé.
            fm = _frame_metrics(cfg)
            cy = y + h - fm["mb"] // 2 - 1
            if show_tally:
                r = max(3, min(max(3, fm["band"] // 3), int(round(m["size"] * 0.45))))
                stL = tally_state.get(f"{{i}}_L", "off")
                stR = tally_state.get(f"{{i}}_R", "off")
                fL, oL = _PILL_COLORS.get(stL, _PILL_COLORS["off"])
                fR, oR = _PILL_COLORS.get(stR, _PILL_COLORS["off"])
                _render_pill(d, x + fm["ml"] + r + 4,     cy, r, fL, oL)
                _render_pill(d, x + w - fm["mr"] - r - 4, cy, r, fR, oR)
            if is_proto and proto_txt:
                font = _fit_text(d, proto_txt, m, w - 2 * fm["ml"] - 4 * m["pad"])
                d.text((x + w // 2, cy - 1), proto_txt, font=font,
                       fill=(8, 8, 10, 255), anchor="mm")
                d.text((x + w // 2, cy), proto_txt, font=font,
                       fill=_TALLY_TEXT_BY_DOMINANT[dom], anchor="mm")

        elif FRAME_STYLE == "viewfinder":
            # Chip nom arrondi haut-gauche + pastille d'état (équerres → render_border).
            if show_label:
                g = _video_rect(cfg)
                fm = g["fm"]
                txt = proto_txt if is_proto else (cfg.get("name", "") or "")
                chip_h = fm["band"]
                dot_r = max(2, int(round(m["size"] * 0.3))) if show_tally else 0
                try:
                    tw = int(d.textlength(txt, font=m["font"])) if txt else 0
                except Exception:
                    tw = 0
                chip_w = max(20, min(w - 8, tw + 4 * tally_pad
                                     + ((dot_r * 2 + tally_pad) if dot_r else 0)))
                cx0 = g["vx"]
                cy0 = y + 1
                d.rounded_rectangle([cx0, cy0, cx0 + chip_w - 1, cy0 + chip_h - 1],
                                    radius=max(2, chip_h // 2), fill=(10, 10, 12, 215))
                tx = cx0 + 2 * tally_pad
                if dot_r:
                    dot_fill = _TALLY_BORDER_RGBA[dom] if dom != "off" else (90, 90, 98, 255)
                    d.ellipse([tx, cy0 + chip_h // 2 - dot_r,
                               tx + 2 * dot_r, cy0 + chip_h // 2 + dot_r], fill=dot_fill)
                    tx += 2 * dot_r + tally_pad
                if txt:
                    d.text((tx, cy0 + chip_h // 2), txt,
                           font=_fit_text(d, txt, m, cx0 + chip_w - tx - 2 * tally_pad),
                           fill=(240, 240, 245, 255), anchor="lm")

        elif FRAME_STYLE == "flat":
            # Le soulignement (couleur tally) est dans render_border ; ici seul le
            # texte protocole (le nom statique est dans render_static).
            if is_proto and proto_txt:
                g = _video_rect(cfg)
                fm = g["fm"]
                d.text((g["vx"] + 2, y + h - fm["t"] - 2 - fm["band"] // 2), proto_txt,
                       font=_fit_text(d, proto_txt, m, g["vw"] - 4),
                       fill=(235, 235, 240, 255), anchor="lm")

        else:  # "none" — comportement historique
            bar_top = y + h - m["bar_h"]
            if is_proto:
                text_l, text_r = x, x + w
                tally_color = tally_state.get(f"{{i}}_L", "off")
                if show_tally:
                    text_l += tally_pad + tally_sz + 4
                    text_r -= tally_pad + tally_sz + 4
                bg = TALLY_TEXT_BG.get(tally_color)
                if bg:
                    d.rectangle([text_l, bar_top, text_r, y + h], fill=bg)
                txt_fill = TALLY_TEXT_COLORS.get(tally_color, (200, 200, 200, 255))
                d.text(((text_l + text_r) // 2, bar_top + m["bar_h"] // 2), proto_txt,
                       font=_fit_text(d, proto_txt, m, text_r - text_l),
                       fill=txt_fill, anchor="mm")
            if show_tally:
                # Pavés tally centrés sur le rectangle VIDÉO (vx/vw), pas la cellule entière
                # (qui inclut la bande VU audio).
                g = _video_rect(cfg)
                vx, vw = g["vx"], g["vw"]
                ty = bar_top + (m["bar_h"] - tally_sz) // 2
                stL = tally_state.get(f"{{i}}_L", "off")
                stR = tally_state.get(f"{{i}}_R", "off")
                cL = TALLY_COLORS[stL]; bL = TALLY_BORDER_COLORS[stL]
                cR = TALLY_COLORS[stR]; bR = TALLY_BORDER_COLORS[stR]
                d.rectangle([vx + tally_pad, ty,
                             vx + tally_pad + tally_sz, ty + tally_sz],
                            fill=cL, outline=bL)
                d.rectangle([vx + vw - tally_pad - tally_sz, ty,
                             vx + vw - tally_pad, ty + tally_sz],
                            fill=cR, outline=bR)
    return img   # RGBA — consolidation : converti une seule fois après alpha_composite

def render_border():
    """Couche bordure SEULE, blendée juste APRÈS la vidéo : au-dessus de l'image
    mais SOUS le bandeau, les labels, les pavés tally et les VU-mètres (elle ne
    barre plus la zone de texte). Cerne le rectangle image (_video_rect, même
    géométrie que la boucle composite) selon le style courant. Dépend du tally
    pour 'tally_border'."""
    img = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, cfg in enumerate(FLUX_CONFIG):
        if cfg.get("hidden"):
            continue
        g = _video_rect(cfg)
        x, y, w, h = g["x"], g["y"], g["w"], g["h"]
        vx, vy, vw, vh = g["vx"], g["vy"], g["vw"], g["vh"]
        fm = g["fm"]
        dom = _window_tally_dominant(i)

        if FRAME_STYLE == "stylized":
            # Bezel « moniteur » : corps sombre arrondi plein sur la cellule, trou
            # percé pour l'écran (ImageDraw écrit les pixels : fill transparent =
            # trou), lèvre interne claire autour de l'écran.
            t = fm["t"]
            rad = max(3, t)
            inset = max(1, t // 2)
            d.rounded_rectangle([x, y, x + w - 1, y + h - 1], radius=rad,
                                fill=(46, 46, 53, 255))
            d.rounded_rectangle([x + inset, y + inset,
                                 x + w - 1 - inset, y + h - 1 - inset],
                                radius=max(2, rad - inset), fill=(27, 27, 32, 255))
            d.rectangle([vx, vy, vx + vw - 1, vy + vh - 1], fill=(0, 0, 0, 0))
            d.rectangle([vx - 1, vy - 1, vx + vw, vy + vh], outline=(98, 98, 108, 255))

        elif FRAME_STYLE == "classic":
            # UMD : cadre fin neutre cernant l'image.
            f = max(2, fm["t"] // 2)
            _render_border_colored(d, vx - f, vy - f, vw + 2 * f, vh + 2 * f,
                                   _FRAME_NEUTRAL, f)

        elif FRAME_STYLE == "tally_border":
            # Bordure FINE (3 px) TOUJOURS visible (neutre au repos), cernant SEULEMENT l'image.
            # L'onglet nom est posé AU-DESSUS du cadre (dehors) → render_dynamic.
            b = fm["ml"]
            color = _TALLY_BORDER_RGBA[dom] if dom != "off" else _FRAME_NEUTRAL
            _render_border_colored(d, vx - b, vy - b, vw + 2 * b, vh + 2 * b, color, b)

        elif FRAME_STYLE == "viewfinder":
            # Équerres de viseur aux 4 coins (blanches au repos, couleur tally sinon).
            bt = max(2, fm["t"] // 2)
            gap = max(2, fm["t"] // 2)
            color = _TALLY_BORDER_RGBA[dom] if dom != "off" else (225, 225, 232, 255)
            arm = max(8, int(round(min(vw, vh) * 0.14)))
            x0, y0 = vx - gap, vy - gap
            x1, y1 = vx + vw - 1 + gap, vy + vh - 1 + gap
            for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                                   (x0, y1, 1, -1), (x1, y1, -1, -1)):
                ex, ey = cx + sx * arm, cy + sy * (bt - 1)
                d.rectangle([min(cx, ex), min(cy, ey), max(cx, ex), max(cy, ey)],
                            fill=color)
                ex, ey = cx + sx * (bt - 1), cy + sy * arm
                d.rectangle([min(cx, ex), min(cy, ey), max(cx, ex), max(cy, ey)],
                            fill=color)

        elif FRAME_STYLE == "flat":
            # Soulignement plein pleine largeur, collé au bas de la cellule.
            color = _TALLY_BORDER_RGBA[dom] if dom != "off" else _FLAT_NEUTRAL
            d.rectangle([vx, y + h - fm["t"], vx + vw - 1, y + h - 1], fill=color)

        else:  # "none" — bordure globale historique
            if BORDER_W > 0:
                for k in range(BORDER_W):
                    d.rectangle([vx + k, vy + k, vx + vw - 1 - k, vy + vh - 1 - k],
                                outline=BORDER_COLOR)
    return img   # RGBA — consolidation : converti une seule fois après alpha_composite

def _meter_label_tile(status, bx0, by0, bx1, by1, mx, my, mw, mh):
    """TUILE YUV séparée : étiquette VERTICALE « SILENCE » (signal muet) ou « ABSENCE » (flux coupé),
    petite, centrée sur la zone des barres — pour SAVOIR pourquoi il n'y a pas de son. Posée en tuile
    INDÉPENDANTE (PIL→YUV), donc rendu IDENTIQUE sur les chemins GPU et CPU, et payée uniquement hors
    état « ok ». Renvoie un tuple tuile (même bbox que le meter) ou None."""
    txt = "ABSENCE" if status == "absence" else "SILENCE"
    col = (236, 72, 60) if status == "absence" else (240, 184, 44)   # rouge / ambre
    tw_, th_ = bx1 - bx0, by1 - by0
    if tw_ <= 0 or th_ <= 0:
        return None
    bars_mh = max(20, mh - 12)
    # Caractères DROITS (non tournés), EMPILÉS verticalement (1 par ligne, lu de haut en bas).
    # PETIT et DISCRET : taille ≈ celle des numéros de canaux (pas plus gros), bornée à la largeur
    # du meter. Centré verticalement sur la zone des barres.
    nch = max(1, len(txt))
    # Largeur de la zone des BARRES (hors bande des graduations/dB à gauche, METER_TICK_W).
    bars_w = max(1, mw - METER_TICK_W)
    fsz = max(7, min(10, bars_w, (bars_mh - 2) // nch))
    f = _font(fsz)
    line_h = fsz + 1
    img = Image.new("RGBA", (tw_, th_), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Centré horizontalement sur les CANAUX (milieu des barres, pas du meter entier) : entre les
    # canaux 4 et 5 sur 8, entre 1 et 2 sur 2, etc.
    cx = (mx - bx0) + METER_TICK_W + bars_w // 2
    # Aligné EN BAS de la zone des barres (juste au-dessus des numéros de canaux).
    y0 = (my - by0) + bars_mh - line_h // 2 - (nch - 1) * line_h
    for i, ch in enumerate(txt):
        cyc = y0 + i * line_h
        d.text((cx + 1, cyc + 1), ch, font=f, fill=(8, 8, 10, 255), anchor="mm")   # liseré sombre
        d.text((cx, cyc), ch, font=f, fill=col + (255,), anchor="mm")
    oy, ou, ov, oa, oa2 = rgba_to_yuv(img)
    return (bx0, by0, bx1, by1, oy, ou, ov, oa, oa2)

def render_meters(now):
    """Re-rendu par frame des peak meters. Renvoie une LISTE de TUILES YUV prêtes à blender
    [(bx0, by0, bx1, by1, y, u, v, a, a2), ...] — UNE tuile par meter, sur sa propre bbox locale
    (alignée chroma paire), PAS une couche plein écran. Les VU changent à chaque trame (jamais
    cachés) : convertir/blender chaque petit rectangle (≈ une bande VU dans une cellule) au lieu
    d'une bbox-union quasi plein écran (meters dispersés sur un mur 16 fenêtres) supprime le gros
    du coût de l'habillage per-frame. Renvoie None si aucun meter activé."""
    tiles = []
    for i, cfg in enumerate(FLUX_CONFIG):
        n = int(cfg.get("meter_channels") or 0)
        if n == 0 or cfg.get("hidden"):
            continue
        # État audio lazy-init depuis le shm vidéo dérivé
        st = audio_states.get(i)
        if st is None:
            st = _open_audio_state(i, cfg.get("path") or "")
            audio_states[i] = st  # peut être None (audio absent) → on dessine quand même un meter "vide"
        peaks = holds = None
        status = "absence"   # pas de flux audio du tout → absence (st None)
        if st is not None:
            peaks, holds, status = _update_peaks(st, n, now)
        if peaks is None:
            peaks = np.full(n, METER_MIN_DB)
            holds = np.full(n, METER_MIN_DB)
        # Geometry du meter dans la cellule (géométrie partagée _video_rect). Le meter
        # occupe toute la hauteur de la ZONE hors bandeau (ah), pas celle de la vidéo
        # letterboxée — la bande VU réserve sa largeur, la vidéo est réduite au ratio.
        g = _video_rect(cfg)
        x, y, w, h = g["x"], g["y"], g["w"], g["h"]
        mw = _meter_layout(n)
        mh = max(20, g["ah"] - 4)
        if mw >= w or mh < 20:
            continue
        inside = bool(cfg.get("meter_inside"))
        # Bande VU posée DANS l'habillage (entre la marge du cadre et l'image),
        # alignée verticalement sur la zone disponible (ay/ah).
        fm = g.get("fm")
        ml = fm["ml"] if fm else 0
        mr = fm["mr"] if fm else 0
        if cfg.get("meter_position") == "left":
            mx = x + ml + 2
        else:
            mx = x + w - mr - mw - 2
        my = g["ay"] + 2
        opacity_pct = int(cfg.get("meter_opacity") or 70)
        if not inside:
            opacity_pct = 100  # hors image = totalement opaque (zone réservée)
        # bbox locale du meter (_draw_meter dessine jusqu'à mx+mw / my+mh inclus → +1), bornée à
        # la sortie et alignée chroma : origine ramenée à un multiple de _CW/_CH, dimensions
        # complétées au multiple supérieur (rgba_to_yuv sous-échantillonne par _CW/_CH).
        bx0 = max(0, mx); by0 = max(0, my)
        bx1 = min(OUT_WIDTH, mx + mw + 1); by1 = min(OUT_HEIGHT, my + mh + 1)
        bx0 -= bx0 % _CW; by0 -= by0 % _CH
        if (bx1 - bx0) % _CW: bx1 = min(OUT_WIDTH, bx1 + (_CW - (bx1 - bx0) % _CW))
        if (by1 - by0) % _CH: by1 = min(OUT_HEIGHT, by1 + (_CH - (by1 - by0) % _CH))
        if bx1 <= bx0 or by1 <= by0:
            continue
        if GPU:
            # Chemin GPU : statique caché en VRAM + barres composées en RGBA (xp), aucune PIL par trame.
            oy, ou, ov, oa, oa2 = _meter_tile_gpu(bx1 - bx0, by1 - by0, mx - bx0, my - by0, mw, mh, n,
                                                  peaks, holds, cfg.get("meter_scale") or "dbfs", opacity_pct)
        else:
            # Chemin CPU ÉPROUVÉ — INCHANGÉ (PIL par trame). Ne pas modifier (garantie 'sans GPU identique').
            tile = Image.new("RGBA", (bx1 - bx0, by1 - by0), (0, 0, 0, 0))
            _draw_meter(tile, mx - bx0, my - by0, mw, mh, n, peaks, holds,
                        cfg.get("meter_scale") or "dbfs", opacity_pct)
            oy, ou, ov, oa, oa2 = rgba_to_yuv(tile)
        tiles.append((bx0, by0, bx1, by1, oy, ou, ov, oa, oa2))
        # Étiquette SILENCE / ABSENCE par-dessus (tuile séparée, même bbox → blend après le meter).
        # Cosmétique → ne jamais casser le rendu d'une trame si l'étiquette échoue.
        if status in ("silence", "absence"):
            try:
                lab = _meter_label_tile(status, bx0, by0, bx1, by1, mx, my, mw, mh)
            except Exception:
                lab = None
            if lab is not None:
                tiles.append(lab)
    return tiles or None


# ─── Couche « info » : NO SIGNAL / FREEZE / format source ────────────────────
# Cachée par signature (statuts + textes) : le coût PIL n'est payé qu'aux
# transitions d'état, pas par frame. Posée SOUS l'habillage (bordure, labels et
# meters passent par-dessus), dans le rectangle IMAGE de chaque fenêtre.

def _fmt_chip_txt(cfg, src):
    """« 1920×1080p25 » — format DÉCLARÉ par le câblage (in_w/in_h/in_scan/in_fps,
    résolus par l'orchestrateur depuis le producteur du shm). Repli : dims réelles
    du mmap, sans cadence, si le producteur est inconnu en DB."""
    w = int(cfg.get("in_w") or 0); h = int(cfg.get("in_h") or 0)
    if not (w and h) and src is not None:
        w, h = src["in_w"], src["in_h"]
    if not (w and h):
        return ""
    fps = cfg.get("in_fps")
    if not fps:
        base = f"{{w}}×{{h}}"
    else:
        n, d = _rate_nd(fps)
        fps_txt = str(n) if d == 1 else str(round(n / d, 2))
        scan = "i" if str(cfg.get("in_scan") or "p").startswith("i") else "p"
        base = f"{{w}}×{{h}}{{scan}}{{fps_txt}}"
    colo = str(cfg.get("in_colorimetry") or "").strip()
    if colo:
        base += " " + (("BT." + colo) if colo.isdigit() else colo.upper())
    return base

def render_info(statuses):
    """Layer RGBA plein canvas : placeholders NO SIGNAL, badges FREEZE, chips format.
    `statuses` = [(idx, statut, fmt_txt)] — exactement la signature qui a déclenché le bake."""
    img = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, status, fmt_txt, proxy in statuses:
        if i >= len(FLUX_CONFIG):
            continue
        g = _video_rect(FLUX_CONFIG[i])
        vx, vy, vw, vh = g["vx"], g["vy"], g["vw"], g["vh"]
        if proxy:
            # Badge proxy (mode ingénierie) en haut-gauche, couleur selon la classe de coût.
            _plabel, _pcost = proxy
            _pcol = {{"copy": (40, 170, 90, 235), "strided": (210, 150, 0, 235)}}.get(_pcost, (200, 70, 55, 235))
            _pglyph = {{"copy": "✓", "strided": "~"}}.get(_pcost, "↯")
            ptxt = f"{{_plabel}} {{_pglyph}}"
            fnt = _font(max(8, min(14, vh // 14)))
            pad = max(2, vh // 120)
            tb = d.textbbox((0, 0), ptxt, font=fnt)
            bw, bh = (tb[2] - tb[0]) + 2 * pad + 2, (tb[3] - tb[1]) + 2 * pad
            d.rounded_rectangle([vx + 3, vy + 3, vx + 3 + bw, vy + 3 + bh], radius=2, fill=_pcol)
            d.text((vx + 3 + bw // 2, vy + 3 + bh // 2 + 1), ptxt, font=fnt, fill=(12, 14, 10, 255), anchor="mm")
        if status == "nosignal":
            # Fond gris sombre : distingue « pas de source » d'une vidéo noire légitime.
            d.rectangle([vx, vy, vx + vw - 1, vy + vh - 1], fill=(22, 24, 28, 255))
            fnt = _font(max(10, min(48, vh // 8)))
            d.text((vx + vw // 2, vy + vh // 2), "NO SIGNAL",
                   font=fnt, fill=(150, 153, 160, 255), anchor="mm")
            continue   # pas de chip format sans source
        # Chip format (mode ingénierie) en HAUT-DROITE.
        _stack_top = vy + 3   # bas du dernier badge haut-droite → empilement (FREEZE en dessous)
        if fmt_txt:
            fnt = _font(max(8, min(14, vh // 14)))
            pad = max(2, vh // 120)
            tb = d.textbbox((0, 0), fmt_txt, font=fnt)
            cw, ch = (tb[2] - tb[0]) + 2 * pad + 2, (tb[3] - tb[1]) + 2 * pad
            cx, cy = vx + vw - cw - 3, vy + 3
            d.rounded_rectangle([cx, cy, cx + cw, cy + ch], radius=2, fill=(0, 0, 0, 170))
            d.text((cx + cw // 2, cy + ch // 2 + 1), fmt_txt,
                   font=fnt, fill=(210, 212, 218, 255), anchor="mm")
            _stack_top = cy + ch + 3
        if status == "freeze":
            fnt = _font(max(8, min(20, vh // 12)))
            pad = max(3, vh // 80)
            tb = d.textbbox((0, 0), "FREEZE", font=fnt)
            bw, bh = (tb[2] - tb[0]) + 2 * pad, (tb[3] - tb[1]) + 2 * pad
            bx, by = vx + vw - bw - 4, _stack_top + 1   # sous le chip format (haut-droite empilé)
            d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=3, fill=(214, 132, 0, 235))
            d.text((bx + bw // 2, by + bh // 2 + 1), "FREEZE",
                   font=fnt, fill=(24, 18, 6, 255), anchor="mm")
    return img   # RGBA — consolidation : converti une seule fois après alpha_composite

_info_sig = None      # signature de la couche info bakée (None = re-bake forcé)
_info_layer = None


# ─── Serveur TSL 5.0 (TCP) ───────────────────────────────────
# Chaque entrée dans le flux TCP (format observé) :
#   SOM(2=0xFE02) + VER(1) + FLAGS(1) + SCREEN(2LE) + INDEX(2LE)
#   + EXTRA(2) + CONTROL(2LE) + LENGTH(2LE) + TEXT(LENGTH bytes Latin-1)
# Total fixe par entrée sans texte = 14 bytes.
# CONTROL : bits 0-1 = rh-tally (off/red/green/amber)
#           bits 2-3 = text-tally (couleur label)
#           bits 4-5 = lh-tally
#           bits 6-7 = brightness
TSL_SOM = b"\xfe\x02"  # Start-Of-Message

def _tsl_color(val):
    """TSL 5.0 : 0=off, 1=red, 2=green, 3=amber."""
    if val == 0: return "off"
    if val == 2: return "green"
    if val == 3: return "amber"
    return "red"  # 1=red

def _tally_dominant(rh, lh, tt):
    """Couleur dominante à afficher sur les deux boîtes.
    Rouge + vert simultanés → ambre. Text tally = fallback si RH/LH off."""
    has_red   = any(v in (1, 3) for v in (rh, lh, tt))
    has_green = any(v == 2      for v in (rh, lh, tt))
    has_amber = any(v == 3      for v in (rh, lh, tt))
    if has_amber or (has_red and has_green):
        return "amber"
    if has_red:
        return "red"
    if has_green:
        return "green"
    return "off"

def _apply_tsl(index, control, text):
    """Met à jour tally_state / tsl_text pour toutes les fenêtres écoutant cet index.

    Protocole delta : chaque packet ne contient que les slots actifs à cet instant.
    TTL adaptatif : mesuré sur les arrivées successives du même slot (TSL_SLOT_TTL_FACTOR ×
    intervalle observé). Le contrôleur n'envoie plus le slot → il expire après TTL.
    """
    rh = control & 0x03
    tt = (control >> 2) & 0x03
    lh = (control >> 4) & 0x03
    now = time.monotonic()
    if (control & 0x3F) == 0:
        # VSM envoie toujours un control=0 explicite → clear immédiat, on lui fait confiance.
        for s in ('rh', 'tt', 'lh'):
            _tsl_slots.pop((index, s), None)
    else:
        for s, v in (('rh', rh), ('tt', tt), ('lh', lh)):
            if v:
                key = (index, s)
                prev = _tsl_slots.get(key)
                if prev is not None:
                    _pv, prev_ts, prev_iv = prev
                    raw_iv = now - prev_ts
                    iv = raw_iv if prev_iv is None else (0.5 * prev_iv + 0.5 * raw_iv)
                else:
                    iv = None
                _tsl_slots[key] = [v, now, iv]
        # Mode combiné : si le contrôleur a déjà envoyé plusieurs signaux dans un même
        # paquet, chaque paquet est un snapshot complet → slot absent = éteint immédiat.
        # Mode séparé (keepalives indépendants) : seuil 90 % de l'intervalle pour éviter
        # les faux positifs dus à la gigue.
        if sum(1 for v in (rh, tt, lh) if v) > 1:
            _tsl_combined[index] = True
        active_in_packet = {{s for s, v in (('rh', rh), ('tt', tt), ('lh', lh)) if v}}
        combined = _tsl_combined.get(index, False)
        stale = []
        for k, (sv, ts, iv) in _tsl_slots.items():
            if k[0] != index or iv is None:
                continue
            age = now - ts
            if k[1] not in active_in_packet:
                if combined or age >= iv * 0.9:
                    stale.append(k)
            elif age > max(TSL_SLOT_TTL_MIN, iv * TSL_SLOT_TTL_FACTOR):
                stale.append(k)
        for k in stale:
            del _tsl_slots[k]
    # Expose l'état des slots dans le debug
    now2 = time.monotonic()
    tsl_debug["slots"] = {{
        f"{{idx}}:{{s}}": {{"v": sv, "age_ms": round((now2 - ts) * 1000),
                            "iv_ms": round(iv * 1000) if iv is not None else None}}
        for (idx, s), (sv, ts, iv) in _tsl_slots.items()
    }}
    rh = _tsl_slots.get((index, 'rh'), [0])[0]
    tt = _tsl_slots.get((index, 'tt'), [0])[0]
    lh = _tsl_slots.get((index, 'lh'), [0])[0]
    # Mode Direct : Rouge = TT (on-air), Vert = LH (preview). Couleur FORCÉE si le champ est actif.
    red_active   = tt != 0
    green_active = lh != 0
    if tsl_text_by_index.get(index) != text:
        tsl_text_by_index[index] = text   # exposé aux overlays texte sourcés TSL
        overlay_dirty.set()               # re-bake la couche overlay cachée (texte sourcé TSL)
    changed = False
    for i, cfg in enumerate(FLUX_CONFIG):
        if int(cfg.get("tsl_index", 0) or 0) != index:
            continue
        new_L = "red"   if (cfg.get("tally_red")   and red_active)   else "off"
        new_R = "green" if (cfg.get("tally_green") and green_active) else "off"
        if tally_state.get(f"{{i}}_L") != new_L:
            tally_state[f"{{i}}_L"] = new_L; changed = True
        if tally_state.get(f"{{i}}_R") != new_R:
            tally_state[f"{{i}}_R"] = new_R; changed = True
        if _is_protocol_label(cfg) and tsl_text.get(i) != text:
            tsl_text[i] = text; changed = True
    if changed:
        tally_dirty.set()

def _handle_tsl_client(conn):
    tsl_debug["connections"] += 1
    buf = bytearray()
    try:
        with conn:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buf.extend(chunk)
                tsl_debug["last_raw_hex"] = buf[:22].hex()
                # Parcours de toutes les entrées complètes dans le buffer
                while True:
                    som = buf.find(TSL_SOM)
                    if som < 0:
                        # Garde le dernier byte au cas où c'est 0xfe (demi-SOM)
                        buf = bytearray(buf[-1:]) if buf else bytearray()
                        break
                    if som > 0:
                        buf = buf[som:]   # saute les bytes avant le SOM
                    # Il faut au moins 14 bytes (SOM + header fixe 12 bytes)
                    if len(buf) < 14:
                        break
                    # Offsets vérifiés sur le fil VSM : SCREEN@6, INDEX@8, CONTROL@10, LENGTH@12, TEXT@14
                    control = struct.unpack_from("<H", buf, 10)[0]
                    length  = struct.unpack_from("<H", buf, 12)[0]
                    total   = 14 + length
                    if len(buf) < total:
                        break
                    ver     = buf[2]
                    index   = struct.unpack_from("<H", buf, 8)[0]
                    text    = buf[14:14 + length].decode("latin-1", errors="replace") if length else ""
                    tsl_debug["last_ver"]     = ver
                    tsl_debug["last_index"]   = index
                    tsl_debug["last_control"] = control
                    tsl_debug["last_text"]    = text
                    tsl_debug["last_error"]   = None
                    buf = buf[total:]
                    _apply_tsl(index, control, text)
    except Exception as e:
        tsl_debug["last_error"] = str(e)

def _tsl_server():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", TSL_PORT))
            s.listen(8)
            while True:
                conn, _ = s.accept()
                threading.Thread(target=_handle_tsl_client, args=(conn,), daemon=True).start()
        except Exception:
            try: s.close()
            except Exception: pass
            time.sleep(2)

if TSL_MODE == "direct" and TSL_PORT > 0:
    threading.Thread(target=_tsl_server, daemon=True).start()

# ─── Overlays : Texte / Horloge / Image (outils non-vidéo) ───
# Liste séparée de flux_config (objets purement visuels : non câblés, non tally-slot).
OVERLAYS = CONFIG.get("overlays") or []
overlay_dirty = threading.Event()
overlay_dirty.set()
overlay_bg_layer = None              # couche images de fond (cachée, re-bakée sur overlay_dirty)
_base_y = _base_u = _base_v = None   # canvas de base PRÉ-BLENDÉ avec le fond (copié par trame ; None = fond absent → neutre)
_overlay_img_cache = {{}}            # id → (signature_b64, PIL RGBA)
_chrono_state = {{}}                 # id → {{"running": bool, "base": s, "since": epoch|None}}

# Polices embarquées dans l'image compute (cf. _compute_runtime/Dockerfile). Chaque clé →
# liste de chemins candidats ; repli DejaVu Bold puis police Pillow par défaut si absente.
_FONT_FILES = {{
    "dejavu-sans":          ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "dejavu-sans-bold":     ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "dejavu-serif":         ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
    "dejavu-mono":          ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
    "liberation-sans":      ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
    "liberation-sans-bold": ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"],
    "liberation-mono":      ["/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"],
    "inter":    ["/usr/share/fonts/truetype/inter/Inter-Regular.otf",
                 "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
                 "/usr/share/fonts/truetype/inter/InterVariable.ttf"],
    "roboto":   ["/usr/share/fonts/truetype/roboto/unhinted/Roboto-Regular.ttf",
                 "/usr/share/fonts/truetype/roboto/hinted/Roboto-Regular.ttf",
                 "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf"],
    "firacode": ["/usr/share/fonts/truetype/firacode/FiraCode-Regular.ttf",
                 "/usr/share/fonts/opentype/firacode/FiraCode-Regular.otf"],
}}
_DEFAULT_FONT_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_ofont_cache = {{}}
def _overlay_font(key, size):
    size = max(6, int(size))
    ck = (key, size)
    f = _ofont_cache.get(ck)
    if f is not None:
        return f
    for path in _FONT_FILES.get(key or "", []):
        try:
            f = ImageFont.truetype(path, size); break
        except Exception:
            f = None
    if f is None:
        try:
            f = ImageFont.truetype(_DEFAULT_FONT_FILE, size)
        except Exception:
            try: f = ImageFont.load_default(size)
            except Exception: f = ImageFont.load_default()
    _ofont_cache[ck] = f
    return f

def _hex_rgb(s, default=(255, 255, 255)):
    try:
        s = (s or "").strip().lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except Exception:
        return default

def _overlay_get(ov, key, default):
    v = ov.get(key)
    return default if v is None else v

def _tsl_index_dominant(index):
    rh = _tsl_slots.get((index, 'rh'), [0])[0]
    tt = _tsl_slots.get((index, 'tt'), [0])[0]
    lh = _tsl_slots.get((index, 'lh'), [0])[0]
    return _tally_dominant(rh, lh, tt)

def _overlay_geom(ov):
    x = max(0, min(int(ov.get("x") or 0), OUT_WIDTH - 1))
    y = max(0, min(int(ov.get("y") or 0), OUT_HEIGHT - 1))
    w = max(2, min(int(ov.get("w") or 2), OUT_WIDTH - x))
    h = max(2, min(int(ov.get("h") or 2), OUT_HEIGHT - y))
    return x, y, w, h

def _overlay_active(ov):
    """État Tally On : True si l'overlay est allumé.
    Central : flag poussé par l'orchestrateur (résolu depuis la ligne + niveau de Tally).
    Direct  : dérivé de l'index TSL local (tally_index)."""
    if TSL_MODE != "direct":
        return bool(overlay_central.get(str(ov.get("id") or ""), {{}}).get("active"))
    ti = int(ov.get("tally_index") or 0)
    if ti <= 0:
        return False
    return _tsl_index_dominant(ti) != "off"

def _draw_text_overlay(d, ov, text, color_override=None):
    x, y, w, h = _overlay_geom(ov)
    active = _overlay_active(ov)
    col = _hex_rgb(color_override or (ov.get("color_on") if active else ov.get("color")), (255, 255, 255))
    bg_hex = (ov.get("bg_color_on") if active else ov.get("bg_color")) or ""
    bg_op = max(0, min(100, int(_overlay_get(ov, "bg_opacity", 100))))
    pad = max(2, h // 12)
    if bg_hex.strip():
        br, bgc, bb = _hex_rgb(bg_hex, (0, 0, 0))
        d.rectangle([x, y, x + w - 1, y + h - 1],
                    fill=(br, bgc, bb, int(255 * bg_op / 100)))
    text = text or ""
    if not text:
        return
    fs = int(_overlay_get(ov, "font_size", 0) or 0)
    if fs <= 0:
        fs = max(6, int(h * 0.7))
    fkey = ov.get("font") or "dejavu-sans-bold"
    font = _overlay_font(fkey, fs)
    # Multiligne : le texte peut contenir des sauts de ligne (\n). multiline_text gère aussi le
    # mono-ligne. Police ajustée pour tenir en largeur ET en hauteur (le bloc grandit en vertical).
    maxw = w - 2 * pad
    maxh = h - 2 * pad
    try:
        bb = d.multiline_textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        scale = 1.0
        if tw > maxw > 0:
            scale = min(scale, maxw / tw)
        if th > maxh > 0:
            scale = min(scale, maxh / th)
        if scale < 1.0:
            fs = max(6, int(fs * scale))
            font = _overlay_font(fkey, fs)
    except Exception:
        pass
    fill = col + (255,)
    cy = y + h // 2
    align = ov.get("align") or "center"
    if align == "left":
        d.multiline_text((x + pad, cy), text, font=font, fill=fill, anchor="lm", align="left")
    elif align == "right":
        d.multiline_text((x + w - pad, cy), text, font=font, fill=fill, anchor="rm", align="right")
    else:
        d.multiline_text((x + w // 2, cy), text, font=font, fill=fill, anchor="mm", align="center")

def _overlay_text_value(ov):
    if (ov.get("text_source") or "local") == "tsl":
        # Central : texte poussé par l'orchestrateur (ligne + colonne du tableau /labels).
        if TSL_MODE != "direct":
            return overlay_central.get(str(ov.get("id") or ""), {{}}).get("text", "") or ""
        return tsl_text_by_index.get(int(ov.get("tsl_index") or 0), "") or ""
    return ov.get("text") or ""

def _parse_tc_seconds(s):
    # Champs séparés par ":" alignés à DROITE. 4 champs → HH:MM:SS:FF (le dernier = images) ;
    # ≤3 champs → HH:MM:SS (pas d'images). "00:01:30" = 90 s, "00:00:00:12" = 12 images.
    try:
        f = [int(x) for x in str(s or "").split(":") if x != ""]
    except Exception:
        return 0.0
    if not f:
        return 0.0
    ff = 0
    if len(f) >= 4:
        f = f[-4:]; ff = f[3]; f = f[:3]
    f = f[-3:]
    while len(f) < 3:
        f = [0] + f
    hh, mm, ss = f
    return hh * 3600 + mm * 60 + ss + ff * _FD / _FN

def _chrono_elapsed(ov, now):
    cid = ov.get("id")
    st = _chrono_state.get(cid)
    if st is None:
        run = _as_bool(ov.get("chrono_running"))
        st = {{"running": run, "base": 0.0, "since": now if run else None}}
        _chrono_state[cid] = st
    base = st["base"]
    if st["running"] and st["since"] is not None:
        base += now - st["since"]
    return base

def _countdown_remaining(ov, now):
    """Secondes restantes d'un décompte, ou None si l'overlay n'est pas un décompte."""
    if (ov.get("clock_source") or "ptp") != "countdown":
        return None
    return max(0.0, _parse_tc_seconds(ov.get("chrono_start")) - _chrono_elapsed(ov, now))

def _countdown_color(ov, now):
    """Couleur d'ALERTE d'un décompte proche de 0 (hex), ou None pour garder la couleur normale.
    Orange de cd_warn_orange→cd_warn_red s, rouge en deçà (terminé = rouge). Activé par cd_warn
    (défaut on). Les seuils par défaut suivent la demande : orange à 10 s, rouge à 5 s."""
    rem = _countdown_remaining(ov, now)
    if rem is None or not _as_bool(_overlay_get(ov, "cd_warn", True)):
        return None
    t_red    = float(_overlay_get(ov, "cd_warn_red", 5) or 0)
    t_orange = float(_overlay_get(ov, "cd_warn_orange", 10) or 0)
    if rem <= t_red:
        return _overlay_get(ov, "cd_color_red", "#ff3030") or "#ff3030"
    if rem <= t_orange:
        return _overlay_get(ov, "cd_color_orange", "#ff9000") or "#ff9000"
    return None

# ── Horloge source ANC : timecode embarqué (RP188/ATC) lu du flux ANC (2110-40) ──
# MXL-aligné : l'ANC est un FLUX DE DONNÉES séparé (futur ST 2038), pas un champ de l'en-tête
# vidéo. On dérive le shm ANC de l'entrée vidéo (miroir des VU-mètres) et on décode le RP188
# côté consommateur. Format du slot écrit par mtl_rx.c data_rx_thread :
#   [u32 meta_num][u32 udw_fill][anc_meta_rec×meta_num][udw octets] ; anc_meta_rec = 8×uint16 (16 o).
ANC_SLOT     = 8192
ANC_RING     = 8
ANC_HDR      = 64
ANC_TOTAL    = ANC_HDR + ANC_RING * ANC_SLOT
ANC_META_REC = 16
anc_states = {{}}   # flux_idx → {{"for_path", "state": {{shm_f, shm, path}}|None}}

def _derive_anc_shm_path(video_path):
    """`/dev/shm/mtl_0` → `/dev/shm/mtl_anc_0` (miroir 2110_io hooks._derive_anc_shm)."""
    m = re.match(r"(/dev/shm/.+?)_(\d+)$", video_path)
    if m:
        return f"{{m.group(1)}}_anc_{{m.group(2)}}"
    return None

def _open_anc_state(video_path):
    p = _derive_anc_shm_path(video_path)
    if not p:
        return None
    try:
        rd = bobimxl.Reader(inst, p.removeprefix("/dev/shm/"))  # flux ANC data MXL (lève si absent)
        return {{"reader": rd, "path": p}}
    except Exception:
        return None

def _decode_anc_tc(reader):
    """Dernier grain ANC (flux MXL data) → ATC RP188 (DID/SDID 0x60, 16 UDW BCD) → (hh,mm,ss,ff,df).
    Le grain = UNE payload ANC à l'offset 0 : [u32 meta_num][u32 udw_fill][meta×16][udw]
    (le ring/header sont gérés par MXL, plus de slot à calculer)."""
    try:
        got = reader.get_latest()
        if got is None:
            return None
        v = bytes(got[2])
        mn, _fill = struct.unpack_from("<II", v, 0)
        if mn == 0 or mn > 256:
            return None
        meta_off = 8
        udw_base = meta_off + mn * ANC_META_REC
        for m in range(mn):
            did, sdid, _line, _hori, udw_size, udw_offset, _c, _s = struct.unpack_from(
                "<8H", v, meta_off + m * ANC_META_REC)
            if did != 0x60 or sdid != 0x60 or udw_size < 16:
                continue
            w = udw_base + udw_offset
            if w + 16 > len(v):
                return None
            b = [(v[w + 2*i] & 0x0f) | ((v[w + 2*i + 1] & 0x0f) << 4) for i in range(8)]
            frames  = (b[0] & 0x0f) + ((b[0] >> 4) & 0x03) * 10
            seconds = (b[2] & 0x0f) + ((b[2] >> 4) & 0x07) * 10
            minutes = (b[4] & 0x0f) + ((b[4] >> 4) & 0x07) * 10
            hours   = (b[6] & 0x0f) + ((b[6] >> 4) & 0x03) * 10
            return (hours, minutes, seconds, frames, bool((b[0] >> 6) & 0x01))
    except Exception:
        return None
    return None

def _anc_tc_for(ov):
    """(hh,mm,ss,ff,df) de l'entrée vidéo référencée par l'horloge ANC (tc_source), ou None."""
    try:
        idx = int(ov.get("tc_source") or 0)
    except (TypeError, ValueError):
        return None
    with state_lock:
        cfg = FLUX_CONFIG[idx] if 0 <= idx < len(FLUX_CONFIG) else None
        vpath = (cfg.get("path") or "") if cfg else ""
    rec = anc_states.get(idx)
    # (ré)ouvre si l'entrée a changé de source, ou tant que le flux ANC n'est pas encore là.
    if rec is None or rec["for_path"] != vpath or rec["state"] is None:
        if rec and rec["state"]:
            try: rec["state"]["reader"].close()
            except Exception: pass
        rec = {{"for_path": vpath, "state": _open_anc_state(vpath) if vpath else None}}
        anc_states[idx] = rec
    return _decode_anc_tc(rec["state"]["reader"]) if rec["state"] else None

def _fmt_clock_fields(ov, hh, mm, ss, ff, df=False, dash=False):
    """Met en forme HH/MM/SS/II selon les cases activées ; séparateur images ';' si drop-frame.
    dash=True → tirets (timecode ANC indisponible)."""
    cell = (lambda _v: "--") if dash else (lambda v: "%02d" % v)
    parts = []
    if _as_bool(_overlay_get(ov, "show_hh", True)): parts.append(cell(hh))
    if _as_bool(_overlay_get(ov, "show_mm", True)): parts.append(cell(mm))
    if _as_bool(_overlay_get(ov, "show_ss", True)): parts.append(cell(ss))
    base = ":".join(parts)
    if _as_bool(_overlay_get(ov, "show_ff", False)):
        base = (base + (";" if df else ":") + cell(ff)) if base else cell(ff)
    return base or ("--:--:--" if dash else "%02d:%02d:%02d" % (hh, mm, ss))

def _format_clock(ov, now):
    src = ov.get("clock_source") or "ptp"
    if src == "anc":   # timecode embarqué de la source (RP188/ATC), lu du flux ANC dérivé
        tc = _anc_tc_for(ov)
        if tc is None:
            return _fmt_clock_fields(ov, 0, 0, 0, 0, dash=True)
        hh, mm, ss, ff, df = tc
        return _fmt_clock_fields(ov, hh, mm, ss, ff, df=df)
    if src in ("chrono", "countdown"):
        elapsed = _chrono_elapsed(ov, now)
        if src == "countdown":
            val = max(0.0, _parse_tc_seconds(ov.get("chrono_start")) - elapsed)
        else:  # chrono : compte À PARTIR de la valeur de départ (0 par défaut)
            val = _parse_tc_seconds(ov.get("chrono_start")) + elapsed
    else:  # ptp : heure du jour (CLOCK_REALTIME disciplinée phc2sys) + offset signé
        val = (now + int(_overlay_get(ov, "offset_ms", 0)) / 1000.0) % 86400
    hh = int(val // 3600)
    mm = int((val % 3600) // 60)
    ss = int(val % 60)
    ff = int((val - int(val)) * _FN / _FD)
    return _fmt_clock_fields(ov, hh, mm, ss, ff)

def _overlay_image(ov):
    b64 = ov.get("image_b64") or ""
    if not b64:
        return None
    cid = ov.get("id")
    sig = (len(b64), b64[:24])
    c = _overlay_img_cache.get(cid)
    if c and c[0] == sig:
        return c[1]
    try:
        pim = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
    except Exception:
        pim = None
    _overlay_img_cache[cid] = (sig, pim)
    return pim

def _draw_image_overlay(img, ov):
    pim = _overlay_image(ov)
    if pim is None:
        return
    x, y, w, h = _overlay_geom(ov)
    fit = ov.get("fit") or "contain"
    iw, ih = pim.size
    if iw <= 0 or ih <= 0:
        return
    if fit == "stretch":
        rim = pim.resize((w, h)); ox, oy = x, y
    elif fit == "cover":
        scale = max(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        rim = pim.resize((nw, nh))
        left = (nw - w) // 2; top = (nh - h) // 2
        rim = rim.crop((left, top, left + w, top + h)); ox, oy = x, y
    else:  # contain
        scale = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        rim = pim.resize((nw, nh))
        ox = x + (w - nw) // 2; oy = y + (h - nh) // 2
    op = max(0, min(100, int(_overlay_get(ov, "opacity", 100))))
    if op < 100:
        rim.putalpha(rim.split()[3].point(lambda v: int(v * op / 100)))
    img.alpha_composite(rim, (ox, oy))

def render_overlays_bg():
    bg = [ov for ov in OVERLAYS if (not ov.get("hidden"))
          and ov.get("kind") == "image" and ov.get("layer") == "background"]
    if not bg:
        return None
    img = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
    for ov in bg:
        _draw_image_overlay(img, ov)
    return img   # RGBA — consolidation : converti une seule fois après alpha_composite

def render_overlays_fg_static():
    """Overlays premier-plan dont la VALEUR ne change pas par trame (texte fixe/TSL, images) →
    BAKÉS dans le chrome caché (rendu une seule fois, re-baké sur édition/tally/MAJ TSL). Sortir le
    texte de la boucle per-frame supprime un re-dessin + une reconversion RGBA→YUV à CHAQUE image
    (coût ∝ nombre de caractères / surface)."""
    fg = [ov for ov in OVERLAYS if (not ov.get("hidden")) and ov.get("kind") in ("text", "image")
          and not (ov.get("kind") == "image" and ov.get("layer") == "background")]
    if not fg:
        return None
    img = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    for ov in fg:
        if ov.get("kind") == "text":
            _draw_text_overlay(d, ov, _overlay_text_value(ov))
        else:
            _draw_image_overlay(img, ov)
    return img   # RGBA — consolidation : converti une seule fois après alpha_composite

def render_overlays_fg(now):
    """PER-FRAME : uniquement les horloges (valeur qui change à chaque trame). Le texte/les images
    fixes sont dans le chrome caché (render_overlays_fg_static)."""
    clk = [ov for ov in OVERLAYS if (not ov.get("hidden")) and ov.get("kind") == "clock"]
    if not clk:
        return None
    img = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    for ov in clk:
        _draw_text_overlay(d, ov, _format_clock(ov, now), color_override=_countdown_color(ov, now))
    return img   # RGBA — consolidation : converti une seule fois après alpha_composite

def render_clock_tiles(now):
    """Horloges en TUILES YUV — UNE par horloge sur sa propre bbox (alignée chroma), comme
    render_meters. Évite la bbox-UNION quasi plein écran quand les horloges sont DISPERSÉES (le blend
    per-frame coûtait ∝ surface englobante : 4 horloges éparpillées → ¾ d'écran blendés à chaque trame
    alors que chaque horloge fait ~60 k px). Le rendu PIL + conversion YUV ne sont payés qu'au
    changement de valeur (gate appelant) ; seul le blend par petite bbox reste per-frame. Renvoie
    [(bx0, by0, bx1, by1, y, u, v, a, a2), ...] ou None."""
    clk = [ov for ov in OVERLAYS if (not ov.get("hidden")) and ov.get("kind") == "clock"]
    if not clk:
        return None
    tiles = []
    for ov in clk:
        x, y, w, h = _overlay_geom(ov)
        # bbox locale = rect de l'overlay, bornée à la sortie et alignée chroma (rgba_to_yuv
        # sous-échantillonne par _CW/_CH) : origine ramenée à un multiple, dimensions complétées.
        bx0, by0, bx1, by1 = x, y, x + w, y + h
        bx0 -= bx0 % _CW; by0 -= by0 % _CH
        if (bx1 - bx0) % _CW: bx1 = min(OUT_WIDTH, bx1 + (_CW - (bx1 - bx0) % _CW))
        if (by1 - by0) % _CH: by1 = min(OUT_HEIGHT, by1 + (_CH - (by1 - by0) % _CH))
        if bx1 <= bx0 or by1 <= by0:
            continue
        tile = Image.new("RGBA", (bx1 - bx0, by1 - by0), (0, 0, 0, 0))
        dd = ImageDraw.Draw(tile, "RGBA")
        ovs = dict(ov); ovs["x"] = x - bx0; ovs["y"] = y - by0   # même horloge, coords LOCALES à la tuile
        _draw_text_overlay(dd, ovs, _format_clock(ov, now), color_override=_countdown_color(ov, now))
        oy, ou, ovv, oa, oa2 = rgba_to_yuv(tile)
        tiles.append((bx0, by0, bx1, by1, oy, ou, ovv, oa, oa2))
    return tiles or None

static_rgba  = render_static()   # couches d'habillage CACHÉES, conservées en RGBA (PIL)
dyn_rgba     = None
overlay_fg_rgba = render_overlays_fg_static()   # overlays texte/images fixes, bakés dans le chrome
border_rgba  = render_border()
_chrome_rgba = None              # info+bordure+statique+dynamique pré-composés en UNE image RGBA (caché)
_chrome_yuv  = None              # sa conversion YUV+alpha (refaite seulement sur changement)
_chrome_pre  = None              # opérandes de blend PRÉ-CALCULÉS du chrome (inv_a, src_a par plan) — chemin rapide
_chrome_dirty = True             # force la 1re composition du chrome
# Cache de la couche PER-FRAME des HORLOGES (les VU-mètres ont leur propre chemin par tuiles, jamais
# caché) : YUV+bbox réutilisés tant que la valeur affichée ne change pas (re-render+convert
# DÉTERMINISTE, 1×/s sans le champ images).
_pf_cache_sig = None
_pf_tiles     = None   # tuiles YUV des horloges (une par horloge) — recalculées au changement de valeur

# ─── Boucle de mix ───────────────────────────────────────────

# ─── Hot-input : source de chaque fenêtre re-câblable à chaud via :8082 ──────
# mv_state["inputs"][i] = path courant ; ensure_input(i) rouvre le mmap au changement
# (même résolution — un changement de résolution passe par un redéploiement côté
# orchestrateur, qui réécrit in_w/in_h).
# Géométrie (x/y/w/h) modifiable à chaud via :8082/window : on mute FLUX_CONFIG[idx]
# en place sous state_lock (la boucle vidéo le voit à la frame suivante) et on lève
# geom_dirty pour re-baker la couche statique (bordures/bandeaux/labels). Le STYLE
# (border, résolution de sortie, taille de label) reste lui figé au déploiement.
state_lock = threading.Lock()
geom_dirty = threading.Event()
mv_state = {{"inputs": [cfg.get("path", "") for cfg in FLUX_CONFIG]}}
sources = [None] * len(FLUX_CONFIG)
_in_track = {{}}  # i → {{"path", "fi", "t"}} : dernier avancement du frame_index source (détection freeze)

def ensure_input(i, want_path=None, want_w=None, want_h=None):
    """Ouvre/maintient la source de la fenêtre i. Si (want_path,want_w,want_h) est fourni
    (proxy choisi par _select_input), on ouvre CE shm ; sinon la source pleine câblée."""
    with state_lock:
        if i >= len(mv_state["inputs"]) or i >= len(sources):
            return None
        cur = sources[i]
        if want_path is not None:
            wanted = want_path; cfg_iw = want_w; cfg_ih = want_h
        else:
            wanted = mv_state["inputs"][i]
            cfg_iw = FLUX_CONFIG[i].get("in_w", 640) if i < len(FLUX_CONFIG) else 640
            cfg_ih = FLUX_CONFIG[i].get("in_h", 640) if i < len(FLUX_CONFIG) else 360
    if not wanted:
        if cur is not None:
            try: cur["reader"].close()
            except Exception: pass
            with state_lock:
                if i < len(sources): sources[i] = None
        return None
    if cur is not None and cur.get("path") == wanted:
        return cur
    if cur is not None:
        try: cur["reader"].close()
        except Exception: pass
        with state_lock:
            if i < len(sources): sources[i] = None
    src = open_source({{"path": wanted, "in_w": cfg_iw, "in_h": cfg_ih}})
    if src is not None:
        src["path"] = wanted
        with state_lock:
            if i < len(sources): sources[i] = src
    return src

def _drop_input(i, rd):
    """Ferme le Reader de l'entrée i et oublie la source (→ ensure_input la ROUVRE à la frame
    suivante sur la génération courante du flux). Utilisé pour reconnecter un Reader périmé
    (flux amont recréé sous le même nom) que le cache de ensure_input ne rouvrirait jamais."""
    try: rd.close()
    except Exception: pass
    # GC OBLIGATOIRE entre close et réouverture (même parade que le moteur, tx_reopen_if_stale) :
    # le flux périmé reste résolvable PAR NOM tant qu'il n'est pas collecté — sans GC, la
    # réouverture retombe sur L'ORPHELIN qu'on vient de lâcher (mesuré : boucle drop/reopen à
    # ½ cadence sur un shard dont le proxy amont avait été recréé).
    try: inst.garbage_collect()
    except Exception: pass
    with state_lock:
        if i < len(sources): sources[i] = None
    _in_track.pop(i, None)
    _stale_since.pop(i, None)

_PROXY_UPSCALE_TOL = 0.08   # un proxy jusqu'à ~8% trop petit peut être AGRANDI (zoom léger) au

_OCT_GLYPH = {{2: "½", 4: "¼", 8: "⅛", 16: "1/16"}}

def _classify(path, base, w, h, tw, th):
    """Métadonnée du choix de proxy pour le monitoring (P2/P3/P4) : {{read, cost, kind}}.
    kind = full|oct|custom ; cost = copy (taille exacte) | strided (downscale entier) | gather."""
    if path == base:
        kind, read = "full", "plein"
    elif "__s" in path:
        kind, read = "custom", f"{{w}}×{{h}}"
    elif "__p" in path:
        kind = "oct"
        try: L = int(path.rsplit("__p", 1)[1])
        except (ValueError, IndexError): L = 0
        read = _OCT_GLYPH.get(L, (f"1/{{L}}" if L else "oct"))
    else:
        kind, read = "proxy", f"{{w}}×{{h}}"
    if w == tw and h == th:
        cost = "copy"
    elif w >= tw and h >= th and tw and th and w % tw == 0 and h % th == 0:
        cost = "strided"
    else:
        cost = "gather"
    return {{"read": read, "cost": cost, "kind": kind}}

def _select_input(i, cfg, target_w, target_h):
    """Choisit la MEILLEURE source pour une tuile `target_w×target_h` parmi la source pleine et
    ses proxies pyramide (octaves + tailles sur-mesure). Critère = COÛT du redimensionnement :
      0. « convient » = couvre la tuile à la TOLÉRANCE D'UPSCALE près (≤8% d'agrandissement) →
         un proxy un poil trop petit est préféré à l'octave au-dessus (moins d'octets, zoom léger
         imperceptible sur un mur de monitoring) ;
      1. MATCH EXACT (w==tuile, h==tuile) → resize_plane renvoie le plan tel quel = COPIE PURE ;
      2. sinon ratio ENTIER downscale (strided zéro-copie ; le gather np.ix_ non-entier est lent) ;
      3. sinon la PLUS PETITE aire (moins d'octets lus + gather le plus court).
    OPPORTUNISTE : entrée sans `proxies` = comportement classique. Repli systématique sur le
    plein (un proxy absent/inadapté ne fige jamais la tuile). Renvoie (path, in_w, in_h)."""
    with state_lock:
        base = mv_state["inputs"][i] if i < len(mv_state["inputs"]) else ""
    base_w = cfg.get("in_w", 640); base_h = cfg.get("in_h", 360)
    if not base:
        # Cellule SANS source câblée (décâblée) → AUCUN candidat proxy. Sinon un proxy pyramide
        # RÉSIDUEL (créé quand la cellule était câblée, son flux encore produit tant que la source
        # amont vit) serait choisi malgré la source vide → la tuile reste « vivante » après
        # suppression de la source. base vide → ensure_input("") ferme le Reader → No Signal.
        return "", base_w, base_h, _classify("", "", base_w, base_h, target_w or base_w, target_h or base_h)
    if target_w <= 0 or target_h <= 0:
        return base, base_w, base_h, _classify(base, base, base_w, base_h, target_w or base_w, target_h or base_h)
    cands = [(base, base_w, base_h)]
    for p in (cfg.get("proxies") or []):
        try:
            pw = int(p.get("w") or 0); ph = int(p.get("h") or 0)
        except (TypeError, ValueError):
            continue
        pp = p.get("path") or ""
        # Existence = le FLUX MXL du proxy est présent (≠ fichier shm) — la pyramide migrée
        # produit des flux MXL ; un proxy retiré (drop) cesse d'être candidat → repli source pleine.
        if pp and pw > 0 and ph > 0 and _flow_exists(pp.removeprefix("/dev/shm/")):
            cands.append((pp, pw, ph))
    min_w = target_w / (1.0 + _PROXY_UPSCALE_TOL)
    min_h = target_h / (1.0 + _PROXY_UPSCALE_TOL)
    valid = [c for c in cands if c[1] >= min_w and c[2] >= min_h]
    if not valid:
        return base, base_w, base_h, _classify(base, base, base_w, base_h, target_w, target_h)
    def _key(c):
        exact   = (c[1] == target_w and c[2] == target_h)
        strided = (c[1] >= target_w and c[2] >= target_h and
                   c[1] % target_w == 0 and c[2] % target_h == 0)
        return (0 if exact else (1 if strided else 2), c[1] * c[2])
    best = min(valid, key=_key)
    return best[0], best[1], best[2], _classify(best[0], base, best[1], best[2], target_w, target_h)


def _flow_exists(name):
    """Le flux MXL `name` est-il présent ? (dossier <uuid>.mxl-flow dans le domaine)."""
    try:
        return os.path.exists(os.path.join(inst.domain, bobimxl.flow_id(name) + ".mxl-flow"))
    except Exception:
        return False

def _scan_proxies_for(cfg):
    """Découvre les proxies pyramide DISPONIBLES pour la source d'une fenêtre en énumérant les
    FLUX MXL du domaine (le `flow_def.json` porte le NOM dans `label` → on retrouve `<src>__…`).
    Octaves `<src>__pL` + sur-mesure `<src>__s<w>x<h>` ; dims = frame_width/height du flow_def
    (exactes). Renvoie [{{path,w,h}}] ou None si la source n'est pas résolue."""
    path = cfg.get("path") or ""
    src = path.removeprefix("/dev/shm/") if path.startswith("/dev/shm/") else path
    if not src:
        return None
    pref = src + "__"
    try:
        entries = os.listdir(inst.domain)
    except OSError:
        return None
    out = []
    for ent in entries:
        if not ent.endswith(".mxl-flow"):
            continue
        try:
            with open(os.path.join(inst.domain, ent, "flow_def.json")) as _f:
                fd = json.load(_f)
        except Exception:
            continue
        name = fd.get("label") or ""
        if not name.startswith(pref):
            continue
        w = int(fd.get("frame_width") or 0); h = int(fd.get("frame_height") or 0)
        if w >= 2 and h >= 2:
            out.append({{"path": "/dev/shm/" + name, "w": w, "h": h}})
    return out

def _proxy_scan_loop():
    """Rafraîchit `cfg['proxies']` de chaque fenêtre en continu (toutes ~1.5 s) → injection À CHAUD :
    le multiview consomme les nouveaux proxies (sur-mesure régénérés par le reconcile) ou cesse
    d'utiliser ceux retirés, SANS redéploiement (plus de settle 2-passes)."""
    while True:
        time.sleep(1.5)
        try:
            with state_lock:
                fc = list(FLUX_CONFIG)
            for cfg in fc:
                px = _scan_proxies_for(cfg)
                if px is not None:
                    cfg["proxies"] = px
        except Exception:
            pass


class MvControlHandler(BaseHTTPRequestHandler):
    def _json(self):
        n = int(self.headers.get("Content-Length") or 0)
        try: return json.loads(self.rfile.read(n).decode()) if n else {{}}
        except Exception: return {{}}
    def do_POST(self):
        if self.path == "/window":
            return self._do_window()
        if self.path == "/style":
            return self._do_style()
        if self.path == "/reconfigure":
            return self._do_reconfigure()
        if self.path == "/overlays":
            return self._do_overlays()
        if self.path == "/chrono":
            return self._do_chrono()
        if self.path != "/input":
            self.send_response(404); self.end_headers(); return
        b = self._json()
        try: idx = int(b.get("idx"))
        except Exception:
            self.send_response(400); self.end_headers(); return
        shm = (b.get("shm") or "").strip()
        path = ("/dev/shm/" + shm) if shm else ""
        ok = False
        with state_lock:
            if 0 <= idx < len(mv_state["inputs"]):
                mv_state["inputs"][idx] = path; ok = True
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({{"ok": ok}}).encode())
    def _do_window(self):
        # {{idx, x?, y?, w?, h?, + params visuels par fenêtre}} : modifie à chaud.
        b = self._json()
        try: idx = int(b.get("idx"))
        except Exception:
            self.send_response(400); self.end_headers(); return
        ok = False
        with state_lock:
            if 0 <= idx < len(FLUX_CONFIG):
                cfg = FLUX_CONFIG[idx]
                for k in ("x", "y", "w", "h", "tsl_index", "meter_channels", "meter_opacity"):
                    if k in b and b[k] is not None:
                        try: cfg[k] = int(b[k])
                        except (TypeError, ValueError): pass
                for k in ("name", "meter_position", "meter_scale"):
                    if k in b and b[k] is not None:
                        cfg[k] = str(b[k])
                for k in ("show_label", "show_tally", "meter_inside", "hidden", "label_proportional"):
                    if k in b and b[k] is not None:
                        cfg[k] = _as_bool(b[k])
                ok = True
                geom_dirty.set()
                tally_dirty.set()
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({{"ok": ok}}).encode())
    def _do_style(self):
        # Params globaux visuels : border_w, border_color, overlay_below, label_size,
        # frame_style, show_no_signal, freeze_detect_s, show_format.
        b = self._json()
        with state_lock:
            global BORDER_W, BORDER_COLOR, OVERLAY_BELOW, LABEL_SIZE, FRAME_STYLE
            global SHOW_NO_SIGNAL, FREEZE_DETECT_S, SHOW_FORMAT, SHOW_PROXY
            if "border_w" in b:
                try: BORDER_W = max(0, int(b["border_w"]))
                except (TypeError, ValueError): pass
            if "border_color" in b:
                BORDER_COLOR = str(b["border_color"])
            if "overlay_below" in b:
                OVERLAY_BELOW = _as_bool(b["overlay_below"])
            if "label_size" in b:
                # Tailles dérivées (bandeau/tally/police) recalculées PAR FENÊTRE au
                # rendu via _label_metrics — seul le réglage de base change ici.
                try: LABEL_SIZE = max(6, int(b["label_size"]))
                except (TypeError, ValueError): pass
            if "frame_style" in b:
                FRAME_STYLE = str(b["frame_style"])
            if "show_no_signal" in b:
                SHOW_NO_SIGNAL = _as_bool(b["show_no_signal"])
            if "freeze_detect_s" in b:
                try: FREEZE_DETECT_S = max(0.0, float(b["freeze_detect_s"]))
                except (TypeError, ValueError): pass
            if "show_format" in b:
                SHOW_FORMAT = _as_bool(b["show_format"])
            if "show_proxy" in b:
                SHOW_PROXY = _as_bool(b["show_proxy"])
            geom_dirty.set()
            tally_dirty.set()
        self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({{"ok": True}}).encode())
    def _do_reconfigure(self):
        # Remplacement atomique de toute la liste de sources + géométrie.
        # Permet l'ajout/suppression de fenêtres à chaud depuis l'éditeur.
        b = self._json()
        new_fc = b.get("flux_config") or []
        with state_lock:
            # PRÉSERVATION des Readers inchangés : un ajout/déplacement/retrait de PiP ne doit PAS
            # figer les AUTRES tuiles. L'ancien code fermait TOUS les Readers + remettait sources à
            # None → chaque tuile inchangée se figeait le temps de la ré-ouverture (et ne récupérait
            # pas toujours seule → re-câblage manuel). On ne ferme désormais QUE les Readers dont la
            # source (chemin câblé) a changé ou qui disparaissent ; les tuiles inchangées gardent
            # leur Reader VIVANT. Les indices de banque sont stables (0.9.0) → appariement par index.
            old_sources = list(sources)
            old_inputs  = list(mv_state["inputs"])
            new_inputs  = [cfg.get("path", "") for cfg in new_fc]
            new_sources = [None] * len(new_fc)
            for i in range(len(new_fc)):
                if (i < len(old_sources) and old_sources[i] is not None
                        and i < len(old_inputs) and old_inputs[i] == new_inputs[i] and new_inputs[i]):
                    new_sources[i] = old_sources[i]   # même source → Reader conservé (pas de gel)
                    old_sources[i] = None             # marqué conservé → pas fermé ci-dessous
            for src in old_sources:                   # Readers non conservés (source changée/retirée)
                if src is not None:
                    try: src["reader"].close()
                    except Exception: pass
            FLUX_CONFIG[:] = new_fc
            mv_state["inputs"][:] = new_inputs
            sources[:] = new_sources
            _in_track.clear()
            _stale_since.clear()
            geom_dirty.set()
            tally_dirty.set()
        self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({{"ok": True}}).encode())
    def _do_overlays(self):
        # Remplacement atomique de la liste d'overlays (texte/horloge/image). Hot-apply.
        b = self._json()
        new_ov = b.get("overlays")
        if not isinstance(new_ov, list):
            self.send_response(400); self.end_headers(); return
        with state_lock:
            OVERLAYS[:] = new_ov
            live = {{ov.get("id") for ov in new_ov}}
            for cid in list(_overlay_img_cache):
                if cid not in live:
                    _overlay_img_cache.pop(cid, None)
            for cid in list(_chrono_state):
                if cid not in live:
                    _chrono_state.pop(cid, None)
            overlay_dirty.set()
        self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({{"ok": True}}).encode())
    def _do_chrono(self):
        # Pilotage live d'un chrono/décompte : {{id, action: start|stop|reset}}.
        b = self._json()
        cid = b.get("id"); action = (b.get("action") or "").lower()
        now = time.time(); ok = False
        with state_lock:
            st = _chrono_state.get(cid)
            if st is None:
                st = {{"running": False, "base": 0.0, "since": None}}
                _chrono_state[cid] = st
            if action == "start":
                if not st["running"]:
                    st["running"] = True; st["since"] = now
                ok = True
            elif action == "stop":
                if st["running"] and st["since"] is not None:
                    st["base"] += now - st["since"]
                st["running"] = False; st["since"] = None
                ok = True
            elif action == "reset":
                # Raz = retour à la valeur de départ ET arrêt (jamais de relance auto).
                st["base"] = 0.0
                st["running"] = False
                st["since"] = None
                ok = True
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({{"ok": ok}}).encode())
    def do_GET(self):
        with state_lock: inp = list(mv_state["inputs"])
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({{"inputs": inp}}).encode())
    def log_message(self, *a): pass

threading.Thread(
    target=lambda: HTTPServer(("0.0.0.0", 8082), MvControlHandler).serve_forever(),
    daemon=True).start()

# Injection de proxies pyramide À CHAUD : scrute /dev/shm en continu (cf. _proxy_scan_loop).
threading.Thread(target=_proxy_scan_loop, daemon=True).start()

# Sortie MXL : index tai en genlock-grille, sinon compteur libre (input-locked / cadence libre).
_out_mode  = "tai" if (GENLOCK and not INPUT_LOCKED) else "free"
# Mode tranche : le flowDef porte slice_height → libmxl publie le grain en N tranches égales
# (commit progressif). Writer forwarde **flow_kw à build_flow_def. Sans slice : inchangé (1 tranche).
out_writer = bobimxl.Writer(inst, SHM_OUT_NAME, OUTPUT_W, OUTPUT_H, CHROMA, BIT_DEPTH,
                            _FN, _FD, index_mode=_out_mode,
                            **({{"slice_height": SLICE_LINES}} if SLICE_ON else {{}}))
if SLICE_ON:
    print(f"multiview: MODE TRANCHE actif — {{SLICE_LINES}} lignes/bande "
          f"({{OUT_HEIGHT // SLICE_LINES}} tranches/trame)")
    metrics["slice_mode"] = True
out_frame_index = 0
start_time = time.time()
_fps_last_idx = 0          # fps en fenêtre glissante (delta depuis le dernier report)
_fps_last_t   = start_time
next_frame_time = _grid_next(start_time, FRAME_INTERVAL) if GENLOCK else start_time
_last_in_fi = {{}}         # mode input-locked : dernier frame_index COMPOSÉ par source (barrière)
_lag_frames = {{}}         # mode input-locked : retard par shm = nb de composes consécutifs SANS
                           # avancement du frame_index source (0 = synchrone). Exposé inputs_lag_frames.

# Backoff des entrées mortes (mode tranche) : ti → [fi_bloqué, timeouts_consécutifs]. Persistant
# ENTRE les trames (module) ; réarmé dès que la tête de la tuile dépasse fi_bloqué.
_sl_backoff = {{}}

def _compose_bands(cy, cu, cv, batch, chrome_pre, meter_tiles, pf_tiles, fi_out=None):
    """MODE TRANCHE — compose et PUBLIE la trame BANDE PAR BANDE (SLICE_LINES lignes/bande).
    Pour chaque bande de sortie : attend (get_slice, réveil au commit partiel du producteur) les
    lignes SOURCE nécessaires de chaque tuile intersectante, resize+place la bande (même mapping
    nearest que resize_plane → octet-identique au whole-frame), blende l'habillage intersectant
    (chrome pré-calculé / VU / horloges), copie la bande dans le grain de sortie et committe
    validSlices=k+1 → l'étage AVAL (TX 2110 slice, multiview chaîné) démarre sur la 1ʳᵉ bande.
    Budget total borné : une entrée en retard bascule sur son dernier grain COMPLET (tuile
    décalée d'1 image) — la sortie n'est JAMAIS bloquée par une source."""
    nb = OUT_HEIGHT // SLICE_LINES
    global _sl_dbg
    wait_ns = 0          # cumul des ATTENTES get_slice — renvoyé à l'appelant : own_latency_ms
                         # doit les EXCLURE (le tissu/monitoring y lit la SATURATION du worker,
                         # pas le suivi du fil — cf. cap réactif pyramide, même contrat)
    # FLOW : la sortie est écrite À L'INDEX D'EPOCH ciblé (alignement du tissu) ; sinon index
    # du Writer (tai genlock / compteur input-locked) comme avant.
    _gidx, gi_o, vw_o = out_writer.open_grain(index=fi_out)
    ysz = OUT_WIDTH * OUT_HEIGHT * _BPS
    csz = (OUT_WIDTH // _CW) * (OUT_HEIGHT // _CH) * _BPS
    ov_y = vw_o[:ysz].view(_NP_DT).reshape(OUT_HEIGHT, OUT_WIDTH)
    ov_u = vw_o[ysz:ysz + csz].view(_NP_DT).reshape(OUT_HEIGHT//_CH, OUT_WIDTH//_CW)
    ov_v = vw_o[ysz + csz:ysz + 2*csz].view(_NP_DT).reshape(OUT_HEIGHT//_CH, OUT_WIDTH//_CW)
    _deadl = time.monotonic() + FRAME_INTERVAL * 1.5   # garde-fou GLOBAL (borne dure de la trame)
    _sl_dbg = [len(batch), -1, 0, 0, 0]   # [tuiles, valid0, attentes, replis, dormantes]
    # état mutable par tuile : plans (rebindés au repli grain complet), tranches valides, colonnes.
    # FAST-PATH ratio ENTIER (grilles 2×2/3×3…) : vues STRIDÉES pré-calculées (comme resize_plane)
    # → placement de bande = simple slice-assign (~8 µs vs ~120 µs np.ix_ 2D, mesuré : le np.ix_
    # par bande×tuile coûtait ~15 ms/trame et faisait dériver le compose d'une demi-trame).
    def _tile_views(sy, su, sv, in_h, in_w, vh, vw):
        if vh > 0 and vw > 0 and in_h % vh == 0 and in_w % vw == 0:
            _sy, _sx = in_h // vh, in_w // vw
            return ("v", sy[::_sy, ::_sx], su[::_sy, ::_sx], sv[::_sy, ::_sx], None, None)
        cxi = (np.arange(vw) * in_w) // vw
        cci = (np.arange(vw // _CW) * (in_w // _CW)) // (vw // _CW)
        return ("g", sy, su, sv, cxi, cci)
    st = []
    for (ti, rd, fi, gi, sy, su, sv, in_h, in_w, vy, vh, vx, vw) in batch:
        total = int(gi.totalSlices or 1)
        valid0 = int(gi.validSlices or total)
        # BACKOFF entrées mortes (durcissement) : une tuile qui a timeouté 2 trames de suite SUR LE
        # MÊME grain (tête figée = producteur mort mi-grain) passe DORMANTE : bascule immédiate sur
        # son dernier grain complet, ZÉRO sonde — elle ne consomme plus aucun budget tant que sa
        # tête n'avance pas. Toute avance de tête (fi > enregistré) réarme la tuile.
        _bk = _sl_backoff.get(ti)
        if _bk is not None and fi > _bk[0]:
            _sl_backoff.pop(ti, None); _bk = None
        if _bk is not None and _bk[1] >= 2:
            _sl_dbg[4] += 1
            g = rd.get(fi - 1, timeout_ns=2_000_000)
            if g is not None:
                _v = g[2]
                _yb = in_w * in_h * _BPS
                _ub = (in_w // _CW) * (in_h // _CH) * _BPS
                sy = _v[:_yb].view(_NP_DT).reshape(in_h, in_w)
                su = _v[_yb:_yb + _ub].view(_NP_DT).reshape(in_h//_CH, in_w//_CW)
                sv = _v[_yb + _ub:_yb + 2*_ub].view(_NP_DT).reshape(in_h//_CH, in_w//_CW)
            valid0 = total                       # plus aucune attente pour cette tuile
        st.append([ti, rd, fi, sy, su, sv, in_h, in_w, vy, vh, vx, vw,
                   total, valid0, max(1, in_h // max(1, total)),
                   _tile_views(sy, su, sv, in_h, in_w, vh, vw),
                   int(FRAME_INTERVAL * 1e9)])   # budget d'attente PAR TUILE (découplage)
    for k in range(nb):
        b0 = k * SLICE_LINES; b1 = b0 + SLICE_LINES
        for t in st:
            ti, rd, fi, sy, su, sv, in_h, in_w, vy, vh, vx, vw, total, valid, islh, tv = t[:16]
            a = max(vy, b0); b = min(vy + vh, b1)
            if a >= b:
                continue
            # dernière ligne SOURCE (plan Y) requise par cette bande (nearest = même mapping que
            # resize_plane) → nb de tranches source nécessaires (convention : k tranches = lignes
            # [0, k·islh) valides sur les 3 plans, chroma compris).
            need_row = ((b - 1 - vy) * in_h) // vh
            need_k = min(total, need_row // islh + 1)
            if _sl_dbg[1] < 0:
                _sl_dbg[1] = valid
            if need_k > valid:
                _sl_dbg[2] += 1
                # Budget PAR TUILE (t[16], durcissement) : une tuile en retard n'entame QUE son
                # propre budget — les autres gardent le leur (plus de couplage par le budget
                # global, qui reste la borne dure de la trame).
                _left = min(_deadl - time.monotonic(), t[16] / 1e9)
                _w0 = time.monotonic()
                g = rd.get_slice(fi, need_k, timeout_ns=max(1, int(_left * 1e9))) if _left > 0 else None
                _dw = int((time.monotonic() - _w0) * 1e9)
                wait_ns += _dw; t[16] -= _dw
                if g is not None:
                    valid = t[13] = max(need_k, int(g[1].validSlices or need_k))
                else:
                    # budget épuisé / producteur en retard → REPLI sur le dernier grain COMPLET
                    # (les bandes déjà posées de cette tuile restent celles du grain fi → très
                    # léger tearing d'UNE image sur la tuile en retard, jamais de blocage) +
                    # comptage backoff : 2 timeouts consécutifs sur le MÊME grain → dormante.
                    _bk = _sl_backoff.get(ti)
                    _sl_backoff[ti] = [fi, (_bk[1] + 1 if _bk and _bk[0] == fi else 1)]
                    g = rd.get(fi - 1, timeout_ns=2_000_000)
                    if g is not None:
                        _v = g[2]
                        _yb = in_w * in_h * _BPS
                        _ub = (in_w // _CW) * (in_h // _CH) * _BPS
                        t[3] = sy = _v[:_yb].view(_NP_DT).reshape(in_h, in_w)
                        t[4] = su = _v[_yb:_yb + _ub].view(_NP_DT).reshape(in_h//_CH, in_w//_CW)
                        t[5] = sv = _v[_yb + _ub:_yb + 2*_ub].view(_NP_DT).reshape(in_h//_CH, in_w//_CW)
                        t[15] = _tile_views(sy, su, sv, in_h, in_w, vh, vw)
                    tv = t[15]
                    _sl_dbg[3] += 1
                    valid = t[13] = total          # plus aucune attente pour les bandes suivantes
            # placement de la bande : vue STRIDÉE (ratio entier, ~8 µs) sinon gather chaîné
            # lignes→colonnes (~33 µs — np.ix_ 2D mesuré à ~120 µs, prohibitif à bande×tuile)
            r0 = a - vy; r1 = r0 + (b - a)
            ca0, cb0 = a // _CH, b // _CH
            _cx0 = vx // _CW
            if tv[0] == "v":
                cy[a:b, vx:vx + vw] = tv[1][r0:r1]
                if cb0 > ca0:
                    rc0 = r0 // _CH
                    cu[ca0:cb0, _cx0:_cx0 + vw//_CW] = tv[2][rc0:rc0 + (cb0 - ca0)]
                    cv[ca0:cb0, _cx0:_cx0 + vw//_CW] = tv[3][rc0:rc0 + (cb0 - ca0)]
            else:
                ry = ((np.arange(r0, r1) * in_h) // vh)
                cy[a:b, vx:vx + vw] = tv[1][ry][:, tv[4]]
                if cb0 > ca0:
                    rc = ((np.arange(r0 // _CH, r0 // _CH + (cb0 - ca0)) * (in_h // _CH)) // (vh // _CH))
                    cu[ca0:cb0, _cx0:_cx0 + vw//_CW] = tv[2][rc][:, tv[5]]
                    cv[ca0:cb0, _cx0:_cx0 + vw//_CW] = tv[3][rc][:, tv[5]]
        # habillage intersectant la bande : 1) chrome statique pré-calculé (bbox)
        if chrome_pre is not None:
            bx0, by0, bx1, by1, _piY, _saY, _piC, _saU, _saV = chrome_pre
            a = max(by0, b0); b = min(by1, b1)
            if a < b:
                l0 = a - by0; l1 = b - by0
                cy[a:b, bx0:bx1] = blend_pre(cy[a:b, bx0:bx1], _piY[l0:l1], _saY[l0:l1])
                ca0, cb0 = a // _CH, b // _CH
                lc0 = ca0 - by0 // _CH; lc1 = lc0 + (cb0 - ca0)
                cx0, cx1 = bx0 // _CW, bx1 // _CW
                if cb0 > ca0:
                    cu[ca0:cb0, cx0:cx1] = blend_pre(cu[ca0:cb0, cx0:cx1], _piC[lc0:lc1], _saU[lc0:lc1])
                    cv[ca0:cb0, cx0:cx1] = blend_pre(cv[ca0:cb0, cx0:cx1], _piC[lc0:lc1], _saV[lc0:lc1])
        # 2) VU-mètres puis 3) horloges (z-ordre conservé), bornés aux lignes de la bande
        for _tiles in (meter_tiles, pf_tiles):
            if not _tiles:
                continue
            for (bx0, by0, bx1, by1, _oy, _ou, _ovv, _oa, _oa2) in _tiles:
                a = max(by0, b0); b = min(by1, b1)
                if a >= b:
                    continue
                l0 = a - by0; l1 = b - by0
                cy[a:b, bx0:bx1] = blend(cy[a:b, bx0:bx1], _oy[l0:l1], _oa[l0:l1])
                ca0, cb0 = a // _CH, b // _CH
                lc0 = ca0 - by0 // _CH; lc1 = lc0 + (cb0 - ca0)
                cx0, cx1 = bx0 // _CW, bx1 // _CW
                if cb0 > ca0:
                    cu[ca0:cb0, cx0:cx1] = blend(cu[ca0:cb0, cx0:cx1], _ou[lc0:lc1], _oa2[lc0:lc1])
                    cv[ca0:cb0, cx0:cx1] = blend(cv[ca0:cb0, cx0:cx1], _ovv[lc0:lc1], _oa2[lc0:lc1])
        # publication de la bande : copie canvas→grain + commit PROGRESSIF (réveille l'aval)
        ov_y[b0:b1] = cy[b0:b1]
        if b1 // _CH > b0 // _CH:
            ov_u[b0//_CH:b1//_CH] = cu[b0//_CH:b1//_CH]
            ov_v[b0//_CH:b1//_CH] = cv[b0//_CH:b1//_CH]
        out_writer.commit(gi_o, valid_slices=None if k == nb - 1 else k + 1)
    return wait_ns

def _peek_inputs():
    """frame_index courants des sources OUVERTES (mode input-locked). Lit les 8 premiers octets de
    chaque mmap déjà ouvert (peu coûteux, sans rouvrir) ; -1 si fermée."""
    sig = []
    for _s in sources:
        if _s is not None:
            try:
                _hi = _s["reader"].head_index()
                sig.append(-1 if _hi == bobimxl.MXL_UNDEFINED_INDEX else _hi)
            except Exception: sig.append(-1)
        else:
            sig.append(-1)
    return sig

# SIGBUS : indispensable pour un multiview CHAÎNÉ (qui lit le shm d'un autre plugin). Quand le
# producteur amont recrée/redimensionne son shm (redéploiement, changement de format), une lecture
# d'un mmap momentanément tronqué lève SIGBUS → SANS handler, le process est TUÉ net (pas une
# exception Python). Le handler referme les sources → ensure_input les rouvre à l'état courant.
_bus_error = threading.Event()
def _on_sigbus(signum, frame):
    _bus_error.set()
try:
    signal.signal(signal.SIGBUS, _on_sigbus)
except (ValueError, OSError):
    pass   # pas dans le thread principal / plateforme sans SIGBUS
_last_frame_err = 0.0   # throttle du log du garde-fou de boucle (try/except du corps per-frame)
_last_emit_m = time.monotonic()   # mode input-locked : instant (monotone) de la dernière émission
# RÉ-OUVERTURE des Readers PÉRIMÉS : un producteur amont (2110_io RX (re)sub/relock, redeploy d'une
# source…) DÉTRUIT puis RECRÉE son flux MXL SOUS LE MÊME NOM. ensure_input garde alors le handle en
# cache (path inchangé → jamais rouvert) et get_latest() reste bloqué sur l'ancien ring détruit →
# « No Signal » (ou freeze) PERMANENT, sans SIGBUS pour déclencher la reconnexion. On suit donc par
# entrée l'instant (monotone) où elle est devenue « stale » (aucun grain lisible OU index figé) et,
# au-delà de REOPEN_STALE_S, on FERME + ré-ouvre le Reader (recrée mxlCreateFlowReader sur la
# génération courante du flux). Idempotent : si le flux est réellement absent, open_source échoue et
# on retentera au prochain palier. Couvre la disparition silencieuse (recréation sans SIGBUS).
_stale_since = {{}}   # i → instant monotone où l'entrée est devenue stale (None tant qu'elle lit)
REOPEN_STALE_S = 2.0

while True:
    now = time.time()
    now_m = time.monotonic()   # horloge MONOTONE pour les durées (détection freeze) — insensible aux sauts de CLOCK_REALTIME (genlock garde time.time())
    if _bus_error.is_set():
        _bus_error.clear()
        with state_lock:
            for _s in sources:
                if _s is not None:
                    try: _s["reader"].close()
                    except Exception: pass
            sources[:] = [None] * len(sources)
            _in_track.clear()
        time.sleep(0.02)
        continue   # source(s) refermée(s) → ensure_input rouvre proprement à la frame suivante
    if INPUT_LOCKED:
        # DATA-DRIVEN, cadence BORNÉE À 1 IMAGE : on compose dès que TOUTES les entrées ouvertes ont
        # avancé (chemin rapide, latence minimale, alignement des tuiles), MAIS au plus tard 1 image
        # après l'émission précédente. On n'attend JAMAIS une entrée au-delà de ce budget : une
        # entrée en retard/morte est simplement DÉCALÉE d'1 image (sa tuile garde sa dernière trame,
        # la passe de compositing relit son frame_index courant) et rattrape au tick suivant. Ainsi
        # une seule entrée HS ne bride plus la cadence (avant : deadline 2× ré-armée → ~½ cadence),
        # et la latence ajoutée par étage reste < 1 image (= budget du tissu). Pas d'attente de grille.
        # MODE TRANCHE : deadline élargie (1,25×T). Avec deadline = T exactement (= la période des
        # grains PTP), la barrière peut se VERROUILLER en phase « deadline » (exit à mi-grain, la
        # tête n'avançant que ~1 ms plus tard, à jamais — aucune dérive entre les deux horloges) →
        # la composition démarre à mi-grain au lieu de la 1ʳᵉ tranche (mesuré : valid0≈14/30). La
        # marge laisse l'avance de tête GAGNER systématiquement → phase recalée sur la 1ʳᵉ tranche.
        # Sans source vivante, la cadence de secours devient T×1,25 (40 fps à 50p) — acceptable
        # pour un mur sans entrée. Whole-frame : deadline historique inchangée.
        _deadline = _last_emit_m + FRAME_INTERVAL * (1.25 if SLICE_ON else 1.0)
        while True:
            _cur = _peek_inputs()
            _any_open = False
            _all_advanced = True
            for _i, _fi in enumerate(_cur):
                if _fi < 0:
                    continue
                _any_open = True
                if _fi <= _last_in_fi.get(_i, -1):
                    _all_advanced = False
            # Compose dès que toutes les entrées OUVERTES ont avancé (chemin rapide), sinon au plus tard
            # à la deadline (1 image). PAS de compose immédiat quand AUCUNE entrée n'est ouverte (ou
            # toutes gelées) : on se cale sur la deadline → ~cadence nominale. Sinon un nœud sans entrée
            # (sources absentes) FREE-RUN à plusieurs centaines de fps → sature la bande passante mémoire
            # et ralentit tout le nœud aval (assembleur). Cf. shard sans sources → 480 fps observé.
            if (_any_open and _all_advanced) or time.monotonic() >= _deadline:
                # Commit : MAJ du dernier frame_index composé + retard par shm (0 si avancé, sinon +1).
                for _i, _fi in enumerate(_cur):
                    if _fi < 0:
                        continue
                    _src = sources[_i] if _i < len(sources) else None
                    _p = (_src or {{}}).get("path") or ""
                    _key = _p[len("/dev/shm/"):] if _p.startswith("/dev/shm/") else _p
                    if _key:
                        _lag_frames[_key] = 0 if _fi > _last_in_fi.get(_i, -1) else _lag_frames.get(_key, 0) + 1
                    _last_in_fi[_i] = _fi
                break
            time.sleep(0.0005)
        _last_emit_m = time.monotonic()
    else:
        wait = next_frame_time - now
        if wait > 0:
            time.sleep(wait)

    ts_cycle_start = time.time_ns()   # début du compositing (après l'attente de grille) → own_latency
    # FLOW : index d'epoch CIBLE de cette trame (grille TAI) — les tuiles visent le grain fi_out
    # de LEUR source et la sortie est écrite à ce même index (alignement inter-étages du tissu).
    # _next_index en mode tai = lecture pure de la grille (aucun effet de bord).
    _fi_out = out_writer._next_index() if FLOW else None
    ts_out = ts_cycle_start           # défini AVANT le try : si le rendu plante (garde-fou), own_lat.push
                                      # ci-dessous ne lève plus NameError (le crash qui tuait la boucle).
    _sl_waited = 0                    # mode tranche : attentes get_slice de la trame (exclues de own)
    ts_in_per_input = {{}}  # path → transit_ms (âge à la lecture), rempli par les lectures réussies
    _statuses = []          # (idx, statut, chip format, proxy) → signature de la couche info
    _pu = {{}}              # idx → {{src, read, cost, kind}} (monitoring pyramide, cette frame)
    _pread = []             # noms de shm proxy réellement lus cette frame

    with state_lock:
        _fc = list(FLUX_CONFIG)   # snapshot stable pour cette frame

    # Images de fond (layer=background) : sous la vidéo. Couche cachée, re-bakée sur changement.
    # Idem couche overlay PREMIER-PLAN cachée (texte/images fixes) → re-bakée ici (édition / MAJ TSL),
    # puis intégrée au chrome (coût par-trame nul). Le fond étant STATIQUE, on le PRÉ-BLENDE une seule
    # fois (à la re-bake) dans un canvas de base caché `_base_*` ; la boucle par-trame fait alors une
    # simple COPIE de ce base au lieu de re-blender plein écran 3 plans À CHAQUE trame (≈30 ms → ≈1 ms,
    # le coût dominant des assembleurs/murs avec image de fond). Cf. caching du chrome premier-plan.
    if overlay_dirty.is_set():
        with state_lock:
            _bg_rgba = render_overlays_bg()
            overlay_fg_rgba = render_overlays_fg_static()
        overlay_bg_layer = rgba_to_yuv(_bg_rgba) if _bg_rgba is not None else None
        overlay_dirty.clear()
        _chrome_dirty = True
        if overlay_bg_layer is not None:
            # _base pré-blendé RÉSIDENT BACKEND (device si GPU) : uploadé une seule fois à la re-bake
            # (rare), la boucle par-trame n'en fait qu'une copie. _to_xp uploade les plans d'overlay.
            _oby, _obu, _obv, _oba, _oba2 = overlay_bg_layer
            _base_y = blend(xp.zeros((OUT_HEIGHT, OUT_WIDTH), dtype=_NP_DT), _to_xp(_oby), _to_xp(_oba))
            _base_u = blend(xp.full((OUT_HEIGHT//_CH, OUT_WIDTH//_CW), _NEUTRAL, dtype=_NP_DT), _to_xp(_obu), _to_xp(_oba2))
            _base_v = blend(xp.full((OUT_HEIGHT//_CH, OUT_WIDTH//_CW), _NEUTRAL, dtype=_NP_DT), _to_xp(_obv), _to_xp(_oba2))
        else:
            _base_y = _base_u = _base_v = None

    # Canvas de la trame (RÉSIDENT BACKEND : VRAM si GPU) : COPIE du base pré-blendé (fond) si présent,
    # sinon neutre (cas sans fond, inchangé/rapide). Une copie d'un buffer existant est ~1 ms vs ~30 ms.
    if _base_y is not None:
        canvas_y = _base_y.copy(); canvas_u = _base_u.copy(); canvas_v = _base_v.copy()
    else:
        canvas_y = xp.zeros((OUT_HEIGHT, OUT_WIDTH), dtype=_NP_DT)
        canvas_u = xp.full((OUT_HEIGHT//_CH, OUT_WIDTH//_CW), _NEUTRAL, dtype=_NP_DT)
        canvas_v = xp.full((OUT_HEIGHT//_CH, OUT_WIDTH//_CW), _NEUTRAL, dtype=_NP_DT)
    _gpu_batch = []   # entrées lues cette trame : (src_y,src_u,src_v numpy, géométrie) → upload groupé
    _slice_batch = []  # MODE TRANCHE : (rd, fi, gi, vues zéro-copie, géométrie) → _compose_bands

    for i, cfg in enumerate(_fc):
        if cfg.get("hidden"):
            continue   # entrée de la banque non affichée : source câblée conservée, pas de rendu
        # Géométrie partagée (bandeau sous l'image SANS déformation, bande VU hors image)
        g = _video_rect(cfg)
        vy, vh = g["vy"], g["vh"]
        video_x, video_w = g["vx"], g["vw"]
        # Pyramide : lire le proxy pré-réduit le mieux dimensionné pour cette tuile (sinon plein).
        _ep, _ew, _eh, _einfo = _select_input(i, cfg, video_w, vh)
        src = ensure_input(i, _ep, _ew, _eh)   # rouvre le mmap si la source/proxy a changé
        # Monitoring : choix de proxy de cette tuile (badge P2 + métriques P3/P4).
        _srcname = (cfg.get("path") or "").removeprefix("/dev/shm/")
        _pu[i] = {{"src": _srcname, "read": _einfo["read"], "cost": _einfo["cost"], "kind": _einfo["kind"]}}
        if _einfo["kind"] != "full":
            _pread.append(_ep.removeprefix("/dev/shm/"))
        _proxy_chip = (_einfo["read"], _einfo["cost"]) if SHOW_PROXY else None

        if src is None:
            canvas_y[vy:vy+vh, video_x:video_x+video_w] = 0
            if SHOW_NO_SIGNAL:
                _statuses.append((i, "nosignal", "", None))
            continue

        try:
            rd      = src["reader"]
            in_w    = src["in_w"]
            in_h    = src["in_h"]
            frame_size = src["frame_size"]
            if SLICE_ON and not src.get("interlaced"):
                # MODE TRANCHE : viser le grain de TÊTE (peut être EN COURS d'écriture — un RX
                # 2110 slice committe progressivement). On n'attend ICI que la 1ʳᵉ tranche ; les
                # bandes suivantes sont attendues AU FIL du compose (_compose_bands). Tête pas
                # encore ouverte / flux whole-frame → dernier grain complet (dégénéré, 0 attente).
                got = None
                _hi = rd.head_index()
                if FLOW:
                    # FLOW : cibler L'INDEX D'EPOCH fi_out (alignement du tissu). Source sur la
                    # même grille (|tête−fi_out| ≤ 2) ou pas encore ouverte → attendre la 1ʳᵉ
                    # tranche du grain fi_out (3 ms couvrent la phase d'arrivée ~1,6 ms) ; source
                    # HORS-GRILLE (index libre, ex. flux legacy) → suivre sa tête comme avant.
                    # GÉNÉRATION PÉRIMÉE : sur un flux de la grille, une tête TRÈS en retard
                    # (> ~5 s) et figée = reader sur l'ancien ring d'un producteur recréé sous le
                    # même nom (pyramide/RX redéployé) — le grain reste LISIBLE donc ni le stale
                    # « got is None » ni le freeze (souvent désactivé sur les nœuds du tissu) ne
                    # reconnectent jamais (mesuré : shard 37 min en retard). On DROP → rouvert à
                    # la trame suivante sur la génération courante (inoffensif hors-grille : un
                    # flux à index libre est reconnu par sa tête qui AVANCE, pas par sa valeur).
                    if (_hi != bobimxl.MXL_UNDEFINED_INDEX and _fi_out - _hi > 250
                            and _hi == _in_track.get(i, {{}}).get("fi")):
                        _drop_input(i, rd)
                        canvas_y[vy:vy+vh, video_x:video_x+video_w] = 0
                        if SHOW_NO_SIGNAL:
                            _statuses.append((i, "nosignal", "", None))
                        continue
                    _tgt = (_fi_out if (_hi == bobimxl.MXL_UNDEFINED_INDEX or abs(_hi - _fi_out) <= 2)
                            else _hi)
                    got = rd.get_slice(_tgt, 1, timeout_ns=3_000_000)
                    if got is None and _tgt != _hi and _hi != bobimxl.MXL_UNDEFINED_INDEX:
                        got = rd.get_slice(_hi, 1, timeout_ns=1_000_000)   # retard → grain courant
                elif _hi != bobimxl.MXL_UNDEFINED_INDEX:
                    got = rd.get_slice(_hi, 1, timeout_ns=2_000_000)
                if got is None:
                    got = rd.get_latest()
            else:
                got = rd.get_latest()
            # ENTRELACÉ : on VERROUILLE la PARITÉ sur le champ HAUT (index pair). Sans ça, get_latest
            # renvoie alternativement top/bottom (50/s) → scalés à la tuile, les deux champs ont un
            # décalage vertical d'½ ligne → SCINTILLEMENT des bords horizontaux. Un seul champ (top) →
            # bob progressif STABLE à la cadence trame (25/s), suffisant pour un mur de monitoring.
            if got is not None and src.get("interlaced") and (got[0] % 2 == 1):
                _gt = rd.get(got[0] - 1)        # champ haut apparié (déjà commité → retour immédiat)
                if _gt is not None:
                    got = _gt
            if got is None:        # flux ouvert mais aucun grain lisible (vide OU Reader périmé)
                canvas_y[vy:vy+vh, video_x:video_x+video_w] = 0
                if SHOW_NO_SIGNAL:
                    _statuses.append((i, "nosignal", "", None))
                # Reconnexion : si le « No Signal » persiste au-delà de REOPEN_STALE_S, le Reader
                # est probablement périmé (flux amont recréé sous le même nom → handle sur l'ancien
                # ring) ; ensure_input ne le rouvrirait jamais (path inchangé). On le DROP → rouvert
                # à la frame suivante. Réarmé en boucle → retente tant que le flux est absent.
                _t0 = _stale_since.get(i)
                if _t0 is None:
                    _stale_since[i] = now_m
                elif now_m - _t0 > REOPEN_STALE_S:
                    _drop_input(i, rd)
                continue
            _stale_since.pop(i, None)   # lecture réussie → réarme le compteur de péremption
            fi = got[0]; src_view = got[2]
            # TRANSIT (arrivée) = âge de la trame d'entrée (now_tai − dernière écriture producteur).
            ts_in_per_input[src.get("path", cfg["path"])] = (bobimxl.now_tai() - rd.last_write_time()) / 1e6
            # Suivi freeze : t = dernier instant où l'index de grain a avancé.
            tr = _in_track.get(i)
            if tr is None or tr.get("path") != src["path"]:
                tr = {{"path": src["path"], "fi": fi, "t": now_m}}
                _in_track[i] = tr
            elif fi != tr["fi"]:
                tr["fi"] = fi; tr["t"] = now_m
            _st = "freeze" if (FREEZE_DETECT_S > 0 and now_m - tr["t"] > FREEZE_DETECT_S) else ""
            _chip = _fmt_chip_txt(cfg, src) if SHOW_FORMAT else ""
            if _st or _chip or _proxy_chip:
                _statuses.append((i, _st, _chip, _proxy_chip))
            # Freeze PROLONGÉ : même cause racine que le « No Signal » (Reader périmé sur flux amont
            # recréé) — l'ancien ring peut renvoyer indéfiniment son dernier grain (index figé) au
            # lieu de None. Au-delà de FREEZE_DETECT_S + REOPEN_STALE_S, on reconnecte le Reader.
            # Sans danger pour une source légitimement statique : elle relit simplement son dernier
            # grain après ré-ouverture. (drop = continue : la tuile garde sa dernière trame.)
            if FREEZE_DETECT_S > 0 and now_m - tr["t"] > FREEZE_DETECT_S + REOPEN_STALE_S:
                _drop_input(i, rd)
                continue

            _yb  = in_w * in_h * _BPS               # octets du plan Y
            _uvb = (in_w // _CW) * (in_h // _CH) * _BPS   # octets d'un plan chroma
            if SLICE_ON:
                # MODE TRANCHE : vues ZÉRO-COPIE sur le grain (les lignes au-delà de validSlices ne
                # sont PAS lues ici — _compose_bands attend chaque bande avant de la toucher ; le
                # handler SIGBUS couvre la recréation amont, comme pour les mmaps historiques).
                src_y = src_view[:_yb].view(_NP_DT).reshape(in_h, in_w)
                src_u = src_view[_yb:_yb + _uvb].view(_NP_DT).reshape(in_h//_CH, in_w//_CW)
                src_v = src_view[_yb + _uvb:_yb + 2 * _uvb].view(_NP_DT).reshape(in_h//_CH, in_w//_CW)
                _slice_batch.append((i, rd, fi, got[1], src_y, src_u, src_v, in_h, in_w,
                                     vy, vh, video_x, video_w))
                continue
            src_y = np.frombuffer(bytes(src_view[:_yb]),
                                  dtype=_NP_DT).reshape(in_h, in_w)
            src_u = np.frombuffer(bytes(src_view[_yb:_yb + _uvb]),
                                  dtype=_NP_DT).reshape(in_h//_CH, in_w//_CW)
            src_v = np.frombuffer(bytes(src_view[_yb + _uvb:_yb + 2 * _uvb]),
                                  dtype=_NP_DT).reshape(in_h//_CH, in_w//_CW)
            # On COLLECTE (plans numpy depuis le shm + géométrie de destination) au lieu de
            # resize+place inline. Le placement se fait APRÈS la boucle : en GPU, un seul upload
            # GROUPÉ épinglé des plans collectés (1 H2D/trame, sinon le GPU régresse — banc Phase 0) ;
            # en CPU, simple report (resize+place identiques, tuiles disjointes → octet-identique).
            _gpu_batch.append((src_y, src_u, src_v, vy, vh, video_x, video_w))
        except Exception:
            canvas_y[vy:vy+vh, video_x:video_x+video_w] = 0
            if SHOW_NO_SIGNAL:
                _statuses.append((i, "nosignal", "", None))

    # Placement des tuiles vidéo collectées : 1 upload GPU groupé épinglé (ou resize numpy direct en CPU).
    _place_batch(canvas_y, canvas_u, canvas_v, _gpu_batch)

    # Publication du monitoring proxy de cette frame (swap de référence = atomique pour le lecteur).
    _proxy_usage_latest = _pu
    _proxy_read_latest = sorted(set(_pread))

    _t_after_inputs = time.time_ns()   # profiling : fin des entrées vidéo (lecture+resize+blend tuiles)

    try:
        # Re-bake des couches d'habillage CACHÉES (RGBA) sur changement, puis (re)composition du
        # « chrome » consolidé (z-ordre info < bordure < statique < dynamique) en UNE image RGBA cachée.
        if geom_dirty.is_set():
            with state_lock:
                static_rgba = render_static()
            geom_dirty.clear(); tally_dirty.set(); _info_sig = None; _chrome_dirty = True

        if tally_dirty.is_set():
            dyn_rgba    = render_dynamic()
            border_rgba = render_border()
            overlay_fg_rgba = render_overlays_fg_static()   # texte tally-réactif (couleur on/off)
            tally_dirty.clear(); _chrome_dirty = True

        _sig = tuple(_statuses)
        if _sig != _info_sig:
            with state_lock:
                _info_layer = render_info(_statuses) if _statuses else None
            _info_sig = _sig; _chrome_dirty = True

        if _chrome_dirty:
            _ch = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
            for _lyr in (_info_layer, border_rgba, static_rgba, dyn_rgba, overlay_fg_rgba):   # z-ordre conservé (overlays fixes au-dessus)
                if _lyr is not None:
                    _ch.alpha_composite(_lyr)
            # Bornage à la BBOX réelle du chrome (chroma-alignée) : on ne blende plus tout l'écran à
            # chaque trame quand l'habillage est ÉPARS (ex. assembleur du tissu = 0 ou 1 texte). Chrome
            # entièrement transparent → _chrome_pre None → blend totalement sauté (cas assembleur).
            _cbb = _ch.getbbox()
            if _cbb is None:
                _chrome_rgba = None; _chrome_pre = None
            else:
                bx0, by0, bx1, by1 = _cbb
                bx0 -= bx0 % _CW; by0 -= by0 % _CH
                if bx1 % _CW: bx1 = min(OUT_WIDTH, bx1 + (_CW - bx1 % _CW))
                if by1 % _CH: by1 = min(OUT_HEIGHT, by1 + (_CH - by1 % _CH))
                _chrome_rgba = _ch
                # Opérandes de blend pré-calculés sur la BBOX (statiques jusqu'au prochain changement) :
                # inv_a=(255−α) et src_a=src·α par plan (Y, puis U/V via l'alpha sous-échantillonnée).
                _cy, _cu, _cv, _ca, _ca2 = rgba_to_yuv(_ch.crop((bx0, by0, bx1, by1)))
                # Opérandes UPLOADÉS au backend (device si GPU) UNE fois ici (chrome caché, re-bake rare)
                # → le blend_pre par-trame reste 100 % en VRAM. _to_xp = identité en CPU (inchangé).
                _chrome_pre = (bx0, by0, bx1, by1,
                               _to_xp(255 - _ca.astype(_ACC)),  _to_xp(_cy.astype(_ACC) * _ca),
                               _to_xp(255 - _ca2.astype(_ACC)), _to_xp(_cu.astype(_ACC) * _ca2), _to_xp(_cv.astype(_ACC) * _ca2))
            _chrome_dirty = False

        # Habillage = chrome STATIQUE (caché) + VU-mètres (tuiles per-frame) + horloges (cachées).
        _ts_ov0 = time.time_ns()
        # VU-mètres : rendus + convertis en TUILES locales (une bbox par meter, cf. render_meters).
        # Inhérents per-frame (le niveau change à chaque trame) → jamais cachés, mais chaque tuile ne
        # couvre que la bande VU d'une cellule (≪ bbox-union quasi plein écran de l'ancien chemin).
        _meter_tiles = render_meters(now)
        # Couche PER-FRAME des overlays fg = HORLOGES, en TUILES (une bbox par horloge, cf. render_meters
        # ; texte/images fixes bakés dans le chrome). Rendu PIL + conversion YUV refaits SEULEMENT au
        # changement de VALEUR (signature = chaînes formatées, déterministe → 1×/s sans le champ images
        # FF) ; le blend, lui, se fait par PETITE bbox d'horloge à chaque trame (au lieu de la bbox-UNION
        # quasi plein écran quand les horloges sont dispersées → gros gain de blend per-frame).
        _pf_sig = tuple((_format_clock(ov, now), _countdown_color(ov, now)) for ov in OVERLAYS
                        if (not ov.get("hidden")) and ov.get("kind") == "clock")
        if _pf_sig != _pf_cache_sig:
            _pf_cache_sig = _pf_sig
            _pf_tiles = render_clock_tiles(now)
        _ts_ov1 = time.time_ns()   # fin rendu PIL + conversion YUV (tuiles VU + horloges AU CHANGEMENT)
        # 1) Chrome statique : blendé via opérandes pré-calculés sur sa BBOX (SANS conversion). Borné à
        # la zone réellement habillée → un assembleur sans chrome (bbox None) ne paie RIEN ici.
        # MODE TRANCHE : blends + écriture faits BANDE PAR BANDE dans _compose_bands (plus bas).
        if _chrome_pre is not None and not SLICE_ON:
            bx0, by0, bx1, by1, _piY, _saY, _piC, _saU, _saV = _chrome_pre
            canvas_y[by0:by1, bx0:bx1] = blend_pre(canvas_y[by0:by1, bx0:bx1], _piY, _saY)
            cy0, cy1, cx0, cx1 = by0 // _CH, by1 // _CH, bx0 // _CW, bx1 // _CW
            canvas_u[cy0:cy1, cx0:cx1] = blend_pre(canvas_u[cy0:cy1, cx0:cx1], _piC, _saU)
            canvas_v[cy0:cy1, cx0:cx1] = blend_pre(canvas_v[cy0:cy1, cx0:cx1], _piC, _saV)
        _ts_ov2 = time.time_ns()   # fin du blend chrome
        # 2) VU-mètres puis 3) horloges : blend de chaque TUILE sur sa propre bbox (z-ordre conservé :
        # meters sous les horloges fg). Chaque tuile ne couvre que son petit rectangle local.
        for _src_tiles in (() if SLICE_ON else (_meter_tiles, _pf_tiles)):
            if not _src_tiles:
                continue
            for (bx0, by0, bx1, by1, _oy, _ou, _ov, _oa, _oa2) in _src_tiles:
                # Petites tuiles per-frame : uploadées au backend pour le blend en VRAM (_to_xp=identité CPU).
                _oy, _ou, _ov, _oa, _oa2 = (_to_xp(_oy), _to_xp(_ou), _to_xp(_ov), _to_xp(_oa), _to_xp(_oa2))
                canvas_y[by0:by1, bx0:bx1] = blend(canvas_y[by0:by1, bx0:bx1], _oy, _oa)
                cy0, cy1, cx0, cx1 = by0 // _CH, by1 // _CH, bx0 // _CW, bx1 // _CW
                canvas_u[cy0:cy1, cx0:cx1] = blend(canvas_u[cy0:cy1, cx0:cx1], _ou, _oa2)
                canvas_v[cy0:cy1, cx0:cx1] = blend(canvas_v[cy0:cy1, cx0:cx1], _ov, _oa2)

        _t_after_overlays = time.time_ns()   # profiling : ov_convert=blend chrome, ov_blend=blend VU+horloges (bbox)
        _t_ov_render.push((_ts_ov1 - _ts_ov0) / 1e6)
        _t_ov_convert.push((_ts_ov2 - _ts_ov1) / 1e6)
        _t_ov_blend.push((_t_after_overlays - _ts_ov2) / 1e6)
        if SLICE_ON:
            # MODE TRANCHE : placement des tuiles + habillage + écriture sortie BANDE PAR BANDE,
            # avec commit MXL progressif (l'aval démarre sur la 1ʳᵉ bande). CPU/paysage (gaté).
            _sl_waited = _compose_bands(canvas_y, canvas_u, canvas_v, _slice_batch,
                                        _chrome_pre, _meter_tiles, _pf_tiles,
                                        fi_out=_fi_out) or 0
        else:
            if _PORTRAIT:                        # compose en portrait → tourne 90° vers la trame paysage
                canvas_y, canvas_u, canvas_v = _rotate_out(canvas_y, canvas_u, canvas_v)
            out_frame = xp.concatenate([canvas_y.ravel(), canvas_u.ravel(), canvas_v.ravel()])
            # Grain de sortie MXL (zéro-copie : vue uint8 de l'array _NP_DT). En genlock-grille,
            # l'index tai vient du Writer ; en input-locked/libre, compteur interne du Writer.
            _gidx, _gi_o, _vw_o = out_writer.open_grain()
            if GPU:
                # Download D2H ÉPINGLÉ (.get(out=pinned), rapide) puis copie vers la vue grain. Un .get
                # direct dans le grain (non épinglé) serait ~4× plus lent (banc Phase 0 : dl 5,8→1,3 ms).
                _n = out_frame.size
                if _outpin["buf"] is None or _outpin["buf"].size < _n:
                    _outpin["buf"] = np.frombuffer(cp.cuda.alloc_pinned_memory(_n * _BPS), dtype=_NP_DT, count=_n)
                out_frame.get(out=_outpin["buf"][:_n])
                _vw_o[:OUT_FRAME_SIZE] = _outpin["buf"][:_n].view(np.uint8)
            else:
                _vw_o[:OUT_FRAME_SIZE] = out_frame.view(np.uint8)
            out_writer.commit(_gi_o)
        ts_out = time.time_ns()
        # Détail du compositing : entrées (lecture+resize+blend tuiles) / habillage / assemblage sortie.
        _t_inputs.push((_t_after_inputs - ts_cycle_start) / 1e6)
        _t_overlays.push((_t_after_overlays - _t_after_inputs) / 1e6)
        _t_output.push((ts_out - _t_after_overlays) / 1e6)
    except Exception as _e:
        # Garde-fou : une exception transitoire (rendu overlay/chrome, écriture sortie…) ne doit
        # PAS tuer le process. On saute cette frame (la sortie garde sa dernière image via le ring)
        # et la cadence avance normalement ci-dessous. Log throttlé pour ne pas inonder.
        _nowe = time.time()
        if _nowe - _last_frame_err > 5.0:
            print(f"multiview: erreur de rendu ignorée (frame {{out_frame_index}}) : {{_e}}")
            _last_frame_err = _nowe
    # Latence par PiP : TRANSIT (arrivée) déjà calculé à la lecture (ts_read − ts_in producteur).
    for path, transit_ms in ts_in_per_input.items():
        key = path[len("/dev/shm/"):] if path.startswith("/dev/shm/") else path
        if key not in lat_in:
            lat_in[key] = RollingMs()
        lat_in[key].push(transit_ms)
    # Traitement PROPRE du nœud (compositing) = ts_out − début du cycle.
    # own = TRAVAIL de compositing : en mode tranche les attentes get_slice (suivi du fil) sont
    # EXCLUES — le tissu (décisions de sharding) et le monitoring y lisent la saturation du
    # worker, pas la période de la source (même contrat que la pyramide / cap réactif).
    own_lat.push((ts_out - ts_cycle_start - _sl_waited) / 1e6)
    out_frame_index += 1
    if GENLOCK:
        next_frame_time += FRAME_INTERVAL
        if next_frame_time < time.time():           # retard → recale sur la grille
            # FLOW : suivre un grain occupe ~toute la période (dernière bande arrive à ~T−0,6 ms)
            # — dépasser de quelques centaines de µs est NORMAL. Le recale historique saute à la
            # PROCHAINE frontière → le tick retombe en fin de période → la composition lit des
            # grains COMPLETS (mesuré valid0=30 : whole-frame déguisé, +1 trame). En flow on
            # RATTRAPE (tick immédiat, les attentes par bande absorbent le retard, la phase
            # reconverge seule) tant que le retard reste < 1 période ; au-delà, recale normal.
            if not (FLOW and time.time() - next_frame_time < FRAME_INTERVAL):
                next_frame_time = _grid_next(time.time(), FRAME_INTERVAL)
    else:
        next_frame_time = start_time + (out_frame_index * FRAME_INTERVAL)
    if out_frame_index % 25 == 0:
        # fps en fenêtre glissante (débit sur les ~25 dernières trames) au lieu d'une moyenne
        # cumulée depuis le démarrage (qui masquait les variations / sous-estimait après warmup).
        _now_fps = time.time()
        _dt_fps = _now_fps - _fps_last_t
        if _dt_fps > 0:
            metrics["fps"] = round((out_frame_index - _fps_last_idx) / _dt_fps, 1)
        _fps_last_idx = out_frame_index; _fps_last_t = _now_fps
        _refresh_lat_metrics()
        _sd = globals().get('_sl_dbg')
        if _sd:
            # Observabilité recette (TISSU_SLICE.md) : compteurs slice promus en métriques :8080.
            metrics["slice"] = {{"tiles": _sd[0], "valid0": _sd[1], "waits": _sd[2],
                               "fallbacks": _sd[3], "dormant": _sd[4],
                               "backoff": len(_sl_backoff)}}
        print(f"Mix frame {{out_frame_index}} — {{metrics['fps']}} fps"
              + (f" [slice: tuiles={{_sd[0]}} valid0={{_sd[1]}} waits={{_sd[2]}} replis={{_sd[3]}} dorm={{_sd[4]}}]" if _sd else ""))
