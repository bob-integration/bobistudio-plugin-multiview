# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 BOBI SAS, France
# Auteur : Cyril Mazouer, pour le compte de BOBI SAS
# Distribué sous licence GNU GPL v3 (ou ultérieure) ; voir le fichier LICENSE.

import mmap, socket, struct, time, numpy as np, threading, json, os, re, base64, io
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image, ImageDraw, ImageFont

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

lat_in = {{}}  # {{shm_name: RollingMs}}

# ─── Config injectée (contrat plugin) ───────────────────────
CONFIG         = {config}
HOSTNAME       = "{hostname}"
PLUGIN_VERSION = "{plugin_version}"

FLUX_CONFIG   = CONFIG.get("flux_config") or []
SHM_OUT       = "/dev/shm/" + (CONFIG.get("shm_out") or "mxl_mix")
OUT_WIDTH     = int(CONFIG.get("out_width") or CONFIG.get("width") or 1280)
OUT_HEIGHT    = int(CONFIG.get("out_height") or CONFIG.get("height") or 720)
def _as_bool(v):
    # bool("False") == True : on parse explicitement les chaînes de CONFIG.
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)

BORDER_W      = int(CONFIG.get("border_w") or 0)        # bordure globale (px)
BORDER_COLOR  = CONFIG.get("border_color") or "#ffffff" # couleur de bordure globale
OVERLAY_BELOW = _as_bool(CONFIG.get("overlay_below"))   # bandeau sous l'image vs par-dessus
LABEL_SIZE    = int(CONFIG.get("label_size") or 14)     # taille du texte du label (px)
TSL_PORT      = int(CONFIG.get("tsl_port") or 0)        # port TCP TSL 5.0 (0 = désactivé)
TSL_REMOTE    = _as_bool(CONFIG.get("tsl_remote"))      # True = TSL géré par l'orchestrateur (désactive le serveur local)

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
OUT_FRAME_SIZE = (OUT_WIDTH * OUT_HEIGHT + 2 * (OUT_WIDTH // _CW) * (OUT_HEIGHT // _CH)) * _BPS
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
FRAME_STYLE = CONFIG.get("frame_style") or "none"  # none | classic | tally_border | stylized

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

# audio_states[flux_idx] = {{shm_f, shm, last_chunk, peaks (np), holds (np), hold_ts (np)}}
audio_states = {{}}

def _derive_audio_shm_path(video_path):
    """`/dev/shm/mire1_0` → `/dev/shm/mire1_audio_0`. None si aucun match."""
    m = re.match(r"(/dev/shm/.+?)_(\d+)$", video_path)
    if m:
        return f"{{m.group(1)}}_audio_{{m.group(2)}}"
    return None

def _open_audio_state(flux_idx, video_path):
    """Tente d'ouvrir le shm audio dérivé du shm vidéo. Renvoie le dict state ou None."""
    p = _derive_audio_shm_path(video_path)
    if not p or not os.path.exists(p):
        return None
    try:
        if os.path.getsize(p) < A_TOTAL_SIZE:
            return None
        f = open(p, "r+b")
        shm = mmap.mmap(f.fileno(), A_TOTAL_SIZE)
        return {{"shm_f": f, "shm": shm, "last_chunk": -1,
                 "peaks": None, "holds": None, "hold_ts": None,
                 "path": p}}
    except Exception:
        return None

def _update_peaks(state, n_channels, now):
    """Lit le dernier chunk audio et met à jour peaks + holds (avec decay)."""
    shm = state["shm"]
    try:
        chunk_index, _ = struct.unpack("QQ", bytes(shm[0:16]))
    except Exception:
        return None, None
    # Lit le chunk le plus récent (même si == last_chunk, on a quand même un signal valide)
    slot = chunk_index % A_RING_SIZE
    off  = A_HEADER_SIZE + slot * A_CHUNK_SIZE
    chunk = bytes(shm[off:off + A_CHUNK_SIZE])
    # 3 bytes per sample, BIG-endian signed (wire-native), 8 channels interleaved
    arr = np.frombuffer(chunk, dtype=np.uint8).reshape(A_SAMPLES_PER_CHUNK, A_CHANNELS_MAX, 3)
    samples = ((arr[:, :, 0].astype(np.int32) << 16)
               | (arr[:, :, 1].astype(np.int32) << 8)
               | arr[:, :, 2].astype(np.int32))
    samples = np.where(samples & 0x800000, samples - (1 << 24), samples)
    peaks_lin = np.max(np.abs(samples), axis=0).astype(np.float64)  # (channels,)
    full_scale = float(1 << (A_BIT_DEPTH - 1))
    peak_db = np.where(peaks_lin > 0,
                       20.0 * np.log10(peaks_lin / full_scale),
                       METER_MIN_DB - 1)
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
    state["last_chunk"] = chunk_index
    state["peaks"] = peak_db
    state["holds"] = holds
    state["hold_ts"] = hold_ts
    return peak_db, holds

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

# état tally : {{"<idx>_L": "red"|"green"|"amber"|"off", "<idx>_R": ...}}
tally_state = {{}}
# texte dynamique reçu par TSL, indexé par flux idx (utilisé si label_source == 'protocol')
tsl_text = {{}}
# texte TSL indexé par index TSL brut (utilisé par les overlays texte sourcés TSL, hors fenêtres)
tsl_text_by_index = {{}}
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
metrics = {{"fps": 0.0, "inputs_latency_ms": {{}}}}

def _refresh_lat_metrics():
    out = {{}}
    for shm_name, rm in lat_in.items():
        out[shm_name] = rm.avg()
    metrics["inputs_latency_ms"] = out
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
                slot  = str(data["slot"]).upper()
                color = str(data.get("color", "off")).lower()
                if slot not in ("L", "R") or color not in TALLY_COLORS:
                    raise ValueError("slot/color invalide")
                tally_state[f"{{idx}}_{{slot}}"] = color
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
                if changed:
                    tally_dirty.set()
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

# ─── Préparation du shm de sortie ────────────────────────────

with open(SHM_OUT, "wb") as f:
    f.write(b"\x00" * OUT_TOTAL)

# ─── Helpers ─────────────────────────────────────────────────

def open_source(cfg):
    try:
        in_w = cfg.get("in_w", 640)
        in_h = cfg.get("in_h", 360)
        frame_size = (in_w * in_h + 2 * (in_w // _CW) * (in_h // _CH)) * _BPS
        total = HEADER_SIZE + frame_size * RING_SIZE
        f = open(cfg["path"], "r+b")
        shm = mmap.mmap(f.fileno(), total)
        return {{"f": f, "shm": shm, "view": memoryview(shm),
                 "in_w": in_w, "in_h": in_h, "frame_size": frame_size}}
    except Exception as e:
        print(f"Erreur ouverture {{cfg['path']}}: {{e}}")
        return None

def resize_plane(plane, target_h, target_w):
    from_h, from_w = plane.shape
    row_idx = (np.arange(target_h) * from_h / target_h).astype(int)
    col_idx = (np.arange(target_w) * from_w / target_w).astype(int)
    return plane[np.ix_(row_idx, col_idx)]

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
    car en 10/12 bits dst*(255-a) déborde uint16."""
    a32 = alpha.astype(np.uint32)
    return ((dst.astype(np.uint32) * (255 - a32) + src.astype(np.uint32) * a32) // 255).astype(_NP_DT)

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
    Formules miroir côté éditeur : multiview.js dessiner()."""
    h = max(2, int(cfg.get("h") or 0))
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
    # Zone disponible : hauteur amputée du bandeau (mode « sous l'image »), largeur
    # amputée de la bande VU « hors image ».
    bar_on = bool(cfg.get("show_label") or cfg.get("show_tally"))
    avail_h = h - _bar_reserved_h(cfg, m) if (OVERLAY_BELOW and bar_on) else h
    if avail_h < 2:
        avail_h = h  # cellule trop petite pour un bandeau sous — fallback overlay
    meter_n = int(cfg.get("meter_channels") or 0)
    moff = 0
    if meter_n > 0 and not cfg.get("meter_inside"):
        moff = _meter_layout(meter_n) + 4
        moff += moff % 2
    avail_w = max(2, w - moff)
    ax = x + (moff if cfg.get("meter_position") == "left" else 0)
    # Fit homothétique du ratio de la cellule (w×h) dans la zone, centré.
    scale = min(avail_w / w, avail_h / h)
    vw = max(2, int(w * scale)); vw -= vw % 2
    vh = max(2, int(h * scale)); vh -= vh % 2
    vx = ax + (avail_w - vw) // 2
    vx -= vx % 2   # alignement chroma (vx//_CW)
    vy = y + (avail_h - vh) // 2
    vy -= vy % 2   # alignement chroma (vy//_CH en 4:2:0)
    return {{"x": x, "y": y, "w": w, "h": h,
             "vx": vx, "vy": vy, "vw": vw, "vh": vh,
             "ah": avail_h,    # hauteur de la ZONE hors bandeau (pour les VU-mètres)
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
            # Barre noire épaisse en bas + label centré. Lampes L/R en dynamique.
            cbar = m["cbar"]
            if bar_on:
                bar_top = y + h - cbar
                d.rectangle([x, bar_top, x + w, y + h], fill=(10, 10, 12, 235))
                d.line([x, bar_top, x + w, bar_top], fill=(80, 80, 90, 255), width=1)
                if static_label:
                    tl, tr = x + m["pad"] + m["tally"] + 6, x + w - m["pad"] - m["tally"] - 6
                    d.text(((tl + tr) // 2, bar_top + cbar // 2), name,
                           font=_fit_text(d, name, m, tr - tl),
                           fill=(240, 240, 245, 255), anchor="mm")
            # Cadre fin gris → couche bordure (render_border), au-dessus du bandeau.

        elif FRAME_STYLE == "tally_border":
            # La bordure colorée est dynamique. Ici : juste le label sur fond translucide
            # posé en overlay dans le bas de l'image (pas de barre dédiée).
            if static_label:
                lab_h = m["bar_h"]
                _render_gradient_bar(img, x, y + h - lab_h, w, lab_h, (0, 0, 0), 0, 200)
                d.text((x + w // 2, y + h - lab_h // 2), name,
                       font=_fit_text(d, name, m, w - 8),
                       fill=(245, 245, 250, 255), anchor="mm")

        elif FRAME_STYLE == "stylized":
            # Dégradé bas pour l'UMD + cadre fin. Pastilles L/R en dynamique.
            if bar_on:
                grad_h = m["grad_h"]
                _render_gradient_bar(img, x, y + h - grad_h, w, grad_h, (0, 0, 0), 0, 210)
                if static_label:
                    d.text((x + w // 2, y + h - grad_h // 2 + 2), name,
                           font=_fit_text(d, name, m, w - 8),
                           fill=(245, 245, 250, 255), anchor="mm")
            # Cadre fin → couche bordure (render_border), au-dessus du bandeau.

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
    return rgba_to_yuv(img)

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
            cbar = m["cbar"]
            bar_top = y + h - cbar
            # Fond de barre teinté selon dominante
            if show_tally and dom != "off":
                tint = {{"red": (120, 20, 20, 235), "green": (20, 90, 35, 235),
                        "amber": (120, 80, 0, 235)}}[dom]
                d.rectangle([x, bar_top, x + w, y + h], fill=tint)
            # Lampes carrées L/R
            if show_tally:
                ty = bar_top + (cbar - tally_sz) // 2
                for st, lx in ((tally_state.get(f"{{i}}_L", "off"), x + tally_pad),
                               (tally_state.get(f"{{i}}_R", "off"),
                                x + w - tally_pad - tally_sz)):
                    fill = _TALLY_BORDER_RGBA.get(st, (0, 0, 0, 0))
                    if fill[3] == 0:
                        fill = (40, 40, 44, 200)
                    d.rectangle([lx, ty, lx + tally_sz, ty + tally_sz],
                                fill=fill, outline=(220, 220, 225, 255))
            if is_proto and proto_txt:
                tl, tr = x + tally_pad + tally_sz + 6, x + w - tally_pad - tally_sz - 6
                d.text(((tl + tr) // 2, bar_top + cbar // 2), proto_txt,
                       font=_fit_text(d, proto_txt, m, tr - tl),
                       fill=(245, 245, 250, 255), anchor="mm")

        elif FRAME_STYLE == "tally_border":
            # La bordure colorée est tracée par render_border (couche du dessus).
            if is_proto and proto_txt:
                lab_h = m["bar_h"]
                _render_gradient_bar(img, x, y + h - lab_h, w, lab_h, (0, 0, 0), 0, 200)
                d.text((x + w // 2, y + h - lab_h // 2), proto_txt,
                       font=_fit_text(d, proto_txt, m, w - 8),
                       fill=_TALLY_TEXT_BY_DOMINANT[dom], anchor="mm")

        elif FRAME_STYLE == "stylized":
            # Pastilles rondes PGM (gauche) / PVW (droite) en haut
            if show_tally:
                r = max(5, int(round(m["size"] * 0.5)))
                cy = y + r + 6
                stL = tally_state.get(f"{{i}}_L", "off")
                stR = tally_state.get(f"{{i}}_R", "off")
                fL, oL = _PILL_COLORS.get(stL, _PILL_COLORS["off"])
                fR, oR = _PILL_COLORS.get(stR, _PILL_COLORS["off"])
                _render_pill(d, x + r + 6,     cy, r, fL, oL)
                _render_pill(d, x + w - r - 6, cy, r, fR, oR)
            if is_proto and proto_txt:
                grad_h = m["grad_h"]
                _render_gradient_bar(img, x, y + h - grad_h, w, grad_h, (0, 0, 0), 0, 210)
                d.text((x + w // 2, y + h - grad_h // 2 + 2), proto_txt,
                       font=_fit_text(d, proto_txt, m, w - 8),
                       fill=_TALLY_TEXT_BY_DOMINANT[dom], anchor="mm")

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
                ty = bar_top + (m["bar_h"] - tally_sz) // 2
                stL = tally_state.get(f"{{i}}_L", "off")
                stR = tally_state.get(f"{{i}}_R", "off")
                cL = TALLY_COLORS[stL]; bL = TALLY_BORDER_COLORS[stL]
                cR = TALLY_COLORS[stR]; bR = TALLY_BORDER_COLORS[stR]
                d.rectangle([x + tally_pad, ty,
                             x + tally_pad + tally_sz, ty + tally_sz],
                            fill=cL, outline=bL)
                d.rectangle([x + w - tally_pad - tally_sz, ty,
                             x + w - tally_pad, ty + tally_sz],
                            fill=cR, outline=bR)
    return rgba_to_yuv(img)

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
        bx, by, bw, bh = g["vx"], g["vy"], g["vw"], g["vh"]
        if FRAME_STYLE == "classic":
            d.rectangle([bx, by, bx + bw - 1, by + bh - 1], outline=(60, 60, 68, 255))
        elif FRAME_STYLE == "stylized":
            d.rectangle([bx, by, bx + bw - 1, by + bh - 1], outline=(90, 90, 100, 220))
        elif FRAME_STYLE == "tally_border":
            thick = max(4, int(round(g["m"]["size"] * 0.45)))
            _render_border_colored(d, bx, by, bw, bh,
                                   _TALLY_BORDER_RGBA[_window_tally_dominant(i)], thick)
        else:  # "none"
            if BORDER_W > 0:
                for k in range(BORDER_W):
                    d.rectangle([bx + k, by + k, bx + bw - 1 - k, by + bh - 1 - k],
                                outline=BORDER_COLOR)
    return rgba_to_yuv(img)

def render_meters(now):
    """Re-rendu par frame : layer RGBA contenant tous les peak meters.
    Renvoie (y, u, v, a, a2) prêt à blender, ou None si aucun meter activé."""
    has_meters = any((cfg.get("meter_channels") or 0) > 0 and not cfg.get("hidden")
                     for cfg in FLUX_CONFIG)
    if not has_meters:
        return None
    img = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
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
        if st is not None:
            peaks, holds = _update_peaks(st, n, now)
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
        if cfg.get("meter_position") == "left":
            mx = x + 2
        else:
            mx = x + w - mw - 2
        my = y + 2
        opacity_pct = int(cfg.get("meter_opacity") or 70)
        if not inside:
            opacity_pct = 100  # hors image = totalement opaque (zone réservée)
        _draw_meter(img, mx, my, mw, mh, n, peaks, holds,
                    cfg.get("meter_scale") or "dbfs", opacity_pct)
    return rgba_to_yuv(img)


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
    color = _tally_dominant(rh, lh, tt)
    tsl_text_by_index[index] = text   # exposé aux overlays texte sourcés TSL
    changed = False
    for i, cfg in enumerate(FLUX_CONFIG):
        if int(cfg.get("tsl_index", 0) or 0) != index:
            continue
        new_L = color
        new_R = color
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
                    # CONTROL à +10, LENGTH à +12 (2 bytes EXTRA entre INDEX et CONTROL)
                    control = struct.unpack_from("<H", buf, 10)[0]
                    length  = struct.unpack_from("<H", buf, 12)[0]
                    total   = 14 + length
                    if len(buf) < total:
                        break
                    ver     = buf[2]
                    index   = struct.unpack_from("<H", buf, 6)[0]
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

if TSL_PORT > 0 and not TSL_REMOTE:
    threading.Thread(target=_tsl_server, daemon=True).start()

# ─── Overlays : Texte / Horloge / Image (outils non-vidéo) ───
# Liste séparée de flux_config (objets purement visuels : non câblés, non tally-slot).
OVERLAYS = CONFIG.get("overlays") or []
overlay_dirty = threading.Event()
overlay_dirty.set()
overlay_bg_layer = None              # couche images de fond (cachée, re-bakée sur overlay_dirty)
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
    """État Tally On : True si l'index TSL référencé a un tally actif (≠ off)."""
    ti = int(ov.get("tally_index") or 0)
    if ti <= 0:
        return False
    return _tsl_index_dominant(ti) != "off"

def _draw_text_overlay(d, ov, text):
    x, y, w, h = _overlay_geom(ov)
    active = _overlay_active(ov)
    col = _hex_rgb(ov.get("color_on") if active else ov.get("color"), (255, 255, 255))
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
    try:
        tw = d.textlength(text, font=font)
        maxw = w - 2 * pad
        if tw > maxw > 0:
            fs = max(6, int(fs * maxw / tw))
            font = _overlay_font(fkey, fs)
    except Exception:
        pass
    fill = col + (255,)
    cy = y + h // 2
    align = ov.get("align") or "center"
    if align == "left":
        d.text((x + pad, cy), text, font=font, fill=fill, anchor="lm")
    elif align == "right":
        d.text((x + w - pad, cy), text, font=font, fill=fill, anchor="rm")
    else:
        d.text((x + w // 2, cy), text, font=font, fill=fill, anchor="mm")

def _overlay_text_value(ov):
    if (ov.get("text_source") or "local") == "tsl":
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

def _format_clock(ov, now):
    src = ov.get("clock_source") or "ptp"
    if src in ("chrono", "countdown"):
        elapsed = _chrono_elapsed(ov, now)
        if src == "countdown":
            val = max(0.0, _parse_tc_seconds(ov.get("chrono_start")) - elapsed)
        else:
            val = elapsed
    else:  # ptp : heure du jour (CLOCK_REALTIME disciplinée phc2sys) + offset signé
        val = (now + int(_overlay_get(ov, "offset_ms", 0)) / 1000.0) % 86400
    hh = int(val // 3600)
    mm = int((val % 3600) // 60)
    ss = int(val % 60)
    ff = int((val - int(val)) * _FN / _FD)
    parts = []
    if _as_bool(_overlay_get(ov, "show_hh", True)): parts.append("%02d" % hh)
    if _as_bool(_overlay_get(ov, "show_mm", True)): parts.append("%02d" % mm)
    if _as_bool(_overlay_get(ov, "show_ss", True)): parts.append("%02d" % ss)
    if _as_bool(_overlay_get(ov, "show_ff", False)): parts.append("%02d" % ff)
    return ":".join(parts) if parts else "%02d:%02d:%02d" % (hh, mm, ss)

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
    return rgba_to_yuv(img)

def render_overlays_fg(now):
    fg = [ov for ov in OVERLAYS if (not ov.get("hidden"))
          and not (ov.get("kind") == "image" and ov.get("layer") == "background")]
    if not fg:
        return None
    img = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    for ov in fg:
        k = ov.get("kind")
        if k == "text":
            _draw_text_overlay(d, ov, _overlay_text_value(ov))
        elif k == "clock":
            _draw_text_overlay(d, ov, _format_clock(ov, now))
        elif k == "image":
            _draw_image_overlay(img, ov)
    return rgba_to_yuv(img)

static_y, static_u, static_v, static_a, static_a2 = render_static()
dyn_y = dyn_u = dyn_v = dyn_a = dyn_a2 = None
border_y, border_u, border_v, border_a, border_a2 = render_border()

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

def ensure_input(i):
    with state_lock:
        if i >= len(mv_state["inputs"]) or i >= len(sources):
            return None
        wanted = mv_state["inputs"][i]
        cur    = sources[i]
        cfg_iw = FLUX_CONFIG[i].get("in_w", 640) if i < len(FLUX_CONFIG) else 640
        cfg_ih = FLUX_CONFIG[i].get("in_h", 640) if i < len(FLUX_CONFIG) else 360
    if not wanted:
        if cur is not None:
            try: cur["shm"].close(); cur["f"].close()
            except Exception: pass
            with state_lock:
                if i < len(sources): sources[i] = None
        return None
    if cur is not None and cur.get("path") == wanted:
        return cur
    if cur is not None:
        try: cur["shm"].close(); cur["f"].close()
        except Exception: pass
        with state_lock:
            if i < len(sources): sources[i] = None
    src = open_source({{"path": wanted, "in_w": cfg_iw, "in_h": cfg_ih}})
    if src is not None:
        src["path"] = wanted
        with state_lock:
            if i < len(sources): sources[i] = src
    return src

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
                for k in ("show_label", "show_tally", "meter_inside", "hidden"):
                    if k in b and b[k] is not None:
                        cfg[k] = _as_bool(b[k])
                ok = True
                geom_dirty.set()
                tally_dirty.set()
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({{"ok": ok}}).encode())
    def _do_style(self):
        # Params globaux visuels : border_w, border_color, overlay_below, label_size, frame_style.
        b = self._json()
        with state_lock:
            global BORDER_W, BORDER_COLOR, OVERLAY_BELOW, LABEL_SIZE, FRAME_STYLE
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
            for src in list(sources):
                if src is not None:
                    try: src["shm"].close(); src["f"].close()
                    except Exception: pass
            FLUX_CONFIG[:] = new_fc
            mv_state["inputs"][:] = [cfg.get("path", "") for cfg in new_fc]
            sources[:] = [None] * len(new_fc)
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
                st["base"] = 0.0
                st["since"] = now if st["running"] else None
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

f_out = open(SHM_OUT, "r+b")
shm_out = mmap.mmap(f_out.fileno(), OUT_TOTAL)
shm_out_view = memoryview(shm_out)
out_frame_index = 0
start_time = time.time()
next_frame_time = _grid_next(start_time, FRAME_INTERVAL) if GENLOCK else start_time

while True:
    now = time.time()
    wait = next_frame_time - now
    if wait > 0:
        time.sleep(wait)

    canvas_y = np.zeros((OUT_HEIGHT, OUT_WIDTH), dtype=_NP_DT)
    canvas_u = np.full((OUT_HEIGHT//_CH, OUT_WIDTH//_CW), _NEUTRAL, dtype=_NP_DT)
    canvas_v = np.full((OUT_HEIGHT//_CH, OUT_WIDTH//_CW), _NEUTRAL, dtype=_NP_DT)
    ts_in_per_input = {{}}  # path → ts_in_ns, rempli par les lectures réussies

    with state_lock:
        _fc = list(FLUX_CONFIG)   # snapshot stable pour cette frame

    # Images de fond (layer=background) : sous la vidéo. Couche cachée, re-bakée sur changement.
    if overlay_dirty.is_set():
        with state_lock:
            overlay_bg_layer = render_overlays_bg()
        overlay_dirty.clear()
    if overlay_bg_layer is not None:
        _oby, _obu, _obv, _oba, _oba2 = overlay_bg_layer
        canvas_y = blend(canvas_y, _oby, _oba)
        canvas_u = blend(canvas_u, _obu, _oba2)
        canvas_v = blend(canvas_v, _obv, _oba2)

    for i, cfg in enumerate(_fc):
        if cfg.get("hidden"):
            continue   # entrée de la banque non affichée : source câblée conservée, pas de rendu
        src = ensure_input(i)   # rouvre le mmap si la source a changé (hot-input)
        # Géométrie partagée (bandeau sous l'image SANS déformation, bande VU hors image)
        g = _video_rect(cfg)
        vy, vh = g["vy"], g["vh"]
        video_x, video_w = g["vx"], g["vw"]

        if src is None:
            canvas_y[vy:vy+vh, video_x:video_x+video_w] = 0
            continue

        try:
            shm     = src["shm"]
            shm_view = src["view"]
            in_w    = src["in_w"]
            in_h    = src["in_h"]
            frame_size = src["frame_size"]
            fi, ts_in = struct.unpack("QQ", bytes(shm[0:16]))
            ts_in_per_input[src.get("path", cfg["path"])] = ts_in
            slot   = fi % RING_SIZE
            offset = HEADER_SIZE + slot * frame_size

            _yb  = in_w * in_h * _BPS               # octets du plan Y
            _uvb = (in_w // _CW) * (in_h // _CH) * _BPS   # octets d'un plan chroma
            src_y = np.frombuffer(bytes(shm_view[offset:offset + _yb]),
                                  dtype=_NP_DT).reshape(in_h, in_w)
            src_u = np.frombuffer(bytes(shm_view[offset + _yb:offset + _yb + _uvb]),
                                  dtype=_NP_DT).reshape(in_h//_CH, in_w//_CW)
            src_v = np.frombuffer(bytes(shm_view[offset + _yb + _uvb:offset + frame_size]),
                                  dtype=_NP_DT).reshape(in_h//_CH, in_w//_CW)

            dst_y = resize_plane(src_y, vh,      video_w)
            dst_u = resize_plane(src_u, vh//_CH, video_w//_CW)
            dst_v = resize_plane(src_v, vh//_CH, video_w//_CW)

            canvas_y[vy:vy+vh,                     video_x:video_x+video_w]                  = dst_y
            canvas_u[vy//_CH:vy//_CH+vh//_CH,      video_x//_CW:video_x//_CW+video_w//_CW]   = dst_u
            canvas_v[vy//_CH:vy//_CH+vh//_CH,      video_x//_CW:video_x//_CW+video_w//_CW]   = dst_v
        except Exception:
            canvas_y[vy:vy+vh, video_x:video_x+video_w] = 0

    # Géométrie modifiée à chaud (:8082/window) → re-baker la couche statique
    if geom_dirty.is_set():
        with state_lock:
            static_y, static_u, static_v, static_a, static_a2 = render_static()
        geom_dirty.clear()
        tally_dirty.set()   # labels protocole + pavés tally ont aussi bougé

    # Re-rendu dynamique (labels protocole + tally) si l'état a changé.
    # La couche bordure dépend aussi du tally (style tally_border) et de la géo/
    # style (geom_dirty arme tally_dirty) → on la re-bake ici en même temps,
    # AVANT son blend (la bordure est désormais la première couche d'habillage).
    if tally_dirty.is_set():
        dyn_y, dyn_u, dyn_v, dyn_a, dyn_a2 = render_dynamic()
        border_y, border_u, border_v, border_a, border_a2 = render_border()
        tally_dirty.clear()

    # Bordure SOUS l'habillage : au-dessus de la vidéo seulement — le bandeau de
    # texte, les pavés tally et les VU-mètres passent PAR-DESSUS (elle ne barre
    # plus la zone de texte d'une ligne).
    if border_a is not None:
        canvas_y = blend(canvas_y, border_y, border_a)
        canvas_u = blend(canvas_u, border_u, border_a2)
        canvas_v = blend(canvas_v, border_v, border_a2)

    # Blend overlay statique (bandeaux + noms)
    canvas_y = blend(canvas_y, static_y,  static_a)
    canvas_u = blend(canvas_u, static_u,  static_a2)
    canvas_v = blend(canvas_v, static_v,  static_a2)

    if dyn_a is not None:
        canvas_y = blend(canvas_y, dyn_y, dyn_a)
        canvas_u = blend(canvas_u, dyn_u, dyn_a2)
        canvas_v = blend(canvas_v, dyn_v, dyn_a2)

    # Peak meters : re-rendu à chaque frame (les niveaux changent en continu)
    m_layer = render_meters(now)
    if m_layer is not None:
        m_y, m_u, m_v, m_a, m_a2 = m_layer
        canvas_y = blend(canvas_y, m_y, m_a)
        canvas_u = blend(canvas_u, m_u, m_a2)
        canvas_v = blend(canvas_v, m_v, m_a2)

    # Overlays texte/horloge/logo (layer=foreground) : par-dessus TOUT, re-rendus chaque frame
    # (l'horloge avance en continu, le texte/couleur peut suivre le tally TSL).
    _ovfg = render_overlays_fg(now)
    if _ovfg is not None:
        _ofy, _ofu, _ofv, _ofa, _ofa2 = _ovfg
        canvas_y = blend(canvas_y, _ofy, _ofa)
        canvas_u = blend(canvas_u, _ofu, _ofa2)
        canvas_v = blend(canvas_v, _ofv, _ofa2)

    out_frame = np.concatenate([canvas_y.flatten(), canvas_u.flatten(), canvas_v.flatten()])
    slot   = out_frame_index % RING_SIZE
    offset = HEADER_SIZE + slot * OUT_FRAME_SIZE
    shm_out_view[offset:offset + OUT_FRAME_SIZE] = out_frame.tobytes()
    ts_out = time.time_ns()
    # write_ts = instant de présentation (grille PTP) en genlock, sinon horloge mur.
    _wts = int(next_frame_time * 1e9) if GENLOCK else ts_out
    shm_out[0:16] = struct.pack("QQ", out_frame_index, _wts)
    # Latence par PiP : Δ ts_out − ts_in pour chaque input lu ce cycle
    for path, ts_in in ts_in_per_input.items():
        key = path[len("/dev/shm/"):] if path.startswith("/dev/shm/") else path
        if key not in lat_in:
            lat_in[key] = RollingMs()
        lat_in[key].push((ts_out - ts_in) / 1e6)
    out_frame_index += 1
    if GENLOCK:
        next_frame_time += FRAME_INTERVAL
        if next_frame_time < time.time():           # retard → recale sur la grille
            next_frame_time = _grid_next(time.time(), FRAME_INTERVAL)
    else:
        next_frame_time = start_time + (out_frame_index * FRAME_INTERVAL)
    if out_frame_index % 25 == 0:
        elapsed = time.time() - start_time
        metrics["fps"] = round(out_frame_index / elapsed, 1)
        _refresh_lat_metrics()
        print(f"Mix frame {{out_frame_index}} — {{metrics['fps']}} fps")
