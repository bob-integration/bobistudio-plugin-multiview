import mmap, socket, struct, time, numpy as np, threading, json, os, re
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
BORDER_W      = int(CONFIG.get("border_w") or 0)        # bordure globale (px)
BORDER_COLOR  = CONFIG.get("border_color") or "#ffffff" # couleur de bordure globale
OVERLAY_BELOW = bool(CONFIG.get("overlay_below"))       # bandeau sous l'image vs par-dessus
LABEL_SIZE    = int(CONFIG.get("label_size") or 14)     # taille du texte du label (px)
TSL_PORT      = int(CONFIG.get("tsl_port") or 0)        # port TCP TSL 5.0 (0 = désactivé)

# Chroma uniforme du pipeline (entrées ET sortie ont le même layout ; défaut 4:2:2).
CHROMA = str(CONFIG.get("chroma") or "422")
_CW = {{"420": 2, "422": 2, "444": 1}}.get(CHROMA, 2)   # diviseur largeur chroma
_CH = {{"420": 2, "422": 1, "444": 1}}.get(CHROMA, 1)   # diviseur hauteur chroma
PIX_FMT = {{"420": "yuv420p", "422": "yuv422p", "444": "yuv444p"}}.get(CHROMA, "yuv422p")
OUT_FRAME_SIZE = OUT_WIDTH * OUT_HEIGHT + 2 * (OUT_WIDTH // _CW) * (OUT_HEIGHT // _CH)
HEADER_SIZE    = 64
RING_SIZE      = 10
OUT_TOTAL      = HEADER_SIZE + (OUT_FRAME_SIZE * RING_SIZE)
FRAME_INTERVAL = 1.0 / 25

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
TALLY_SIZE = max(8,  int(round(LABEL_SIZE * 1.4)))
TALLY_PAD  = max(2,  int(round(LABEL_SIZE * 0.35)))
BAR_H      = max(14, int(round(LABEL_SIZE * 2)))

# ─── Peak meters (audio) ─────────────────────────────────────
# Format shm audio cohérent avec receiver_nmos.py : header 64B + ring 100 chunks
# de 1152 bytes (1 ms à L24/48k/8ch).
A_SAMPLE_RATE       = 48000
A_CHANNELS_MAX      = 8
A_BIT_DEPTH         = 24
A_SAMPLES_PER_CHUNK = A_SAMPLE_RATE // 1000        # 48
A_CHUNK_SIZE        = A_SAMPLES_PER_CHUNK * A_CHANNELS_MAX * (A_BIT_DEPTH // 8)  # 1152
A_HEADER_SIZE       = 64
A_RING_SIZE         = 100
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
    # 3 bytes per sample, little-endian signed, 8 channels interleaved
    arr = np.frombuffer(chunk, dtype=np.uint8).reshape(A_SAMPLES_PER_CHUNK, A_CHANNELS_MAX, 3)
    samples = (arr[:, :, 0].astype(np.int32)
               | (arr[:, :, 1].astype(np.int32) << 8)
               | (arr[:, :, 2].astype(np.int32) << 16))
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
        if self.path != "/tally":
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            data = json.loads(self.rfile.read(length).decode() or "{{}}")
            idx  = int(data["flux_idx"])
            slot = str(data["slot"]).upper()
            color = str(data.get("color", "off")).lower()
            if slot not in ("L", "R") or color not in TALLY_COLORS:
                raise ValueError("slot/color invalide")
            tally_state[f"{{idx}}_{{slot}}"] = color
            tally_dirty.set()
            self._send_json({{"status": "ok"}})
        except Exception as e:
            self.send_response(400); self.end_headers()
            self.wfile.write(str(e).encode())

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
        frame_size = in_w * in_h + 2 * (in_w // _CW) * (in_h // _CH)
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
    y = ( 0.299 * r + 0.587 * g + 0.114 * b      ).clip(0, 255).astype(np.uint8)
    u = (-0.169 * r - 0.331 * g + 0.500 * b + 128).clip(0, 255).astype(np.uint8)
    v = ( 0.500 * r - 0.419 * g - 0.081 * b + 128).clip(0, 255).astype(np.uint8)
    # Sous-échantillonnage chroma selon le format du pipeline (_CW/_CH) : moyenne pour U/V,
    # max pour l'alpha. 4:2:0 → bloc 2×2 ; 4:2:2 → 1×2 (hauteur pleine) ; 4:4:4 → identité.
    def _sub_avg(p):
        pp = p.astype(np.uint16)
        if _CW == 2: pp = (pp[:, 0::2] + pp[:, 1::2] + 1) // 2
        if _CH == 2: pp = (pp[0::2, :] + pp[1::2, :] + 1) // 2
        return pp.astype(np.uint8)
    def _sub_max(p):
        if _CW == 2: p = np.maximum(p[:, 0::2], p[:, 1::2])
        if _CH == 2: p = np.maximum(p[0::2, :], p[1::2, :])
        return p
    u2 = _sub_avg(u); v2 = _sub_avg(v); a2 = _sub_max(a)
    return y, u2, v2, a, a2

def blend(dst, src, alpha):
    """uint8 + uint8 avec alpha uint8 → uint8."""
    a16 = alpha.astype(np.uint16)
    return ((dst.astype(np.uint16) * (255 - a16) + src.astype(np.uint16) * a16) // 255).astype(np.uint8)

# ─── Rendu d'overlay (PIL) ───────────────────────────────────

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", LABEL_SIZE)
except Exception:
    FONT = ImageFont.load_default()

def _is_protocol_label(cfg):
    return cfg.get("show_label") and cfg.get("label_source") == "protocol"

def render_static():
    """Pré-rendu une fois : bordures + bandeau noir + labels statiques (hostname/mxl_path).
    Les labels en mode 'protocol' et les pavés tally sont rendus dynamiquement."""
    img = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for cfg in FLUX_CONFIG:
        x, y, w, h = cfg["x"], cfg["y"], cfg["w"], cfg["h"]
        if BORDER_W > 0:
            for i in range(BORDER_W):
                d.rectangle([x + i, y + i, x + w - 1 - i, y + h - 1 - i], outline=BORDER_COLOR)
        bar_on = bool(cfg.get("show_label") or cfg.get("show_tally"))
        if bar_on:
            bar_top = y + h - BAR_H
            d.rectangle([x, bar_top, x + w, y + h], fill=(0, 0, 0, 180))
            if cfg.get("show_label") and not _is_protocol_label(cfg):
                text_l, text_r = x, x + w
                if cfg.get("show_tally"):
                    text_l += TALLY_PAD + TALLY_SIZE + 4
                    text_r -= TALLY_PAD + TALLY_SIZE + 4
                name = cfg.get("name", "") or ""
                d.text(((text_l + text_r) // 2, bar_top + BAR_H // 2),
                       name, font=FONT, fill=(255, 255, 255, 255), anchor="mm")
    return rgba_to_yuv(img)

def render_dynamic():
    """Re-rendu à chaque changement Tally/TSL : labels protocole + pavés L/R."""
    img = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, cfg in enumerate(FLUX_CONFIG):
        if not (cfg.get("show_label") or cfg.get("show_tally")):
            continue
        x, y, w, h = cfg["x"], cfg["y"], cfg["w"], cfg["h"]
        bar_top = y + h - BAR_H
        if _is_protocol_label(cfg):
            text_l, text_r = x, x + w
            tally_color = tally_state.get(f"{{i}}_L", "off")  # L et R ont la même couleur
            if cfg.get("show_tally"):
                text_l += TALLY_PAD + TALLY_SIZE + 4
                text_r -= TALLY_PAD + TALLY_SIZE + 4
            bg = TALLY_TEXT_BG.get(tally_color)
            if bg:
                d.rectangle([text_l, bar_top, text_r, y + h], fill=bg)
            txt = tsl_text.get(i, "") or ""
            txt_fill = TALLY_TEXT_COLORS.get(tally_color, (200, 200, 200, 255))
            d.text(((text_l + text_r) // 2, bar_top + BAR_H // 2),
                   txt, font=FONT, fill=txt_fill, anchor="mm")
        if cfg.get("show_tally"):
            ty = bar_top + (BAR_H - TALLY_SIZE) // 2
            stL = tally_state.get(f"{{i}}_L", "off")
            stR = tally_state.get(f"{{i}}_R", "off")
            cL  = TALLY_COLORS[stL];        bL = TALLY_BORDER_COLORS[stL]
            cR  = TALLY_COLORS[stR];        bR = TALLY_BORDER_COLORS[stR]
            d.rectangle([x + TALLY_PAD,                  ty,
                         x + TALLY_PAD + TALLY_SIZE,     ty + TALLY_SIZE],
                        fill=cL, outline=bL)
            d.rectangle([x + w - TALLY_PAD - TALLY_SIZE, ty,
                         x + w - TALLY_PAD,              ty + TALLY_SIZE],
                        fill=cR, outline=bR)
    return rgba_to_yuv(img)

def render_meters(now):
    """Re-rendu par frame : layer RGBA contenant tous les peak meters.
    Renvoie (y, u, v, a, a2) prêt à blender, ou None si aucun meter activé."""
    has_meters = any((cfg.get("meter_channels") or 0) > 0 for cfg in FLUX_CONFIG)
    if not has_meters:
        return None
    img = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
    for i, cfg in enumerate(FLUX_CONFIG):
        n = int(cfg.get("meter_channels") or 0)
        if n == 0:
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
        # Geometry du meter dans la cellule
        x, y, w, h = cfg["x"], cfg["y"], cfg["w"], cfg["h"]
        x = max(0, min(x, OUT_WIDTH  - 1))
        y = max(0, min(y, OUT_HEIGHT - 1))
        w = max(2, min(w, OUT_WIDTH  - x))
        h = max(2, min(h, OUT_HEIGHT - y))
        bar_on = bool(cfg.get("show_label") or cfg.get("show_tally"))
        vh = h - BAR_H if (OVERLAY_BELOW and bar_on) else h
        mw = _meter_layout(n)
        mh = max(20, vh - 4)
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

if TSL_PORT > 0:
    threading.Thread(target=_tsl_server, daemon=True).start()

static_y, static_u, static_v, static_a, static_a2 = render_static()
dyn_y = dyn_u = dyn_v = dyn_a = dyn_a2 = None

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
        wanted = mv_state["inputs"][i]
    cur = sources[i]
    if not wanted:
        if cur is not None:
            try: cur["shm"].close(); cur["f"].close()
            except Exception: pass
            sources[i] = None
        return None
    if cur is not None and cur.get("path") == wanted:
        return cur
    if cur is not None:
        try: cur["shm"].close(); cur["f"].close()
        except Exception: pass
        sources[i] = None
    cfg = FLUX_CONFIG[i]
    src = open_source({{"path": wanted, "in_w": cfg.get("in_w", 640), "in_h": cfg.get("in_h", 360)}})
    if src is not None:
        src["path"] = wanted
        sources[i] = src
    return src

class MvControlHandler(BaseHTTPRequestHandler):
    def _json(self):
        n = int(self.headers.get("Content-Length") or 0)
        try: return json.loads(self.rfile.read(n).decode()) if n else {{}}
        except Exception: return {{}}
    def do_POST(self):
        if self.path == "/window":
            return self._do_window()
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
        # {{idx, x?, y?, w?, h?}} : repositionne/redimensionne une fenêtre à chaud.
        b = self._json()
        try: idx = int(b.get("idx"))
        except Exception:
            self.send_response(400); self.end_headers(); return
        ok = False
        with state_lock:
            if 0 <= idx < len(FLUX_CONFIG):
                cfg = FLUX_CONFIG[idx]
                for k in ("x", "y", "w", "h"):
                    if k in b and b[k] is not None:
                        try: cfg[k] = int(b[k])
                        except (TypeError, ValueError): pass
                ok = True
                geom_dirty.set()   # re-baker la couche statique (bordures/labels)
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
next_frame_time = start_time

while True:
    now = time.time()
    wait = next_frame_time - now
    if wait > 0:
        time.sleep(wait)

    canvas_y = np.zeros((OUT_HEIGHT, OUT_WIDTH), dtype=np.uint8)
    canvas_u = np.full((OUT_HEIGHT//_CH, OUT_WIDTH//_CW), 128, dtype=np.uint8)
    canvas_v = np.full((OUT_HEIGHT//_CH, OUT_WIDTH//_CW), 128, dtype=np.uint8)
    ts_in_per_input = {{}}  # path → ts_in_ns, rempli par les lectures réussies

    for i in range(len(FLUX_CONFIG)):
        src = ensure_input(i)   # rouvre le mmap si la source a changé (hot-input)
        cfg = FLUX_CONFIG[i]
        x, y, w, h = cfg["x"], cfg["y"], cfg["w"], cfg["h"]
        x = max(0, min(x, OUT_WIDTH  - 1))
        y = max(0, min(y, OUT_HEIGHT - 1))
        w = max(2, min(w, OUT_WIDTH  - x))
        h = max(2, min(h, OUT_HEIGHT - y))
        w = w if w % 2 == 0 else w - 1
        h = h if h % 2 == 0 else h - 1

        # Si bandeau "sous l'image", on rétrécit la zone vidéo verticalement
        bar_on = bool(cfg.get("show_label") or cfg.get("show_tally"))
        vh = h - BAR_H if (OVERLAY_BELOW and bar_on) else h
        vh = vh if vh % 2 == 0 else vh - 1
        if vh < 2:
            vh = h  # cellule trop petite pour un bandeau sous — fallback overlay

        # Si meter "hors image", on rétrécit la zone vidéo horizontalement et on
        # décale son origine X selon la position (gauche/droite). Le meter sera
        # dessiné par render_meters() dans l'espace restant.
        meter_n = int(cfg.get("meter_channels") or 0)
        meter_off = 0
        if meter_n > 0 and not cfg.get("meter_inside"):
            meter_off = _meter_layout(meter_n) + 4
            meter_off = meter_off if meter_off % 2 == 0 else meter_off + 1
        video_w = max(2, w - meter_off)
        video_w = video_w if video_w % 2 == 0 else video_w - 1
        video_x = x + (meter_off if cfg.get("meter_position") == "left" else 0)

        if src is None:
            canvas_y[y:y+vh, video_x:video_x+video_w] = 0
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

            src_y = np.frombuffer(bytes(shm_view[offset:offset + in_w*in_h]),
                                  dtype=np.uint8).reshape(in_h, in_w)
            in_uv = (in_w // _CW) * (in_h // _CH)   # taille d'un plan chroma d'entrée
            src_u = np.frombuffer(bytes(shm_view[offset + in_w*in_h:offset + in_w*in_h + in_uv]),
                                  dtype=np.uint8).reshape(in_h//_CH, in_w//_CW)
            src_v = np.frombuffer(bytes(shm_view[offset + in_w*in_h + in_uv:offset + frame_size]),
                                  dtype=np.uint8).reshape(in_h//_CH, in_w//_CW)

            dst_y = resize_plane(src_y, vh,      video_w)
            dst_u = resize_plane(src_u, vh//_CH, video_w//_CW)
            dst_v = resize_plane(src_v, vh//_CH, video_w//_CW)

            canvas_y[y:y+vh,                       video_x:video_x+video_w]                  = dst_y
            canvas_u[y//_CH:y//_CH+vh//_CH,        video_x//_CW:video_x//_CW+video_w//_CW]   = dst_u
            canvas_v[y//_CH:y//_CH+vh//_CH,        video_x//_CW:video_x//_CW+video_w//_CW]   = dst_v
        except Exception:
            canvas_y[y:y+vh, video_x:video_x+video_w] = 0

    # Géométrie modifiée à chaud (:8082/window) → re-baker la couche statique
    if geom_dirty.is_set():
        with state_lock:
            static_y, static_u, static_v, static_a, static_a2 = render_static()
        geom_dirty.clear()
        tally_dirty.set()   # labels protocole + pavés tally ont aussi bougé

    # Blend overlay statique (bordures + noms)
    canvas_y = blend(canvas_y, static_y,  static_a)
    canvas_u = blend(canvas_u, static_u,  static_a2)
    canvas_v = blend(canvas_v, static_v,  static_a2)

    # Re-rendu dynamique (labels protocole + tally) si l'état a changé
    if tally_dirty.is_set():
        dyn_y, dyn_u, dyn_v, dyn_a, dyn_a2 = render_dynamic()
        tally_dirty.clear()

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

    out_frame = np.concatenate([canvas_y.flatten(), canvas_u.flatten(), canvas_v.flatten()])
    slot   = out_frame_index % RING_SIZE
    offset = HEADER_SIZE + slot * OUT_FRAME_SIZE
    shm_out_view[offset:offset + OUT_FRAME_SIZE] = out_frame.tobytes()
    ts_out = time.time_ns()
    shm_out[0:16] = struct.pack("QQ", out_frame_index, ts_out)
    # Latence par PiP : Δ ts_out − ts_in pour chaque input lu ce cycle
    for path, ts_in in ts_in_per_input.items():
        key = path[len("/dev/shm/"):] if path.startswith("/dev/shm/") else path
        if key not in lat_in:
            lat_in[key] = RollingMs()
        lat_in[key].push((ts_out - ts_in) / 1e6)
    out_frame_index += 1
    next_frame_time = start_time + (out_frame_index * FRAME_INTERVAL)
    if out_frame_index % 25 == 0:
        elapsed = time.time() - start_time
        metrics["fps"] = round(out_frame_index / elapsed, 1)
        _refresh_lat_metrics()
        print(f"Mix frame {{out_frame_index}} — {{metrics['fps']}} fps")
