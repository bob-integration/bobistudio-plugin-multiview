# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 BOBI SAS, France
# Auteur : Cyril Mazouer, pour le compte de BOBI SAS
# Distribué sous licence GNU GPL v3 (ou ultérieure) ; voir le fichier LICENSE.

import mmap, socket, struct, sys, time, numpy as np, threading, json, os, re, base64, io, signal, gc
import datetime   # horloges : conversion civile PAR FUSEAU (zoneinfo), cf. _civil_hms
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image, ImageDraw, ImageFont
import bobimxl   # migration MXL Phase 1 : entrées vidéo+ANC via Reader, sortie via Writer

# ── TRACE DE PLANTAGE NATIF ────────────────────────────────────────────────────────────────────
# Un mur qui meurt en `code -11` (SIGSEGV) sans rien laisser est INDÉBOGABLE. Mesuré le
# 2026-08-06 sur le mur 333 : 34 morts dans la journée, journal du conteneur muet — que des
# lignes de fonctionnement normal, puis « [agent] script terminé (code -11) ». On ne pouvait
# même pas dire QUEL appel C fautait (noyau mvk fusionné, binding MXL, numpy).
# `faulthandler` imprime sur stderr la pile Python de TOUS les threads au moment du signal, puis
# laisse le signal suivre son cours (comportement inchangé, le process meurt pareil). Coût nul
# tant qu'aucun signal n'arrive : rien ne s'exécute.
# ⚠ Il pose aussi un handler SIGBUS, mais `signal.signal(SIGBUS, _on_sigbus)` plus bas s'exécute
# APRÈS et reprend la main : la reconnexion sur mmap tronqué reste le comportement en vigueur.
import faulthandler
faulthandler.enable(all_threads=True)


def _mxl_lib_state():
    """Variante libmxl réellement chargée (baseline / x86-64-v3) — diagnostic seul, ne doit
    JAMAIS faire échouer /state."""
    try:
        return bobimxl.lib_info()
    except Exception:
        return None
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
    def mx(self):
        """PIC de la fenêtre. Une moyenne ne dit rien d'une cadence : sur une grille de 20 ms,
        c'est la QUEUE de la distribution qui fait tomber les trames, pas le centre. Un mur à
        7 ms de moyenne d'habillage qui pointe à 60 ms rate une trame sur trois."""
        if not self.d: return None
        if time.time_ns() - self.last_ns > 2_000_000_000: return None
        return round(max(self.d), 1)

lat_in = {{}}  # {{shm_name: RollingMs}} — TRANSIT par entrée (ts_read − ts_in producteur) = arrivée
own_lat = RollingMs()  # traitement PROPRE du nœud (ts_out − ts_cycle_start), exposé own_latency_ms
# Profiling du compositing (où vont les ms de own_latency) : entrées vidéo / habillage / sortie.
_t_inputs = RollingMs(); _t_overlays = RollingMs(); _t_output = RollingMs()
# Sous-ventilation des ENTRÉES : `inputs` est un agrégat de deux natures très différentes —
# la MOISSON (sélection du proxy, lecture du grain MXL, copie hôte des plans) et le PLACEMENT
# (_place_batch : upload GPU groupé + resize + pose dans le canvas). Sans les séparer on ne peut
# pas décider ce qu'il y a lieu de replier hors du chemin critique : la moisson est du travail
# CPU/mémoire parallélisable, le placement est une file GPU qui ne gagne rien à être threadée.
_t_in_read = RollingMs(); _t_in_place = RollingMs()
# Comptage des TUILES per-frame de l'habillage. Sur GPU chaque tuile blendée coûte TROIS
# lancements de kernel (un par plan), et le banc gate a montré que le coût de LANCEMENT domine
# le travail utile dès qu'on multiplie les petites tuiles (M3 : 45,6 ms à 30 bandes contre 3,0 ms
# en pleine trame). Sans ce comptage, `ov_blend` ne dit pas s'il est cher parce qu'il y a beaucoup
# de pixels ou parce qu'il y a beaucoup d'appels — deux problèmes aux remèdes opposés.
_n_meters = RollingMs(); _n_hist = RollingMs(); _n_clock = RollingMs(); _n_px_blend = RollingMs()
# Dans `ov_blend`, deux coûts cohabitent et appellent des remèdes OPPOSÉS : le TRANSFERT des tuiles
# vers la VRAM (5 tableaux par tuile, depuis de la mémoire hôte pageable) et les LANCEMENTS de
# kernel du blend lui-même (3 par tuile). Si c'est le transfert qui domine, le remède est de garder
# côté GPU les tuiles qui n'ont pas changé (frises et horloges sont déjà cachées côté hôte, et
# pourtant re-téléversées à chaque trame) ; si ce sont les lancements, le remède est de grouper.
_t_ov_upload = RollingMs()
# `ov_clock` agrège deux choses payées à des rythmes différents : le calcul de la SIGNATURE des
# tuiles per-frame (fait à CHAQUE trame — il décode les paquets ANC pour savoir si la valeur a
# bougé) et le RENDU proprement dit (payé seulement quand elle a bougé).
_t_pf_sig = RollingMs()
# Sous-ventilation de l'habillage (overlays) : rendu PIL meters/fg / conversion RGBA→YUV / blend.
_t_ov_render = RollingMs(); _t_ov_convert = RollingMs(); _t_ov_blend = RollingMs()
# Instrumentation des RE-BAKES d'habillage (couches cachées) : ces coûts sont épisodiques mais
# PLEIN CADRE (PIL + rgba_to_yuv + uploads). Compteurs de FRÉQUENCE (par seconde) + coût moyen :
# c'est la seule façon de voir un habillage re-baké en rafale (churn TSL/statuts) — invisible dans
# la ventilation inputs/overlays/output où il se noie.
# Ventilation FINE de ov_render (banc docs/chantiers/MULTIVIEW_BENCH.md) : VU-mètres / frises / horloges+ANC.
_t_ov_meters = RollingMs(); _t_ov_hist = RollingMs(); _t_ov_clock = RollingMs()
_t_ov_bake = RollingMs()   # re-bake du chrome (Image.new + alpha_composite + rgba_to_yuv bbox)
_t_ov_bg   = RollingMs()   # re-bake couche fond/fg statique (overlay_dirty) — compté dans `inputs`
_bake_ctr = {{"chrome": 0, "bg": 0, "tally": 0, "geom": 0, "info": 0, "hist": 0, "frames": 0,
              "t0": time.time()}}
_bake_rate = {{}}          # dernier instantané par seconde (exposé sur :8080)

# ─── Config injectée (contrat plugin) ───────────────────────
CONFIG         = {config}
HOSTNAME       = "{hostname}"
_START_TS      = time.time()   # origine de la variable de texte %duree%
PLUGIN_VERSION = "{plugin_version}"

# ─── Niveau de log ─────────────────────────────────────────────────────────
# `log_level` (config_schema du plugin, défaut « info ») filtre les impressions du script.
# Le critère n'est PAS « verbeux vs silencieux » mais ÉVÉNEMENT vs MÉTRIQUE :
#   debug   — le lance-flammes : par trame, par bande, décisions internes
#   info    — ÉVÉNEMENTS rares et signifiants  ← DÉFAUT (toujours visible) : démarrage/
#             arrêt, session ouverte/fermée, changement de format, reconnexion, repli sur
#             un chemin dégradé, entrée qui apparaît/disparaît, rebascule.
#   warning — anomalies et replis subis
#   error   — échecs
# RÈGLE 1 : après une panne, le journal PAR DÉFAUT doit permettre de RECONSTITUER
#   l'histoire. Élever le niveau après coup ne récupère RIEN : ce qui n'a pas été écrit
#   est perdu. On ne coupe donc pas l'information, on coupe la redondance.
# RÈGLE 2 : une MÉTRIQUE PÉRIODIQUE (fps, compteurs) ne se journalise PAS — elle est déjà
#   publiée sur :8080 et échantillonnée par l'orchestrateur. La journaliser duplique la
#   mesure ET consomme la fenêtre de rétention (journal Docker non roté : le bruit purge
#   les lignes utiles anciennes). Au mieux `debug`.
# RÈGLE 3 : un événement qui peut partir EN RAFALE s'AGRÈGE sur une fenêtre et sort en UNE
#   ligne périodique (« N frames lentes sur la dernière minute, pire … ») — le signal
#   reste, le spam disparaît.
# Réglable à chaud, sans redéployer, quand le plugin expose l'endpoint de contrôle :
# POST :8082/log_level {{"level": "debug"}} (exposé aux macros via param_tree/actions).
_LOG_ORDER = {{"debug": 10, "info": 20, "warning": 30, "error": 40}}
LOG_LEVEL = str(CONFIG.get("log_level") or "info").strip().lower()
if LOG_LEVEL not in _LOG_ORDER:
    LOG_LEVEL = "info"
_LOG_MIN = _LOG_ORDER[LOG_LEVEL]


def log(msg, niveau="info"):
    """Impression gatée par le niveau de log courant (défaut du message : « info »)."""
    if _LOG_ORDER.get(niveau, 20) >= _LOG_MIN:
        print(msg, flush=True)


def set_log_level(niveau):
    """Change le niveau à chaud. Renvoie True si le niveau est reconnu."""
    global LOG_LEVEL, _LOG_MIN
    lv = str(niveau or "").strip().lower()
    if lv not in _LOG_ORDER:
        return False
    LOG_LEVEL, _LOG_MIN = lv, _LOG_ORDER[lv]
    return True



# Filet anti-régression : `force_cpu` force le chemin numpy ÉPROUVÉ (xp=np) MÊME sur un nœud GPU,
# sans changer d'image. Réassigne les globals AVANT toute boucle de rendu → repli instantané identique
# au multiview 100% CPU si le chemin GPU posait souci. Absent/false → aucun effet (auto-détection).
if CONFIG.get("force_cpu"):
    xp = np
    GPU = False
    _GPU_NAME = None

# ─── Kernel compose fusionné C (libbobi_mvk, chantier fusion numpy→C 2026-07) ─────────────
# Chemin CPU uniquement (le GPU garde ses ElementwiseKernel fusionnés) : blend / blend_pre /
# place nearest en UNE passe mémoire chacun via bobimxl.mvk_* (image bobi-compute ≥ 0.11,
# bit-exact au numpy — selftest mvk_selftest.py). Lib absente (vieille image) OU wrappers non
# applicables → repli numpy intégral : le repli EST l'ancien code, octet-identique. Threads
# OpenMP posés par bobimxl au chargement (env BOBI_MVK_THREADS, sinon cœurs physiques du
# cpuset HT-aware). getattr : un bobimxl d'ancienne image n'a pas mvk_available.
# ⚠ INTERRUPTEUR `mvk` (config_schema, défaut activé). Le noyau fusionné est du C : quand il
# écrit hors bornes, il corrompt le tas et le process meurt PLUS TARD, AILLEURS — mesuré le
# 2026-08-06 sur le mur 333, 4 traces, 3 sites de crash différents (mvk_rgba2yuv, une affectation
# numpy, PIL.tobytes), toujours avec les boulangers chrome/frises actifs. Sans moyen de le
# désactiver, impossible de trancher entre « c'est lui » et « c'est autre chose », ni de tenir un
# mur debout en attendant. Le repli numpy EST l'ancien code, octet-identique (cf. ci-dessus) :
# le couper coûte de la vitesse, jamais de la justesse.
# (évaluation littérale : `_as_bool` est défini plus bas dans le fichier)
_MVK_ON = str(CONFIG.get("mvk", True)).strip().lower() not in ("0", "false", "no", "off", "none", "")
_MVK = (not GPU) and _MVK_ON and bool(getattr(bobimxl, "mvk_available", lambda: False)())
# Gate HÔTE (par-appel) : certaines passes restent du numpy CPU MÊME sur un mur GPU — le
# re-bake du chrome (rgba_to_yuv sur l'image PIL pleine trame, à chaque tally/statut) en tête.
# Le gate global `not GPU` de _MVK les privait du kernel C (observé sur le 163 : ov_render
# 5-7 ms de re-bakes en numpy pur). Les call-sites qui l'utilisent vérifient eux-mêmes que le
# tableau est bien hôte (isinstance cupy) — le chemin VRAM garde ses kernels cupy.
_MVK_HOST = _MVK_ON and bool(getattr(bobimxl, "mvk_available", lambda: False)())

FLUX_CONFIG   = CONFIG.get("flux_config") or []
# Blocs VU-mètres posés DIRECTEMENT sur le layout du MUR (indépendants de toute fenêtre) :
# {{x,y,w,h (fractions 0..1 du MUR ENTIER — pas d'une cellule), channels, ch_start, scale,
# opacity, align, width_mode, audio_path, label?}}. Même vocabulaire que le composant `meters`
# des modèles de PiP (cf. section « Modèles de PiP ») — le rendu réutilise EXACTEMENT le même
# corps de dessin (_comp_rect/_meter_fit_dims/_meter_tiles_at) via un cfg synthétique couvrant
# tout le canvas (cf. render_meters). Modifiable à chaud via :8082/reconfigure (comme flux_config).
METER_BLOCKS  = CONFIG.get("meter_blocks") or []
# Blocs d'HISTORIQUE posés sur le MUR (mêmes conventions que meter_blocks : fractions 0..1 du mur,
# source câblée en propre page Câbles, modifiables à chaud via :8082/reconfigure) :
#   video_history_blocks : {{x,y,w,h, duration(10|30|60|120), opacity, events, path, label?}}
#   audio_history_blocks : {{x,y,w,h, duration, channels, ch_start, opacity, audio_path, label?}}
# Mêmes clés que les composants `video_history`/`audio_history` des modèles de PiP → UN SEUL
# code de rendu (cf. section « Historique vidéo / audio »).
VHIST_BLOCKS  = CONFIG.get("video_history_blocks") or []
AHIST_BLOCKS  = CONFIG.get("audio_history_blocks") or []
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
def _as_bool(v, default=False):
    # bool("False") == True : on parse explicitement les chaînes de CONFIG.
    # `default` rendu pour v=None (les appels 0.28.0 `_as_bool(x, False)` levaient TypeError →
    # tout affichage ANC par cellule crashait la trame ; corrigé en 0.29.0).
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)

# L'habillage GLOBAL de mur (border_w/border_color/overlay_below/label_size/frame_style/
# show_format) a été MIGRÉ dans les MODÈLES DE PIP (0.33.0) : le cadre est une propriété du
# composant `video`, le « texte sous l'image » un layout de modèle. Le chemin de rendu
# classique est SUPPRIMÉ ; le repli sans modèle = modèle « Classique » GÉNÉRÉ (_classic_comps).
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
SHOW_PROXY    = _as_bool(CONFIG.get("show_proxy"))      # badge proxy lu par tuile (mode ingénierie pyramide)
# Heure CIVILE des horloges « PTP » : l'horloge du nœud est sur l'échelle PTP/TAI (docs/reference/PTP_CLOCK.md)
# → heure civile = horloge − tai_utc_offset_s (mesuré, injecté par before_deploy), au fuseau du
# contrôleur (`tz` — les images runtime sont en UTC). N'affecte QUE l'affichage des horloges ;
# la grille genlock/TAI du compositing est intouchée.
_TZ_NAME = str(CONFIG.get("tz") or "").strip()
if _TZ_NAME:
    os.environ["TZ"] = _TZ_NAME
    try:
        time.tzset()
    except Exception:
        pass

# FUSEAU PAR HORLOGE. `tz` du CONFIG = fuseau du mur (hérité du réglage global du système, injecté
# au déploiement) ; chaque horloge peut le SURCHARGER par son propre champ `tz` — mur d'horloges
# Paris / New York / Tokyo, usage courant en régie. On ne peut donc PAS se contenter du TZ global du
# process (`time.localtime`) : la conversion doit être explicite, par zone, à chaque horloge.
_ZONE_CACHE = {{}}     # nom IANA → ZoneInfo | None (None = introuvable, déjà signalé)
_ZONE_BAD   = {{}}     # nom IANA refusé → nb d'horloges concernées (exposé sur :8080)

def _zone(name):
    """ZoneInfo d'un nom IANA, mis en cache. Renvoie None si le nom est vide, si `zoneinfo` est
    indisponible, ou si la base tzdata de l'image ne connaît pas ce fuseau — l'appelant retombe
    alors sur le fuseau du mur. Un fuseau refusé est COMPTÉ (`clock_tz_unknown` sur :8080) : une
    horloge qui affiche silencieusement la mauvaise heure est pire qu'une horloge absente."""
    name = (name or "").strip()
    if not name:
        return None
    if name in _ZONE_CACHE:
        z = _ZONE_CACHE[name]
        if z is None:
            _ZONE_BAD[name] = _ZONE_BAD.get(name, 0) + 1
        return z
    z = None
    try:
        from zoneinfo import ZoneInfo
        z = ZoneInfo(name)
    except Exception as _e:
        log("horloge : fuseau « %s » inconnu de l'image (%s) — repli sur le fuseau du mur"
            % (name, _e), "warning")
        _ZONE_BAD[name] = _ZONE_BAD.get(name, 0) + 1
    _ZONE_CACHE[name] = z
    return z

def _civil_hms(ts, tz_name):
    """(heure, minute, seconde) civiles d'un instant epoch UTC dans `tz_name`. Fuseau absent ou
    inconnu → fuseau du PROCESS (celui du mur, posé par os.environ['TZ'] ci-dessus) : identique au
    comportement historique."""
    z = _zone(tz_name)
    if z is not None:
        d = datetime.datetime.fromtimestamp(ts, z)
        return d.hour, d.minute, d.second
    _lt = time.localtime(ts)
    return _lt.tm_hour, _lt.tm_min, _lt.tm_sec
try:
    TAI_UTC_OFFSET_S = max(0, int(CONFIG.get("tai_utc_offset_s") or 0))
except (TypeError, ValueError):
    TAI_UTC_OFFSET_S = 0

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
# ─── Filtre de RÉDUCTION des tuiles vidéo ────────────────────────────────────────────────────
# Une tuile de mur est une RÉDUCTION de sa source : 1920×1080 dans une case 640×360 = ratio 3,
# soit UN pixel de sortie pour NEUF d'entrée. Historiquement le placement prenait un pixel sur
# les neuf et jetait les huit autres (décimation `plane[::3, ::3]`) : c'est du sous-échantillon-
# nage sans pré-filtre, donc du repliement de spectre. Tout ce qui est fin dans la source — texte
# incrusté, mires, contours — devient du crénelage ou disparaît purement et simplement.
#   "box"     : moyenne du bloc source (filtre d'AIRE), le pré-filtre CORRECT pour une réduction.
#   "nearest" : décimation historique — conservée pour comparer et pour dégager du CPU.
# ⚠ COÛT MESURÉ (banc numpy, plan Y 1920×1080→640×360) : décimation+copie 0,13 ms, box 2,8 ms.
# La box LIT tous les pixels source là où la décimation n'en lisait qu'un sur neuf : c'est une
# opération memory-bound et le facteur ~10 est structurel, pas une maladresse d'implémentation.
# Sur un gros mur ça ne tient pas 50 fps en numpy — d'où le réglage. La suite est un kernel C
# fusionné (réduction+placement en une passe, cf. mvcompose.c/mvk_place) sur le modèle des
# autres passes de compositing, qui rend le surcoût négligeable.
SCALE_FILTER = str(CONFIG.get("scale_filter") or "box").strip().lower()
if SCALE_FILTER not in ("box", "nearest"):
    SCALE_FILTER = "box"
_BOX = (SCALE_FILTER == "box")
# ─── Traitement des sources ENTRELACÉES ──────────────────────────────────────────────────────
# Le mur sort TOUJOURS en progressif. Une source 1080i doit donc être désentrelacée.
#   "weave" : les DEUX champs appariés sont retissés en une trame pleine (défaut). Rend la
#             RÉSOLUTION VERTICALE COMPLÈTE — sur une source 1080i c'est un facteur 2 rendu à
#             l'image AVANT toute mise à l'échelle, bien plus que ce qu'un filtre peut donner.
#   "bob"   : champ HAUT seul, comportement historique. Moitié de la résolution verticale, mais
#             strictement insensible au mouvement.
# Le weave peigne sur le mouvement : deux champs distants de 20 ms cousus ligne à ligne. Ici
# c'est acceptable parce que la tuile est TOUJOURS une forte réduction — le filtre box vertical
# qui suit moyenne des lignes des DEUX champs et absorbe l'essentiel du peigne. Ce raisonnement
# NE VAUT PAS pour une sortie 1:1 : ne pas réutiliser ce weave ailleurs sans le rechiffrer.
INTERLACE_MODE = str(CONFIG.get("interlace_mode") or "weave").strip().lower()
if INTERLACE_MODE not in ("weave", "bob"):
    INTERLACE_MODE = "weave"
PIX_FMT = ({{"420": "yuv420p", "422": "yuv422p", "444": "yuv444p"}}.get(CHROMA, "yuv422p")
           + (("12le" if BIT_DEPTH >= 12 else "10le") if _DEEP else ""))
OUT_FRAME_SIZE = (OUTPUT_W * OUTPUT_H + 2 * (OUTPUT_W // _CW) * (OUTPUT_H // _CH)) * _BPS

# ─── BALAYAGE DE SORTIE (progressif / ENTRELACÉ) ─────────────────────────────────────────────
# `scan`/`field_order` sont portés HORS-BANDE par le deploy_config (cf. app/scripts.py) et
# DÉCLARÉS à tout l'aval (NMOS, slot TX du moteur 2110, SDP). Jusqu'en 0.40.x le script les
# IGNORAIT : un mur « 1080i50 » écrivait une TRAME PLEINE PROGRESSIVE (flowDef
# interlace_mode=progressive, grain = 4 147 200 o) — mesuré au banc. L'aval champ-natif (mtl_rx :
# `plane_h = height/2`, `reader_field`) lisait alors la MOITIÉ du payload = la moitié HAUTE de
# l'image. D'où : sortie « i » en réalité progressive ET images coupées en deux côté consommateur.
#
# CONTRAT MXL ENTRELACÉ (identique au txgen du moteur 2110, seule référence qui fait foi) :
#   flowDef : frame_height PLEINE (1080) + interlace_mode=interlaced_tff/bff
#             + grain_rate = cadence TRAME (25/1)
#   libmxl  : dimensionne chaque grain à 1 CHAMP (½ hauteur) et DOUBLE la cadence de grain (50/s)
#   écriture: 2 grains-CHAMPS par trame, aux index CHAMP = trame×2 + parité
#             (index PAIR = lignes PAIRES = champ HAUT ; index IMPAIR = lignes impaires)
# L'ordre de champ ne change PAS cette correspondance index↔parité de ligne : il dit seulement
# lequel des deux est émis EN PREMIER sur le fil (le moteur s'en charge).
SCAN = str(CONFIG.get("scan") or "p").strip().lower()
INTERLACED = (SCAN == "i") and OUTPUT_H % 2 == 0
FIELD_ORDER = str(CONFIG.get("field_order") or "").strip().lower()
if FIELD_ORDER not in ("tff", "bff"):
    # Défaut par résolution, MÊME convention que app.scripts.field_order (HD/UHD = TFF, SD = BFF).
    FIELD_ORDER = "bff" if 0 < OUTPUT_H <= 576 else "tff"
IL_MODE = (("interlaced_bff" if FIELD_ORDER == "bff" else "interlaced_tff")
           if INTERLACED else "progressive")
# Taille d'un grain-CHAMP (½ hauteur) — ce que libmxl alloue réellement en entrelacé.
FIELD_H = OUTPUT_H // 2
OUT_FIELD_SIZE = (OUTPUT_W * FIELD_H + 2 * (OUTPUT_W // _CW) * (FIELD_H // _CH)) * _BPS

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
# ENTRELACÉ : le format de réglage stocke la cadence CHAMP (« HD 1080i50 » → fps=50), comme le
# reste de la chaîne (cf. moteur 2110 : `if scan == "i" and fps > 30: fps /= 2`). Le flowDef et la
# boucle de composition travaillent en cadence TRAME (25) — 1 trame composée = 2 champs émis.
if INTERLACED and (_FN / _FD) > 30:
    if _FN % 2 == 0: _FN //= 2            # 50/1 → 25/1
    else:            _FD *= 2             # 60000/1001 → 30000/1001 (59,94i → 29,97)
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
# ── GPU SLICE (chantier TISSU_SLICE_GPU.md, squelette banc gate) ──────────────────────────────
# `gpu_slice` (OPT-IN, défaut off) lève l'exclusion GPU du mode tranche : composition bande par
# bande EN VRAM — H2D par bande (staging épinglé groupé par bande, _gpu_place_band), blends
# chrome/VU/horloges par bande en VRAM (opérandes déjà résidents), D2H par bande épinglé + commit
# progressif. Sémantique STRICTEMENT identique au slice CPU (_compose_bands est commun : attentes
# get_slice, budgets PAR TUILE, backoff, ciblage flow — seuls les octets transitent par la VRAM).
# GATE : ne PAS activer en prod avant le verdict du banc dl360-2/T4 (tools/bench_gpu_slice_gate.py)
# — le banc Phase 0 a montré que des transferts fins mal faits font RÉGRESSER le GPU. Sans le flag,
# comportement inchangé : GPU ⇒ whole-frame (upload groupé), le repli documenté docs/chantiers/TISSU_SLICE.md §4.
GPU_SLICE_REQ = _as_bool(CONFIG.get("gpu_slice", False))
# Repli lot PIL (0.31.0) : `meters_pil=true` restaure le rendu PIL par-trame HISTORIQUE des
# VU-mètres (avant 0.31.0 le chemin tuile — statique caché + barres peintes — était GPU-only).
METERS_PIL = _as_bool(CONFIG.get("meters_pil", False))
# ENTRELACÉ : le mode TRANCHE est INCOMPATIBLE (un grain = 1 champ, pas une trame à découper en
# bandes) → désactivé, comme dans le moteur 2110 (`if slice_wanted() && !s->interlaced`).
SLICE_ON = (SLICE_MODE and (not GPU or GPU_SLICE_REQ) and not _PORTRAIT and not INTERLACED
            and SLICE_LINES > 0 and OUT_HEIGHT % SLICE_LINES == 0 and SLICE_LINES % _CH == 0)
GPU_SLICE = GPU and SLICE_ON      # chemin tranche VRAM actif (exige GPU + flag + éligibilité)
# ── MICRO-BATCH GPU (TISSU_SLICE_GPU.md §b variante 2, verdict banc gate GO-MEGA-135) ─────────
# Le banc gate (T4, sous charge) a REJETÉ le bande-à-bande 36 l (trame 15,26 ms — le coût de
# LANCEMENT des kernels domine : compose 10,6 ms à 30 lancements vs 0,71 ms pleine trame) et voté
# méga-bandes 135 l (trame 5,66 ms, 1ʳᵉ bande 0,58 ms, surcoût transferts +0,12 ms). Implémentation
# retenue : micro-batch — `gpu_batch_bands` = NOMBRE DE BANDES MXL (SLICE_LINES) PAR LOT GPU,
# dernier lot PARTIEL si nb_bandes % k ≠ 0. Sémantique choisie (vs « nb de lots par trame ») car
# le paramètre reste stable quelle que soit la hauteur de sortie et colle au grain MXL ; à
# 1080p/36 l, k=4 → 8 lots (7×4 bandes + 1×2) de 144 l ≈ les 135 l mesurés GO (1080/8 n'est PAS
# un multiple de 36 — 144 l est le grain MXL le plus proche ; 8 lancements/trame, pas 30).
# get_slice (attentes amont) et commit MXL restent au grain SLICE_LINES : l'amont/aval ne voit
# AUCUNE différence de protocole, seule la granularité des opérations GPU grossit (H2D groupé par
# lot, kernels par lot, D2H par lot RECOUVERT — cf. _compose_bands). k=1 = squelette 0.26.0.
GPU_BATCH_BANDS = 4
try:
    GPU_BATCH_BANDS = max(1, int(CONFIG.get("gpu_batch_bands", 4) or 1))
except Exception:
    pass
if SLICE_MODE and not SLICE_ON:
    log(f"multiview: slice_mode demandé mais inéligible (gpu={{GPU}} sans gpu_slice "
        f"portrait={{_PORTRAIT}} h={{OUT_HEIGHT}}%{{SLICE_LINES}}) — whole-frame", "warning")
if GPU_SLICE:
    log(f"multiview: GPU SLICE actif (opt-in banc, TISSU_SLICE_GPU.md) — bandes de "
        f"{{SLICE_LINES}} lignes sur {{_GPU_NAME}}, micro-batch {{GPU_BATCH_BANDS}} bande(s)/lot",
        "info")
# ── CADENCE « flow » (tissu en tranches, docs/chantiers/TISSU_SLICE.md) ──────────────────────────────────────
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
    log("multiview: cadence=flow exige le mode tranche éligible — repli genlock", "warning")
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
# Couleurs de bordure épaisse selon tally dominant (bordures tally des composants video)
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
# Largeur de la COLONNE DE GRADUATIONS d'un VU-mètre (libellés dB + petit trait de repère).
# ★ MESURÉE, pas choisie : le trait occupe les 4 derniers pixels, et le libellé le plus large
# ("+12" en échelle PPM, police par défaut) fait 19 px. Il faut donc 1 (marge gauche) + 19 + 2
# (garde) + 4 (trait) = 26. À 16 px — la valeur historique — TOUT libellé à 3 caractères
# ("-12", "-18", "-30"…) mesure 15 px et finissait DANS le trait de repère ; "+12" débordait
# même de 4 px sur les barres. Rétrécir la police n'est pas une porte de sortie : il faudrait
# descendre à la taille 6, soit des chiffres de 4 px de haut, illisibles sur un mur.
# Largeur SELON L'ÉCHELLE : le « +12 » du PPM (19 px) est le seul libellé à imposer 26 ; en dBFS
# le pire est « -18 » (15 px), 22 suffisent. Rendre ces 4 px aux murs en dBFS ne coûte rien.
METER_TICK_W_DBFS    = 22
METER_TICK_W_PPM     = 26
METER_TICK_W_MARKS   = 6    # traits seuls (4 px de trait + 2 de garde), sans les chiffres
METER_TICK_W         = METER_TICK_W_PPM   # défaut des signatures = le pire cas (jamais trop étroit)
# Marge réservée entre un VU-mètre et le BORD de sa cellule : sans elle, un mètre aligné sur le
# bord touche visuellement la fenêtre voisine et on ne sait plus à quel PiP l'audio appartient.
METER_EDGE_MARGIN    = 4
METER_MIN_DB         = -60.0
METER_DECAY_DB_PER_S = 20.0    # vitesse de chute du peak hold

# audio_states[flux_idx] = {{ar (bobimxl.AudioReader), name, peaks (np), holds (np), hold_ts (np)}}
audio_states = {{}}

def _derive_audio_name(video_path, flow=0):
    """`mire1_0` (ou `/dev/shm/mire1_0`) → nom de flux audio `mire1_audio_0`. None si pas de _N final.
    `flow` : décalage de flux — les composants meters des modèles de PiP adressent un espace de
    16 canaux = 2 FLUX de 8 (canaux 1-8 = flux dérivé, 9-16 = flux dérivé SUIVANT `_audio_{{N+1}}`)."""
    name = (video_path or "").removeprefix("/dev/shm/")
    m = re.match(r"(.+?)_(\d+)$", name)
    return f"{{m.group(1)}}_audio_{{int(m.group(2)) + flow}}" if m else None

# Sentinel `audio_path` (composer/multiview.js, sélecteur « Source audio ») : distingue
# EXPLICITEMENT « aucune » (VU-mètres coupés) de la valeur vide historique (= auto, dérivée de
# la vidéo). Convention purement UI/config — jamais un vrai nom de shm.
AUDIO_PATH_NONE = "__none__"

def _audio_name_for(cfg, flow=0):
    """Nom du flux audio de la fenêtre : le port CÂBLÉ (`audio_path`, page Câbles OU sélecteur
    du composer) sinon la dérivation historique depuis la vidéo. Trois états `audio_path` :
    absent/vide = auto (dérivé) ; `AUDIO_PATH_NONE` = explicitement aucun (renvoie None, pas de
    repli sur la dérivation) ; toute autre valeur = flux explicite. `flow`=1 : flux SUIVANT
    (canaux 9-16) — dérivé en bumpant l'index final du nom du flux 0 (câblé ou dérivé)."""
    wired = (cfg.get("audio_path") or "").strip() if isinstance(cfg.get("audio_path"), str) else ""
    if wired == AUDIO_PATH_NONE:
        return None
    wired = wired.removeprefix("/dev/shm/")
    if wired:
        if flow == 0:
            return wired
        m = re.match(r"(.+?)_(\d+)$", wired)
        return f"{{m.group(1)}}_{{int(m.group(2)) + flow}}" if m else None
    return _derive_audio_name(cfg.get("path") or "", flow)

def _open_audio_state(flux_idx, cfg, flow=0):
    """Ouvre le FLUX MXL audio de la fenêtre (bobimxl.AudioReader). Renvoie state ou None."""
    nm = _audio_name_for(cfg, flow)
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
        try: _gc_mxl()
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
        try: _gc_mxl()
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
        try: _gc_mxl()
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
    # `np.where` évalue les DEUX branches : sur un canal muet, log10(0) est calculé puis jeté, en
    # émettant un RuntimeWarning à chaque relevé. Le résultat était juste, mais le journal se
    # remplissait — et un journal bruyant est exactement ce qui masque les vraies pannes.
    with np.errstate(divide="ignore"):
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

def _meter_grad(comp_or_blk, scale, rw):
    """Niveau de graduations EFFECTIF d'un VU-mètre et largeur de colonne associée.
    Renvoie (niveau, tick_w) avec niveau ∈ "full" (traits + chiffres) | "marks" (traits seuls) |
    "none" (barres nues).

    Le réglage `graduations` vaut "auto" (défaut) ou l'un des trois niveaux, qui s'impose alors.
    En AUTO, on décide sur la PLACE RÉELLEMENT ALLOUÉE au mètre (`rw`, largeur du composant) : la
    colonne ne doit pas dominer, seuil 40 %. C'est la place donnée par le concepteur du modèle qui
    compte, pas la taille intrinsèque du mètre — sinon un mur à 2 canaux dégraderait toujours
    (22/(22+11) = 67 %) même dans une cellule large.
    Mesuré sur le mur 379 : zone de 125 px → 18 %, on garde les chiffres ; petit PiP à ~40 px →
    55 %, on passe aux traits seuls ; sous ~15 px il ne reste que les barres."""
    mode = str((comp_or_blk or {{}}).get("graduations") or "auto").strip().lower()
    tw_full = METER_TICK_W_PPM if scale == "ppm" else METER_TICK_W_DBFS
    if mode == "full":
        return "full", tw_full
    if mode == "marks":
        return "marks", METER_TICK_W_MARKS
    if mode == "none":
        return "none", 0
    if rw > 0 and tw_full <= rw * 0.40:
        return "full", tw_full
    if rw > 0 and METER_TICK_W_MARKS <= rw * 0.40:
        return "marks", METER_TICK_W_MARKS
    return "none", 0


def _meter_inset(rx, rw, cell_x, cell_w):
    """Rentre le mètre de METER_EDGE_MARGIN là où son rectangle TOUCHE le bord de la cellule.
    Uniquement sur le ou les côtés concernés : un mètre déjà écarté du bord n'est pas déplacé, et
    on ne mange jamais plus du tiers de la largeur allouée."""
    cap = max(0, rw // 3)
    m = min(METER_EDGE_MARGIN, cap)
    if m <= 0:
        return rx, rw
    if rx <= cell_x:
        rx += m; rw -= m
    if rx + rw >= cell_x + cell_w:
        rw -= m
    return rx, max(2, rw)


def _meter_layout(n_channels, tick_w=METER_TICK_W, bar_w=METER_BAR_W, gap=METER_GAP):
    """Renvoie la largeur totale d'un meter à N canaux pour les dimensions données. Sans bordure."""
    return tick_w + n_channels * bar_w + (n_channels - 1) * gap

def _meter_fit_dims(n_channels, rw, tick_w=None):
    """Dimensions (tick_w, bar_w, gap, mw_effectif) d'un meter mode `fit` : SEULES les barres de
    canaux s'élargissent pour occuper `rw` — la zone de graduations (tick_w) et l'espacement
    inter-canaux (gap) restent FIXES (METER_TICK_W/METER_GAP), sinon l'échelle dBFS/PPM et ses
    repères se déforment. `mw_effectif` ≤ rw (arrondi entier des barres) : c'est la largeur
    réellement dessinée, à utiliser comme `mw` par l'appelant (jamais `rw` tel quel).
    Repli : si `rw` est trop étroit pour tenir graduations + barres au minimum (≥ 2 px), retombe
    proprement sur la largeur INTRINSÈQUE (mode `auto`, jamais de barre à 0 px)."""
    n = max(1, n_channels)
    MIN_BAR = 2
    tick_w = METER_TICK_W if tick_w is None else tick_w
    gap = METER_GAP
    avail_bars = rw - tick_w - (n - 1) * gap
    bar_w = avail_bars // n if n > 0 else 0
    if bar_w < MIN_BAR:
        bar_w = METER_BAR_W   # repli intrinsèque : mêmes dims que le mode auto
        mw = _meter_layout(n, tick_w, bar_w, gap)
    else:
        mw = _meter_layout(n, tick_w, bar_w, gap)   # ≤ rw (reste éventuel non dessiné, à droite)
    return tick_w, bar_w, gap, mw

def _draw_meter(img, mx, my, mw, mh, n_channels, peaks_db, holds_db, scale, opacity_pct, ch0=0,
                 tick_w=METER_TICK_W, bar_w=METER_BAR_W, gap=METER_GAP,
                 grad="full", grad_side="left"):
    """Dessine un peak meter sur l'image RGBA. opacity_pct 10..100.
    Réserve 12 px en bas pour afficher le numéro de canal sous chaque barre.
    `ch0` : décalage d'étiquetage (composants meters à affectation de canaux : la barre k
    affiche le n° réel ch0+k+1 — le rendu des barres est inchangé).
    `tick_w`/`bar_w`/`gap` : dimensions effectives (mode `fit` : mises à l'échelle de `rw` via
    `_meter_fit_dims` ; par défaut = constantes historiques, mode `auto`)."""
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
    # CÔTÉ des graduations : à gauche (défaut) la colonne occupe [mx, mx+tick_w) et le trait ses
    # 4 derniers pixels, collé aux barres ; à droite tout est en miroir — la colonne est en fin de
    # mètre et le trait à son bord GAUCHE, de nouveau côté barres. Un mètre posé au bord droit
    # d'un PiP a ainsi son échelle tournée vers l'image, pas vers la fenêtre voisine.
    _right = (grad_side == "right")
    _grad_x0 = (mx + mw - tick_w) if _right else mx        # origine de la colonne
    _tick_x0 = _grad_x0 if _right else (_grad_x0 + tick_w - 4)
    # Mode superposé : AUCUNE colonne latérale ici, les graduations sont une tuile à part
    # blendée après les barres (cf. _meter_grad_tile). Sans ce garde-fou, tick_w = 0 ferait
    # écrire le trait à mx-4, hors du mètre.
    _in_bars = (grad_side == "inside")
    for tick_dbfs, lbl in (ticks_dbfs if (grad != "none" and not _in_bars) else ()):
        f = to_frac(tick_dbfs)
        y_tick = my + bars_mh - int(round(f * bars_mh))
        # Petite ligne 3px du côté des barres (donc dans la zone des barres, devant)
        d.line([_tick_x0, y_tick, _tick_x0 + 3, y_tick],
               fill=(180, 180, 180, a_text))
        if grad != "full":
            continue
        # Label seulement si suffisamment d'espace vertical avec le précédent
        if abs(y_tick - last_label_y) >= 9 and y_tick - 4 >= my and y_tick + 4 <= bars_bottom:
            # Libellé ALIGNÉ À DROITE, terminé 2 px avant le trait de repère. L'ancrage à gauche
            # (`mx + 1`) faisait dépendre la fin du libellé de sa LONGUEUR : "0" tenait, "-18"
            # finissait dans le trait, "+12" débordait sur les barres. Aligné à droite, la marge
            # au trait est constante quel que soit le texte, et les chiffres s'alignent entre eux.
            # Bornage à `mx + 1` : un libellé plus large que la colonne reste dans le meter.
            _lf = ImageFont.load_default()
            try:
                _lw = d.textlength(lbl, font=_lf)
            except Exception:
                _lw = 0
            # Le libellé fuit toujours le trait : aligné à droite quand la colonne est à gauche,
            # à gauche quand elle est à droite. La marge au trait reste constante dans les deux
            # sens, quelle que soit la longueur du texte.
            if _right:
                _lx = min(_grad_x0 + tick_w - 1 - int(round(_lw)), _tick_x0 + 6)
                _lx = max(_grad_x0 + 5, _lx)
            else:
                _lx = max(_grad_x0 + 1, _grad_x0 + tick_w - 6 - int(round(_lw)))
            d.text((_lx, y_tick - 4), lbl, font=_lf, fill=(220, 220, 220, a_text))
            last_label_y = y_tick
    # Barres + numéro de canal en bas
    # Barres : après la colonne quand elle est à gauche, dès le bord du mètre quand elle est à
    # droite (ou absente — tick_w vaut alors 0 et les deux expressions coïncident).
    bar_x0 = mx if _right else (mx + tick_w)
    green_top_px  = int(round(green_top  * bars_mh))
    yellow_top_px = int(round(yellow_top * bars_mh))
    for ch in range(n_channels):
        bx = bar_x0 + ch * (bar_w + gap)
        peak_h = int(round(to_frac(peaks_db[ch]) * bars_mh))
        hold_h = int(round(to_frac(holds_db[ch]) * bars_mh))
        # Green zone
        gh = min(peak_h, green_top_px)
        if gh > 0:
            d.rectangle([bx, bars_bottom - gh, bx + bar_w - 1, bars_bottom],
                        fill=(60, 200, 60, a_bar))
        # Yellow zone
        if peak_h > green_top_px:
            yh = min(peak_h, yellow_top_px) - green_top_px
            if yh > 0:
                d.rectangle([bx, bars_bottom - green_top_px - yh, bx + bar_w - 1, bars_bottom - green_top_px],
                            fill=(220, 180, 40, a_bar))
        # Red zone
        if peak_h > yellow_top_px:
            rh = peak_h - yellow_top_px
            if rh > 0:
                d.rectangle([bx, bars_bottom - yellow_top_px - rh, bx + bar_w - 1, bars_bottom - yellow_top_px],
                            fill=(230, 60, 60, a_bar))
        # Peak hold (ligne fine)
        if hold_h > 0:
            yh = bars_bottom - hold_h
            d.line([bx, yh, bx + bar_w - 1, yh], fill=(255, 255, 255, a_hold), width=1)
        # Numéro de canal sous la barre (centré sur la barre, ch0+ch+1 = 1-indexé réel)
        ch_label = str(ch0 + ch + 1)
        # ImageFont.load_default() est très petit, label sur 1 caractère → ~5px wide
        lx = bx + (bar_w // 2) - 2
        ly = bars_bottom + 2
        d.text((lx, ly), ch_label,
               font=ImageFont.load_default(), fill=(220, 220, 220, a_text))

# ─── Peak meters : chemin GPU (cupy) ─────────────────────────────────────────
# Graduations/fond/labels/n° de canal = STATIQUES → rendus PIL UNE fois, ★ convertis en YUV et
# cachés SOUS CETTE FORME ★. Barres + peak-hold = DYNAMIQUES → peints par trame DIRECTEMENT dans
# les plans YUV (couleur unie ⇒ YUV constant). Plus aucun PIL NI aucune conversion RGBA→YUV par
# trame (0.40.0 : c'était 86 % du coût des VU au banc — la tuile entière était re-convertie alors
# que seules quelques barres de couleur unie changeaient).
_meter_static_yuv_cache = {{}}   # key -> (Y, U, V, alpha, alpha_sub) HÔTE du fond STATIQUE (sans barres)
_MET_GREEN = (60, 200, 60); _MET_YELLOW = (220, 180, 40); _MET_RED = (230, 60, 60); _MET_WHITE = (255, 255, 255)
_MET_YUV_CACHE = {{}}   # (r,g,b) -> (Y, U, V) natifs — constantes de couleur des barres

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

def _draw_meter_static(img, mx, my, mw, mh, n_channels, scale, opacity_pct, ch0=0,
                        tick_w=METER_TICK_W, bar_w=METER_BAR_W, gap=METER_GAP,
                        grad="full", grad_side="left"):
    """Partie STATIQUE du meter (fond + graduations + labels dB + n° de canal), SANS les barres —
    identique à _draw_meter hors boucle barres/hold. Rendu une seule fois (caché).
    `tick_w`/`bar_w`/`gap` : voir _draw_meter (mode `fit` vs dimensions historiques).
    ⚠ C'est LE chemin par défaut (le cache YUV) : toute correction de graduations faite dans
    _draw_meter doit être faite ICI AUSSI, sinon elle ne touche que le repli `meters_pil`."""
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
    _right = (grad_side == "right")
    _grad_x0 = (mx + mw - tick_w) if _right else mx
    _tick_x0 = _grad_x0 if _right else (_grad_x0 + tick_w - 4)
    _lf = ImageFont.load_default()
    _in_bars = (grad_side == "inside")   # cf. _draw_meter : colonne latérale supprimée
    for tick_dbfs, lbl in (ticks_dbfs if (grad != "none" and not _in_bars) else ()):
        y_tick = my + bars_mh - int(round(to_frac(tick_dbfs) * bars_mh))
        d.line([_tick_x0, y_tick, _tick_x0 + 3, y_tick], fill=(180, 180, 180, a_text))
        if grad != "full":
            continue
        if abs(y_tick - last_label_y) >= 9 and y_tick - 4 >= my and y_tick + 4 <= bars_bottom:
            try:
                _lw = d.textlength(lbl, font=_lf)
            except Exception:
                _lw = 0
            if _right:
                _lx = max(_grad_x0 + 5, min(_grad_x0 + tick_w - 1 - int(round(_lw)), _tick_x0 + 6))
            else:
                _lx = max(_grad_x0 + 1, _grad_x0 + tick_w - 6 - int(round(_lw)))
            d.text((_lx, y_tick - 4), lbl, font=_lf, fill=(220, 220, 220, a_text))
            last_label_y = y_tick
    for ch in range(n_channels):
        bx = (mx if _right else mx + tick_w) + ch * (bar_w + gap)
        d.text((bx + (bar_w // 2) - 2, bars_bottom + 2), str(ch0 + ch + 1),
               font=ImageFont.load_default(), fill=(220, 220, 220, a_text))

def _meter_static_yuv(W, H, rmx, rmy, mw, mh, n, scale, opacity_pct, ch0=0,
                       tick_w=METER_TICK_W, bar_w=METER_BAR_W, gap=METER_GAP,
                       grad="full", grad_side="left"):
    """Fond STATIQUE du meter (fond + graduations + labels), rendu PIL UNE fois puis caché
    ★ DÉJÀ CONVERTI EN YUV ★ (0.40.0). C'était LE point chaud des VU (banc : 86 % de leur coût) :
    l'ancien chemin gardait le statique en RGBA et re-convertissait TOUTE la tuile en YUV à CHAQUE
    trame, alors que seules les barres changent — et qu'elles sont d'une COULEUR UNIE, donc de YUV
    constant. La conversion par trame était intégralement redondante.
    La clé de cache inclut tick_w/bar_w/gap : un meter `fit` (largeur effective propre à sa cellule)
    ne doit JAMAIS réutiliser le fond caché d'un meter `auto` ou `fit` d'une autre largeur."""
    # grad/grad_side DANS LA CLÉ : ils changent les pixels du fond (colonne présente ou non,
    # côté, chiffres ou non). Les omettre ferait resservir le fond d'un autre réglage.
    key = (W, H, rmx, rmy, mw, mh, n, scale, opacity_pct, ch0, tick_w, bar_w, gap, grad, grad_side)
    p = _meter_static_yuv_cache.get(key)
    if p is None:
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        _draw_meter_static(img, rmx, rmy, mw, mh, n, scale, opacity_pct, ch0, tick_w, bar_w, gap,
                           grad, grad_side)
        p = rgba_to_yuv(img)      # (Y, U, V, alpha, alpha_sub) — hôte, kernel C si dispo
        _meter_static_yuv_cache[key] = p
    return p

def _met_yuv(rgb):
    """(Y, U, V) natifs d'une couleur RGB UNIE, obtenus par le MÊME convertisseur que le reste du
    mur (patch 2×2 → rgba_to_yuv) : la valeur d'une barre est donc BIT-EXACTE avec l'ancien chemin.
    Mémoïsé (4 couleurs)."""
    c = _MET_YUV_CACHE.get(rgb)
    if c is None:
        patch = np.empty((2, 2, 4), dtype=np.uint8)
        patch[..., 0] = rgb[0]; patch[..., 1] = rgb[1]; patch[..., 2] = rgb[2]; patch[..., 3] = 255
        _y, _u, _v, _a, _am = rgba_to_yuv(Image.fromarray(patch, "RGBA"))
        c = (int(_y[0, 0]), int(_u[0, 0]), int(_v[0, 0]))
        _MET_YUV_CACHE[rgb] = c
    return c

def _meter_paint_rect(y, u, v, a8, x0, y0, x1, y1, ycc, a):
    """PEINT une couleur unie DIRECTEMENT dans les plans YUV+alpha de la tuile (bornée).
    ⚠ ÉCART ASSUMÉ (validé produit, 0.40.0) : la LUMA et l'ALPHA sont posées à PLEINE RÉSOLUTION —
    donc strictement identiques à l'ancien chemin (peinture RGBA puis conversion). Le CHROMA, lui,
    est sous-échantillonné : un échantillon chroma coupé en deux par le bord MOBILE d'une barre
    prenait la MOYENNE (fond/barre) et prend désormais la couleur FRANCHE de celui qui le couvre
    majoritairement (arrondi au plus proche). L'écart maximal est donc un liseré chroma d'UN
    échantillon (2 px) sur le bord d'une barre ; la couleur, la hauteur, la position et l'opacité
    de la barre restent inchangées AU BIT PRÈS (la luma, qui porte le contour, est exacte)."""
    Ht = y.shape[0]; Wt = y.shape[1]
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(Wt, x1); y1 = min(Ht, y1)
    if x1 <= x0 or y1 <= y0:
        return
    y[y0:y1, x0:x1] = ycc[0]
    a8[y0:y1, x0:x1] = a
    cx0 = (x0 + _CW // 2) // _CW; cx1 = (x1 + _CW // 2) // _CW
    cy0 = (y0 + _CH // 2) // _CH; cy1 = (y1 + _CH // 2) // _CH
    if cx1 > cx0 and cy1 > cy0:
        u[cy0:cy1, cx0:cx1] = ycc[1]
        v[cy0:cy1, cx0:cx1] = ycc[2]

def _meter_tile_yuv(W, H, rmx, rmy, mw, mh, n, peaks_db, holds_db, scale, opacity_pct, ch0=0,
                     tick_w=METER_TICK_W, bar_w=METER_BAR_W, gap=METER_GAP,
                     grad="full", grad_side="left"):
    """Tuile YUV d'un meter : copie du fond statique CACHÉ EN YUV + barres/hold peints DIRECTEMENT
    en YUV. Plus AUCUNE conversion RGBA→YUV par trame (banc : 2,3 → 0,4 ms sur 4 fenêtres).
    Tuile HÔTE (numpy) : elle est minuscule (une bande de VU), et `_to_xp` l'uploade au blend —
    ce qui, sur un mur GPU, supprime aussi la nuée de micro-noyaux cupy que coûtait la peinture
    RGBA par tranche. `tick_w`/`bar_w`/`gap` : voir _draw_meter."""
    sy, su, sv, sa, _sam = _meter_static_yuv(W, H, rmx, rmy, mw, mh, n, scale, opacity_pct,
                                             ch0, tick_w, bar_w, gap, grad, grad_side)
    y = sy.copy(); u = su.copy(); v = sv.copy(); a8 = sa.copy()
    a_bar = int(220 * opacity_pct / 100); a_hold = int(255 * opacity_pct / 100)
    bars_mh = max(20, mh - 12); bars_bottom = rmy + bars_mh
    to_frac, green_top, yellow_top = _meter_scale_params(scale)
    green_top_px = int(round(green_top * bars_mh)); yellow_top_px = int(round(yellow_top * bars_mh))
    _g = _met_yuv(_MET_GREEN); _yl = _met_yuv(_MET_YELLOW)
    _r = _met_yuv(_MET_RED);   _w = _met_yuv(_MET_WHITE)
    for ch in range(n):
        # Colonne à droite → les barres commencent au bord du mètre (cf. _draw_meter.bar_x0).
        bx = (rmx if grad_side == "right" else rmx + tick_w) + ch * (bar_w + gap)
        peak_h = int(round(to_frac(peaks_db[ch]) * bars_mh))
        hold_h = int(round(to_frac(holds_db[ch]) * bars_mh))
        # NB : PIL d.rectangle est INCLUSIF sur (x1,y1) → bas +1 ici pour égaler les hauteurs.
        gh = min(peak_h, green_top_px)
        if gh > 0:
            _meter_paint_rect(y, u, v, a8, bx, bars_bottom - gh, bx + bar_w, bars_bottom + 1, _g, a_bar)
        if peak_h > green_top_px:
            yh = min(peak_h, yellow_top_px) - green_top_px
            if yh > 0:
                _meter_paint_rect(y, u, v, a8, bx, bars_bottom - green_top_px - yh, bx + bar_w, bars_bottom - green_top_px + 1, _yl, a_bar)
        if peak_h > yellow_top_px:
            rh = peak_h - yellow_top_px
            if rh > 0:
                _meter_paint_rect(y, u, v, a8, bx, bars_bottom - yellow_top_px - rh, bx + bar_w, bars_bottom - yellow_top_px + 1, _r, a_bar)
        if hold_h > 0:
            yh = bars_bottom - hold_h
            _meter_paint_rect(y, u, v, a8, bx, yh, bx + bar_w, yh + 1, _w, a_hold)
    # alpha sous-échantillonnée : RECALCULÉE (max du bloc) depuis l'alpha pleine résolution qu'on
    # vient de peindre → EXACTE, et contiguë (contrat mvk, cf. 0.39.3).
    am = a8
    if _CW == 2: am = np.maximum(am[:, 0::2], am[:, 1::2])
    if _CH == 2: am = np.maximum(am[0::2, :], am[1::2, :])
    return y, u, v, a8, np.ascontiguousarray(am)

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
# ─── CADENCE : compteurs HONNÊTES (0.42.0) ───────────────────────────────────────────────────
# ★ Pourquoi ce bloc existe. Jusqu'à la 0.41.x, `fps` était calculé TOUS LES 25 TRAMES :
# `25 / (t_maintenant − t_du_25e_précédent)`. Le NOMBRE de trames de la fenêtre est donc FIXE et
# c'est sa DURÉE qui varie → un seul tick en retard (20-30 ms : GC, hoquet de grille, pic hôte)
# gonfle la durée d'UNE fenêtre de 0,5 s de 5 % et fait « chuter » fps à 47-48, pendant que la
# fenêtre SUIVANTE, qui rattrape, remonte à 52 — le mur, lui, n'a rien perdu. Le tableau de bord
# n'échantillonne qu'une fenêtre toutes les 5 s : il tombe sur une fenêtre creuse ~1 fois sur 4.
# Mesuré sur le mur 333 (60 relevés) : fps min 47,9 / méd 50,0 / max 50,5, moyenne 49,78 — quand
# `bakes_per_s.frames` (fenêtre ≥ 1 s, MÊME compteur) donnait min 48,9 / méd 49,9, moyenne 49,73.
# **Les deux comptaient la même chose** (les trames composées) : l'écart était 100 % un artefact de
# FENÊTRE. Une métrique qui oscille de ±5 % sans qu'aucune trame ne soit perdue fait perdre une
# heure à chaque incident — donc :
#   • la cadence est mesurée sur une fenêtre de TEMPS (2 s) à partir de compteurs MONOTONES ;
#   • elle est calculée AU MOMENT DU SCRAPE (bord droit = maintenant) → si la boucle de compo
#     meurt, le chiffre DÉCROÎT vers 0 au lieu de rester figé sur sa dernière belle valeur ;
#   • le vrai signal de santé (« ai-je perdu des trames ? ») n'est plus déduit du fps mais COMPTÉ :
#     `frames_missed` = slots de grille genlock réellement sautés.
# ★ ENTRELACÉ (0.41.0) : un mur 1080i50 COMPOSE 25 trames/s et ÉMET 50 champs/s. `fps` expose la
# cadence de SORTIE (grains/s : champs si entrelacé) — c'est la cadence du format déclaré
# (« HD 1080i50 » → 50), donc un mur i50 sain affiche 50 et ne paraît pas à moitié mort ;
# `frames_per_s` expose les trames composées (25). Les deux sont publiés, nommés, et `fps_unit`
# dit lequel est lequel.
# ★ 0.64.0 — POURQUOI LE MUR N'AFFICHAIT JAMAIS 50. La 0.42.0 a posé « bord droit = maintenant »
# pour qu'une boucle morte fasse tomber le chiffre à 0. Mais le compteur lu à cet instant RETARDE
# sur « maintenant » d'une fraction de période (le scrape tombe ENTRE deux commits), alors que le
# bord GAUCHE est un échantillon pris JUSTE APRÈS un commit — donc sans retard. Les deux retards ne
# s'annulent pas : il manque jusqu'à une trame au numérateur pour un dénominateur complet, soit un
# biais SYSTÉMATIQUEMENT bas d'une période sur la fenêtre (20 ms / 1,8 s ≈ 1 %). Un mur exactement
# à 50 affichait 49,3-49,9 et JAMAIS 50 : mesuré côté bus, 50,0000 grain/s et zéro intervalle
# déficitaire sur 9 000 échantillons pendant que le plugin publiait 49,7. Un indicateur qui
# n'atteint jamais sa valeur nominale se lit « on n'y arrive pas » et coûte une enquête à chaque
# fois — le même piège que la fenêtre à nombre de trames fixe, par l'autre bout.
# Correctif : quand la boucle SUIT, les DEUX bords sont des échantillons alignés sur trame → le
# rapport est exact (50,0 franc). Quand elle ne suit plus (aucun échantillon récent), on reprend
# « maintenant » comme bord droit → le chiffre décroît vers 0. La propriété de la 0.42.0 est donc
# conservée là où elle sert VRAIMENT (détecter l'arrêt) et retirée là où elle ne faisait que
# mentir de 1 % (le régime nominal).
FPS_WINDOW_S = 2.0
_RATE_SAMPLE_EVERY = 25          # cf. boucle de compo : `if out_frame_index % 25 == 0: _rate_sample()`
_GRAINS_PER_FRAME = 2 if INTERLACED else 1
# Au-delà de ce retard sans échantillon, la boucle ne suit plus → bord droit = maintenant (le
# chiffre décroît). Dérivé de la cadence réelle (2,5 intervalles d'échantillonnage), jamais figé :
# 50p → 1,25 s ; 1080i50 (25 trames composées/s) → 2,5 s.
RATE_STALE_S = max(0.6, 2.5 * _RATE_SAMPLE_EVERY * FRAME_INTERVAL)
_rate_lock  = threading.Lock()
# `neuves` = trames composées à partir d'au moins UNE entrée qui a AVANCÉ. Distinguer les deux
# est indispensable : un mur peut composer 50 fois par seconde à partir de tuiles inchangées —
# il publie alors 50 fps en toute honnêteté (il compose bien 50 fois), pendant que son émetteur
# aval, qui n'émet que sur changement, tombe à 38. Mesuré le 2026-08-07 : mur à 50,1 fps et
# `frames_missed_per_s` à 0, shards à 24-26 fps ratant la moitié de leurs créneaux, TX à 38.
# La page Câbles montrait 50 et tout le monde concluait que la chaîne allait bien.
# ⚠ MINORANT du changement VISIBLE : une horloge incrustée fait bouger l'image sans qu'aucune
# entrée n'avance. Ce compteur mesure « est-ce que je relaie de la matière NEUVE », pas « est-ce
# que les pixels ont changé » — c'est la question qui compte pour diagnostiquer une chaîne.
_rate_total = {{"frames": 0, "missed": 0, "neuves": 0}}   # compteurs MONOTONES (boucle de compo)
_rate_hist  = deque(maxlen=64)                 # (t_monotone, frames, missed) échantillonnés ~2×/s

def _rate_sample():
    """Échantillon de cadence (appelé par la boucle de compo, ~toutes les 25 trames)."""
    with _rate_lock:
        _rate_hist.append((time.monotonic(), _rate_total["frames"], _rate_total["missed"],
                           _rate_total["neuves"]))

def _update_rate_metrics():
    """Publie fps / frames_per_s / frames_missed_per_s.

    Les DEUX bords sont des échantillons alignés sur trame tant que la boucle suit → le rapport est
    exact (un mur nominal affiche 50,0, pas 49,x ; cf. bloc 0.64.0 ci-dessus). Si plus aucun
    échantillon n'est récent (`RATE_STALE_S`), le bord droit redevient « maintenant » → le chiffre
    décroît vers 0 et une boucle morte reste visible."""
    now = time.monotonic()
    with _rate_lock:
        hist = list(_rate_hist)
        cur_f = _rate_total["frames"]; cur_m = _rate_total["missed"]
        cur_n = _rate_total["neuves"]
    if not hist:
        return
    # Bord DROIT : le dernier échantillon (aligné sur trame, donc sans retard) si la boucle suit ;
    # sinon l'instant présent avec les compteurs courants → la valeur chute au lieu de se figer.
    t_dr, f_dr, m_dr, n_dr = hist[-1]
    if now - t_dr > RATE_STALE_S:
        t_dr, f_dr, m_dr, n_dr = now, cur_f, cur_m, cur_n
    old = None
    for s in hist:
        if t_dr - s[0] <= FPS_WINDOW_S:
            old = s
            break
    if old is None:
        old = hist[-1]      # boucle figée : la fenêtre ne contient plus rien → delta 0 sur un long dt
    dt = t_dr - old[0]
    if dt < 0.3:
        return              # trop tôt (démarrage) ou un seul échantillon : on garde la valeur précédente
    fps_frames = (f_dr - old[1]) / dt
    metrics["frames_per_s"]        = round(fps_frames, 1)
    metrics["fps"]                 = round(fps_frames * _GRAINS_PER_FRAME, 1)
    metrics["frames_missed_per_s"] = round((m_dr - old[2]) / dt, 2)   # mêmes bords que fps
    metrics["frames_missed"]       = cur_m                            # total : on sert le plus frais
    # Cadence de CONTENU NEUF (mêmes bords que fps) : à quelle vitesse ce nœud relaie de la
    # matière nouvelle, par opposition à la vitesse à laquelle il compose. Un écart franc entre
    # les deux dit que les ENTRÉES ne suivent pas — c'est le maillon faible de la chaîne.
    metrics["fps_content"]         = round((n_dr - old[3]) / dt * _GRAINS_PER_FRAME, 1)

metrics = {{"fps": 0.0, "inputs_latency_ms": {{}}, "own_latency_ms": None,
           # VERSION RÉELLEMENT EN COURS D'EXÉCUTION. `deploy_config.plugin_version` n'est qu'une
           # écriture en base : rien ne garantissait qu'elle décrive le script qui tourne. Un mur a
           # tourné une nuit entière sur un script estampillé 0.67.0 mais sans le correctif qu'il
           # portait, et il a fallu un `docker exec … grep` pour s'en apercevoir — la garde de
           # version, elle, le croyait à jour et refusait de le redéployer. Publiée ici, la version
           # devient VÉRIFIABLE : l'orchestrateur compare ce qui tourne au manifeste au lieu de se
           # fier à sa propre comptabilité (cf. `version_en_cours` dans app/deploy.py).
           "plugin_version": PLUGIN_VERSION,
           "fps_nominal": round(_FN / _FD * _GRAINS_PER_FRAME, 2),   # cadence de sortie visée (= format déclaré)
           "fps_unit": "fields" if INTERLACED else "frames",         # unité de `fps` (champs si entrelacé)
           "frames_per_s": 0.0,        # trames COMPOSÉES/s (= fps ÷ 2 en entrelacé)
           "frames_missed": 0,         # cumul des slots de grille genlock SAUTÉS (vraies trames perdues)
           "frames_missed_per_s": 0.0,
           "mvk": _MVK,                          # kernel compose fusionné C actif (chemin CPU)
           "mvk_threads": (getattr(bobimxl, "mvk_threads", lambda: 0)() if _MVK else 0),
           "gpu": GPU, "gpu_name": _GPU_NAME,    # GPU = compositing accéléré cupy (sinon numpy CPU)
           "gpu_slice": GPU_SLICE,               # tranche VRAM active (opt-in gpu_slice, banc gate)
           "gpu_batch_bands": GPU_BATCH_BANDS if GPU_SLICE else None}}   # bandes/lot GPU (micro-batch)

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
    # shm proxy réellement lus (orphan-detect) : boucle de mix (fenêtres) ∪ échantillonneur des
    # frises VIDÉO (une frise dont la source n'alimente AUCUNE fenêtre lit quand même un proxy —
    # sans ça l'orchestrateur le croirait ORPHELIN et le supprimerait sous nos pieds).
    metrics["proxy_read"]  = sorted(set(_proxy_read_latest) | _vh_read_proxies())
    # Filtre de réduction RÉELLEMENT appliqué aux tuiles. Sans ça, un mur déployé avant le
    # réglage et un mur filtré sont indiscernables de l'extérieur — et le surcoût de `inputs`
    # serait inexplicable. `scale_filter_nearest_paths` compte les chemins fusionnés (kernel C)
    # écartés parce qu'ils ne savent faire que du nearest : c'est le prix payé, rendu visible.
    metrics["scale_filter"] = SCALE_FILTER
    metrics["scale_filter_nearest_paths"] = _mvk_miss["place"]
    metrics["interlace_mode"] = INTERLACE_MODE
    # Fuseau du mur + fuseaux d'horloge REFUSÉS par la base tzdata de l'image. Une horloge dont le
    # fuseau est inconnu retombe sur celui du mur : sans ce compteur elle afficherait une heure
    # fausse en silence, et rien à l'écran ne le dirait.
    metrics["tz"] = _TZ_NAME
    # Anatomie du blend d'habillage : combien de TUILES per-frame, donc combien de lancements de
    # kernel (×3 plans sur GPU), et combien de pixels ils couvrent réellement. `us_par_tuile` est
    # le chiffre qui tranche : un coût par tuile très supérieur au travail utile = borné par les
    # lancements (remède : grouper), un coût qui suit la surface = borné par les pixels.
    _nt = _n_meters.avg() + _n_hist.avg() + _n_clock.avg()
    metrics["ov_tiles"] = {{"meters": round(_n_meters.avg(), 1), "hist": round(_n_hist.avg(), 1),
                           "clock_anc": round(_n_clock.avg(), 1), "total": round(_nt, 1),
                           "lancements_gpu": round(_nt * 3, 1) if GPU else 0,
                           "kpixels": round(_n_px_blend.avg() / 1000.0, 1),
                           "us_par_tuile": round(_t_ov_blend.avg() * 1000.0 / _nt, 1) if _nt else None,
                           "upload_ms": round(_t_ov_upload.avg(), 2),
                           "sig_ms": round(_t_pf_sig.avg(), 2),
                           "kernels_ms": round(max(0.0, _t_ov_blend.avg() - _t_ov_upload.avg()), 2)}}
    # PICS par sous-étage sur la fenêtre glissante — c'est eux qui font tomber les trames.
    metrics["compose_peak_ms"] = {{"own": own_lat.mx(), "in_read": _t_in_read.mx(),
                                  "in_place": _t_in_place.mx(), "overlays": _t_overlays.mx(),
                                  "ov_render": _t_ov_render.mx(), "ov_convert": _t_ov_convert.mx(),
                                  "ov_blend": _t_ov_blend.mx(), "ov_meters": _t_ov_meters.mx(),
                                  "ov_hist": _t_ov_hist.mx(), "ov_clock": _t_ov_clock.mx(),
                                  "ov_bake": _t_ov_bake.mx(), "output": _t_output.mx()}}
    # Piles capturées PENDANT un dépassement, les plus fréquentes d'abord. C'est la réponse à
    # « où part le temps », observée et non déduite.
    metrics["piles_pic"] = dict(sorted(_piles.items(), key=lambda kv: -kv[1])[:8])
    metrics["tuile_pin_miss"] = dict(_tuile_pin_miss)   # replis du transfert épinglé — doit rester 0
    metrics["place_miss"] = _place_miss["n"]   # replis du gather fusionné — doit rester à 0
    metrics["hist_prof"] = {{k: dict(v) for k, v in _hist_prof.items()}}
    metrics["cpu_split"] = ({{"compo": [_CPUS[0]], "boulangers": _CPUS[1:]}}
                            if _CPU_SPLIT else None)
    metrics["anc_bake"] = dict(_anc_prof)
    metrics["clock_tz_unknown"] = dict(_ZONE_BAD)
    # Profiling du compositing : ventilation de own_latency (entrées / habillage / sortie).
    metrics["compose_breakdown_ms"] = {{"inputs": _t_inputs.avg(), "overlays": _t_overlays.avg(),
                                       "output": _t_output.avg(),
                                       "in_read": _t_in_read.avg(), "in_place": _t_in_place.avg(),
                                       "ov_render": _t_ov_render.avg(), "ov_convert": _t_ov_convert.avg(),
                                       "ov_blend": _t_ov_blend.avg(),
                                       "ov_bake": _t_ov_bake.avg(), "ov_bg": _t_ov_bg.avg(),
                                       "ov_meters": _t_ov_meters.avg(), "ov_hist": _t_ov_hist.avg(),
                                       "ov_clock": _t_ov_clock.avg()}}
    # Fréquence des RE-BAKES d'habillage (par seconde, fenêtre glissante depuis le dernier appel) :
    # un chrome re-baké en rafale (churn TSL / statuts) est LE piège perf du multiview — il coûte
    # plein cadre (PIL + rgba_to_yuv + upload) et se noyait dans la moyenne `overlays`.
    _now_r = time.time(); _dt_r = _now_r - _bake_ctr["t0"]
    if _dt_r >= 1.0:
        _bake_ctr["hist"] = _hist_bake_ctr[0]; _hist_bake_ctr[0] = 0
        _bake_rate.clear()
        _bake_rate.update({{k: round(_bake_ctr[k] / _dt_r, 1)
                            for k in ("chrome", "bg", "tally", "geom", "info", "hist", "frames")}})
        for k in ("chrome", "bg", "tally", "geom", "info", "hist", "frames"):
            _bake_ctr[k] = 0
        _bake_ctr["t0"] = _now_r
    metrics["bakes_per_s"] = dict(_bake_rate)
    metrics["mvk_host"] = _MVK_HOST
    metrics["mvk_miss"] = dict(_mvk_miss)
    metrics["mvk_why"] = dict(_mvk_why)
    # `hist_bake_ms` = coût UNITAIRE d'un re-bake de frise. ★ 0.40.0 : il est désormais payé par le
    # THREAD BOULANGER, plus par la boucle de composition → il ne coûte PLUS DE TRAME. `async: true`
    # le dit explicitement (sur une version antérieure, ce même chiffre était un pic DANS la trame).
    # Le coût réellement vu par la trame est `compose_breakdown_ms.ov_hist` — il doit être ~0,05 ms.
    metrics["hist_bake_ms"] = dict(_hist_bake, **{{"async": True}})
    # `chrome_bake_ms` = coût UNITAIRE d'une passe de boulange de l'habillage (fond + chrome).
    # ★ 0.42.0 : payé par le THREAD BOULANGER → il ne coûte PLUS DE TRAME (`async: true`). Ce que la
    # trame paie est `compose_breakdown_ms.ov_bake` (ramassage + upload) — il doit rester ~0,1 ms.
    metrics["chrome_bake_ms"] = dict(_chrome_bake, **{{"async": True}})
# debug TSL : dernier paquet reçu (mis à jour par _handle_tsl_client)
tsl_debug = {{"last_raw_hex": None, "last_ver": None, "last_index": None,
              "last_control": None, "last_text": None, "last_error": None,
              "connections": 0, "slots": {{}}}}

# ─── HTTP : metrics + tally ──────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/tally":
            self._send_json(tally_state)
        elif self.path == "/anc":
            # Inventaire ANC par entrée (DIAGNOSTIC — n'incruste RIEN à l'image) : ce que chaque
            # source transporte réellement, ses erreurs de checksum, son timecode.
            try:
                self._send_json(_anc_report())
            except Exception as e:
                self._send_json({{"error": str(e)}})
        elif self.path == "/tsl_debug":
            self._send_json({{**tsl_debug,
                              "tally_state": dict(tally_state),
                              "tsl_text": {{str(k): v for k, v in tsl_text.items()}}}})
        else:
            # Cadence recalculée AU SCRAPE (bord droit = maintenant) : le chiffre servi est frais,
            # et une boucle de compo morte le fait tomber à 0 au lieu de le figer.
            try:
                _update_rate_metrics()
            except Exception:
                pass
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
                # ★ PERF : comme /tally_bulk, ne marquer sale QUE sur changement réel de valeur
                # (un ré-envoi identique — action/macro rejouée — ne doit pas re-baker l'habillage).
                _ch = False
                if "slot" in data:
                    # Forme par-lampe (service TSL) : une lampe L/R, une couleur.
                    slot = str(data["slot"]).upper()
                    if slot not in ("L", "R"):
                        raise ValueError("slot invalide")
                    _k = f"{{idx}}_{{slot}}"
                    if tally_state.get(_k) != color:
                        tally_state[_k] = color
                        _ch = True
                else:
                    # Forme simple (action shotbox/macro, sans slot) : couleur DOMINANTE
                    # de la fenêtre — red (antenne), green (préparation), off (éteint).
                    _l = color if color in ("red", "amber") else "off"
                    _r = "green" if color in ("green", "amber") else "off"
                    if tally_state.get(f"{{idx}}_L") != _l:
                        tally_state[f"{{idx}}_L"] = _l
                        _ch = True
                    if tally_state.get(f"{{idx}}_R") != _r:
                        tally_state[f"{{idx}}_R"] = _r
                        _ch = True
                if _ch:
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
                    # ★ PERF : ne marquer « changé » QUE si la valeur change RÉELLEMENT. Le
                    # distributeur TSL central pousse un tally_bulk toutes les 100 ms MÊME sans
                    # changement (keepalive) : marquer sale à chaque paquet re-bakait l'habillage
                    # PLEIN CADRE 10×/s (PIL + RGBA→YUV + upload GPU ≈ 25 ms) — chaque re-bake
                    # tombant dans une trame, le mur perdait le budget 20 ms (mesuré : mur 333
                    # Horace, 28-36 fps au lieu de 50, `overlays` 13,7 ms dont ~7 de re-bake).
                    if tally_state.get(f"{{idx}}_{{slot}}") != color:
                        tally_state[f"{{idx}}_{{slot}}"] = color
                        changed = True
                    # Texte label optionnel (label_col depuis l'orchestrateur)
                    text = upd.get("text")
                    if text is not None and slot == "L" and tsl_text.get(idx) != str(text):
                        tsl_text[idx] = str(text)
                        changed = True
                # Overlays texte central : id → texte + état actif (résolu côté orchestrateur)
                ov_changed = False
                for ov in (data.get("overlays") or []):
                    oid = str(ov.get("id") or "")
                    if not oid:
                        continue
                    _ovv = {{"text": str(ov.get("text") or ""),
                             "active": bool(ov.get("active"))}}
                    if overlay_central.get(oid) != _ovv:   # idem : re-bake SEULEMENT sur changement
                        overlay_central[oid] = _ovv
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

def open_source(cfg, own_instance=False):
    """Ouvre une source comme flux MXL (Reader). Le NOM = chemin sans le préfixe /dev/shm/.
    Format LU DU flow_def du producteur (source de vérité) → in_w/in_h/chroma/bit_depth.

    `own_instance=True` ouvre le Reader sur une Instance MXL **DÉDIÉE** au lieu de l'Instance
    globale du process. C'est l'ÉCHAPPATOIRE du décrochage de génération (cf. _drop_input) : MXL
    met les flux en cache PAR INSTANCE et `garbage_collect()` ne réclame PAS un flux encore
    RÉFÉRENCÉ DANS CETTE INSTANCE — s'il reste un reader ouvert sur la génération morte (autre
    tuile sur le même flux, handle non libéré), le ré-open retombe sur l'orphelin, indéfiniment
    (mesuré au banc). Une Instance neuve n'a aucun cache : elle résout le flux sur disque, donc
    sur la génération VIVANTE. (Un lecteur d'un AUTRE process, lui, ne bloque PAS le GC : vérifié
    au banc — close+GC+reopen suffit tant que notre propre handle est bien lâché.)"""
    _own = None
    try:
        name = cfg["path"].removeprefix("/dev/shm/")
        _own = bobimxl.Instance() if own_instance else None
        rd = bobimxl.Reader(_own or inst, name)   # lève si le flux n'existe pas encore
        fmt = rd.format()
        if not fmt:
            rd.close()
            if _own is not None:
                try: _own.close()
                except Exception: pass
            return None
        # ENTRELACÉ NATIF : un grain = 1 CHAMP (½ hauteur ; in_h = hauteur de CHAMP via format()).
        # Le multiview sort TOUJOURS en PROGRESSIF (mur de monitoring) → on « bobe » : un champ scalé
        # à la hauteur de tuile (resize_plane) = désentrelacement par champ, suffisant pour un preview.
        in_w, in_h = fmt["width"], fmt["height"]
        frame_size = (in_w * in_h + 2 * (in_w // _CW) * (in_h // _CH)) * _BPS
        return {{"reader": rd, "in_w": in_w, "in_h": in_h, "frame_size": frame_size,
                 "interlaced": bool(fmt.get("interlaced")), "own_inst": _own}}
    except Exception:
        try:
            if _own is not None: _own.close()
        except Exception: pass
        return None

def _weave_fields(vt, vb, in_w, in_h):
    """ENTRELACÉ : recompose une TRAME PLEINE à partir des deux champs appariés.
    Convention de parité du fichier : index PAIR = lignes PAIRES = champ HAUT, index IMPAIR =
    lignes impaires. L'ordre de champ (TFF/BFF) ne change PAS cette correspondance — il dit
    seulement lequel part en premier sur le fil — donc le tissage se fait sur la PARITÉ D'INDEX
    et reste correct dans les deux ordres.
    `in_h` est la hauteur de CHAMP (ce que rend format() sur un flux entrelacé natif) ; la trame
    rendue fait donc 2·in_h. En 4:2:2 la chroma a la même hauteur que le luma et se tisse pareil ;
    en 4:2:0 le tissage vertical de la chroma est une APPROXIMATION (le siting chroma entrelacé
    n'est pas une simple alternance de lignes) — acceptable sur un mur de monitoring, à ne pas
    reprendre tel quel pour une sortie de diffusion."""
    _yb  = in_w * in_h * _BPS
    _uvb = (in_w // _CW) * (in_h // _CH) * _BPS
    ch, cw = in_h // _CH, in_w // _CW
    y = np.empty((in_h * 2, in_w), dtype=_NP_DT)
    y[0::2] = np.frombuffer(bytes(vt[:_yb]), dtype=_NP_DT).reshape(in_h, in_w)
    y[1::2] = np.frombuffer(bytes(vb[:_yb]), dtype=_NP_DT).reshape(in_h, in_w)
    u = np.empty((ch * 2, cw), dtype=_NP_DT)
    v = np.empty((ch * 2, cw), dtype=_NP_DT)
    u[0::2] = np.frombuffer(bytes(vt[_yb:_yb + _uvb]), dtype=_NP_DT).reshape(ch, cw)
    u[1::2] = np.frombuffer(bytes(vb[_yb:_yb + _uvb]), dtype=_NP_DT).reshape(ch, cw)
    v[0::2] = np.frombuffer(bytes(vt[_yb + _uvb:_yb + 2 * _uvb]), dtype=_NP_DT).reshape(ch, cw)
    v[1::2] = np.frombuffer(bytes(vb[_yb + _uvb:_yb + 2 * _uvb]), dtype=_NP_DT).reshape(ch, cw)
    return y, u, v

def _box_acc_dtype(n):
    """dtype d'accumulation d'une somme de `n` échantillons : uint16 tant qu'elle ne PEUT PAS
    déborder (255·9 = 2295 en 8 bits, 1023·9 = 9207 en 10 bits), uint32 au-delà. uint16 brasse
    2× moins d'octets sur une opération memory-bound — mais un débordement serait SILENCIEUX et
    donnerait des tuiles fausses, donc le seuil est CALCULÉ sur la profondeur réelle, pas supposé."""
    return np.uint16 if n * _MAXV <= 65535 else np.uint32

def _box_reduce(plane, sy, sx):
    """Réduction d'un plan par MOYENNE du bloc sy×sx (filtre d'aire) — ratio ENTIER uniquement.
    Somme par additions successives des sy·sx sous-vues stridées du bloc : mesuré 2,8 ms sur un
    plan Y 1920×1080→640×360, contre 14 ms pour le `.sum(axis=(1, 3))` naïf, qui matérialise un
    intermédiaire 4D non contigu — la formulation compte plus que l'algorithme ici. Arrondi au
    plus proche (+n//2). Écrit dans le backend de `plane` (sous-vues + accumulation) → marche
    identiquement en numpy et en cupy, canvas GPU compris."""
    if sy <= 1 and sx <= 1:
        return plane
    h, w = plane.shape
    th, tw = h // sy, w // sx
    q = plane[:th * sy, :tw * sx].reshape(th, sy, tw, sx)
    n = sy * sx
    acc = q[:, 0, :, 0].astype(_box_acc_dtype(n))
    for i in range(sy):
        for j in range(sx):
            if i or j:
                acc += q[:, i, :, j]
    return ((acc + n // 2) // n).astype(_NP_DT)

def resize_plane(plane, target_h, target_w):
    from_h, from_w = plane.shape
    if target_h <= 0 or target_w <= 0:
        return plane[:1, :1]
    # Downscale à ratio ENTIER (grilles 2×2/3×3/4×4… → tuile = ½, ⅓, ¼ de la source).
    if from_h % target_h == 0 and from_w % target_w == 0:
        sy, sx = from_h // target_h, from_w // target_w
        if _BOX and (sy > 1 or sx > 1):
            return _box_reduce(plane, sy, sx)   # moyenne du bloc = anticrénelage correct
        # nearest : slicing à pas constant `plane[::sy, ::sx]` (une VUE, zéro copie) au lieu du
        # gather np.ix_ (alloue+copie). Octet-identique au nearest-neighbor générique ci-dessous
        # (arange*from/target = arange*s pour s entier).
        return plane[::sy, ::sx]
    # Gather (ratio non entier/upscale) : indices via le MÊME backend que `plane` (xp) — un index
    # numpy sur un tableau cupy lèverait. xp=np en CPU → comportement inchangé (octet-identique).
    _xp = cp if (GPU and isinstance(plane, cp.ndarray)) else np
    # Ratio NON ENTIER en box : on pré-réduit du plus grand facteur ENTIER disponible — c'est là
    # que se joue l'essentiel de l'anticrénelage (un 1920→700 passe par 960) — puis le gather
    # nearest ajuste à la taille exacte. Un ratio non entier n'admet pas de filtre d'aire exact
    # sans pondération par pixel de sortie, hors budget sur ce chemin.
    if _BOX:
        py, px = max(1, from_h // target_h), max(1, from_w // target_w)
        if py > 1 or px > 1:
            plane = _box_reduce(plane, py, px)
            from_h, from_w = plane.shape
    row_idx = (_xp.arange(target_h) * from_h / target_h).astype(int)
    col_idx = (_xp.arange(target_w) * from_w / target_w).astype(int)
    return plane[_xp.ix_(row_idx, col_idx)]

def rgba_to_yuv(img):
    """Image PIL RGBA → (Y full-res, U sub 2x2, V sub 2x2, alpha full, alpha sub 2x2).
    Y/U/V fusionnés en C (mvk ABI 2) quand disponible — bit-exact au numpy ci-dessous
    (mêmes expressions float32, -ffp-contract=off) ; l'alpha (uint8 brut) reste numpy.
    Gros bénéficiaire : le re-bake du chrome PLEINE TRAME à chaque bascule tally."""
    arr = np.array(img)
    if _MVK_HOST:   # gate HÔTE : arr vient de PIL, toujours numpy — mur GPU compris
        _got = getattr(bobimxl, "mvk_rgba2yuv", lambda *a: None)(
            arr, _NP_DT, _SCALE, _MAXV, _CW, _CH)
        if _got is not None:
            _y, _u2, _v2 = _got
            # ⚠ PERF : `arr[..., 3]` est une vue de PAS 4 (RGBA entrelacé). Les kernels mvk
            # exigent le DERNIER AXE CONTIGU (_mvk_ok2d) → un alpha stridé faisait échouer
            # SILENCIEUSEMENT mvk_blend_into sur le plan Y de CHAQUE tuile per-frame (frises,
            # horloges) → repli numpy à 20× le coût (banc : 3,4 ms au lieu de 0,11 ms sur une
            # frise 1920×250). Les plans chroma, eux, passaient : leur alpha sous-échantillonnée
            # est un tableau NEUF. On COMPACTE l'alpha ici, une fois (0,16 ms plein cadre, payé
            # au bake), pour que tous les blends aval prennent le kernel C.
            _a = np.ascontiguousarray(arr[..., 3])
            _am = _a
            if _CW == 2: _am = np.maximum(_am[:, 0::2], _am[:, 1::2])
            if _CH == 2: _am = np.maximum(_am[0::2, :], _am[1::2, :])
            return _y, _u2, _v2, _a, _am
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    a = np.ascontiguousarray(arr[..., 3])   # cf. note PERF ci-dessus : alpha compactée (mvk)
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

# Blends GPU FUSIONNÉS (ElementwiseKernel) : UN SEUL lancement par plan, au lieu des ~5 kernels
# émis par la décomposition cupy (astype/mul/sub/add/floordiv/astype). Même arithmétique ENTIÈRE
# (accumulation 32 bits non signée) → résultat OCTET-IDENTIQUE aux versions numpy ci-dessous
# (vérifié au banc micro-batch). Indispensable au mode tranche GPU : à N lots × (chrome + VU +
# horloges) × 3 plans, le coût de LANCEMENT des kernels dominait le compose (banc gate M3 : 10,6 ms
# à 30 lancements vs 0,71 ms pleine trame — la décomposition cupy multipliait encore ce coût ×5).
if GPU:
    _blend_k = cp.ElementwiseKernel(
        "T dst, T src, uint8 alpha", "T out",
        "out = (T)(((unsigned int)dst * (255u - (unsigned int)alpha) + "
        "(unsigned int)src * (unsigned int)alpha) / 255u)", "bobi_blend")
    _blend_pre_k = cp.ElementwiseKernel(
        "T dst, A inv_a, A src_a", "T out",
        "out = (T)(((unsigned int)dst * (unsigned int)inv_a + (unsigned int)src_a) / 255u)",
        "bobi_blend_pre")

def blend(dst, src, alpha):
    """dst/src YUV (uint8/uint16) + alpha uint8 (0..255) → même dtype. Accumulateur uint32
    car en 10/12 bits dst*(255-a) déborde uint16. NB : `// 255` (une seule ufunc vectorisée) est
    PLUS rapide en numpy que l'astuce entière `(t+(t>>8)+1)>>8` (4 passes mémoire = memory-bound)."""
    if GPU and isinstance(dst, cp.ndarray):
        return _blend_k(dst, src, alpha)
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
    if GPU and isinstance(dst, cp.ndarray):
        return _blend_pre_k(dst, inv_a, src_a)
    return ((dst.astype(_ACC) * inv_a + src_a) // 255).astype(_NP_DT)

# Variantes IN-PLACE pour les call-sites par-trame dont la destination est une VUE du canvas :
# mvk = 1 passe mémoire directe dans la vue (plus d'intermédiaire ni de ré-assignation) ; repli
# = strictement l'assignation d'origine `vue[...] = blend(...)` (mêmes octets). Ne PAS utiliser
# sur un dst partagé/caché (mvk mute dst) — uniquement les vues canvas de la trame courante.
_mvk_miss = {{"blend": 0, "blend_pre": 0, "place": 0, "calls": 0}}   # diagnostic : replis numpy
_mvk_why = {{}}   # diagnostic : signature (shape/dtype) des replis
_hist_bake = {{"last": 0.0, "max": 0.0}}   # coût UNITAIRE du re-bake d'une frise (pic, cf. render_history_tiles)

def _blend_into(dv, s, aa):
    _mvk_miss["calls"] += 1
    if _MVK and bobimxl.mvk_blend_into(dv, s, aa):
        return
    if GPU and isinstance(dv, cp.ndarray):
        # ÉCRITURE DIRECTE dans la vue canvas : `dv[...] = blend(dv, s, aa)` allouait un tableau
        # temporaire puis le RECOPIAIT dans la vue — une allocation et une passe mémoire de plus
        # par plan et par tuile, soit une trentaine par trame sur un mur habillé. Le kernel étant
        # élément-à-élément (chaque sortie ne dépend que de l'entrée de MÊME index), le passer en
        # sortie de lui-même est sûr, y compris sur une vue non contiguë.
        _blend_k(dv, s, aa, dv)
        return
    # Un repli numpy ICI coûte 20× le kernel C (banc : 4,6 ms vs 0,19 ms sur une tuile de frise
    # 1920×250, 3 plans) → il doit rester à ZÉRO. Compté et exposé (`mvk_miss` sur :8080).
    if _MVK:
        _mvk_miss["blend"] += 1
        _k = "dv%s/%s s%s/%s a%s/%s" % (dv.shape, dv.dtype, s.shape, s.dtype, aa.shape, aa.dtype)
        _mvk_why[_k] = _mvk_why.get(_k, 0) + 1
    dv[...] = blend(dv, s, aa)

def _blend_pre_into(dv, ia, sa):
    if _MVK and bobimxl.mvk_blend_pre_into(dv, ia, sa):
        return
    if GPU and isinstance(dv, cp.ndarray):
        _blend_pre_k(dv, ia, sa, dv)   # même raison que _blend_into : pas de temporaire, pas de recopie
        return
    dv[...] = blend_pre(dv, ia, sa)

# ─── Cache VRAM des tuiles d'habillage ───────────────────────────────────────
# Les frises et les horloges/ANC sont DÉJÀ cachées côté hôte : elles ne sont re-dessinées qu'au
# changement de valeur (une frise toutes les quelques secondes, une horloge ~1×/s). Elles n'en
# étaient pas moins RE-TÉLÉVERSÉES vers la VRAM à chaque trame — cinq tableaux par tuile, depuis de
# la mémoire pageable. Mesuré le 2026-08-07 sur les shards : 4,6 ms de transfert sur 8,2 ms de
# `ov_blend`, pour 8 tuiles sur 10 qui n'avaient pas changé.
# Le cache est keyé sur l'IDENTITÉ de la tuile hôte, et il garde une référence forte sur elle : sans
# cette référence, l'objet pourrait être collecté et son id RÉATTRIBUÉ à une autre tuile, qui se
# verrait alors servir une copie VRAM étrangère (une frise à la place d'une horloge). C'est le seul
# piège de cette approche, et il est silencieux — d'où la référence, qui rend la réattribution
# impossible par construction.
# Les VU-mètres, eux, changent RÉELLEMENT à chaque trame : ils ne passent pas par ici (ils
# n'auraient que des défauts de cache, et feraient churner la VRAM pour rien).
_ov_dev = {{}}          # id(tuile hôte) → (tuile hôte, tuile téléversée)
_ov_dev_vus = set()    # ids touchés par la trame courante (base de l'éviction)

# ─── Téléversement ÉPINGLÉ des tuiles per-frame ──────────────────────────────
# Les tuiles qui changent à chaque trame (VU-mètres) montent en VRAM par cinq `cp.asarray`
# successifs sur des tableaux numpy ORDINAIRES. Une source pageable oblige CUDA à passer par un
# tampon intermédiaire et rend la copie SYNCHRONE : la boucle attend. Le chien de garde l'a mesuré
# comme 30 % des dépassements de trame — le deuxième poste après le point de synchronisation.
# Motif déjà employé dans ce fichier pour les plans vidéo (`_place_batch`) : on rassemble tout
# dans un tampon hôte ÉPINGLÉ persistant, un SEUL transfert, puis on redécoupe des vues côté
# carte. Les cinq tableaux n'ont pas le même type (YUV en _NP_DT, alphas en uint8) : on travaille
# donc en OCTETS, avec des offsets alignés pour que les vues typées restent valides.
_pin_tuile = {{"host": None, "dev": None, "cap": 0}}
_tuile_pin_miss = {{"n": 0, "why": None}}

def _tuile_vram(t):
    """Tuile per-frame montée en VRAM en UN transfert épinglé. None = non applicable (repli)."""
    try:
        bx0, by0, bx1, by1, y, u, v, a, a2 = t
        arrs = (y, u, v, a, a2)
        tailles = [x.nbytes for x in arrs]
        # Offsets alignés sur 8 octets : une vue uint16 sur un offset impair serait invalide.
        offs, o = [], 0
        for n in tailles:
            offs.append(o); o += (n + 7) & ~7
        total = o
        if _pin_tuile["cap"] < total:
            cap = max(total, _pin_tuile["cap"] * 2)
            _pin_tuile["host"] = np.frombuffer(cp.cuda.alloc_pinned_memory(cap), dtype=np.uint8,
                                               count=cap)
            _pin_tuile["dev"] = cp.empty(cap, dtype=cp.uint8)
            _pin_tuile["cap"] = cap
        host = _pin_tuile["host"]; dev = _pin_tuile["dev"]
        for x, off, n in zip(arrs, offs, tailles):
            host[off:off + n] = np.ascontiguousarray(x).view(np.uint8).ravel()
        dev[:total].set(host[:total])            # UN seul H2D, depuis de la mémoire épinglée
        vues = []
        for x, off, n in zip(arrs, offs, tailles):
            vues.append(dev[off:off + n].view(x.dtype).reshape(x.shape))
        return (bx0, by0, bx1, by1) + tuple(vues)
    except Exception as _e:                                                # noqa: BLE001
        _tuile_pin_miss["n"] += 1
        _tuile_pin_miss["why"] = repr(_e)[:160]
        return None


def _tuile_dev(t):
    """Tuile prête pour le backend, en réutilisant sa copie VRAM tant que la tuile hôte est LA MÊME.
    En CPU `_to_xp` est l'identité : on rend la tuile telle quelle, coût strictement nul."""
    if not GPU:
        return t
    _k = id(t)
    _ov_dev_vus.add(_k)
    _hit = _ov_dev.get(_k)
    if _hit is not None and _hit[0] is t:
        return _hit[1]
    bx0, by0, bx1, by1, _y, _u, _v, _a, _a2 = t
    _up = (bx0, by0, bx1, by1, _to_xp(_y), _to_xp(_u), _to_xp(_v), _to_xp(_a), _to_xp(_a2))
    _ov_dev[_k] = (t, _up)
    return _up

def _tuiles_dev_evict():
    """Libère les copies VRAM des tuiles qui n'ont pas servi à cette trame (frise retirée, horloge
    re-dessinée…). Sans ça le cache grossirait à chaque re-bake."""
    if not _ov_dev:
        return
    for _k in [_k for _k in _ov_dev if _k not in _ov_dev_vus]:
        _ov_dev.pop(_k, None)
    _ov_dev_vus.clear()

def _mvk_place_plane(dstv, plane, th, tw):
    """resize_plane + assignation FUSIONNÉS (mvk_place → écrit la vue canvas en 1 passe).
    Indices nearest = MÊMES formules que resize_plane (pas entier, sinon troncature float),
    calculées ici — le C ne fait que le gather. False → repli resize_plane (bit-exact).
    ⚠ Le kernel ne sait faire QUE le gather nearest : sous filtre box il n'a pas le droit de
    servir, sinon la tuile serait décimée pendant que le reste du plugin croit filtrer. On rend
    la main à resize_plane, et on COMPTE le repli (`mvk_miss.place` sur :8080) pour que le
    surcoût soit lisible au lieu d'être subi en silence."""
    if not _MVK or th <= 0 or tw <= 0:
        return False
    if _BOX:
        _mvk_miss["place"] += 1
        return False
    fh, fw = plane.shape
    if fh % th == 0 and fw % tw == 0:
        ri = (np.arange(th) * (fh // th)).astype(np.int32)
        return bobimxl.mvk_place_into(dstv, plane, ri, col0=0, col_step=fw // tw)
    ri = (np.arange(th) * fh / th).astype(np.int32)
    ci = (np.arange(tw) * fw / tw).astype(np.int32)
    return bobimxl.mvk_place_into(dstv, plane, ri, col_idx=ci)

# ─── Placement groupé des tuiles vidéo (CPU direct / GPU upload épinglé groupé) ───────────────
_pin = {{"host": None, "dev": None, "cap": 0}}   # buffers persistants GPU (staging épinglé entrées + device)
_outpin = {{"buf": None}}                        # buffer hôte ÉPINGLÉ persistant pour le download sortie (GPU)

def _out_host(arr):
    """Tableau planar prêt à écrire dans un grain, côté HÔTE. En GPU : download D2H ÉPINGLÉ
    (.get(out=pinned) — un .get direct dans le grain, mmap non épinglé, serait ~4× plus lent,
    banc Phase 0). En CPU : renvoyé tel quel (zéro copie). Chemin PARTAGÉ par la sortie
    progressive (1 trame pleine) et la sortie entrelacée (2 champs) — le buffer épinglé est
    dimensionné sur la plus grosse demande vue, donc réutilisé d'un champ à l'autre."""
    if not GPU:
        return arr
    _n = arr.size
    if _outpin["buf"] is None or _outpin["buf"].size < _n:
        _outpin["buf"] = np.frombuffer(cp.cuda.alloc_pinned_memory(_n * _BPS),
                                       dtype=_NP_DT, count=_n)
    arr.get(out=_outpin["buf"][:_n])
    return _outpin["buf"][:_n]

def _out_host_plans(plans):
    """Mêmes garanties que `_out_host`, mais SANS concaténer d'abord.

    L'ancien chemin construisait une trame contiguë EN VRAM (`xp.concatenate`) uniquement pour
    offrir un tableau unique au download — soit une allocation plein cadre et une recopie de plus
    par trame. Or le tampon épinglé est déjà contigu : il suffit d'y déverser chaque plan à son
    offset. Trois downloads au lieu d'un, mais plus d'allocation ni de recopie côté carte.
    Mesuré comme le poste dominant des dépassements (la moitié, avec l'assemblage) : l'étage de
    sortie est le point où la boucle SYNCHRONISE, donc là où toute la file accumulée se paie."""
    if not GPU:
        return np.concatenate([p.ravel() for p in plans])
    _n = sum(p.size for p in plans)
    if _outpin["buf"] is None or _outpin["buf"].size < _n:
        _outpin["buf"] = np.frombuffer(cp.cuda.alloc_pinned_memory(_n * _BPS),
                                       dtype=_NP_DT, count=_n)
    _off = 0
    for _p in plans:
        _k = _p.size
        _p.ravel().get(out=_outpin["buf"][_off:_off + _k])
        _off += _k
    return _outpin["buf"][:_n]
# GPU SLICE : staging épinglé PAR LOT (entrées) + buffers épinglés de SORTIE (D2H) — bande seule
# (hy/hu/hv, squelette k=1) ou lots ×2 double-buffer + stream D2H dédié (hb/stream, micro-batch).
# Persistants (l'allocation pinned est coûteuse) ; capacité entrée en croissance ×2 comme _pin.
_slgpu = {{"hin": None, "din": None, "cap": 0, "hy": None, "hu": None, "hv": None,
          "hb": None, "hb_grp": 0, "stream": None}}

def _gpu_place_band(cy, cu, cv, stage):
    """MODE TRANCHE GPU (gpu_slice) : place dans le canvas VRAM les bandes source d'un LOT de
    bandes de sortie (1 bande au grain squelette, gpu_batch_bands en micro-batch — verdict
    GO-MEGA-135). UN SEUL H2D par lot : les rangées source (déjà réduites par les vues stridées/
    gathers hôte de _compose_bands) sont concaténées dans le staging hôte ÉPINGLÉ puis uploadées
    groupées — même leçon que le banc Phase 0 (un transfert par tuile, ou pageable, fait RÉGRESSER
    le GPU), déclinée au lot. Point d'ancrage (a) du banc gate TISSU_SLICE_GPU.md."""
    total = 0
    for it in stage:
        total += it[0].size + (2 * it[1].size if it[1] is not None else 0)
    if _slgpu["cap"] < total:                    # (ré)alloue staging épinglé + device (croissance ×2)
        cap = max(total, (_slgpu["cap"] * 2) or total)
        _slgpu["hin"] = np.frombuffer(cp.cuda.alloc_pinned_memory(cap * _BPS), dtype=_NP_DT, count=cap)
        _slgpu["din"] = cp.empty(cap, dtype=_NP_DT)
        _slgpu["cap"] = cap
    hin = _slgpu["hin"]; din = _slgpu["din"]
    off = 0; meta = []
    # Entrée de stage : (by, bu, bv, a, b, ca0, cb0, vx, vw, cx0, csx, csc) — csx/csc = PAS de
    # décimation COLONNE restant à appliquer en VRAM (fast-path ratio entier : les rangées sont
    # uploadées PLEINE LARGEUR — copies hôte contiguës par rangée, le gather colonne-stridé
    # coûtait ~5,6 ms/trame au banc micro-batch — et la décimation colonne se fait au placement).
    for (by_, bu_, bv_, a, b, ca0, cb0, vx, vw, cx0, csx, csc) in stage:
        ny = by_.size
        hin[off:off+ny].reshape(by_.shape)[...] = by_          # gather hôte → épinglé (1 seule copie)
        nu = 0
        if bu_ is not None:
            nu = bu_.size
            hin[off+ny:off+ny+nu].reshape(bu_.shape)[...] = bu_
            hin[off+ny+nu:off+ny+2*nu].reshape(bv_.shape)[...] = bv_
        meta.append((off, by_.shape, bu_.shape if bu_ is not None else None,
                     a, b, ca0, cb0, vx, vw, cx0, csx, csc))
        off += ny + 2 * nu
    din[:off].set(hin[:off])                     # UN seul H2D par lot (épinglé → device)
    for (o, shy, shu, a, b, ca0, cb0, vx, vw, cx0, csx, csc) in meta:
        ny = shy[0] * shy[1]
        cy[a:b, vx:vx + vw] = din[o:o+ny].reshape(shy)[:, ::csx]
        if shu is not None:
            nu = shu[0] * shu[1]
            cu[ca0:cb0, cx0:cx0 + vw//_CW] = din[o+ny:o+ny+nu].reshape(shu)[:, ::csc]
            cv[ca0:cb0, cx0:cx0 + vw//_CW] = din[o+ny+nu:o+ny+2*nu].reshape(shu)[:, ::csc]

# ─── Placement GPU fusionné : indices cachés + gather écrit DANS le canvas ───
# `dst[...] = resize_plane(src, h, w)` coûtait, par plan : deux `arange`, deux multiplications,
# deux `astype`, un gather qui ALLOUE, puis une recopie dans la vue — soit une huitaine de
# lancements et deux tableaux temporaires. À trois plans par tuile et deux tuiles, une quarantaine
# de lancements par trame, alors que le GPU est à 40 % : on ne le sature pas, on l'attend.
# Deux corrections, de la même famille que celles qui ont payé ce soir :
#   • les INDICES ne dépendent que des tailles (source, destination) — donc ils ne changent
#     JAMAIS d'une trame à l'autre. Ils sont calculés une fois et gardés en VRAM ;
#   • le gather écrit DIRECTEMENT dans la vue du canvas, en UN lancement, au lieu de produire un
#     tableau qu'on recopie ensuite.
# Repli sur le chemin d'origine à la moindre surprise (forme, dtype, backend) — et le repli est
# COMPTÉ (`place_miss` sur :8080), sinon une régression silencieuse coûterait des millisecondes
# sans jamais le dire.
_idx_cache = {{}}          # (from_h, from_w, th, tw) → (row_idx, col_idx) en VRAM
_place_miss = {{"n": 0}}

if GPU:
    _gather_k = cp.ElementwiseKernel(
        "raw T src, raw int32 ry, raw int32 cx, int32 sw, int32 dw", "T out",
        "int r = i / dw; int c = i - r * dw; out = src[ry[r] * (long long)sw + cx[c]];",
        "bobi_gather_place")

def _indices(from_h, from_w, th, tw):
    """Indices nearest (mêmes formules que resize_plane, donc rendu identique), en VRAM et cachés.
    Les calculer à chaque trame était l'essentiel du coût : six lancements pour des valeurs
    constantes."""
    _k = (from_h, from_w, th, tw)
    _v = _idx_cache.get(_k)
    if _v is None:
        _v = ((cp.arange(th) * from_h / th).astype(cp.int32),
              (cp.arange(tw) * from_w / tw).astype(cp.int32))
        _idx_cache[_k] = _v
    return _v

def _place_gpu_plane(dst, plane, th, tw):
    """Redimensionne `plane` vers la vue `dst` (th×tw) en UN lancement. False = non applicable."""
    try:
        if not GPU or not isinstance(plane, cp.ndarray) or th <= 0 or tw <= 0:
            return False
        fh, fw = plane.shape
        if _BOX:
            py, px = max(1, fh // th), max(1, fw // tw)
            if py > 1 or px > 1:
                plane = _box_reduce(plane, py, px)
                fh, fw = plane.shape
        if (fh, fw) == (th, tw):
            dst[...] = plane            # même taille : simple recopie, rien à interpoler
            return True
        ry, cx = _indices(fh, fw, th, tw)
        _src = plane if plane.flags.c_contiguous else cp.ascontiguousarray(plane)
        _gather_k(_src.ravel(), ry, cx, np.int32(_src.shape[1]), np.int32(tw), dst)
        return True
    except Exception:                                                      # noqa: BLE001
        _place_miss["n"] += 1
        return False


def _place_batch(cy, cu, cv, batch):
    """Resize + place les tuiles vidéo collectées dans le canvas (cy/cu/cv, backend xp).
    CPU : resize numpy direct (octet-identique à l'inline d'origine). GPU : UN seul upload H2D des
    plans collectés via un buffer hôte ÉPINGLÉ persistant (1 H2D/trame — sinon régression, banc
    Phase 0), puis resize+place en VRAM. Tuiles disjointes → l'ordre de placement n'importe pas."""
    if not batch:
        return
    if not GPU:
        for (sy, su, sv, vy, vh, vx, vw) in batch:
            # Chemin fusionné mvk (3 plans) ; l'échec d'UN plan (forme/contiguïté inattendue)
            # rebascule la tuile ENTIÈRE sur le repli numpy (ré-écrit les 3 plans : sûr).
            if _mvk_place_plane(cy[vy:vy+vh, vx:vx+vw], sy, vh, vw) \
                    and _mvk_place_plane(cu[vy//_CH:vy//_CH+vh//_CH, vx//_CW:vx//_CW+vw//_CW],
                                         su, vh//_CH, vw//_CW) \
                    and _mvk_place_plane(cv[vy//_CH:vy//_CH+vh//_CH, vx//_CW:vx//_CW+vw//_CW],
                                         sv, vh//_CH, vw//_CW):
                continue
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
        _dy = cy[vy:vy+vh, vx:vx+vw]
        _du = cu[vy//_CH:vy//_CH+vh//_CH, vx//_CW:vx//_CW+vw//_CW]
        _dv = cv[vy//_CH:vy//_CH+vh//_CH, vx//_CW:vx//_CW+vw//_CW]
        # Chemin fusionné : gather écrit dans la vue, indices cachés. Repli plan par plan — un
        # échec sur la chroma ne doit pas jeter la luma déjà posée.
        if not _place_gpu_plane(_dy, gy, vh, vw):
            cy[vy:vy+vh, vx:vx+vw] = resize_plane(gy, vh, vw)
        if not _place_gpu_plane(_du, gu, vh//_CH, vw//_CW):
            cu[vy//_CH:vy//_CH+vh//_CH, vx//_CW:vx//_CW+vw//_CW] = resize_plane(gu, vh//_CH, vw//_CW)
        if not _place_gpu_plane(_dv, gv, vh//_CH, vw//_CW):
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
    """Géométrie UNIQUE de la cellule — partagée par la boucle composite et les couches
    d'habillage. Le rectangle vidéo vient du composant `video` du MODÈLE de la cellule
    (toute cellule en a un : explicite, défaut du mur, ou « Classique » généré). Renvoie :
      x/y/w/h    : cellule clampée au canvas (dimensions paires) ;
      vx/vy/vw/vh: rectangle de l'IMAGE vidéo (fit contain = homothétique au ratio SOURCE,
                   centré — letterbox/pillarbox, on ne déforme jamais l'image) ;
      ay/ah      : zone verticale de l'image (compat consommateurs historiques).
    Modèle sans composant vidéo → rect dégénéré (vw=0), la boucle composite saute la lecture."""
    x, y, w, h = cfg["x"], cfg["y"], cfg["w"], cfg["h"]
    x = max(0, min(x, OUT_WIDTH - 1)); y = max(0, min(y, OUT_HEIGHT - 1))
    w = max(2, min(w, OUT_WIDTH - x)); h = max(2, min(h, OUT_HEIGHT - y))
    w -= w % 2; h -= h % 2
    vr = _tpl_video_comp(cfg)
    if vr is None:
        return {{"x": x, "y": y, "w": w, "h": h, "vx": x, "vy": y, "vw": 0, "vh": 0,
                 "fx": x, "fy": y, "fw": 0, "fh": 0, "ay": y, "ah": h}}
    rx, ry, rw, rh = _comp_rect(cfg, vr)
    # ★ HABILLAGE QUI RÉSERVE SA PLACE. Le style « viseur » dessine ses équerres AUTOUR de
    # l'image ; sans réservation, elles n'ont de marge que là où le fit `contain` en laisse
    # (letterbox OU pillarbox, jamais les deux) et retombent sur l'image sur les deux autres
    # côtés. On retire donc l'épaisseur des équerres du rectangle vidéo AVANT le fit : l'image
    # est réduite d'autant, les quatre côtés ont leur marge, et rien n'est recouvert — c'est la
    # même convention que les autres habillages de fenêtre (réduction homothétique).
    # Fait ICI et pas au dessin : _video_rect est la géométrie de RÉFÉRENCE (boucle composite,
    # habillage, tailles réclamées à la pyramide). L'insérer ailleurs ferait diverger le
    # rectangle réellement composé de celui que l'habillage croit border.
    _frame_out = 0        # ce que le style de bordure dessine À L'EXTÉRIEUR de l'image
    if (vr.get("border") or "none") == "viewfinder":
        try:                       # MÊME formule que _tpl_draw_video_border (bw clampé, puis bt) —
            _bw = max(1, min(24, int(vr.get("border_w") or 3)))   # une marge qui ne correspondrait
        except (TypeError, ValueError):                            # pas à l'épaisseur dessinée
            _bw = 3                                                # laisserait un liseré d'image.
        _bt = max(2, _bw)
        _frame_out = _bt
        # ★ `_bt + 1` et non `_bt` : plus bas, vx/vy sont ramenés au PAIR INFÉRIEUR (alignement
        # chroma). Ce recul d'un pixel mangeait EXACTEMENT la marge réservée en haut et à gauche,
        # et les équerres y retombaient sur l'image — le pixel supplémentaire l'absorbe.
        # Jamais au point de faire disparaître l'image : on plafonne la marge au quart du côté.
        _m = _bt + 1
        _mx = min(_m, max(0, rw // 4)); _my = min(_m, max(0, rh // 4))
        rx += _mx; ry += _my; rw = max(2, rw - 2 * _mx); rh = max(2, rh - 2 * _my)
    if (vr.get("fit") or "fill") == "contain":
        # Homothétique au ratio SOURCE (in_w/in_h résolus par l'orchestrateur), centré.
        sw = int(cfg.get("in_w") or 0); sh = int(cfg.get("in_h") or 0)
        if sw > 1 and sh > 1:
            sc = min(rw / sw, rh / sh)
            nw = max(2, int(sw * sc)); nh = max(2, int(sh * sc))
            rx += (rw - nw) // 2; ry += (rh - nh) // 2
            rw, rh = nw, nh
    vw = max(2, min(rw, OUT_WIDTH - rx)); vw -= vw % 2
    vh = max(2, min(rh, OUT_HEIGHT - ry)); vh -= vh % 2
    vx = rx - rx % 2
    vy = ry - ry % 2
    # ★ Rectangle du CADRE : l'image PLUS ce que la bordure dessine à l'extérieur d'elle. Le
    # style « viseur » pose ses équerres dans la marge réservée — mesuré sur le mur : l'équerre
    # occupe les colonnes 4-5 et l'image commence à la 6. Le bord VISUEL du bloc vidéo est donc
    # celui de l'équerre, pas celui de l'image. C'est sur lui que l'habillage doit s'aligner :
    # se caler sur l'image laisse les tallies en retrait du cadre, et ça se voit.
    # Les autres styles dessinent VERS L'INTÉRIEUR → cadre = image, aucun décalage.
    _fx = max(0, vx - _frame_out); _fy = max(0, vy - _frame_out)
    _fw = max(2, min(OUT_WIDTH - _fx, vw + 2 * _frame_out))
    _fh = max(2, min(OUT_HEIGHT - _fy, vh + 2 * _frame_out))
    return {{"x": x, "y": y, "w": w, "h": h, "vx": vx, "vy": vy, "vw": vw, "vh": vh,
             "fx": _fx, "fy": _fy, "fw": _fw, "fh": _fh,
             "ay": vy, "ah": vh}}

def _render_pill(d, cx, cy, r, fill, outline):
    """Pastille ronde centrée (cx, cy) de rayon r."""
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=2)

def render_dynamic():
    """Re-rendu à chaque changement Tally/TSL ou de géométrie : composants BAKÉS des modèles
    de PiP (umd / tally / text / format / bordure video). Toute cellule a un modèle (explicite,
    défaut du mur, ou « Classique » généré — cf. _tpl_comps) : c'est l'UNIQUE moteur d'habillage."""
    img = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, cfg in enumerate(FLUX_CONFIG):
        if cfg.get("hidden"):
            continue
        _tpl_render_dynamic(d, img, i, cfg, _tpl_comps(cfg))
    return img   # RGBA — consolidation : converti une seule fois après alpha_composite

def _meter_label_tile(status, bx0, by0, bx1, by1, mx, my, mw, mh):
    """TUILE YUV séparée : étiquette VERTICALE « SILENCE » (signal muet) ou « ABSENCE » (flux coupé),
    petite, centrée sur la zone des barres — pour SAVOIR pourquoi il n'y a pas de son. Posée en tuile
    INDÉPENDANTE (PIL→YUV), donc rendu IDENTIQUE sur les chemins GPU et CPU, et payée uniquement hors
    état « ok ». Renvoie un tuple tuile (même bbox que le meter) ou None.
    (Banc 2026-07-14 : mesurée à < 0,05 ms par trame et par meter muet, même sur un mur 16 cellules
    — la cacher ne rapporte RIEN de mesurable. Ne pas « optimiser » ce chemin.)"""
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

_meter_grad_tile_cache = {{}}

def _meter_grad_tile(bx0, by0, bx1, by1, mx, my, mw, mh, scale, opacity_pct, grad, n_channels):
    """Graduations SUPERPOSÉES aux barres (grad_side = "inside") — tuile YUV séparée, blendée
    APRÈS le mètre. C'est la seule position possible : les barres sont peintes par-dessus le fond
    statique caché, donc des graduations dessinées dans ce fond seraient recouvertes.

    Intérêt : la colonne latérale disparaît complètement (tick_w = 0) et TOUTE la largeur va aux
    barres — c'est le mode qui rend le plus de place, là où « traits seuls » n'en rendait qu'une
    partie. Traits fins semi-transparents sur toute la largeur des barres (repère de lecture sans
    masquer le niveau) ; en niveau « complet », les valeurs sont posées à gauche sur un fond sombre
    translucide, sans quoi un chiffre clair sur une barre jaune serait illisible.

    Tuile CACHÉE : elle ne dépend que de la géométrie et de l'échelle, jamais des niveaux — seul
    le blend est payé par trame (même contrat que le fond statique)."""
    key = (bx1 - bx0, by1 - by0, mx - bx0, my - by0, mw, mh, scale, opacity_pct, grad, n_channels)
    got = _meter_grad_tile_cache.get(key)
    if got is None:
        tw_, th_ = bx1 - bx0, by1 - by0
        if tw_ <= 0 or th_ <= 0:
            return None
        bars_mh = max(20, mh - 12)
        rmx, rmy = mx - bx0, my - by0
        img = Image.new("RGBA", (tw_, th_), (0, 0, 0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        to_frac, _gt, _yt = _meter_scale_params(scale)
        if scale == "ppm":
            ticks = [(+12, "+12"), (+9, "+9"), (+6, "+6"), (+3, "+3"), (0, "0"),
                     (-3, "-3"), (-6, "-6"), (-9, "-9"), (-12, "-12")]
            ticks_dbfs = [(ebu - 18, lbl) for ebu, lbl in ticks]
        else:
            ticks_dbfs = [(0, "0"), (-3, "-3"), (-6, "-6"), (-9, "-9"), (-12, "-12"),
                          (-18, "-18"), (-20, "-20"), (-30, "-30"), (-40, "-40"), (-50, "-50")]
        a_line = int(120 * opacity_pct / 100)     # discret : on lit le niveau À TRAVERS le trait
        a_txt  = int(235 * opacity_pct / 100)
        a_chip = int(150 * opacity_pct / 100)
        _lf = ImageFont.load_default()
        last_label_y = -10
        for tick_dbfs, lbl in ticks_dbfs:
            y_tick = rmy + bars_mh - int(round(to_frac(tick_dbfs) * bars_mh))
            d.line([rmx, y_tick, rmx + mw - 1, y_tick], fill=(210, 210, 215, a_line))
            if grad != "full":
                continue
            if abs(y_tick - last_label_y) >= 9 and y_tick - 4 >= rmy and y_tick + 4 <= rmy + bars_mh:
                try:
                    _lw = int(round(d.textlength(lbl, font=_lf)))
                except Exception:
                    _lw = 0
                if _lw and _lw + 4 <= mw:
                    d.rectangle([rmx, y_tick - 5, rmx + _lw + 2, y_tick + 5],
                                fill=(0, 0, 0, a_chip))
                    d.text((rmx + 1, y_tick - 4), lbl, font=_lf, fill=(230, 230, 235, a_txt))
                    last_label_y = y_tick
        got = rgba_to_yuv(img)
        _meter_grad_tile_cache[key] = got
    oy, ou, ov, oa, oa2 = got
    return (bx0, by0, bx1, by1, oy, ou, ov, oa, oa2)

def _tile_peaks_range(i, cfg, start0, count, now):
    """Peaks/holds/statut pour la FENÊTRE de canaux [start0, start0+count) d'un espace de
    16 canaux : canaux 0..7 = flux audio dérivé de la source, 8..15 = flux dérivé SUIVANT
    (`_audio_{{N+1}}` — convention « 2 flux de 8 »). États partagés par (tuile, flux) ; statut
    combiné : ok si AU MOINS un flux lu est frais, sinon silence, sinon absence."""
    peaks = np.full(count, METER_MIN_DB)
    holds = np.full(count, METER_MIN_DB)
    got_ok = got_sil = False
    for flow in (0, 1):
        f0 = flow * A_CHANNELS_MAX
        a = max(start0, f0); b = min(start0 + count, f0 + A_CHANNELS_MAX)
        if a >= b:
            continue
        key = (i, flow)
        st = audio_states.get(key)
        if st is None:
            st = _open_audio_state(i, cfg, flow)
            audio_states[key] = st
        if st is None:
            continue
        p, h, s = _update_peaks(st, A_CHANNELS_MAX, now)
        if p is None:
            continue
        peaks[a - start0:b - start0] = p[a - f0:b - f0]
        holds[a - start0:b - start0] = h[a - f0:b - f0]
        if s == "ok":
            got_ok = True
        elif s == "silence":
            got_sil = True
    return peaks, holds, ("ok" if got_ok else ("silence" if got_sil else "absence"))

def _meter_tiles_at(mx, my, mw, mh, n, peaks, holds, scale, opacity_pct, status, tiles, ch0=0,
                     tick_w=METER_TICK_W, bar_w=METER_BAR_W, gap=METER_GAP,
                     grad="full", grad_side="left"):
    """Tuile(s) YUV d'un meter à la géométrie donnée (bbox chroma-alignée + étiquette
    SILENCE/ABSENCE) — corps commun aux meters legacy et aux composants de modèle.
    bbox locale du meter (_draw_meter dessine jusqu'à mx+mw / my+mh inclus → +1), bornée à
    la sortie et alignée chroma : origine ramenée à un multiple de _CW/_CH, dimensions
    complétées au multiple supérieur (rgba_to_yuv sous-échantillonne par _CW/_CH).
    `tick_w`/`bar_w`/`gap` : dimensions effectives des graduations/barres (mode `fit` → mises à
    l'échelle de `mw` par l'appelant via _meter_fit_dims ; par défaut = mode `auto` historique)."""
    bx0 = max(0, mx); by0 = max(0, my)
    bx1 = min(OUT_WIDTH, mx + mw + 1); by1 = min(OUT_HEIGHT, my + mh + 1)
    bx0 -= bx0 % _CW; by0 -= by0 % _CH
    if (bx1 - bx0) % _CW: bx1 = min(OUT_WIDTH, bx1 + (_CW - (bx1 - bx0) % _CW))
    if (by1 - by0) % _CH: by1 = min(OUT_HEIGHT, by1 + (_CH - (by1 - by0) % _CH))
    if bx1 <= bx0 or by1 <= by0:
        return
    if not METERS_PIL:
        # Chemin TUILE (défaut CPU **ET** GPU) : statique (fond/graduations/labels) rendu PIL UNE
        # fois puis caché ★ EN YUV ★, barres+peak-hold peints par trame DIRECTEMENT dans les plans
        # YUV (0.40.0) — aucune PIL et AUCUNE conversion RGBA→YUV par trame. Luma/alpha bit-exactes,
        # chroma des bords de barre « franc » au lieu de moyenné (cf. _meter_paint_rect).
        # Repli bit-exact intégral : `meters_pil=true` (chemin PIL d'origine, ci-dessous).
        oy, ou, ov, oa, oa2 = _meter_tile_yuv(bx1 - bx0, by1 - by0, mx - bx0, my - by0, mw, mh, n,
                                              peaks, holds, scale, opacity_pct, ch0, tick_w, bar_w, gap,
                                              grad, grad_side)
    else:
        # Repli `meters_pil` : chemin PIL historique VERBATIM (redéployer avec meters_pil=true
        # restaure le comportement d'avant 0.31.0 à l'identique en cas de doute sur un mur).
        tile = Image.new("RGBA", (bx1 - bx0, by1 - by0), (0, 0, 0, 0))
        _draw_meter(tile, mx - bx0, my - by0, mw, mh, n, peaks, holds, scale, opacity_pct, ch0,
                    tick_w, bar_w, gap, grad, grad_side)
        oy, ou, ov, oa, oa2 = rgba_to_yuv(tile)
    tiles.append((bx0, by0, bx1, by1, oy, ou, ov, oa, oa2))
    # Graduations SUPERPOSÉES (mode "inside") : tuile propre blendée APRÈS les barres — dans le
    # fond statique elles seraient recouvertes par les barres. Cosmétique → jamais fatale.
    if grad_side == "inside" and grad != "none":
        try:
            gt = _meter_grad_tile(bx0, by0, bx1, by1, mx, my, mw, mh, scale, opacity_pct, grad, n)
        except Exception:
            gt = None
        if gt is not None:
            tiles.append(gt)
    # Étiquette SILENCE / ABSENCE par-dessus (tuile séparée, même bbox → blend après le meter).
    # Cosmétique → ne jamais casser le rendu d'une trame si l'étiquette échoue.
    if status in ("silence", "absence"):
        try:
            lab = _meter_label_tile(status, bx0, by0, bx1, by1, mx, my, mw, mh)
        except Exception:
            lab = None
        if lab is not None:
            tiles.append(lab)

def render_meters(now):
    """Re-rendu par frame des peak meters. Renvoie une LISTE de TUILES YUV prêtes à blender
    [(bx0, by0, bx1, by1, y, u, v, a, a2), ...] — UNE tuile par meter, sur sa propre bbox locale
    (alignée chroma paire), PAS une couche plein écran. Les VU changent à chaque trame (jamais
    cachés) : convertir/blender chaque petit rectangle (≈ une bande VU dans une cellule) au lieu
    d'une bbox-union quasi plein écran (meters dispersés sur un mur 16 fenêtres) supprime le gros
    du coût de l'habillage per-frame. Renvoie None si aucun meter activé.
    Cellule à MODÈLE de PiP : les meters sont les composants `meters` du modèle (géométrie libre,
    largeur intrinsèque du meter ancrée left/center/right dans le rectangle du composant) ;
    l'état audio (_update_peaks) est calculé UNE fois par tuile au max de canaux demandé."""
    tiles = []
    for i, cfg in enumerate(FLUX_CONFIG):
        if cfg.get("hidden"):
            continue
        comps = _tpl_comps(cfg)
        if comps is not None:
            mcs = [c for c in comps if isinstance(c, dict) and c.get("type") == "meters"
                   and _comp_visible(i, cfg, c)]
            if not mcs:
                continue
            for comp in mcs:
                try:
                    # Affectation de canaux : ch_start (1-based) + channels dans un espace de
                    # 16 canaux (2 flux de 8) — ex. « 1-2 à gauche, 3-4 à droite » = 2 composants.
                    try:
                        n = max(1, min(A_CHANNELS_MAX, int(comp.get("channels") or 2)))
                    except (TypeError, ValueError):
                        n = 2
                    try:
                        s0 = max(0, min(2 * A_CHANNELS_MAX - 1, int(comp.get("ch_start") or 1) - 1))
                    except (TypeError, ValueError):
                        s0 = 0
                    n = min(n, 2 * A_CHANNELS_MAX - s0)
                    peaks, holds, status = _tile_peaks_range(i, cfg, s0, n, now)
                    rx, ry, rw, rh = _comp_rect(cfg, comp)
                    # Marge au bord de la CELLULE : un mètre collé au bord touche visuellement la
                    # fenêtre voisine et on ne sait plus à quel PiP l'audio appartient.
                    _g = _video_rect(cfg)
                    rx, rw = _meter_inset(rx, rw, _g["x"], _g["w"])
                    _scale = comp.get("scale") or "dbfs"
                    _grad, _gtw = _meter_grad(comp, _scale, rw)
                    _gside = str(comp.get("grad_side") or "left").strip().lower()
                    if _gside not in ("left", "right", "inside"):
                        _gside = "left"
                    if _gside == "inside":
                        _gtw = 0        # plus de colonne : toute la largeur va aux barres
                    mh = max(20, rh - 1)
                    width_mode = comp.get("width_mode") or "auto"
                    if width_mode == "fit":
                        # Suit la largeur du composant : SEULES les barres de canaux s'élargissent
                        # pour occuper rw (zone de graduations tick_w et espacement gap FIXES —
                        # sinon l'échelle dBFS/PPM et ses repères se déforment, cf. _meter_fit_dims).
                        # L'alignement devient sans effet — le meter occupe le rectangle (mw ≤ rw).
                        tick_w, bar_w, gap, mw = _meter_fit_dims(n, rw, _gtw)
                        mx = rx
                    else:
                        tick_w, bar_w, gap = _gtw, METER_BAR_W, METER_GAP
                        mw = _meter_layout(n, tick_w, bar_w, gap)
                        al = comp.get("align") or "left"
                        mx = rx + ((rw - mw) // 2 if al == "center"
                                   else (rw - mw if al == "right" else 0))
                    opacity_pct = max(10, min(100, int(comp.get("opacity") or 70)))
                    _meter_tiles_at(mx, ry, mw, mh, n, peaks, holds,
                                    _scale, opacity_pct, status, tiles,
                                    ch0=s0, tick_w=tick_w, bar_w=bar_w, gap=gap,
                                    grad=_grad, grad_side=_gside)
                except Exception:
                    continue
    # Blocs VU-mètres du MUR (deploy_config.params.meter_blocks) : posés sur le canevas de
    # sortie ENTIER, indépendants des fenêtres — PAS une fenêtre déguisée (pas de vidéo, pas de
    # tally/label/freeze/NO SIGNAL). `full_cfg` couvre tout le canvas → _comp_rect (qui attend un
    # rect de CELLULE + des fractions de composant) traite le MUR ENTIER comme la « cellule » et
    # le bloc lui-même comme le « composant » (mêmes clés x/y/w/h fractionnaires que `meters`) :
    # AUCUNE duplication du calcul de géométrie/peaks/dessin.
    if METER_BLOCKS:
        full_cfg = {{"x": 0, "y": 0, "w": OUT_WIDTH, "h": OUT_HEIGHT}}
        for j, blk in enumerate(METER_BLOCKS):
            if not isinstance(blk, dict) or blk.get("hidden"):
                continue
            try:
                try:
                    n = max(1, min(A_CHANNELS_MAX, int(blk.get("channels") or 2)))
                except (TypeError, ValueError):
                    n = 2
                try:
                    s0 = max(0, min(2 * A_CHANNELS_MAX - 1, int(blk.get("ch_start") or 1) - 1))
                except (TypeError, ValueError):
                    s0 = 0
                n = min(n, 2 * A_CHANNELS_MAX - s0)
                # Clé d'état audio DISTINCTE des fenêtres (tuple ("mb", j), jamais un int i de
                # fenêtre) : _tile_peaks_range n'utilise l'index que comme clé de dict, pas
                # comme indice de liste — cf. audio_states[(i, flow)].
                peaks, holds, status = _tile_peaks_range(("mb", j), blk, s0, n, now)
                rx, ry, rw, rh = _comp_rect(full_cfg, blk)
                # Bloc de MUR : mêmes réglages de graduations que le composant de cellule. Pas de
                # marge de bord ici — sa « cellule » est le canevas entier, il n'a pas de voisin
                # dont le séparer, et le rentrer déplacerait un bloc que l'utilisateur a posé.
                _bscale = blk.get("scale") or "dbfs"
                _bgrad, _bgtw = _meter_grad(blk, _bscale, rw)
                _bgside = str(blk.get("grad_side") or "left").strip().lower()
                if _bgside not in ("left", "right", "inside"):
                    _bgside = "left"
                if _bgside == "inside":
                    _bgtw = 0
                mh = max(20, rh - 1)
                width_mode = blk.get("width_mode") or "auto"
                if width_mode == "fit":
                    tick_w, bar_w, gap, mw = _meter_fit_dims(n, rw, _bgtw)
                    mx = rx
                else:
                    tick_w, bar_w, gap = _bgtw, METER_BAR_W, METER_GAP
                    mw = _meter_layout(n, tick_w, bar_w, gap)
                    al = blk.get("align") or "left"
                    mx = rx + ((rw - mw) // 2 if al == "center"
                               else (rw - mw if al == "right" else 0))
                opacity_pct = max(10, min(100, int(blk.get("opacity") or 70)))
                _meter_tiles_at(mx, ry, mw, mh, n, peaks, holds,
                                 _bscale, opacity_pct, status, tiles,
                                 ch0=s0, tick_w=tick_w, bar_w=bar_w, gap=gap,
                                 grad=_bgrad, grad_side=_bgside)
            except Exception:
                continue
    return tiles or None


# ─── Bandeau ANC dans la cellule (opt-in par fenêtre, comme les VU-mètres) ───
# Rien n'est dessiné tant que l'utilisateur n'a coché aucune information ANC sur la fenêtre.
# Le texte change rarement (types/AFD/ST352) ou à la trame (timecode/sous-titres) → le rendu
# PIL+YUV est gaté par SIGNATURE (comme les horloges) ; seul le blend par petite bbox est
# per-frame. Défini APRÈS _format_anc_cell (qui vit avec les helpers ANC, plus bas) : appelé
# depuis la boucle, jamais à l'import.
def _anc_units():
    """Unités d'affichage ANC : (i, flags, rect|None). rect None = bandeau legacy (haut/bas de
    l'image, flags = la cfg de la fenêtre) ; rect = composant `anc` d'un modèle de PiP (flags =
    le composant, mêmes clés anc_*, géométrie libre)."""
    out = []
    for i, cfg in enumerate(FLUX_CONFIG):
        if cfg.get("hidden"):
            continue
        comps = _tpl_comps(cfg)
        if comps is None:
            if _anc_enabled(cfg):
                out.append((i, cfg, None))
            continue
        for comp in comps:
            if (isinstance(comp, dict) and comp.get("type") == "anc"
                    and _comp_visible(i, cfg, comp) and _anc_enabled(comp)):
                try:
                    out.append((i, comp, _comp_rect(FLUX_CONFIG[i], comp)))
                except Exception:
                    continue
    return out

# ─── Atlas de glyphes : le bandeau ANC composé par RECOPIE ───────────────────
# Un bandeau de timecode se re-dessine 25 fois par seconde — le timecode change, par définition.
# Chaque re-dessin coûtait ~8 ms mesurés (4,2 de rendu PIL + 4,1 de conversion RGBA→YUV) pour deux
# bandes de 830×34, soit 71 ns par pixel là où la conversion plein cadre du chrome tourne à 6,5.
# C'est du coût FIXE par appel (création d'image, petits tableaux numpy), pas du travail utile :
# le fond, le cadre et les libellés sont identiques d'une trame à l'autre, seuls les CHIFFRES
# changent, et on redessinait tout.
#
# L'atlas rend chaque caractère UNE fois, déjà converti en YUV et DÉJÀ COMPOSITÉ sur le fond du
# bandeau. C'est possible parce que ce fond est UNIFORME : « glyphe sur fond » ne dépend alors pas
# de la position. Composer une ligne devient une suite de recopies mémoire, sans arithmétique.
#
# ⚠ L'avance est forcée PAIRE : en 4:2:2 un glyphe posé à une abscisse impaire tomberait entre deux
# échantillons de chrominance, et la recopie décalerait la couleur d'un demi-pixel. Les chiffres
# tabulaires d'un timecode s'en accommodent naturellement (c'est ce que fait tout incrustateur) ;
# pour un texte à chasse variable, on retombe sur le rendu PIL complet plutôt que de le
# monospacer en douce.
_glyph_atlas = {{}}       # (police, taille, couleur, a_bg, hauteur) → {{"adv": n, "car": {{c: tuiles}}}}
_ATLAS_MAX = 12          # jeux distincts gardés — au-delà, la config change trop pour qu'il serve

def _atlas_pour(cle_police, taille, couleur, a_bg, h):
    """Jeu de glyphes pré-convertis pour ce style, ou None si la police est illisible."""
    cle = (cle_police, int(taille), couleur, int(a_bg), int(h))
    jeu = _glyph_atlas.get(cle)
    if jeu is not None:
        return jeu
    if len(_glyph_atlas) >= _ATLAS_MAX:
        _glyph_atlas.clear()          # style qui change sans cesse : l'atlas ne sert plus, on repart
    fnt = _overlay_font(cle_police, taille)
    if fnt is None:
        return None
    jeu = {{"adv": 0, "car": {{}}, "fnt": fnt, "couleur": couleur, "a_bg": a_bg, "h": int(h)}}
    _glyph_atlas[cle] = jeu
    return jeu

def _glyphe(jeu, c, pad):
    """Tuiles YUV du caractère `c`, glyphe DÉJÀ composité sur le fond du bandeau. None si échec."""
    t = jeu["car"].get(c)
    if t is not None:
        return t
    try:
        fnt = jeu["fnt"]
        # Avance commune à tous les caractères du jeu, forcée PAIRE (cf. bloc ci-dessus).
        if not jeu["adv"]:
            larg = max(int(fnt.getlength(x)) for x in "0123456789:;")
            jeu["adv"] = larg + (larg % 2)
        adv, h = jeu["adv"], jeu["h"]
        img = Image.new("RGBA", (adv, h), (0, 0, 0, jeu["a_bg"]))
        ImageDraw.Draw(img, "RGBA").text((0, pad), c, font=fnt, fill=jeu["couleur"])
        t = rgba_to_yuv(img)
        jeu["car"][c] = t
        return t
    except Exception:                                                      # noqa: BLE001
        return None

def _bandeau_par_glyphes(txt, larg, h, pad, cle_police, taille, couleur, a_bg):
    """Bandeau composé par recopie de glyphes → (y, u, v, a, a2), ou None si non applicable.

    Non applicable = police illisible, texte trop large pour l'avance fixe, ou un caractère dont
    le rendu échoue. Dans tous ces cas on rend la main au chemin PIL complet : mieux vaut payer
    8 ms que d'afficher un bandeau tronqué ou décalé."""
    jeu = _atlas_pour(cle_police, taille, couleur, a_bg, h)
    if jeu is None:
        return None
    g0 = _glyphe(jeu, "0", pad)
    if g0 is None:
        return None
    adv = jeu["adv"]
    if adv <= 0 or pad + len(txt) * adv > larg:
        return None            # ne tient pas : le rendu PIL sait rétrécir, pas nous
    # Fond uniforme, construit directement en YUV — ni PIL ni conversion.
    y = np.full((h, larg), _NEUTRAL if False else 0, dtype=_NP_DT)
    u = np.full((h // _CH, larg // _CW), _NEUTRAL, dtype=_NP_DT)
    v = np.full((h // _CH, larg // _CW), _NEUTRAL, dtype=_NP_DT)
    a = np.full((h, larg), a_bg, dtype=np.uint8)
    a2 = np.full((h // _CH, larg // _CW), a_bg, dtype=np.uint8)
    x = pad - (pad % _CW)      # l'origine aussi doit être paire
    for c in txt:
        t = _glyphe(jeu, c, pad)
        if t is None:
            return None
        gy, gu, gv, ga, ga2 = t
        if gy.shape != (h, adv):
            return None
        y[:, x:x + adv] = gy
        a[:, x:x + adv] = ga
        u[:, x // _CW:(x + adv) // _CW] = gu
        v[:, x // _CW:(x + adv) // _CW] = gv
        a2[:, x // _CW:(x + adv) // _CW] = ga2
        x += adv
    return y, u, v, a, a2


# Banc : ventilation d'un bake ANC — dessin PIL contre conversion RGBA→YUV, et surface traitée.
# 5,7 ms pour deux bandes de ~830x34 px est deux ordres de grandeur au-dessus du travail utile ;
# sans cette ventilation on ne sait pas lequel des deux étages est en cause.
_anc_prof = {{"total": 0.0, "pil": 0.0, "yuv": 0.0, "px": 0, "n": 0}}

def render_anc_tiles(now):
    """Une tuile YUV par unité ANC (bandeau de cellule ou composant de modèle), ou None."""
    tiles = []
    _ts_anc0 = time.time_ns()
    _anc_prof.update({{"yuv": 0.0, "px": 0, "n": 0}})
    for i, flags, rect in _anc_units():
        txt = _format_anc_cell(i, flags)
        if not txt:
            continue
        cfg = FLUX_CONFIG[i]
        if rect is None:
            g = _video_rect(cfg)
            vx, vy, vw, vh = g["vx"], g["vy"], g["vw"], g["vh"]     # DANS l'image, pas l'habillage
            size = max(10, min(28, vh // 18))
            pad = 4
            bh = size + 2 * pad
            # Position : « bottom » (défaut) = bas de l'image, « top » = haut.
            by = (vy + vh - bh) if (flags.get("anc_position") or "bottom") != "top" else vy
        else:
            vx, by, vw, bh = rect
            try:
                size = int(flags.get("font_size") or 0)
            except (TypeError, ValueError):
                size = 0
            if size <= 0:
                size = max(8, bh - 8)
            pad = max(2, (bh - size) // 2)
        # Police du composant ANC (modèle de PiP) ou du bandeau de cellule — repli DejaVu Bold.
        fnt = _overlay_font(flags.get("font") or "dejavu-sans-bold", size)
        bx0, by0 = max(0, vx), max(0, by)
        bx1, by1 = min(OUT_WIDTH, vx + vw), min(OUT_HEIGHT, by + bh)
        bx0 -= bx0 % _CW; by0 -= by0 % _CH
        if (bx1 - bx0) % _CW: bx1 = min(OUT_WIDTH, bx1 + (_CW - (bx1 - bx0) % _CW))
        if (by1 - by0) % _CH: by1 = min(OUT_HEIGHT, by1 + (_CH - (by1 - by0) % _CH))
        if bx1 <= bx0 or by1 <= by0:
            continue
        a_bg = max(0, min(100, int(flags.get("anc_opacity") or 60))) * 255 // 100
        # Un checksum invalide = métadonnée corrompue → texte en rouge (signal d'alarme).
        bad = "CRC!" in txt
        _coul = (255, 90, 90, 255) if bad else (235, 235, 235, 255)
        _al = (flags.get("align") or "left")
        # CHEMIN RAPIDE : composition par recopie de glyphes pré-convertis. Le fond du bandeau est
        # uniforme, donc chaque caractère peut être composité UNE fois sur ce fond et réutilisé à
        # n'importe quelle position. Un timecode ne coûte alors plus que quelques memcpy, au lieu
        # d'un rendu PIL + une conversion RGBA→YUV de toute la bande à chaque trame.
        _rapide = _bandeau_par_glyphes(txt, bx1 - bx0, by1 - by0, pad,
                                       flags.get("font") or "dejavu-sans-bold", size, _coul, a_bg)
        if _rapide is not None and _al == "left":
            _ts_y = time.time_ns()
            _anc_prof["yuv"] += (time.time_ns() - _ts_y) / 1e6
            _anc_prof["px"] += (bx1 - bx0) * (by1 - by0)
            _anc_prof["n"] += 1
            _anc_prof["glyphes"] = _anc_prof.get("glyphes", 0) + 1
            oy, ou, ov, oa, oa2 = _rapide
            tiles.append((bx0, by0, bx1, by1, oy, ou, ov, oa, oa2))
            continue
        _anc_prof["pil_n"] = _anc_prof.get("pil_n", 0) + 1
        tile = Image.new("RGBA", (bx1 - bx0, by1 - by0), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile, "RGBA")
        d.rectangle([0, 0, bx1 - bx0 - 1, by1 - by0 - 1], fill=(0, 0, 0, a_bg))
        # ALIGNEMENT du bandeau. Les composants texte passent par _draw_text_overlay, qui gère
        # déjà `align` ; l'ANC dessine sa propre tuile et écrivait en dur en haut à GAUCHE. On
        # calcule ici l'abscisse à partir de la largeur mesurée du texte. Bornée à `pad` : un
        # texte plus large que la tuile reste lisible par la gauche au lieu de déborder à gauche.
        _tx = pad
        if _al in ("center", "right"):
            try:
                _tw = d.textlength(txt, font=fnt)
            except Exception:
                _tw = 0
            _avail = (bx1 - bx0) - 2 * pad
            _tx = max(pad, pad + int((_avail - _tw) / (2 if _al == "center" else 1)))
        d.text((_tx, pad), txt, font=fnt, fill=_coul)
        _ts_y = time.time_ns()
        oy, ou, ov, oa, oa2 = rgba_to_yuv(tile)
        _anc_prof["yuv"] += (time.time_ns() - _ts_y) / 1e6
        _anc_prof["px"] += (bx1 - bx0) * (by1 - by0)
        _anc_prof["n"] += 1
        tiles.append((bx0, by0, bx1, by1, oy, ou, ov, oa, oa2))
    _anc_prof["total"] = round((time.time_ns() - _ts_anc0) / 1e6, 2)
    _anc_prof["yuv"] = round(_anc_prof["yuv"], 2)
    _anc_prof["pil"] = round(_anc_prof["total"] - _anc_prof["yuv"], 2)
    return tiles or None

def _anc_sig():
    """Signature des bandeaux ANC (gate du re-rendu PIL/YUV) — vide si aucune unité active.
    L'ALIGNEMENT en fait partie : il change les pixels sans changer le texte, donc l'omettre
    ferait ignorer un changement d'alignement appliqué à chaud jusqu'à la prochaine variation
    de la valeur ANC (parfois jamais, sur une source au timecode figé)."""
    return tuple((_format_anc_cell(i, flags), flags.get("align") or "left")
                 for i, flags, _r in _anc_units())


# ─── Historique vidéo / audio (« que s'est-il passé sur cette source ? ») ────────────────
# DEUX outils, disponibles CHACUN sous deux formes (même code de rendu) :
#   • composant de MODÈLE DE PIP (type `video_history` / `audio_history`, géométrie normalisée
#     à la cellule — la source est celle de la fenêtre) ;
#   • bloc libre du MUR (CONFIG.video_history_blocks[] / audio_history_blocks[], géométrie en
#     fractions du MUR ENTIER, source CÂBLÉE en propre — même motif que meter_blocks).
#
# HISTORIQUE VIDÉO : une vignette par SECONDE (bande) + un RUBAN D'ÉVÉNEMENTS (gel / noir /
# perte de signal) sous la bande. Échantillonnage dans un THREAD dédié (VH_TICK_S), JAMAIS dans
# la boucle de mix : 5 relevés/s sur un proxy pyramide déjà réduit, chaque relevé = un gather de
# 96×54 px (VH_THUMB_*) — pas un resize de trame. La bande n'est RECOMPOSÉE (PIL) qu'à l'arrivée
# d'une vignette (1 Hz) ou à une transition d'événement (cache `_hist_cache`, motif du cache
# statique des VU-mètres).
#   ★ CAPTURE À L'INSTANT DE L'ÉVÉNEMENT : à l'ENTRÉE en gel / noir / perte de signal, la vignette
#     est capturée IMMÉDIATEMENT et ÉPINGLÉE (`pinned`) dans sa case temporelle — l'échantillonnage
#     régulier ne l'écrase plus. La bande montre donc l'image SUR LAQUELLE ça s'est figé (pour la
#     perte de signal : la dernière image VALIDE, cf. `good`), pas une image quelconque de la seconde.
#   Statuts : RÉUTILISE `_tile_status` (gel/perte détectés par la boucle de mix) dès que la source
#   est celle d'une fenêtre ; pour un bloc de mur dont la source n'alimente AUCUNE fenêtre, la MÊME
#   règle (frame_index qui n'avance plus depuis FREEZE_DETECT_S / aucun grain lisible) est appliquée
#   à notre propre reader. Le NOIR (absent du moteur) est mesuré sur la vignette (déjà réduite) :
#   luma moyenne ≤ VH_BLACK_MEAN ET max ≤ VH_BLACK_MAX.
#
# HISTORIQUE AUDIO : enveloppe des crêtes (une colonne = min/max de sa tranche, symétrique) +
# SATURATION persistante (colonne rouge) + SILENCE (bande grisée). Échantillonné dans son propre
# thread à AH_TICK_S, ring de HIST_MAX_S s.
HIST_DURATIONS = (10, 30, 60, 120)     # durées offertes (s) — défaut 30
HIST_MAX_S     = 120                   # profondeur des rings (= plus longue durée offerte)
VH_TICK_S      = 0.2                   # période d'échantillonnage vidéo (5 Hz) → ruban à 200 ms près
# Granularité de l'HORODATAGE de l'échelle dans la signature de la frise. À la seconde, chaque
# unité se re-boulangeait 1×/s — sur un mur qui en porte treize, cela faisait 13 boulangeages par
# seconde, chacun coûtant 3 à 5 ms et pointant à 34. Ces à-coups tombent sur les MÊMES cœurs
# épinglés que la boucle de composition et la stallent : mesuré le 2026-08-08, les pics de `own`
# montaient à 52 ms pour une moyenne de 15, tous les étages enflant ensemble — signature d'un
# blocage global, pas d'une opération chère.
# Sur une frise qui couvre 30 à 120 s, une échelle vieille de quelques secondes est invisible.
# Même raisonnement et même valeur que AH_RECOMPOSE_S côté audio, qui l'appliquait déjà.
VH_LABEL_S     = 5.0                   # …horodatage de l'échelle rafraîchi à 0,2 Hz (5 s)
# ★ CADENCE DES VIGNETTES = DÉDUITE DE LA PLACE (0.39.0 — plus de « une par seconde » figée, qui
# donnait des vignettes-timbres illisibles et 1 gather/s pour rien). On calcule combien de vignettes
# tiennent dans la bande SANS DÉFORMATION (largeur d'une vignette = hauteur de bande × ratio de la
# SOURCE) ni chevauchement ; l'intervalle en découle : step = durée / nb_vignettes (≥ 1 s). Une
# frise 30 s de 1720×190 sur du 16:9 → ~5 vignettes à ~6 s d'intervalle. Un bloc plus étroit, plus
# haut, ou une durée de 120 s s'adaptent tout seuls. Le prélèvement suit la MÊME cadence (30× moins
# de gathers qu'en 0.37.0) ; seules la détection d'événement et la sonde de noir restent à 5 Hz.
VH_CELL_MIN_W  = 48                    # plancher : sous ~48 px de large une vignette n'apprend plus rien
VH_CELL_GAP    = 2                     # garde entre deux vignettes (px)
VH_THUMB_H     = 144                   # hauteur de vignette STOCKÉE ; la largeur suit le RATIO SOURCE
VH_THUMB_W_MAX = 256                   # (bornes de largeur : anamorphoses/portrait restent sains)
VH_THUMB_W_MIN = 24
# ★ 0.40.0 — DÉFINITION REMONTÉE (54 → 144 px de haut). Les cases d'une frise pleine largeur font
# ~160 px de haut : une vignette de 54 px y était AGRANDIE ×3, au plus proche voisin, à partir d'un
# prélèvement déjà crénelé. On stockait donc du flou pixellisé et on l'étirait. 144 px couvre la
# hauteur de case usuelle → le redimensionnement d'affichage devient marginal (cf. _vh_render).
# Coût : le prélèvement et la recomposition sont TOUS DEUX hors de la boucle de composition
# (thread d'échantillonnage + thread boulanger) — vérifié au mur 333, own_latency inchangé.
VH_PROBE_W     = 32                    # sonde de LUMA (noir) — prélevée à 5 Hz, plan Y seul : 576
VH_PROBE_H     = 18                    # points, ~10× moins cher qu'une vignette complète
VH_BLACK_MEAN  = 16.0                  # luma moyenne (échelle 8 bits) sous laquelle l'image est « noire »
VH_BLACK_MAX   = 48.0                  # …et pic de luma sous lequel on ne voit RIEN (ni mire, ni logo)
VH_RESCAN_S    = 5.0                   # re-scan des proxies pyramide de la source
# Âge maximal toléré de la dernière écriture producteur (now_tai − lastWriteTime) avant de
# considérer le Reader de l'ÉCHANTILLONNEUR décroché et de le reconnecter — cf. _vh_sample. Même
# valeur et même critère que `STALE_REOPEN_MS` côté boucle de composition ; redéclaré ICI, avec les
# autres constantes de l'échantillonneur, parce que `STALE_REOPEN_MS` est défini bien plus bas dans
# le module, APRÈS le démarrage du thread `_vhist_loop` — le premier relevé lèverait un NameError
# aussitôt avalé par son `except`, et le garde-fou serait mort-né en silence.
VH_STALE_REOPEN_MS = 5000.0
AH_TICK_S      = 0.02                  # période d'échantillonnage audio (50 Hz, = cadence trame)
AH_COL_PX      = 1                     # ENVELOPPE : une colonne PAR PIXEL (finesse d'origine). La
                                       # finesse ne coûte RIEN (les crêtes sont déjà calculées ; le
                                       # coût était dans les écritures numpy stridées, cf. _hist_fill).
                                       # Cran de repli si un jour on veut épaissir le trait.
# ⚠ NE PAS RALENTIR pour gagner de la cadence — essayé et ÉCARTÉ le 2026-08-08. Cette valeur est
# bien la fréquence de boulangeage dominante d'un mur (à 0,2 s, deux frises audio font 10 passes
# par seconde à elles seules). Mais la passer à 1 Hz n'a rapporté qu'UNE image par seconde
# (43,7 fps contre 42,6) : les à-coups sont devenus cinq fois plus rares SANS DEVENIR PLUS PETITS —
# les pics de `own` sont restés à 42-45 ms, la valeur d'un seul boulangeage. Ce n'est donc pas la
# FRÉQUENCE des passes qui coûte la cadence, c'est le fait qu'UNE passe stalle la composition
# pendant des dizaines de millisecondes. Ralentir la frise la rendrait saccadée pour rien.
AH_RECOMPOSE_S = 0.2                   # …redessinée au changement de colonne, 5 Hz max (~2,3 ms/passe)
AH_MAX         = int(HIST_MAX_S / AH_TICK_S)   # 6000 relevés (120 s) par flux audio
AH_LANE_MIN_PX = 10                    # hauteur MINIMALE d'une piste de canal : en dessous, on
                                       # replie sur une piste unique (max des canaux) plutôt que
                                       # d'empiler des traits d'un pixel illisibles.
# NUMÉROTATION DES CANAUX (bandeau tout à gauche de la frise audio) — « quand il y a la place »
# est un SEUIL DUR, pas une intention : en dessous, on n'écrit RIEN (jamais de chiffre tronqué,
# jamais d'enveloppe rognée). Deux conditions, l'une de hauteur, l'autre de largeur.
AH_NUM_MIN_PX   = 12                   # hauteur mini d'une DEMI-bande pour porter un chiffre
                                       # (ImageFont.load_default() ≈ 11 px de haut + 1 px d'air)
AH_NUM_MAX_FRAC = 12                   # le bandeau ne doit pas manger plus de 1/12 de la LARGEUR
# SATURATION (critère retenu, défendable et documenté) : un échantillon est « pleine échelle » si
# |x| ≥ AH_CLIP_LEVEL (≈ −0,009 dBFS, soit le dernier LSB en 20 bits) ; une colonne est marquée
# SATURÉE si un canal suivi présente AH_CLIP_RUN échantillons pleine échelle CONSÉCUTIFS — la
# règle classique des détecteurs de clip numériques (3 FS d'affilée = écrêtage, contre 1 seul qui
# peut être un pic légitime). Le marqueur est PERSISTANT : porté par la colonne, il reste à l'écran
# tant que la colonne est dans la fenêtre de temps (une saturation de 3 ms reste visible 30 s).
AH_CLIP_LEVEL  = 0.999
AH_CLIP_RUN    = 3
# Couleurs des événements (ruban + liseré de la vignette épinglée) — mêmes teintes que les
# badges de statut du produit (rouge=perte, ambre=gel, indigo=noir).
_HIST_EVT_RGB  = {{"nosignal": (236, 72, 60), "freeze": (240, 184, 44), "black": (108, 132, 236)}}
_HIST_BG       = (14, 16, 20)          # fond des deux frises
_HIST_WAVE     = (96, 210, 140)        # enveloppe audio (vert)
_HIST_SILENCE  = (58, 60, 68)          # plage silencieuse (gris)
_HIST_CLIP     = (236, 72, 60)         # colonne saturée (rouge)

_hist_lock  = threading.Lock()
_vh         = {{}}   # nom de flux vidéo → état d'historique (ring de vignettes + ruban)
_ah         = {{}}   # nom de flux audio → état d'historique (rings de crêtes/saturation)
_hist_cache = {{}}   # clé d'unité → (signature, [tuiles YUV]) — LU/ÉCRIT par la boucle de compo SEULE

# ★ 0.40.0 — LA RECOMPOSITION DES FRISES SORT DE LA BOUCLE DE COMPOSITION.
# Le re-bake d'une frise (dessin RGBA + conversion YUV) coûte 6 à 27 ms (mesuré `hist_bake_ms.max`
# = 26,8 ms sur le mur 333) et tombait ENTIER dans la trame qui le déclenchait, ~5×/s : 5 trames
# perdues par seconde sur 50 → le mur plafonnait à ~45 fps sans jamais être compute-bound en
# moyenne (own_latency 11 ms pour un budget de 20). Il n'était pas lent : il était ASSOMMÉ
# PÉRIODIQUEMENT. Dégrader la frise n'aurait rien réglé (le pic aurait juste été plus petit).
# Modèle : un thread BOULANGER fabrique la tuile hors trame ; la boucle de compo ne fait plus que
# (a) déposer une DEMANDE quand la signature change, (b) ramasser la tuile PRÊTE, (c) blender.
#   • Cohérence : une tuile est un tuple de tableaux numpy NEUFS, publié par UNE affectation de
#     dict (atomique sous le GIL) — la boucle ne peut jamais voir une tuile à moitié écrite.
#   • Pas de course sur `_hist_cache` : le boulanger n'y touche PAS (il publie dans `_hist_ready`,
#     que la boucle draine elle-même). Un seul écrivain par structure.
#   • Vol de CPU : le conteneur a 3 cœurs (cpuset) pour UN thread de compo → il reste de la place.
#     Le vrai risque est le GIL : un boulanger qui le garde 5 ms (switchinterval par défaut) ferait
#     attendre la compo. On descend donc l'intervalle de commutation à 0,5 ms — la compo ne peut
#     plus être bloquée plus longtemps que ça, et le pic de 27 ms disparaît de la trame.
# Entre la demande et la livraison (une trame ou deux), la boucle réutilise la tuile PÉRIMÉE : sur
# une frise de 30-120 s, 20-40 ms de retard sont rigoureusement invisibles.
sys.setswitchinterval(0.0005)
_hb_cv     = threading.Condition()   # garde `_hist_want` / `_hist_ready`
_hist_want = {{}}   # clé d'unité → demande de bake (sig, kind, unit, rect, cfg_src, name, now)
_hist_ready = {{}}  # clé d'unité → (signature, [tuiles YUV]) publiées par le boulanger
_hist_bake_ctr = [0]   # nombre de bakes livrés (→ bakes_per_s.hist)

_hist_errs = {{}}   # (clé d'unité, signature d'erreur) → {{kind, rect, err, n, t}} — THROTTLÉ

def _hist_warn(key, kind, rect, err, trace=False):
    """Échec de rendu d'une frise : loggé UNE fois par (unité, signature d'erreur) — pas 50×/s —
    et exposé sur :8080 (`history_errors`). Une frise qui ne peut pas se rendre doit le DIRE."""
    k = (key, err)
    e = _hist_errs.get(k)
    now = time.time()
    if e is None:
        e = {{"kind": kind, "rect": list(rect), "err": err, "n": 0, "t": now}}
        _hist_errs[k] = e
        log(f"multiview: HISTORIQUE {{kind}} {{key}} rect={{rect}} — RENDU IMPOSSIBLE : {{err}}",
            "warning")
        if trace:
            try:
                import traceback as _tb
                _tb.print_exc()
            except Exception:
                pass
    e["n"] += 1
    e["t"] = now
    metrics["history_errors"] = [
        {{"unit": "/".join(str(x) for x in kk[0]), "kind": vv["kind"], "rect": vv["rect"],
          "err": vv["err"], "count": vv["n"]}}
        for kk, vv in _hist_errs.items()]

def _hist_dur(cfg):
    """Durée (s) d'un composant/bloc d'historique, bornée aux durées offertes (défaut 30)."""
    try:
        d = int(cfg.get("duration") or 30)
    except (TypeError, ValueError):
        return 30
    return d if d in HIST_DURATIONS else min(HIST_DURATIONS, key=lambda v: abs(v - d))

def _hist_rect(cfg, comp):
    """Rectangle d'une frise, SNAPPÉ sur la grille chroma (origine ET taille multiples de
    _CW/_CH). La frise est composée dans son propre RGBA local puis blendée telle quelle
    (cf. _hist_tile) : un rect à coordonnée impaire imposerait un recadrage — et c'est ce
    recadrage qui, mal borné, rendait la frise invisible. On aligne donc une bonne fois ici."""
    rx, ry, rw, rh = _comp_rect(cfg, comp)
    x0 = rx - (rx % _CW); y0 = ry - (ry % _CH)
    x1 = min(OUT_WIDTH, rx + rw); y1 = min(OUT_HEIGHT, ry + rh)
    x1 -= (x1 - x0) % _CW
    y1 -= (y1 - y0) % _CH
    return x0, y0, max(_CW, x1 - x0), max(_CH, y1 - y0)

def _hist_units(kind):
    """Unités d'historique à rendre : [(key, unit, rect, cfg_source, win_idx|None)].
    `unit` = le composant de modèle OU le bloc de mur (mêmes clés) ; `cfg_source` = le dict qui
    porte la SOURCE (la fenêtre pour un composant, le bloc lui-même pour un bloc de mur) ;
    `win_idx` = index de fenêtre (réutilisation de `_tile_status`) ou None pour un bloc."""
    ctype = "video_history" if kind == "video" else "audio_history"
    blocks = VHIST_BLOCKS if kind == "video" else AHIST_BLOCKS
    out = []
    for i, cfg in enumerate(FLUX_CONFIG):
        if cfg.get("hidden"):
            continue
        for comp in (_tpl_comps(cfg) or ()):
            if not (isinstance(comp, dict) and comp.get("type") == ctype):
                continue
            if not _comp_visible(i, cfg, comp):
                continue
            try:
                out.append((("w", i, str(comp.get("id") or ctype)), comp, _hist_rect(cfg, comp), cfg, i))
            except Exception:
                continue
    full_cfg = {{"x": 0, "y": 0, "w": OUT_WIDTH, "h": OUT_HEIGHT}}
    for j, blk in enumerate(blocks):
        if not isinstance(blk, dict) or blk.get("hidden"):
            continue
        try:
            out.append(((kind[0] + "b", j, ""), blk, _hist_rect(full_cfg, blk), blk, None))
        except Exception:
            continue
    return out

# ─── Historique VIDÉO : échantillonnage ──────────────────────────────────────
# COÛT (tranché en 0.39.0 — « rapide et visuel, la qualité n'a aucune importance ») :
#   • AUCUN prélèvement dans la boucle de mix : tout se passe dans le thread _vhist_loop ;
#   • la vignette n'est capturée qu'à la cadence des CASES (≈ 6 s sur une frise 30 s de 1720 px —
#     cf. _vh_slots), plus une capture hors-cadence à chaque ÉVÉNEMENT (épinglage) : ~30× moins de
#     prélèvements qu'en 0.37.0 (qui capturait 5×/s) ;
#   • le prélèvement lit le PLUS PETIT proxy pyramide disponible (_vh_pick_path) — jamais la pleine
#     résolution si un proxy existe, même mal dimensionné (crénelage assumé) ;
#   • un prélèvement = un gather de tw×th POINTS (nearest, ~60×54) sur une vue zéro-copie — pas un
#     resize de trame. Mesuré (numpy, plan 1920×1080 8 bits, cf. rapport) : ≈ 0,25 ms la vignette,
#     ≈ 0,04 ms la sonde de luma → à une vignette / 6 s + 5 sondes/s, < 0,03 % d'un cœur ;
#   • aucune taille sur-mesure n'est réclamée à la pyramide (proxy_needs) : un proxy dédié tournerait
#     à 50 Hz pour servir un relevé toutes les 6 s. En revanche le proxy LU est publié dans
#     `proxy_read` (cf. _vh_read_proxies) — sans quoi l'orchestrateur le croirait orphelin.

def _vh_read_proxies():
    """Proxies pyramide RÉELLEMENT lus par l'échantillonneur des frises vidéo (≠ flux pleins)."""
    try:
        with _hist_lock:
            paths = [st.get("path") or "" for st in _vh.values()]
    except Exception:
        return set()
    return {{p for p in paths if p and "__" in p}}

def _vh_new():
    # `cells` = ring de vignettes DATÉES (t absolu) — plus de case « une par seconde » : chaque
    # frise range ces vignettes dans SES propres cases temporelles (nb déduit de sa géométrie).
    return {{"src": None, "path": "", "scan_t": 0.0,
             "cells": deque(maxlen=HIST_MAX_S + 32),  # {{t, img (RGB uint8), pinned, evt}}
             "evt": deque(maxlen=int(HIST_MAX_S / VH_TICK_S) + 8),   # (t, code) — ruban
             "last_fi": None, "last_fi_t": 0.0, "status": "",
             "tw": VH_THUMB_W_MAX, "th": VH_THUMB_H, "aspect": 16.0 / 9.0,
             "cell_t": 0.0, "thumb": None, "good": None, "ver": 0}}

def _vh_thumb_dims(src):
    """Dimensions de STOCKAGE d'une vignette : hauteur fixe, largeur au RATIO DE LA SOURCE (une
    vignette n'est JAMAIS anamorphosée — ni au stockage ni au rendu). Entrelacé natif : un grain =
    un CHAMP (½ hauteur) → le ratio de l'IMAGE est in_w / (2·in_h)."""
    in_w = max(1, int(src.get("in_w") or 1)); in_h = max(1, int(src.get("in_h") or 1))
    if src.get("interlaced"):
        in_h *= 2
    ar = float(in_w) / float(in_h)
    th = max(8, min(VH_THUMB_H, in_h))          # jamais d'UPSCALE : un proxy minuscule est pris tel quel
    tw = max(VH_THUMB_W_MIN, min(VH_THUMB_W_MAX, in_w, int(round(th * ar))))
    return tw, th, float(tw) / float(th)

def _vh_close(st):
    src = st.get("src")
    if src is not None:
        _close_source(src)
    st["src"] = None; st["path"] = ""
    try: _gc_mxl()
    except Exception: pass

def _vh_pick_path(name):
    """Source la MOINS CHÈRE pour une vignette : le PLUS PETIT proxy pyramide disponible, POINT.
    Une vignette d'historique fait ~60 px de large et répond à « c'était quoi, à ce moment-là ? » :
    sa QUALITÉ N'A AUCUNE IMPORTANCE (cadrage utilisateur). On préfère donc systématiquement un
    proxy trop petit (crénelé, tant pis) au flux PLEIN — jamais l'inverse. Le flux plein n'est lu
    que si la pyramide ne produit rien pour cette source. Aucune taille sur-mesure n'est réclamée
    (proxy_needs) : faire produire un proxy dédié à 50 Hz pour servir un relevé toutes les ~6 s
    coûterait au nœud bien plus cher que le prélèvement qu'il économise."""
    best = name; ba = None
    for p in (_scan_proxies_for({{"path": "/dev/shm/" + name}}) or ()):
        w = int(p.get("w") or 0); h = int(p.get("h") or 0)
        if w >= VH_THUMB_W_MIN and h >= 16 and (ba is None or w * h < ba):
            ba = w * h; best = (p.get("path") or "").removeprefix("/dev/shm/")
    return best or name

def _vh_planes(src, view):
    """Vues (zéro-copie) des plans Y/U/V d'un grain."""
    in_w, in_h = src["in_w"], src["in_h"]
    yb  = in_w * in_h * _BPS
    uvb = (in_w // _CW) * (in_h // _CH) * _BPS
    y = view[:yb].view(_NP_DT).reshape(in_h, in_w)
    u = view[yb:yb + uvb].view(_NP_DT).reshape(in_h // _CH, in_w // _CW)
    v = view[yb + uvb:yb + 2 * uvb].view(_NP_DT).reshape(in_h // _CH, in_w // _CW)
    return y, u, v, in_w, in_h

def _vh_luma(src, view):
    """Sonde de LUMA (moyenne, max) sur l'échelle 8 bits — plan Y SEUL, VH_PROBE_W×VH_PROBE_H
    points (576). Prélevée à CHAQUE tick (5 Hz) pour la détection de NOIR : ~10× moins chère
    qu'une vignette complète, qui n'est capturée qu'à la cadence des cases (cf. _vh_sample)."""
    y, _u, _v, in_w, in_h = _vh_planes(src, view)
    ry = (np.arange(VH_PROBE_H) * in_h) // VH_PROBE_H
    rx = (np.arange(VH_PROBE_W) * in_w) // VH_PROBE_W
    ty = y[np.ix_(ry, rx)].astype(np.float32) / _SCALE
    return float(ty.mean()), float(ty.max())

def _vh_thumb(src, view, tw, th):
    """Vignette RGB (th×tw×3), au RATIO DE LA SOURCE (cf. _vh_thumb_dims) — jamais anamorphosée.
    ★ 0.40.0 — VRAI REDIMENSIONNEMENT FILTRÉ (PIL) au lieu du gather de tw×th POINTS (plus proche
    voisin) d'avant. Le sous-échantillonnage brutal d'un 1920×1080 vers 160×54 crénelait tout :
    moiré sur les mires, texte des bandeaux illisible, scintillement d'une vignette à l'autre. On
    avait dégradé la qualité en croyant que le PRÉLÈVEMENT faisait tomber le mur — c'était FAUX :
    le coupable était le PIC de recomposition, sorti de la trame en 0.40.0. Le prélèvement vit dans
    le THREAD d'échantillonnage (jamais dans la boucle de mix, et à ~1/6 Hz par source) : la
    qualité ne coûte donc RIEN à la trame.
    Chaque plan est réduit SÉPARÉMENT par PIL en 8 bits ('L') — LANCZOS sur la luma (qui porte le
    détail), BILINEAR sur la chroma (déjà molle, sous-échantillonnée) — et la conversion YUV→RGB
    se fait sur la PETITE image (tw×th), pas sur la trame. PIL relâche le GIL pendant le
    rééchantillonnage : le thread de composition n'est pas retenu."""
    y, u, v, in_w, in_h = _vh_planes(src, view)
    def _dn(p, flt, gap=None):
        if p.dtype != np.uint8:                 # 10/12 bits → échelle 8 bits
            p = (p // _SCALE).astype(np.uint8)
        im = Image.fromarray(np.ascontiguousarray(p), "L")
        return np.asarray(im.resize((tw, th), flt, reducing_gap=gap), dtype=np.float32)
    # `reducing_gap=3` : PIL pré-réduit en BOX (aire) jusqu'à 3× la cible avant la convolution
    # LANCZOS. Mesuré sur un 1920×1080 : 11,7 → 5,3 ms pour un écart MOYEN de 0,28/255 avec le
    # LANCZOS plein (max 7) — invisible. Deux fois moins de CPU pour la même image.
    ty = _dn(y, Image.LANCZOS, 3.0)
    tu = _dn(u, Image.BILINEAR) - 128.0
    tv = _dn(v, Image.BILINEAR) - 128.0
    r = ty + 1.402 * tv
    g = ty - 0.344136 * tu - 0.714136 * tv
    b = ty + 1.772 * tu
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)

def _vh_sample(name, win, now, step):
    """Un relevé (VH_TICK_S) d'une source vidéo suivie : statut (5 Hz) + vignette (à la cadence
    des cases, `step` — cf. VH_CELL_MIN_W) + ★ ÉPINGLAGE À L'INSTANT DE L'ÉVÉNEMENT (préservé :
    une transition capture la vignette IMMÉDIATEMENT, hors cadence)."""
    with _hist_lock:
        st = _vh.get(name)
        if st is None:
            st = _vh_new(); _vh[name] = st
    if st["src"] is None or (now - st["scan_t"]) > VH_RESCAN_S:
        want = _vh_pick_path(name)
        if st["src"] is None or st["path"] != want:
            _vh_close(st)
            st["src"] = open_source({{"path": want}})
            st["path"] = want if st["src"] is not None else ""
            if st["src"] is not None:
                st["tw"], st["th"], st["aspect"] = _vh_thumb_dims(st["src"])
        st["scan_t"] = now
    got = None
    if st["src"] is not None:
        try:
            got = st["src"]["reader"].get_latest()
        except Exception:
            got = None
    status = ""
    if got is None:
        status = "nosignal"
        st["thumb"] = None
        st["last_fi"] = None
        _vh_close(st)          # reader périmé/flux disparu → rouvert au prochain relevé
    else:
        fi = got[0]
        # ★ READER DÉCROCHÉ (2026-07-28) — MÊME garde-fou que la boucle de composition, qui manquait
        # ICI. Un flux amont RECRÉÉ SOUS LE MÊME NOM (moteur 2110 realigné, producteur redéployé)
        # laisse ce handle sur l'ANCIEN ring : ses grains restent LISIBLES, donc `got is None` ne se
        # déclenche jamais ; et comme `_vh_pick_path` rend le même chemin, la ré-ouverture
        # périodique ne se déclenche pas non plus (`st["path"] == want`). L'échantillonneur relit
        # alors éternellement le même grain.
        # Symptôme observé en prod (Horace, frises du mur 361) : vignettes FIGÉES + bande
        # d'événement « freeze » jaune sous la frise, alors que la boucle de mix — qui a, ELLE, sa
        # reconnexion depuis la 0.45.0 — affichait la source bien vivante (latence d'entrée 0,6 ms).
        # Deux Readers sur la même source rendant des verdicts opposés : signature du handle périmé.
        # CRITÈRE : `lastWriteTime`, le seul fiable et sanctionné par la spec — l'index figé NE
        # SUFFIT PAS (parité entrelacée, grille FLOW), c'est écrit noir sur blanc dans le garde-fou
        # de la boucle de composition. `lw = 0` = information indisponible (producteur qui ne
        # maintient pas lastWriteTime) → on ne conclut rien et on laisse les garde-fous historiques.
        try:
            _lw = st["src"]["reader"].last_write_time()
        except Exception:
            _lw = 0
        if _lw and (bobimxl.now_tai() - _lw) / 1e6 > VH_STALE_REOPEN_MS:
            _age_s = (bobimxl.now_tai() - _lw) / 1e9
            _vh_close(st)                 # rouvert au prochain relevé (200 ms)
            st["last_fi"] = None; st["last_fi_t"] = now
            log(f"multiview: frise {{name}} — Reader décroché (aucune écriture producteur depuis "
                f"{{_age_s:.1f}} s), reconnexion", "warning")
            return
        if fi != st.get("last_fi"):
            st["last_fi"] = fi; st["last_fi_t"] = now
        mean_y = max_y = 0.0; lum_ok = False
        try:
            mean_y, max_y = _vh_luma(st["src"], got[2])   # sonde 32×18 (plan Y) — à CHAQUE tick
            lum_ok = True
        except Exception:
            lum_ok = False
        # Statut : celui DÉJÀ calculé par la boucle de mix si la source alimente une fenêtre
        # (aucune seconde détection) ; sinon MÊME règle appliquée à notre reader. Le noir n'est
        # pas un statut du moteur → mesuré sur la sonde de luma (toujours à 200 ms près).
        ws = _tile_status.get(win) if win is not None else None
        if ws in ("nosignal", "freeze"):
            status = ws
        elif FREEZE_DETECT_S > 0 and (now - st["last_fi_t"]) > FREEZE_DETECT_S:
            status = "freeze"
        elif lum_ok and mean_y <= VH_BLACK_MEAN and max_y <= VH_BLACK_MAX:
            status = "black"
    st["evt"].append((now, status))
    prev = st.get("status") or ""
    st["status"] = status
    trans = (status != prev and status != "")
    periodic = (now - float(st.get("cell_t") or 0.0)) >= step
    # Vignette prélevée SEULEMENT quand elle sert : nouvelle case, ou transition à épingler.
    if got is not None and (periodic or trans):
        try:
            img = _vh_thumb(st["src"], got[2], st["tw"], st["th"])
        except Exception:
            img = None
        if img is not None:
            st["thumb"] = img; st["good"] = img
    if trans:
        # ★ TRANSITION : on ÉPINGLE l'image de l'instant — celle sur laquelle ça s'est figé
        # (gel/noir) ou la dernière VALIDE (perte de signal). Case DATÉE : elle tombera dans la
        # case temporelle de l'événement quelle que soit la cadence de base de la frise.
        img = st.get("good") if status == "nosignal" else st.get("thumb")
        st["cells"].append({{"t": now, "img": img, "pinned": True, "evt": status}})
        st["cell_t"] = now
    elif periodic:
        st["cells"].append({{"t": now, "img": st.get("thumb") if got is not None else None,
                            "pinned": False, "evt": ""}})
        st["cell_t"] = now
    if trans or periodic or status != prev:
        st["ver"] += 1                       # nouvelle case / transition → bande + ruban recomposés

def _vhist_loop():
    """Thread d'échantillonnage vidéo (VH_TICK_S). Aucune trame prélevée dans la boucle de mix."""
    while True:
        time.sleep(VH_TICK_S)
        try:
            units = _hist_units("video")
            wanted = {{}}
            steps = {{}}
            for _k, unit, rect, cfg_src, wi in units:
                nm = (cfg_src.get("path") or "").removeprefix("/dev/shm/")
                if not nm:
                    continue
                wanted.setdefault(nm, wi)
                # Cadence de prélèvement = pas de case de la frise la plus EXIGEANTE de cette
                # source (plusieurs frises peuvent partager le ring : la plus fine gagne).
                with _hist_lock:
                    _stv = _vh.get(nm)
                    _ar = float(_stv["aspect"]) if _stv else 16.0 / 9.0
                _st = _vh_slots(rect, _hist_dur(unit), _as_bool(unit.get("events", True), True), _ar)[1]
                steps[nm] = min(steps.get(nm, 1e9), _st)
            with _hist_lock:
                dead = [nm for nm in _vh if nm not in wanted]
                for nm in dead:
                    _vh_close(_vh.pop(nm))
            # Carte source → fenêtre : un BLOC de mur câblé sur la source d'une fenêtre réutilise
            # AUSSI le statut de cette fenêtre (aucune détection en double).
            wmap = {{}}
            for i, cfg in enumerate(FLUX_CONFIG):
                if cfg.get("hidden"):
                    continue
                p = (cfg.get("path") or "").removeprefix("/dev/shm/")
                if p and p not in wmap:
                    wmap[p] = i
            now = time.time()
            for nm, wi in wanted.items():
                try:
                    _vh_sample(nm, wmap.get(nm, wi), now, max(1.0, steps.get(nm, 1.0)))
                except Exception:
                    pass
        except Exception:
            pass

# ─── Historique AUDIO : échantillonnage ──────────────────────────────────────

def _ah_new():
    return {{"ar": None, "name": "", "w": 0, "n": 0, "last_t": 0.0, "last_head": None,
             "t":    np.zeros(AH_MAX, dtype=np.float64),
             "pk":   np.full((AH_MAX, A_CHANNELS_MAX), METER_MIN_DB, dtype=np.float32),
             "clip": np.zeros((AH_MAX, A_CHANNELS_MAX), dtype=np.uint8)}}

def _ah_push(st, t, pk, clip):
    w = st["w"]
    st["t"][w] = t
    st["pk"][w] = pk
    st["clip"][w] = clip
    st["w"] = (w + 1) % AH_MAX
    st["n"] = min(AH_MAX, st["n"] + 1)

def _ah_sample(name, now):
    """Un relevé (AH_TICK_S) d'un flux audio suivi : crêtes par canal + saturation.
    On lit la fenêtre d'échantillons RÉELLEMENT écoulée depuis le relevé précédent (≈20 ms), pas
    seulement le 1 ms des VU-mètres : sans ça une saturation de quelques ms passerait entre les
    mailles (les VU n'échantillonnent que 1 ms sur 20 — suffisant pour une barre, pas pour un
    détecteur de clip). Coût : ~960×8 float32 = 30 ko + un max/abs → quelques dizaines de µs."""
    with _hist_lock:
        st = _ah.get(name)
        if st is None:
            st = _ah_new(); st["name"] = name; _ah[name] = st
    if st["ar"] is None:
        try:
            _gc_mxl()
        except Exception:
            pass
        try:
            st["ar"] = bobimxl.AudioReader(inst, name)
            st["last_head"] = None
        except Exception:
            st["ar"] = None
    ar = st["ar"]
    mn = np.full(A_CHANNELS_MAX, METER_MIN_DB, dtype=np.float32)
    zc = np.zeros(A_CHANNELS_MAX, dtype=np.uint8)
    if ar is None:
        _ah_push(st, now, mn, zc)           # flux absent → plancher (jamais de crête figée)
        st["last_t"] = now
        return
    try:
        head = int(ar.head_index())
    except Exception:
        head = -1
    if head < 0:
        _ah_push(st, now, mn, zc); st["last_t"] = now
        return
    # Flux COUPÉ (head figé au-delà de ABSENCE_MS) → plancher + reconnexion (même logique que
    # _update_peaks : ni l'AudioWriter MXL ni mtl_rx ne bumpent lastWriteTime).
    if head == st.get("last_head"):
        if (now - float(st.get("last_head_t") or now)) * 1000.0 > ABSENCE_MS:
            try: ar.close()
            except Exception: pass
            st["ar"] = None
            _ah_push(st, now, mn, zc); st["last_t"] = now
            return
    else:
        st["last_head"] = head; st["last_head_t"] = now
    dt = now - (st["last_t"] or (now - AH_TICK_S))
    n = int(max(A_SAMPLES_PER_CHUNK, min(2400, dt * A_SAMPLE_RATE + A_SAMPLES_PER_CHUNK)))
    # Fenêtre glissante FINISSANT au dernier sample commité. `head` est un index
    # ONE-PAST-THE-END : les samples écrits sont [.., head), donc la fenêtre est [head - n, head).
    # L'ancien calcul (head + A_SAMPLES_PER_CHUNK - n) venait du modèle faux « head = début du
    # dernier bloc » que la docstring de bobimxl propageait — il visait un bloc dans le futur.
    start = head - n
    blk = None
    if start >= 0:
        try:
            blk = ar.read_from(start, n)
        except Exception:
            blk = None
    if blk is None:
        try:
            blk = ar.read_latest(A_SAMPLES_PER_CHUNK)   # repli : la seule dernière milliseconde
        except Exception:
            blk = None
    st["last_t"] = now
    if blk is None or blk.size == 0:
        _ah_push(st, now, mn, zc)
        return
    amp = np.abs(blk)
    ch = min(A_CHANNELS_MAX, amp.shape[1])
    pk = mn.copy()
    peak_lin = amp[:, :ch].max(axis=0)
    with np.errstate(divide="ignore"):
        pk[:ch] = np.maximum(METER_MIN_DB, 20.0 * np.log10(np.maximum(peak_lin, 1e-7)))
    cl = zc.copy()
    if amp.shape[0] >= AH_CLIP_RUN:
        fs = amp[:, :ch] >= AH_CLIP_LEVEL          # échantillons pleine échelle
        run = fs[:-(AH_CLIP_RUN - 1)]
        for k in range(1, AH_CLIP_RUN):
            run = run & fs[k:amp.shape[0] - (AH_CLIP_RUN - 1) + k]
        cl[:ch] = run.any(axis=0).astype(np.uint8)  # ≥ AH_CLIP_RUN FS consécutifs = saturation
    _ah_push(st, now, pk, cl)

def _ah_names(unit, cfg_src):
    """Noms des flux audio à suivre pour une unité (espace de 16 canaux = 2 flux de 8)."""
    try:
        s0 = max(0, min(2 * A_CHANNELS_MAX - 1, int(unit.get("ch_start") or 1) - 1))
        n = max(1, min(A_CHANNELS_MAX, int(unit.get("channels") or 2)))
    except (TypeError, ValueError):
        s0, n = 0, 2
    n = min(n, 2 * A_CHANNELS_MAX - s0)
    out = []
    for flow in (0, 1):
        f0 = flow * A_CHANNELS_MAX
        if max(s0, f0) < min(s0 + n, f0 + A_CHANNELS_MAX):
            nm = _audio_name_for(cfg_src, flow)
            out.append((flow, nm))
    return s0, n, out

def _ahist_loop():
    """Thread d'échantillonnage audio (AH_TICK_S = cadence trame)."""
    while True:
        time.sleep(AH_TICK_S)
        try:
            wanted = set()
            for _k, unit, _r, cfg_src, _wi in _hist_units("audio"):
                for _flow, nm in _ah_names(unit, cfg_src)[2]:
                    if nm:
                        wanted.add(nm)
            with _hist_lock:
                for nm in [k for k in _ah if k not in wanted]:
                    stx = _ah.pop(nm)
                    try:
                        if stx.get("ar"):
                            stx["ar"].close()
                    except Exception:
                        pass
            now = time.time()
            for nm in wanted:
                try:
                    _ah_sample(nm, now)
                except Exception:
                    pass
        except Exception:
            pass

# ─── Historique : rendu (tuiles YUV cachées, recomposées au changement seulement) ─────────

def _hist_evt_at(evts, t):
    """Code d'événement au temps t (ruban) — dernier relevé antérieur, "" si hors ring."""
    code = None
    for (ts, c) in evts:
        if ts <= t:
            code = c
        else:
            break
    return code or ""

def _vh_slots(rect, dur, show_evt, aspect):
    """★ Nombre de vignettes DÉDUIT DE LA PLACE, et intervalle qui en découle.
    Une vignette occupe `sh × (sh·ratio_source)` — donc autant de cases que la bande peut en
    aligner SANS DÉFORMATION ni chevauchement (plancher VH_CELL_MIN_W : sous ~48 px de large une
    vignette n'apprend plus rien). step = durée / nb ≥ 1 s. Renvoie (n, step, rb, sh)."""
    rw, rh = rect[2], rect[3]
    sc = _hist_scale_h(rh)                               # bandes de graduations (haut + bas)
    usable = max(8, rh - 2 * sc)
    rb = max(4, min(10, usable // 8)) if show_evt else 0  # hauteur du ruban d'événements
    sh = max(4, usable - rb - (2 if rb else 0))          # hauteur de la bande de vignettes
    cw_min = max(VH_CELL_MIN_W, int(round(sh * max(0.2, aspect))) + VH_CELL_GAP)
    n = max(1, min(int(rw // cw_min), int(dur)))
    return n, dur / float(n), rb, sh, sc

def _hist_fill(arr, y0, y1, rgb, a):
    """Remplit les lignes [y0:y1) d'un RGBA avec une couleur unie — via un GABARIT DE LIGNE
    contigu (arr[y0:y1] = row). ⚠ PERF : `arr[..., 0:3] = (r, g, b)` écrit une vue STRIDÉE et
    coûte ~35× plus cher (3,7 ms vs 0,11 ms sur une frise 1720×200 — mesuré) ; c'est ce motif,
    répété à chaque recomposition, qui faisait tomber le mur sous 50 fps."""
    if y1 <= y0:
        return
    row = np.empty((arr.shape[1], 4), dtype=np.uint8)
    row[:] = (rgb[0], rgb[1], rgb[2], a)
    arr[y0:y1] = row

# ─── Échelle temporelle des frises (graduations) ─────────────────────────────────────────
# DEUX règles, identiques sur les deux frises (elles se lisent l'une SOUS l'autre : mêmes pas,
# mêmes abscisses → une saturation audio se corrèle à l'œil avec un gel image) :
#   • EN HAUT : décompte RELATIF (−30 s … −10 s … 0), 0 = maintenant, tout à droite ;
#   • EN BAS  : HORODATAGE ABSOLU (heure civile, MÊME horloge que les overlays « ptp » :
#     temps du nœud − TAI_UTC_OFFSET_S, fuseau injecté).
# COÛT : les deux bandes sont CACHÉES (motif du fond statique des VU-mètres) — la bande relative ne
# change JAMAIS pour une géométrie donnée, la bande absolue une fois par SECONDE. La clé de cache
# porte tout ce qui la fait varier (rw, h, durée, opacité, pas, seconde) : jamais d'échelle périmée.
_hist_scale_cache = {{}}
_HIST_SCALE_STEPS = (1, 2, 5, 10, 15, 30, 60, 120)
_HIST_SCALE_MINPX = 140                # espacement minimal entre deux graduations (libellé AÉRÉ)

def _hist_scale_h(rh):
    """Hauteur d'une bande de graduations (0 = frise trop basse pour en porter)."""
    return 11 if rh >= 64 else 0

def _hist_scale_step(rw, dur):
    """Pas de graduation DÉDUIT DE LA PLACE (comme le nombre de vignettes) : le plus fin des pas
    offerts qui laisse au moins _HIST_SCALE_MINPX entre deux graduations."""
    for s in _HIST_SCALE_STEPS:
        if rw * s / float(dur) >= _HIST_SCALE_MINPX:
            return s
    return _HIST_SCALE_STEPS[-1]

def _hist_civil(t):
    """Heure civile locale d'un instant du mur — MÊME conversion que l'horloge « ptp » des overlays."""
    lt = time.localtime(t - TAI_UTC_OFFSET_S)
    return "%02d:%02d:%02d" % (lt.tm_hour, lt.tm_min, lt.tm_sec)

def _hist_scale_band(rw, h, dur, a_bg, absolute, now):
    """Bande RGBA (h, rw, 4) de graduations — CACHÉE (rendu PIL payé 1×, ou 1×/s en absolu)."""
    key = (rw, h, dur, a_bg, absolute, int(now) if absolute else 0)
    hit = _hist_scale_cache.get(key)
    if hit is not None:
        return hit
    step = _hist_scale_step(rw, dur)
    img = Image.new("RGBA", (rw, h), (_HIST_BG[0], _HIST_BG[1], _HIST_BG[2], a_bg))
    d = ImageDraw.Draw(img, "RGBA")
    f = ImageFont.load_default()
    k = 0
    while k * step <= dur:
        x = rw - 1 - int(round(k * step * rw / float(dur)))
        if x < 0:
            break
        d.line([x, 0 if absolute else h - 4, x, 3 if absolute else h - 1],
               fill=(150, 156, 168, 255))
        txt = _hist_civil(now - k * step) if absolute else ("0" if k == 0 else "-%ds" % (k * step))
        tw = int(d.textlength(txt, font=f))
        tx = min(rw - tw - 1, max(1, x - tw // 2))
        d.text((tx, 1 if absolute else 0), txt, font=f, fill=(196, 202, 214, 255))
        k += 1
    band = np.asarray(img).copy()          # contigu → blit direct dans la frise
    if len(_hist_scale_cache) > 12:
        _hist_scale_cache.clear()          # (borné : quelques géométries × la seconde courante)
    _hist_scale_cache[key] = band
    return band

def _hist_scales(arr, rw, rh, dur, a_bg, now):
    """Pose les deux bandes de graduations (haut = relatif, bas = absolu) et renvoie leur hauteur."""
    h = _hist_scale_h(rh)
    if h:
        arr[0:h] = _hist_scale_band(rw, h, dur, a_bg, False, now)
        arr[rh - h:rh] = _hist_scale_band(rw, h, dur, a_bg, True, now)
    return h

_ah_num_cache = {{}}   # (rw, wh, nband, band_h, n, s0) → bandeau RGBA (wh, lab_w, 4) | None

def _ah_num_band(rw, wh, nband, band_h, n, s0):
    """Bandeau RGBA des NUMÉROS DE CANAL, collé tout à GAUCHE de la frise audio.

    Le numéro écrit est le numéro RÉEL du canal — `ch_start` pris en compte : `s0 + ch + 1`,
    1-indexé — EXACTEMENT la convention des VU-mètres (`_draw_meter` : `ch0 + k + 1`).
    Une bande porte 2 canaux : le premier dans sa moitié HAUTE, le second dans sa moitié BASSE
    (mise en page identique à celle des demi-enveloppes). Un canal SEUL est centré sur son axe.

    « QUAND IL Y A LA PLACE » = seuil DUR → renvoie None (on n'écrit RIEN) si :
      - une DEMI-bande fait moins de AH_NUM_MIN_PX (un chiffre n'y tiendrait pas sans mordre
        l'enveloppe), ou
      - le bandeau mangerait plus de 1/AH_NUM_MAX_FRAC de la largeur de la frise.
    Dégradation SILENCIEUSE et NETTE : pas de demi-chiffre, pas d'onde rognée.

    ★ CACHÉ, même couche que les graduations (`_hist_scale_band`) : le contenu ne dépend QUE de
    la géométrie, du nombre de canaux et de ch_start → zéro rendu PIL par trame.
    ⚠ La clé PORTE `n` et `s0` (channels/ch_start) : sans eux, changer ch_start afficherait des
    numéros PÉRIMÉS (piège déjà rencontré sur le fond des VU-mètres et les échelles des frises)."""
    key = (rw, wh, nband, band_h, n, s0)
    if key in _ah_num_cache:
        return _ah_num_cache[key]
    half = band_h // 2
    labels = [str(s0 + ch + 1) for ch in range(n)]
    lab_w = 4 + 6 * max(len(l) for l in labels)      # load_default ≈ 6 px/caractère + marge
    band = None
    if half >= AH_NUM_MIN_PX and rw >= lab_w * AH_NUM_MAX_FRAC:
        img = Image.new("RGBA", (lab_w, wh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        f = ImageFont.load_default()
        for bi in range(nband):
            by0 = bi * band_h
            by1 = (bi + 1) * band_h if bi < nband - 1 else wh    # la dernière bande absorbe le reste
            cy = by0 + (by1 - by0) // 2
            for side in (0, 1):
                ch = 2 * bi + side
                if ch >= n:
                    break
                lone = (side == 0 and 2 * bi + 1 >= n)           # canal seul → centré sur l'axe
                if lone:
                    ty = cy - 5
                else:                                            # moitié HAUTE / moitié BASSE
                    ty = (cy - (by1 - by0) // 4 - 5) if side == 0 else (cy + (by1 - by0) // 4 - 5)
                ty = max(by0, min(by1 - 11, ty))
                # Liseré sombre : le chiffre reste lisible quand l'enveloppe passe dessous.
                for _ox, _oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    d.text((2 + _ox, ty + _oy), labels[ch], font=f, fill=(18, 20, 24, 200))
                d.text((2, ty), labels[ch], font=f, fill=(228, 232, 240, 255))
        band = np.asarray(img).copy()
    if len(_ah_num_cache) > 12:
        _ah_num_cache.clear()          # borné (quelques géométries × nombres de canaux)
    _ah_num_cache[key] = band
    return band

def _vh_render(unit, rect, name, now):
    """Bande de vignettes + ruban d'événements → RGBA numpy (rh, rw, 4).
    La recomposition n'a lieu qu'à l'arrivée d'une vignette (tous les `step` s ≈ 6 s) ou à une
    transition d'événement, mais elle tombe DANS une trame du mur (budget 20 ms à 50 fps) → elle
    doit rester à quelques ms. Primitives choisies AU CHRONO (cf. _hist_fill / vignettes PIL).
    Chaque vignette est posée à son RATIO (letterbox/pillarbox dans sa case) : une frise de
    diagnostic qui déforme l'image ment sur ce qu'elle montre."""
    _rx, _ry, rw, rh = rect
    dur = _hist_dur(unit)
    show_evt = _as_bool(unit.get("events", True), True)
    a_bg = max(10, min(100, int(unit.get("opacity") or 85))) * 255 // 100
    with _hist_lock:
        st = _vh.get(name)
        cells = list(st["cells"]) if st else []
        evts = list(st["evt"]) if st else []
        aspect = float(st["aspect"]) if st else 16.0 / 9.0
    n, step, rb, sh, sc = _vh_slots(rect, dur, show_evt, aspect)
    arr = np.empty((rh, rw, 4), dtype=np.uint8)
    _hist_fill(arr, sc, sc + sh, (30, 32, 38), a_bg)    # cases encore vides (avant le 1ᵉʳ relevé)
    _hist_fill(arr, sc + sh, rh - sc, _HIST_BG, a_bg)   # gouttière + fond du ruban
    _hist_scales(arr, rw, rh, dur, a_bg, now)           # graduations (cachées : coût ~nul)
    t0 = now - dur
    cw = rw / float(n)
    strip = arr[sc:sc + sh, :, 0:3]
    # Rangement des vignettes DATÉES dans les cases temporelles de CETTE frise. Une vignette
    # ÉPINGLÉE (capturée à l'instant d'un événement) l'emporte toujours sur un relevé de routine
    # tombé dans la même case — c'est l'image sur laquelle ça s'est figé qu'on veut voir.
    slots = {{}}
    for c in cells:                       # ring DÉJÀ trié par temps → « le dernier gagne »
        k = int((float(c.get("t") or 0.0) - t0) / step)
        if k < 0 or k >= n or c.get("img") is None:
            continue
        prev = slots.get(k)
        if prev is not None and prev.get("pinned") and not c.get("pinned"):
            continue                      # l'ÉPINGLÉE (instant de l'événement) l'emporte
        slots[k] = c
    for k, c in slots.items():
        img = c["img"]
        ih, iw = img.shape[0], img.shape[1]
        x0 = int(round(k * cw)); x1 = min(rw, int(round((k + 1) * cw)) - VH_CELL_GAP)
        box_w = x1 - x0
        if box_w < 4 or ih < 1 or iw < 1:
            continue
        # LETTERBOX / PILLARBOX : la vignette garde SON ratio dans la case (jamais étirée).
        zf = min(box_w / float(iw), sh / float(ih))
        tw = max(2, min(box_w, int(round(iw * zf))))
        th = max(2, min(sh, int(round(ih * zf))))
        ox = x0 + (box_w - tw) // 2
        oy = (sh - th) // 2
        # Mise à l'échelle FILTRÉE (Image.LANCZOS, C) + écriture d'un BLOC RGBA CONTIGU.
        # ★ 0.40.0 : c'était un Image.NEAREST « la qualité n'a aucune importance ici, le crénelage
        # est assumé » — hérité de l'époque où l'on croyait que le coût des frises faisait tomber le
        # mur. C'était FAUX (le coupable était le PIC de recomposition, désormais hors trame) : ce
        # redimensionnement est payé par le THREAD BOULANGER, pas par la trame. Avec des vignettes
        # stockées à 144 px (VH_THUMB_H), le facteur d'échelle est proche de 1 → LANCZOS est à la
        # fois net et bon marché.
        # ★ La vignette est FIGÉE une fois capturée, et sa case ne change pas de taille : son bloc
        # mis à l'échelle est donc TOUJOURS LE MÊME. On le calculait pourtant à chaque bake — un
        # LANCZOS par case, soit une vingtaine par passe sur une frise de 2 minutes, refaits pour
        # rien. Mesuré le 2026-08-08 : 19,7 ms de dessin par bake, contre 4,5 de conversion.
        # Le bloc est mémorisé DANS la case, avec la géométrie qui l'a produit : un changement de
        # taille de frise (ou d'opacité) le périme, et il est recalculé une seule fois.
        _memo = c.get("_blk")
        if _memo is not None and _memo[0] == (tw, th, a_bg):
            blk = _memo[1]
        else:
            up = np.asarray(Image.fromarray(img).resize((tw, th), Image.LANCZOS))
            blk = np.empty((th, tw, 4), dtype=np.uint8)
            blk[..., 0:3] = up
            blk[..., 3] = a_bg
            c["_blk"] = ((tw, th, a_bg), blk)
        arr[sc + oy:sc + oy + th, ox:ox + tw] = blk
        if c.get("pinned") and c.get("evt"):
            # Vignette CAPTURÉE À L'INSTANT de l'événement : liseré à la couleur de l'événement.
            col = _HIST_EVT_RGB.get(c["evt"], (255, 255, 255))
            strip[oy, ox:ox + tw] = col; strip[oy + th - 1, ox:ox + tw] = col
            strip[oy:oy + th, ox] = col; strip[oy:oy + th, ox + tw - 1] = col
    if rb:
        # Ruban : couleur du statut à l'instant de chaque colonne (vide = signal sain). Recherche
        # vectorisée dans le ring d'événements (relevés à VH_TICK_S, donc déjà triés par temps).
        _hist_fill(arr, sc + sh + 1, rh - sc, (44, 48, 56), a_bg)
        band = arr[sc + sh + 1:rh - sc, :, 0:3]
        if evts:
            ets = np.fromiter((e[0] for e in evts), dtype=np.float64, count=len(evts))
            codes = [e[1] for e in evts]
            tcol = t0 + (np.arange(rw) + 0.5) * dur / float(rw)
            j = np.searchsorted(ets, tcol, side="right") - 1
            for code, rgb in _HIST_EVT_RGB.items():
                m = np.zeros(rw, dtype=bool)
                hit = [i for i, c in enumerate(codes) if c == code]
                if not hit:
                    continue
                hset = np.zeros(len(codes), dtype=bool)
                hset[hit] = True
                ok = j >= 0
                m[ok] = hset[j[ok]]
                if m.any():
                    band[:, m] = rgb
    return arr

def _ah_render(unit, rect, cfg_src, now):
    """Enveloppe des crêtes + saturation (rouge, persistante) + silence (gris) → RGBA numpy.
    Agrégation par colonne en O(n) SANS scatter (`np.maximum.at` est lent) : le ring est trié
    par temps → bornes de colonne par searchsorted + reduceat."""
    _rx, _ry, rw, rh = rect
    dur = _hist_dur(unit)
    s0, n, names = _ah_names(unit, cfg_src)
    a_bg = max(10, min(100, int(unit.get("opacity") or 85))) * 255 // 100
    # RÉSOLUTION DE L'ENVELOPPE : une colonne PAR PIXEL (AH_COL_PX = 1). L'enveloppe fine ne coûte
    # RIEN — les crêtes sont déjà calculées par l'échantillonneur, on ne fait que les DESSINER ; le
    # coût de la recomposition était dans les écritures numpy stridées (cf. _hist_fill), pas dans
    # la finesse. AH_COL_PX reste un cran de repli si un jour on veut épaissir le trait.
    # L'agrégation est un MAX (jamais une moyenne) sur la tranche de temps de la colonne, et la
    # saturation un OU logique → une saturation de 3 ms peint TOUTE sa colonne en rouge.
    ncol = max(8, int(rw // max(1, AH_COL_PX)))
    # PISTES PAR CANAL : une bande horizontale par canal suivi (stéréo → canal 1 en haut, canal 2 en
    # bas). Avant, les canaux étaient FUSIONNÉS en une seule enveloppe (max) : on ne pouvait pas
    # distinguer la gauche de la droite. Repli sur une piste unique (fusion, comportement d'avant) si
    # la place manque — une piste sous ~10 px n'apprend plus rien.
    peak = np.full((n, ncol), METER_MIN_DB, dtype=np.float32)
    clip = np.zeros((n, ncol), dtype=bool)
    seen = np.zeros((n, ncol), dtype=bool)
    t0 = now - dur
    need = min(AH_MAX, int(dur / AH_TICK_S) + 8)        # on ne relit QUE la fenêtre affichée…
    snap = []
    with _hist_lock:
        for flow, nm in names:
            st = _ah.get(nm) if nm else None
            if st is None or st["n"] == 0:
                continue
            cnt = min(st["n"], need)
            # …mais la borne est le TEMPS, pas un nombre d'entrées : si les `need` derniers relevés
            # ne remontent pas jusqu'à t0 (thread d'échantillonnage en retard, ou rendu d'un instant
            # antérieur), on élargit au ring entier plutôt que d'afficher une enveloppe TRONQUÉE.
            if cnt < st["n"] and st["t"][(st["w"] - cnt) % AH_MAX] > t0:
                cnt = st["n"]
            idx = (np.arange(cnt) + (st["w"] - cnt)) % AH_MAX
            snap.append((flow, st["t"][idx].copy(), st["pk"][idx].copy(), st["clip"][idx].copy()))
    edges = t0 + np.arange(ncol) * (dur / float(ncol))
    for flow, ts, pks, cls in snap:
        f0 = flow * A_CHANNELS_MAX
        a = max(s0, f0); b = min(s0 + n, f0 + A_CHANNELS_MAX)     # canaux suivis DANS ce flux
        if a >= b or ts.size == 0:
            continue
        st_i = np.searchsorted(ts, edges, side="left")            # début de chaque colonne
        cnt = np.diff(np.append(st_i, ts.size))
        keep = (cnt > 0) & (st_i < ts.size)
        if not keep.any():
            continue
        starts = st_i[keep]
        # Une réduction PAR CANAL (≤ 8 tours) au lieu d'un max qui écrasait les canaux entre eux.
        for g in range(a, b):
            lane = g - s0                                          # piste = rang du canal dans l'unité
            vals = pks[:, g - f0]
            cl = cls[:, g - f0]
            pk_col = np.maximum.reduceat(vals, starts)
            cl_col = np.maximum.reduceat(cl.astype(np.uint8), starts).astype(bool)
            peak[lane][keep] = np.maximum(peak[lane][keep], pk_col)
            clip[lane][keep] |= cl_col
            seen[lane][keep] = True
    if ncol != rw:      # colonnes → pixels (plus proche voisin, AUCUN lissage : un clip reste PLEIN)
        _px = (np.arange(rw) * ncol) // rw
        peak = peak[:, _px]; clip = clip[:, _px]; seen = seen[:, _px]
    silence = seen & (peak <= SILENCE_DB)
    # FOND = un GABARIT DE LIGNE (rw, 4) qui porte DÉJÀ les plages de silence (grisées) → une seule
    # écriture contiguë `arr[:] = row` (0,11 ms) au lieu d'une passe stridée + une passe colonnes.
    arr = np.empty((rh, rw, 4), dtype=np.uint8)
    row = np.empty((rw, 4), dtype=np.uint8)
    row[:] = (_HIST_BG[0], _HIST_BG[1], _HIST_BG[2], a_bg)
    arr[:] = row
    # Graduations : MÊMES pas et MÊMES abscisses que la frise vidéo (elles se lisent l'une sous
    # l'autre) — cachées, donc gratuites par trame.
    sc = _hist_scales(arr, rw, rh, dur, a_bg, now)
    wy0, wy1 = sc, rh - sc                                        # bande utile des enveloppes
    wh = max(2, wy1 - wy0)

    # DEMI-ENVELOPPE PAR CANAL, autour d'un axe commun : une forme d'onde est symétrique, donc une
    # MOITIÉ suffit à voir les problèmes. Une bande porte donc DEUX canaux — le pair vers le HAUT,
    # l'impair vers le BAS (stéréo : canal 1 en haut, canal 2 en bas, sur la même hauteur qu'avant).
    # 4 canaux → 2 bandes, 8 canaux → 4 bandes. Un canal SEUL reste symétrique (haut + bas).
    nband = (n + 1) // 2
    _merged = False
    if (wh // max(1, nband)) < AH_LANE_MIN_PX:      # place insuffisante → tout fusionner (max)
        peak = peak.max(axis=0)[None, :]
        clip = clip.any(axis=0)[None, :]
        seen = seen.any(axis=0)[None, :]
        silence = seen & (peak <= SILENCE_DB)
        n, nband = 1, 1
        _merged = True                             # enveloppe FUSIONNÉE (max des canaux) : la
                                                   # numéroter serait un MENSONGE → aucun numéro

    band_h = wh // nband
    for bi in range(nband):
        by0 = wy0 + bi * band_h
        by1 = by0 + band_h if bi < nband - 1 else wy1             # la dernière bande absorbe le reste
        bh = max(2, by1 - by0)
        cy = by0 + bh // 2
        yy = np.arange(by0, by1, dtype=np.int16) - cy             # SIGNÉ : <0 au-dessus de l'axe
        up, dn = (yy <= 0), (yy >= 0)
        wav = arr[by0:by1]
        for side in (0, 1):                                       # 0 = moitié HAUTE, 1 = moitié BASSE
            ch = 2 * bi + side
            if ch >= n:
                break
            lone = (side == 0 and 2 * bi + 1 >= n)                # canal seul → enveloppe symétrique
            mask_h = np.ones(bh, dtype=bool) if lone else (up if side == 0 else dn)
            half = max(1, (bh // 2 if not lone else bh // 2) - 1)
            pk, cl, sn = peak[ch], clip[ch], seen[ch]
            si = silence[ch]
            # Silence : grisé sur la MOITIÉ du canal concerné (chaque canal a le sien).
            if si.any():
                sub = arr[by0:by1]
                for _c, _v in enumerate(_HIST_SILENCE):
                    sub[..., _c][np.ix_(mask_h, si)] = _v
            frac = np.clip((pk - METER_MIN_DB) / (0.0 - METER_MIN_DB), 0.0, 1.0)  # dBFS → 0..1 (linéaire en dB)
            hgt = (frac * half).astype(np.int16)
            hgt[~sn] = 0
            dist = np.abs(yy)
            band = (dist[:, None] <= hgt[None, :]) & mask_h[:, None]
            # ⚠ PERF : peindre CANAL DE COULEUR PAR CANAL DE COULEUR sous le masque (écritures uint8
            # contiguës) au lieu de `rgb[band] = (r, g, b)` (indexation avancée sur une vue stridée) :
            # 1,3 ms contre 8,8 ms sur une frise 1720×202 — mesuré. Même résultat au pixel près.
            for _c, _v in enumerate(_HIST_WAVE):
                wav[..., _c][band] = _v
            # SATURATION : colonne ROUGE sur la MOITIÉ du canal → on voit LEQUEL a saturé. Le marqueur
            # PERSISTE tant que la colonne est dans la fenêtre (une saturation de 3 ms reste lisible
            # 30 s). Élargi à 2 px : à 120 s de profondeur une colonne fait moins d'un pixel de temps.
            if cl.any():
                wide = cl.copy()
                wide[1:] |= cl[:-1]
                sub = arr[by0:by1]
                for _c, _v in enumerate(_HIST_CLIP):
                    sub[..., _c][np.ix_(mask_h, wide)] = _v
        for _c, _v in enumerate((90, 96, 108)):                   # axe zéro (frontière des 2 canaux)
            np.maximum(arr[cy, :, _c], _v, out=arr[cy, :, _c])
        # Séparateur entre bandes (discret) : sans lui, deux bandes voisines se confondent.
        if bi < nband - 1 and by1 < wy1:
            for _c, _v in enumerate((58, 60, 68)):
                arr[by1 - 1, :, _c] = _v
    # NUMÉROS DE CANAL (tout à gauche) — bandeau CACHÉ, posé APRÈS les enveloppes (sinon l'onde
    # repeindrait par-dessus). None = pas la place → on n'écrit rien (cf. _ah_num_band).
    _nb = None if _merged else _ah_num_band(rw, wh, nband, band_h, n, s0)
    if _nb is not None:
        _lw = _nb.shape[1]
        _sub = arr[wy0:wy1, 0:_lw]
        _al = _nb[..., 3:4].astype(np.uint16)
        _sub[..., :3] = ((_nb[..., :3].astype(np.uint16) * _al
                          + _sub[..., :3].astype(np.uint16) * (255 - _al)) // 255).astype(np.uint8)
    return arr

def _hist_tile(rect, arr):
    """RGBA numpy d'une unité → tuile YUV chroma-alignée (contrat de blend des VU-mètres).
    ⚠ `arr` est en coordonnées LOCALES au rect : la bbox chroma-alignée doit rester CONTENUE
    dans le rect (origine arrondie VERS L'INTÉRIEUR). Arrondir l'origine vers l'extérieur
    (motif des VU-mètres, dont la tuile RGBA est allouée SUR la bbox) donnait ici un offset
    NÉGATIF (`bx0 - rx = -1` pour un rect à x impair) → tranche numpy VIDE → tuile de largeur 0,
    donc frise INVISIBLE (et, selon le backend, exception au blend/conversion). Le rect est
    déjà chroma-aligné par `_hist_rect` : ce bornage ne rogne plus rien, il reste un filet."""
    rx, ry, rw, rh = rect
    bx0 = max(0, rx); by0 = max(0, ry)
    bx1 = min(OUT_WIDTH, rx + rw); by1 = min(OUT_HEIGHT, ry + rh)
    if bx0 % _CW: bx0 += _CW - (bx0 % _CW)      # origine : arrondi VERS L'INTÉRIEUR du rect
    if by0 % _CH: by0 += _CH - (by0 % _CH)
    bx1 -= (bx1 - bx0) % _CW                    # taille : multiple de la grille chroma
    by1 -= (by1 - by0) % _CH
    if bx1 <= bx0 or by1 <= by0:
        return None
    sub = arr[by0 - ry:by1 - ry, bx0 - rx:bx1 - rx]
    if sub.shape[0] < 1 or sub.shape[1] < 1:    # filet : jamais de tuile vide en aval
        return None
    oy, ou, ov, oa, oa2 = rgba_to_yuv(sub)   # rgba_to_yuv accepte un ndarray RGBA (np.array(img))
    return (bx0, by0, bx1, by1, oy, ou, ov, oa, oa2)

# Banc : où vont les millisecondes d'un bake de frise — dessin contre conversion, par essence.
_hist_prof = {{"video": {{"draw": 0.0, "yuv": 0.0, "px": 0}},
              "audio": {{"draw": 0.0, "yuv": 0.0, "px": 0}}}}

def _hist_bake_one(key, kind, unit, rect, cfg_src, name, now):
    """Fabrique la tuile YUV d'UNE frise (dessin RGBA + conversion). ★ APPELÉ PAR LE THREAD
    BOULANGER SEULEMENT ★ — jamais par la boucle de composition (c'est tout l'objet de la 0.40.0).
    Coût 6-27 ms : hors trame, il ne fait plus tomber d'image."""
    _ts_hb = time.time_ns()
    try:
        img = (_vh_render(unit, rect, name, now) if kind == "video"
               else _ah_render(unit, rect, cfg_src, now))
        # Ventilation du bake : DESSIN (agrégation numpy + tracé RGBA) contre CONVERSION en YUV.
        # Les deux appellent des remèdes opposés — rendre le dessin incrémental (ne redessiner que
        # les colonnes neuves) ou rendre la conversion moins chère. Sur le bandeau ANC, la même
        # ventilation avait montré une moitié-moitié qu'on n'aurait pas devinée.
        _ts_conv = time.time_ns()
        _hist_prof[kind]["draw"] = round((_ts_conv - _ts_hb) / 1e6, 2)
        t = _hist_tile(rect, img)
        _hist_prof[kind]["yuv"] = round((time.time_ns() - _ts_conv) / 1e6, 2)
        _hist_prof[kind]["px"] = int(rect[2]) * int(rect[3])
        if t is None:
            _hist_warn(key, kind, rect, "tuile vide (rect hors canvas ou dégénéré)")
    except Exception as _e:
        # ⛔ JAMAIS d'échec muet : une frise qui ne peut pas se rendre le DIT (c'est l'avalage
        # silencieux d'ici qui a masqué le bug de tuile vide de la 0.37.0). Loggé UNE fois par
        # (frise, signature d'erreur) — jamais 50×/s.
        _hist_warn(key, kind, rect, repr(_e), trace=True)
        t = None
    # Coût UNITAIRE du bake (c'est un PIC, la moyenne le dilue) — désormais payé HORS trame.
    _hb = (time.time_ns() - _ts_hb) / 1e6
    _hist_bake["last"] = round(_hb, 2)
    if _hb > _hist_bake["max"]:
        _hist_bake["max"] = round(_hb, 2)
    _hist_bake_ctr[0] += 1
    return [t] if t is not None else []

def _hist_baker_loop():
    """Thread BOULANGER : fabrique les tuiles de frise demandées par la boucle de composition et
    les publie, PRÊTES, dans `_hist_ready`. Une seule à la fois (sérialisé) — inutile d'en faire
    plus : à 5 Hz de recomposition, son taux d'occupation est de quelques pour cent, et rester
    mono-thread garantit qu'il ne dispute jamais deux cœurs à la compo."""
    _nice_baker()   # 0.42.0 : même règle que le boulanger du chrome — la compo passe devant.
    while True:
        with _hb_cv:
            while not _hist_want:
                _hb_cv.wait()
            key = next(iter(_hist_want))
            sig, kind, unit, rect, cfg_src, name, now = _hist_want.pop(key)
        tiles = _hist_bake_one(key, kind, unit, rect, cfg_src, name, now)
        with _hb_cv:
            # Publication ATOMIQUE (une affectation de dict) d'un tuple de tableaux NEUFS : la
            # boucle de compo ne peut pas ramasser une tuile à moitié écrite.
            _hist_ready[key] = (sig, tiles)

def render_history_tiles(now):
    """Tuiles YUV des historiques vidéo/audio (composants de PiP + blocs de mur), ou None.
    ★ NE FABRIQUE PLUS RIEN ★ (0.40.0) : elle ramasse les tuiles prêtes du boulanger, calcule les
    signatures (quelques dizaines de µs) et dépose une DEMANDE quand une frise a changé. Le pic de
    recomposition (6-27 ms) ne tombe plus JAMAIS dans la trame — la boucle ne paie que le blend."""
    if not (VHIST_BLOCKS or AHIST_BLOCKS or _hist_cache or FLUX_CONFIG):
        return None
    # (a) RAMASSAGE des tuiles prêtes → elles deviennent la nouvelle référence de la boucle.
    if _hist_ready:
        with _hb_cv:
            _fresh = list(_hist_ready.items())
            _hist_ready.clear()
        for _k, _v in _fresh:
            _hist_cache[_k] = _v
    tiles = []
    live = set()
    want = []
    for kind in ("video", "audio"):
        for key, unit, rect, cfg_src, _wi in _hist_units(kind):
            rw = rect[2]
            if rw < 8 or rect[3] < 8:
                continue
            live.add(key)
            dur = _hist_dur(unit)
            name = ""
            if kind == "video":
                name = (cfg_src.get("path") or "").removeprefix("/dev/shm/")
                if not name:
                    continue
                with _hist_lock:
                    stv = _vh.get(name)
                    ver = stv["ver"] if stv else -1
                # HORODATAGE ABSOLU de l'échelle : sans lui, la frise afficherait une heure
                # périmée entre deux vignettes (une par ~6 s). Mais à la SECONDE il forçait un
                # re-boulangeage par seconde et par unité, dont le coût retombait en à-coups sur
                # les cœurs de la composition (cf. VH_LABEL_S). Arrondi à VH_LABEL_S : l'échelle
                # peut avoir quelques secondes de retard, ce qui ne se voit pas sur 30 à 120 s.
                sig = (rect, dur, ver, int(unit.get("opacity") or 85),
                       _as_bool(unit.get("events", True), True),
                       int(now / VH_LABEL_S), name)
            else:
                # Pas de recomposition = arrivée d'une nouvelle COLONNE, plafonné à AH_RECOMPOSE_S
                # (5 Hz) : au-delà, l'enveloppe n'avance que de 1-2 px (invisible) et on paierait le
                # dessin 20×/s. À 5 Hz, l'horodatage absolu de l'échelle est toujours à jour.
                _ncol = max(8, int(rw // max(1, AH_COL_PX)))
                col = int(now / max(AH_RECOMPOSE_S, dur / float(_ncol)))
                sig = (rect, dur, col, int(unit.get("opacity") or 85),
                       int(unit.get("channels") or 2), int(unit.get("ch_start") or 1),
                       _audio_name_for(cfg_src, 0) or "")
            hit = _hist_cache.get(key)
            if hit is None or hit[0] != sig:
                # (b) DEMANDE de bake. En attendant la livraison, on réutilise la tuile PÉRIMÉE
                # (une ou deux trames de retard sur une frise de 30-120 s : invisible). Au tout
                # premier passage il n'y a rien à afficher — la frise apparaît une trame plus tard.
                want.append((key, (sig, kind, unit, rect, cfg_src, name, now)))
            if hit is not None:
                tiles.extend(hit[1])
    for k in [k for k in _hist_cache if k not in live]:
        _hist_cache.pop(k, None)
    if want:
        with _hb_cv:
            for k, req in want:
                _hist_want[k] = req      # une demande par frise ; la plus récente écrase l'ancienne
            _hb_cv.notify()
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
        # Consommateurs du TEXTE TSL de la fenêtre : label legacy 'protocol' OU composant umd
        # d'un modèle de PiP sourcé TSL (sinon le re-bake du chrome raterait le changement).
        wants_tsl_text = _is_protocol_label(cfg) or any(
            isinstance(c, dict) and c.get("type") == "umd"
            and (c.get("text_source") or "name") == "tsl"
            for c in (_tpl_comps(cfg) or ()))
        if wants_tsl_text and tsl_text.get(i) != text:
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
_bg_seen = _chrome_seen = None       # dernière publication du boulanger RAMASSÉE par la boucle de compo
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

# ─── Bibliothèque de polices poussée par l'orchestrateur ─────────────────────
# CONFIG["font_library"] = [{{"key": "lib:<sha16>", "name", "family", "ext", "sha256", "b64"}}]
# — injecté au DÉPLOIEMENT par app/fonts.resolve_params() (seules les polices RÉELLEMENT
# référencées par les params voyagent, en base64). Le rootfs du conteneur est ÉPHÉMÈRE : on
# rematérialise donc les fichiers À CHAQUE DÉMARRAGE du script dans un répertoire de travail
# recréé (aucun état à maintenir), puis on les enregistre dans _FONT_FILES → _overlay_font()
# les sert exactement comme une police d'image. Une police illisible est IGNORÉE (log) : la clé
# reste absente de _FONT_FILES et _overlay_font retombe sur DejaVu — jamais de crash du mur.
_FONT_LIB_DIR = "/tmp/mv_fonts"

def _materialize_font_library():
    lib = CONFIG.get("font_library") or []
    if not lib:
        return
    try:
        os.makedirs(_FONT_LIB_DIR, exist_ok=True)
    except OSError as e:
        log("[fonts] répertoire de travail impossible (%s) → repli DejaVu" % e, "warning")
        return
    ok = 0
    for ent in lib:
        if not isinstance(ent, dict):
            continue
        key = str(ent.get("key") or "")
        if not key.startswith("lib:"):
            continue
        ext = (ent.get("ext") or "ttf").lower()
        if ext not in ("ttf", "otf", "ttc"):
            ext = "ttf"
        try:
            data = base64.b64decode(ent.get("b64") or "", validate=True)
            if not data:
                raise ValueError("charge utile vide")
            path = os.path.join(_FONT_LIB_DIR, "%s.%s" % (key[4:], ext))
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            ImageFont.truetype(path, 16)          # chargement RÉEL : une police cassée est écartée ici
            _FONT_FILES[key] = [path]
            ok += 1
        except Exception as e:
            log("[fonts] police %s (%s) illisible : %s → repli DejaVu"
                % (key, ent.get("name") or "?", e), "warning")
    log("[fonts] bibliothèque matérialisée : %d/%d police(s) dans %s"
        % (ok, len(lib), _FONT_LIB_DIR), "info")   # une seule ligne au démarrage → `info`

_materialize_font_library()

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
    return _expand_vars(ov.get("text") or "")

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

# ── ANC par entrée : timecode ET métadonnées (sous-titres, AFD, ST 352, SCTE-104) ──
# MXL-aligné : l'ANC est un FLUX DE DONNÉES séparé (futur ST 2038), pas un champ de l'en-tête
# vidéo. Chaque entrée vidéo a SON port ANC câblable (`anc_path` de flux_config, essence `data`
# de la page Câbles). REPLI historique : si rien n'est câblé, on dérive le shm ANC du chemin
# vidéo (`/dev/shm/mtl_0` → `/dev/shm/mtl_anc_0`) — les murs existants continuent de marcher.
# Le CODAGE du grain est celui annoncé par le producteur (`bobi_anc_format` du flowDef) :
# RFC 8331 (normatif) ou l'ancien format maison → `bobimxl.anc_unpack` aiguille seul (flotte MIXTE).
#
# TOUT AFFICHAGE EST OPT-IN : rien n'apparaît tant que l'utilisateur n'a pas ajouté une
# incrustation de type « anc » et coché les informations qu'il veut voir (cf. _format_anc).
anc_states = {{}}   # flux_idx → {{"for_path", "state": {{reader, path, flow_def}}|None}}

def _derive_anc_shm_path(video_path):
    """`/dev/shm/mtl_0` → `/dev/shm/mtl_anc_0` (miroir 2110_io hooks._derive_anc_shm)."""
    m = re.match(r"(/dev/shm/.+?)_(\d+)$", video_path)
    if m:
        return f"{{m.group(1)}}_anc_{{m.group(2)}}"
    return None

def _anc_path_for(idx):
    """Chemin du flux ANC de l'entrée `idx` : le port CÂBLÉ (`anc_path`) sinon la dérivation
    historique depuis la vidéo. Renvoie ("" si aucun)."""
    with state_lock:
        cfg = FLUX_CONFIG[idx] if 0 <= idx < len(FLUX_CONFIG) else None
        if not cfg:
            return ""
        wired = (cfg.get("anc_path") or "").strip()
        vpath = cfg.get("path") or ""
    return wired or (_derive_anc_shm_path(vpath) or "") if (wired or vpath) else ""

def _open_anc_state(anc_path):
    if not anc_path:
        return None
    try:
        name = anc_path.removeprefix("/dev/shm/")
        rd = bobimxl.Reader(inst, name)         # flux ANC data MXL (lève si absent)
        # Codage annoncé par le producteur — lu UNE fois à l'ouverture (le flowDef est immuable).
        return {{"reader": rd, "path": anc_path, "flow_def": inst.flow_def(name),
                 "gidx": None, "packets": []}}
    except Exception:
        return None

_ANC_STALE_S = 2.0   # index de grain ANC figé au-delà → reader PÉRIMÉ (flux recréé) → réouverture

def _anc_packets(idx):
    """Paquets ANC du DERNIER grain de l'entrée `idx` (liste, éventuellement vide). Le décodage
    est MIS EN CACHE par index de grain : plusieurs incrustations (timecode + sous-titres + AFD…)
    sur la même entrée ne le paient qu'une fois par trame.
    RECONNEXION : un producteur qui DÉTRUIT+RECRÉE son flux ANC sous le même nom (redéploiement
    de la source) laisse le reader collé à l'ancienne génération MORTE — get_latest y rejoue le
    dernier grain à l'infini (timecode FIGÉ) ou rend None, sans exception. Comme pour l'audio
    (head figé) et la vidéo (REOPEN_STALE_S) : index figé/illisible > _ANC_STALE_S → close +
    garbage_collect + réouverture sur la génération vivante."""
    path = _anc_path_for(idx)
    rec = anc_states.get(idx)
    # (ré)ouvre si l'entrée a changé de source, ou tant que le flux ANC n'est pas encore là.
    if rec is None or rec["for_path"] != path or rec["state"] is None:
        if rec and rec["state"]:
            try: rec["state"]["reader"].close()
            except Exception: pass
        rec = {{"for_path": path, "state": _open_anc_state(path)}}
        anc_states[idx] = rec
    st = rec["state"]
    if not st:
        return []
    now_m = time.monotonic()
    def _stale_reopen():
        # Péremption : lâcher le reader + purger l'ancienne génération → réouverture au
        # prochain appel (rec["state"] None). Timer ré-armé = retentes bornées à 1/_ANC_STALE_S.
        try: st["reader"].close()
        except Exception: pass
        try: _gc_mxl()
        except Exception: pass
        rec["state"] = None
        return []
    try:
        got = st["reader"].get_latest()
        if got is None:
            # Aucun grain lisible : flux pas encore écrit OU reader sur une génération morte —
            # dans les deux cas, réouverture bornée (1/_ANC_STALE_S) jusqu'à lecture.
            st.setdefault("fresh_t", now_m)
            if now_m - st["fresh_t"] > _ANC_STALE_S:
                return _stale_reopen()
            return []
        if got[0] != st["gidx"]:                # nouveau grain → décoder une fois
            st["gidx"] = got[0]
            st["fresh_t"] = now_m
            st["packets"] = bobimxl.anc_unpack(bytes(got[2]), st.get("flow_def"))
        elif now_m - st.get("fresh_t", now_m) > _ANC_STALE_S:
            return _stale_reopen()
        return st["packets"]
    except Exception:
        return []

def _anc_idx_of(ov, key="anc_source"):
    try:
        return int(ov.get(key) or 0)
    except (TypeError, ValueError):
        return 0

def _anc_tc_for(ov):
    """(hh,mm,ss,ff,df) de l'entrée référencée par l'horloge ANC (tc_source), ou None."""
    return bobimxl.anc_atc_decode(_anc_packets(_anc_idx_of(ov, "tc_source"))) or None

# Informations ANC affichables SUR L'IMAGE d'une cellule (comme les VU-mètres : un réglage
# PAR FENÊTRE, tout à false par défaut). L'utilisateur coche ce qu'il veut voir, cellule par
# cellule ; une cellule sans case cochée ne dessine RIEN et ne coûte RIEN.
ANC_FIELDS = ("anc_types", "anc_tc", "anc_cc", "anc_afd", "anc_st352", "anc_scte", "anc_crc")

def _anc_enabled(cfg):
    return any(_as_bool(cfg.get(k), False) for k in ANC_FIELDS)

def _format_anc_cell(i, cfg):
    """Ligne ANC d'une cellule : UNIQUEMENT les informations cochées sur CETTE fenêtre.
      anc_types : inventaire des métadonnées réellement portées (ATC, CC/708, AFD…)
      anc_tc    : timecode embarqué (ATC RP 188)
      anc_cc    : sous-titres — texte CEA-608 si décodable, sinon présence
      anc_afd   : format d'image actif (ST 2016-3)
      anc_st352 : format DÉCLARÉ PAR LE SIGNAL (ST 352) — à confronter au SDP
      anc_scte  : déclencheur SCTE-104 (« SPLICE » + opID)
      anc_crc   : nombre de paquets au CHECKSUM INVALIDE (métadonnée corrompue en transit)
    Renvoie "" si rien à afficher (aucune case, ou source sans ANC → « ANC -- » si l'inventaire
    est demandé, pour distinguer « pas d'ANC » de « pas d'affichage »)."""
    pkts = _anc_packets(i)
    if not pkts:
        return "ANC --" if _as_bool(cfg.get("anc_types"), False) else ""
    inv = bobimxl.anc_inventory(pkts)
    parts = []
    if _as_bool(cfg.get("anc_types"), False):
        names = []
        for it in inv:
            if it["name"] not in names:
                names.append(it["name"])
        parts.append(" ".join(names))
    if _as_bool(cfg.get("anc_tc"), False):
        tc = bobimxl.anc_atc_decode(pkts)
        if tc:
            parts.append("%02d:%02d:%02d%s%02d" % (tc[0], tc[1], tc[2], ";" if tc[4] else ":", tc[3]))
    if _as_bool(cfg.get("anc_st352"), False):
        s = bobimxl.anc_decode_st352(pkts)
        if s:
            parts.append(s["label"])
    if _as_bool(cfg.get("anc_afd"), False):
        a = bobimxl.anc_decode_afd(pkts)
        if a:
            parts.append("%s %s" % (a["aspect"], a["label"]))
    if _as_bool(cfg.get("anc_scte"), False):
        sc = bobimxl.anc_scte104(pkts)
        if sc:
            parts.append("SPLICE %d" % sc["op_id"])
    if _as_bool(cfg.get("anc_cc"), False):
        cc = bobimxl.anc_captions(pkts)
        if cc:
            parts.append(cc["cc608"] or ("%s ●" % cc["kind"]))
    if _as_bool(cfg.get("anc_crc"), False):
        bad = sum(1 for it in inv if it["checksum_ok"] is False)
        if bad:
            parts.append("CRC!%d" % bad)
    return "  ".join(p for p in parts if p)

def _anc_report():
    """Inventaire ANC de TOUTES les entrées, pour :8080 et :8082/state (diagnostic — l'UI
    affiche « cette source porte un ATC, des sous-titres… » sans rien incruster à l'image)."""
    out = {{}}
    with state_lock:
        n = len(FLUX_CONFIG)
    for i in range(n):
        pkts = _anc_packets(i)
        if not pkts:
            continue
        inv = bobimxl.anc_inventory(pkts)
        tc = bobimxl.anc_atc_decode(pkts)
        out[str(i)] = {{
            "path": _anc_path_for(i),
            "types": [it["name"] for it in inv],
            "crc_errors": sum(1 for it in inv if it["checksum_ok"] is False),
            "timecode": ("%02d:%02d:%02d%s%02d" % (tc[0], tc[1], tc[2], ";" if tc[4] else ":", tc[3]))
                        if tc else None,
        }}
    return out

# ─── Modèles de PiP (bibliothèque composable — Réglages → PiP) ────────────────
# Une cellule peut porter un MODÈLE (`cfg["template"] = {{"name", "components": […]}}`) qui
# REMPLACE l'habillage legacy de la fenêtre (label/tally/meters/ANC pilotés par flags, positions
# imposées) par une liste de COMPOSANTS librement positionnés. Géométrie NORMALISÉE 0..1
# relative à la cellule → un même modèle sert à une tuile 640×360 comme à une 1920×1080.
# Types : video / umd / tally / meters / anc / clock / text / format.
# Une cellule SANS modèle rend par le chemin historique, STRICTEMENT inchangé.
# Chaque composant peut porter :
#   when  : always | tally_red | tally_green | tally_any | tally_off | no_signal | freeze |
#           signal_ok — condition de visibilité (re-bake sur tally_dirty ; les transitions
#           d'état signal lèvent tally_dirty depuis la boucle, cf. _tile_status) ;
#   min_w : largeur de cellule (px) sous laquelle le composant est MASQUÉ (repli petites tuiles).
_tile_status = {{}}       # i → "" | "nosignal" | "freeze" (état signal par tuile, boucle de mix)
_tpl_status_prev = {{}}   # dernier état publié → détection de transition (re-bake conditions)
_TPL_SIGNAL_CONDS = ("no_signal", "freeze", "signal_ok")

# HÉRITAGE : le MUR peut définir un modèle PAR DÉFAUT (CONFIG.default_template, modifiable à
# chaud via /style). Résolution par cellule : template explicite > modèle par défaut du mur >
# modèle « Classique » GÉNÉRÉ (_classic_comps). Toute cellule a donc TOUJOURS un modèle —
# le chemin de rendu legacy (frame_style/overlay_below/label_size globaux) a été supprimé en
# 0.33.0 (l'habillage de mur vit dans les modèles ; cf. builtin:* côté orchestrateur).
DEFAULT_TEMPLATE = CONFIG.get("default_template") or None

def _tpl_dict_comps(t):
    if not isinstance(t, dict):
        return None
    comps = t.get("components")
    return comps if isinstance(comps, list) and comps else None

# ─── Modèle « Classique » GÉNÉRÉ (repli sans modèle ni défaut de mur) ────────
# Réplique l'habillage historique par défaut (bandeau nom translucide en BAS de l'image +
# pavés tally G/D + bande VU opt-in), GÉNÉRÉ depuis les flags par-fenêtre du composer
# (show_label/show_tally/meter_*) → les cases par-fenêtre continuent de piloter le repli,
# mais TOUT le rendu passe par l'unique moteur de composants. Miroir : app/pip_library.py
# builtin:classic (version statique sélectionnable). Cache par cellule invalidé sur les
# flags (les /window à chaud mutent cfg en place → la clé change, on regénère).
_CLASSIC_KEYS = ("show_label", "show_tally", "label_source", "meter_channels",
                 "meter_position", "meter_inside", "meter_opacity", "meter_scale", "w", "h",
                 "path", "audio_path")

def _classic_comps(cfg):
    key = tuple(str(cfg.get(k)) for k in _CLASSIC_KEYS)
    cached = cfg.get("_classic_gen")
    if cached and cached[0] == key:
        return cached[1]
    w = max(2, int(cfg.get("w") or 0)); h = max(2, int(cfg.get("h") or 0))
    bar_px = min(28, max(14, int(h * 0.18)))          # bandeau ~ historique (label_size 14)
    bh = min(0.40, bar_px / float(h))
    show_label = _as_bool(cfg.get("show_label"))
    show_tally = _as_bool(cfg.get("show_tally"))
    # Fenêtre SANS source vidéo configurée (path vide) mais avec une source audio résolue
    # (câblée explicitement OU dérivée, cf. _audio_name_for) : repli en cellule AUDIO SEULE
    # (VU-mètres pleine largeur + label), plutôt que forcer un composant vidéo qui ne
    # rendrait jamais qu'un NO SIGNAL. CONSERVÉ en 0.36.0 (contrairement au modèle d'usine
    # builtin:audio-only, retiré — cf. app/pip_library.py) : ce repli reste utile pour une
    # fenêtre individuelle, indépendamment du bloc VU-mètres de MUR (meter_blocks). Une
    # fenêtre sans vidéo NI audio garde l'ancien repli vidéo ci-dessous (NO SIGNAL).
    if not (cfg.get("path") or "").strip() and _audio_name_for(cfg):
        comps = []
        if show_label:
            comps.append({{"id": "umd", "type": "umd", "x": 0.0, "y": 1.0 - bh, "w": 1.0, "h": bh,
                          "text_source": ("tsl" if cfg.get("label_source") == "protocol" else "name"),
                          "tally_bg": False, "bg_color": "#000000", "bg_opacity": 70}})
        try:
            n = int(cfg.get("meter_channels") or 0) or 2   # jamais une cellule audio muette
        except (TypeError, ValueError):
            n = 2
        try:
            op = int(cfg.get("meter_opacity") or 100)
        except (TypeError, ValueError):
            op = 100
        comps.append({{"id": "vu", "type": "meters", "channels": n, "ch_start": 1,
                      "scale": cfg.get("meter_scale") or "dbfs", "opacity": op,
                      "align": "center", "width_mode": "fit",
                      "x": 0.04, "y": 0.03, "w": 0.92,
                      "h": max(0.1, 0.94 - (bh if show_label else 0.0))}})
        cfg["_classic_gen"] = (key, comps)
        return comps
    comps = [{{"id": "video", "type": "video", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0,
              "fit": "fill", "border": "none"}}]
    if show_label:
        comps.append({{"id": "umd", "type": "umd", "x": 0.0, "y": 1.0 - bh, "w": 1.0, "h": bh,
                      "text_source": ("tsl" if cfg.get("label_source") == "protocol" else "name"),
                      "tally_bg": False, "bg_color": "#000000", "bg_opacity": 70}})
    if show_tally:
        # Pavés carrés L/R dans le bandeau (côté = ~70 % du bandeau), collés aux bords.
        side = max(6, int(bar_px * 0.7))
        tw = side / float(w); th = side / float(h)
        ty = 1.0 - bh + (bh - th) / 2.0
        pad = max(2, int(bar_px * 0.25)) / float(w)
        comps.append({{"id": "talL", "type": "tally", "shape": "bar", "slot": "L",
                      "x": pad, "y": ty, "w": tw, "h": th}})
        comps.append({{"id": "talR", "type": "tally", "shape": "bar", "slot": "R",
                      "x": 1.0 - pad - tw, "y": ty, "w": tw, "h": th}})
    try:
        n = int(cfg.get("meter_channels") or 0)
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        try:
            op = int(cfg.get("meter_opacity") or 70)
        except (TypeError, ValueError):
            op = 70
        comps.append({{"id": "vu", "type": "meters", "channels": n, "ch_start": 1,
                      "scale": cfg.get("meter_scale") or "dbfs",
                      "opacity": (op if _as_bool(cfg.get("meter_inside")) else 100),
                      "align": ("left" if cfg.get("meter_position") == "left" else "right"),
                      "x": 0.01, "y": 0.02, "w": 0.98,
                      "h": max(0.1, 0.96 - (bh if show_label else 0.0))}})
    cfg["_classic_gen"] = (key, comps)
    return comps

def _tpl_comps(cfg):
    """Composants du modèle EFFECTIF de la cellule (héritage résolu) — jamais None."""
    comps = _tpl_dict_comps(cfg.get("template"))
    if comps is not None:
        return comps
    comps = _tpl_dict_comps(DEFAULT_TEMPLATE)
    if comps is not None:
        return comps
    return _classic_comps(cfg)

def _tpl_video_comp(cfg):
    for c in (_tpl_comps(cfg) or ()):
        if isinstance(c, dict) and c.get("type") == "video":
            return c
    return None

# Types d'HABILLAGE : ceux qui bordent l'image. Par défaut leur axe X suit l'IMAGE (ils doivent
# rester alignés sur ses bords gauche/droit quels que soient le ratio de la source, la taille de la
# cellule et la réserve du viseur), leur axe Y reste sur la CELLULE — c'est le seul moyen d'exprimer
# « sous l'image », que des fractions d'image bornées à [0, 1] ne savent pas dire.
# meters / anc / historiques restent sur la cellule dans les deux axes : ils vivent légitimement
# DANS la marge, à côté de l'image.
_ANCHOR_X_IMAGE_TYPES = ("umd", "tally", "format", "clock", "text")

def _comp_anchor(comp, axis):
    """Ancrage d'un composant sur un AXE : "cell" ou "image". Réglage explicite `anchor_x` /
    `anchor_y` s'il est posé, sinon défaut par type. Le composant `video` est TOUJOURS ancré à la
    cellule : c'est lui qui DÉFINIT l'image (_video_rect l'appelle) — l'ancrer à l'image serait une
    récursion infinie."""
    if (comp or {{}}).get("type") == "video":
        return "cell"
    v = str((comp or {{}}).get("anchor_" + axis) or "").strip().lower()
    if v in ("cell", "image"):
        return v
    if axis == "x" and (comp or {{}}).get("type") in _ANCHOR_X_IMAGE_TYPES:
        return "image"
    return "cell"

def _comp_rect(cfg, comp):
    """Rectangle ABSOLU (px, borné au canvas) d'un composant. Les fractions sont rapportées, PAR
    AXE, soit à la cellule soit au rectangle de l'IMAGE (cf. _comp_anchor)."""
    x, y, w, h = int(cfg["x"]), int(cfg["y"]), int(cfg["w"]), int(cfg["h"])
    x = max(0, min(x, OUT_WIDTH - 1)); y = max(0, min(y, OUT_HEIGHT - 1))
    w = max(2, min(w, OUT_WIDTH - x)); h = max(2, min(h, OUT_HEIGHT - y))
    bx, by, bw, bh = x, y, w, h          # base par défaut : la cellule, sur les deux axes
    ax, ay = _comp_anchor(comp, "x"), _comp_anchor(comp, "y")
    if ax == "image" or ay == "image":
        g = _video_rect(cfg)
        # On s'ancre sur le CADRE (image + bordure extérieure), pas sur l'image nue — cf. la note
        # de _video_rect. Sans bordure débordante les deux rectangles sont identiques.
        if g["fw"] >= 2 and g["fh"] >= 2:     # modèle sans vidéo → on reste sur la cellule
            if ax == "image":
                bx, bw = g["fx"], g["fw"]
            if ay == "image":
                by, bh = g["fy"], g["fh"]
    def _f(k, dflt):
        try:
            return max(0.0, min(1.0, float(comp.get(k, dflt))))
        except (TypeError, ValueError):
            return dflt
    rx = bx + int(round(_f("x", 0.0) * bw))
    ry = by + int(round(_f("y", 0.0) * bh))
    rw = max(2, int(round(_f("w", 1.0) * bw)))
    rh = max(2, int(round(_f("h", 1.0) * bh)))
    rw = max(2, min(rw, bx + bw - rx))
    rh = max(2, min(rh, by + bh - ry))
    return rx, ry, rw, rh

def _comp_visible(i, cfg, comp):
    """Visibilité d'un composant : seuil min_w (repli petites tuiles) + condition `when`."""
    try:
        if int(comp.get("min_w") or 0) > int(cfg.get("w") or 0):
            return False
    except (TypeError, ValueError):
        pass
    when = comp.get("when") or "always"
    if when == "always":
        return True
    if when in _TPL_SIGNAL_CONDS:
        st = _tile_status.get(i) or ""
        return {{"no_signal": st == "nosignal", "freeze": st == "freeze",
                 "signal_ok": st == ""}}.get(when, True)
    dom = _window_tally_dominant(i)
    return {{"tally_red":   dom in ("red", "amber"),
             "tally_green": dom in ("green", "amber"),
             "tally_any":   dom != "off",
             "tally_off":   dom == "off"}}.get(when, True)

def _tpl_pseudo_ov(i, comp, rect, bg_default="", bg_op_default=100):
    """Adapte un composant texte-like au contrat de _draw_text_overlay (pseudo-overlay,
    coordonnées ABSOLUES). bg_color absent → défaut du type ; "" explicite → pas de fond."""
    bg = comp.get("bg_color")
    if bg is None:
        bg = bg_default
    try:
        fs = int(comp.get("font_size") or 0)
    except (TypeError, ValueError):
        fs = 0
    return {{"id": "tpl%s_%s" % (i, comp.get("id") or comp.get("type") or ""),
             "x": rect[0], "y": rect[1], "w": rect[2], "h": rect[3],
             "font": comp.get("font") or "dejavu-sans-bold", "font_size": fs,
             "align": comp.get("align") or "center",
             "color": comp.get("color") or "#ffffff",
             "bg_color": bg, "bg_opacity": comp.get("bg_opacity", bg_op_default)}}

# ─── VARIABLES DE TEXTE ──────────────────────────────────────────────────────────────────────
# Un champ texte (composant `text` d'un modèle, ou overlay texte de mur) peut contenir des
# variables %nom%, remplacées au rendu. Syntaxe en POURCENTS et non en accolades : `script.py` est
# un template str.format, et des accolades dans du texte utilisateur seraient une source de
# confusion permanente (elles ne SERAIENT pas substituées — la substitution est en une passe —
# mais tout lecteur croirait le contraire).
# Un nom inconnu est laissé TEL QUEL : une faute de frappe se voit à l'écran au lieu de produire
# un trou silencieux.
_VAR_CACHE = {{"t": 0.0, "cpu": None, "cpu_prev": None}}

def _read_cgroup_cpu_us():
    """Microsecondes CPU consommées par le CONTENEUR (cgroup v2), ou None. On vise le conteneur
    entier et pas seulement notre processus : c'est ce que l'exploitant compare à ce qu'affiche
    l'orchestrateur."""
    try:
        with open("/sys/fs/cgroup/cpu.stat", "r") as f:
            for line in f:
                if line.startswith("usage_usec"):
                    return int(line.split()[1])
    except Exception:
        pass
    return None

# ─── Échantillonnage des variables de RESSOURCES, hors trame ─────────────────
# `%cpu%` et `%ram%` lisent des pseudo-fichiers cgroup. C'est court (quelques dizaines de µs) mais
# c'est de l'ENTRÉE-SORTIE BLOQUANTE, et elle n'a rien à faire dans le chemin d'une image : sur ce
# genre de valeur, personne n'a besoin de la fraîcheur à la milliseconde, et afficher celle de la
# trame précédente est rigoureusement équivalent. Un fil dédié échantillonne donc à intervalle
# fixe, et les variables ne font plus que LIRE un dictionnaire.
# Bénéfice second, moins visible : le calcul du %CPU est un DELTA entre deux relevés. Appelé
# depuis la trame, son intervalle dépendait du moment où quelqu'un demandait la valeur — donc le
# pourcentage était calculé sur une base irrégulière. À intervalle fixe, il est enfin juste.
def _var_sample_loop():
    while True:
        try:
            now = time.monotonic()
            us = _read_cgroup_cpu_us()
            if us is None:
                try:
                    t = os.times()
                    us = int((t.user + t.system) * 1e6)
                except Exception:                                          # noqa: BLE001
                    us = None
            if us is not None:
                prev = _VAR_CACHE.get("cpu_prev")
                _VAR_CACHE["cpu_prev"] = (now, us)
                if prev and now > prev[0]:
                    pct = (us - prev[1]) / ((now - prev[0]) * 1e6) * 100.0
                    _VAR_CACHE["cpu"] = "%.0f %%" % max(0.0, pct)
            try:
                with open("/sys/fs/cgroup/memory.current", "r") as f:
                    _VAR_CACHE["ram"] = "%d Mo" % (int(f.read().strip()) // (1024 * 1024))
            except Exception:                                              # noqa: BLE001
                pass
        except Exception:                                                  # noqa: BLE001
            pass
        time.sleep(_VARS_CHECK_S)

# ─── Télémétrie POUSSÉE par l'orchestrateur (nœud, contrôleur) ───────────────
# Un conteneur ne peut pas mesurer son NŒUD : il ne voit que son propre cgroup. L'orchestrateur,
# lui, échantillonne déjà la santé de chaque nœud et la sienne. Il POUSSE donc ces valeurs ici
# (POST :8082/telemetry), et les variables ne font que les lire — même contrat que %cpu%/%ram% :
# aucune E/S dans le chemin d'une image, et l'affichage d'une valeur vieille de quelques secondes
# est sans conséquence pour un indicateur de charge.
# Choix ASSUMÉ du sens de la poussée : c'est l'orchestrateur qui parle aux conteneurs (comme pour
# tout le reste du contrôle), et non l'inverse. Un mur qui interrogerait l'API du contrôleur aurait
# besoin de son adresse et d'un jeton, et créerait une dépendance inverse pour un simple affichage.
_TELEMETRIE = {{}}      # {{"noeud": {{...}}, "orchestrateur": {{...}}}} — dernière poussée reçue

def _tel(section, cle, suffixe="", defaut="—", cible=None):
    """Valeur de télémétrie. `cible` = nom d'un nœud ; absente, on prend le nœud de CE mur.
    Un nœud inconnu du cache rend le défaut — jamais la valeur d'un autre nœud, qui donnerait un
    affichage crédible et faux."""
    try:
        if cible:
            src = (_TELEMETRIE.get("noeuds") or {{}}).get(cible)
            if src is None:
                return defaut
        else:
            src = _TELEMETRIE.get(section) or {{}}
        v = src.get(cle)
        return defaut if v is None else ("%s%s" % (v, suffixe))
    except Exception:                                                      # noqa: BLE001
        return defaut


def _var_cpu():
    """Charge CPU du conteneur en %, telle que le fil d'échantillonnage l'a relevée. AUCUNE
    lecture de fichier ici : on rend la dernière valeur connue (cf. _var_sample_loop)."""
    return _VAR_CACHE.get("cpu") or "—"

def _var_ram():
    """Mémoire du conteneur, dernière valeur relevée hors trame."""
    return _VAR_CACHE.get("ram") or "—"

def _var_uptime():
    d = int(time.time() - _START_TS)
    h, rem = divmod(d, 3600); m, sec = divmod(rem, 60)
    return ("%dh%02d" % (h, m)) if h else ("%d min" % m if m else "%d s" % sec)

def _var_format():
    return "%dx%d%s%s" % (OUTPUT_W, OUTPUT_H, "i" if INTERLACED else "p",
                          metrics.get("fps_nominal") and ("%g" % metrics["fps_nominal"]) or "")

def _var_now(fmt):
    """Heure/date CIVILES, même base que les horloges : horloge du nœud (TAI) − offset TAI→UTC,
    dans le fuseau du mur."""
    ts = time.time() - TAI_UTC_OFFSET_S
    z = _zone(_TZ_NAME)
    if z is not None:
        return datetime.datetime.fromtimestamp(ts, z).strftime(fmt)
    return time.strftime(fmt, time.localtime(ts))

_TEXT_VARS = {{
    "conteneur":  lambda: HOSTNAME,
    # PLUGIN_VERSION (placeholder injecté AU RENDU) et non CONFIG["plugin_version"] : cette
    # dernière est la valeur PORTÉE PAR LES PARAMS, donc celle du déploiement PRÉCÉDENT —
    # constaté sur un conteneur rendu en 0.62 dont le CONFIG annonçait encore 0.60.
    "version":    lambda: PLUGIN_VERSION,
    "mur":        lambda: str(CONFIG.get("shm_out") or ""),
    "systeme":    lambda: str(CONFIG.get("system_name") or ""),
    "noeud":      lambda: str(CONFIG.get("node_name") or ""),
    "fuseau":     lambda: _TZ_NAME or "système",
    "heure":      lambda: _var_now("%H:%M"),
    "date":       lambda: _var_now("%d/%m/%Y"),
    "cpu":        _var_cpu,
    "ram":        _var_ram,
    # NŒUD et ORCHESTRATEUR : valeurs poussées par le contrôleur (cf. _TELEMETRIE). « — » tant
    # qu'aucune poussée n'est arrivée — jamais une valeur inventée ni celle du conteneur, qui
    # ferait croire à un nœud au repos alors qu'on n'en sait rien.
    "cpu_noeud":  lambda c=None: _tel("noeud", "cpu_pct", " %", cible=c),
    "ram_noeud":  lambda c=None: _tel("noeud", "ram_pct", " %", cible=c),
    "ram_noeud_mo": lambda c=None: _tel("noeud", "ram_used_mb", " Mo", cible=c),
    "disque_noeud": lambda c=None: _tel("noeud", "disk_pct", " %", cible=c),
    "temp_noeud": lambda c=None: _tel("noeud", "temp_c", " °C", cible=c),
    "charge_noeud": lambda c=None: _tel("noeud", "load1", cible=c),
    "nom_noeud":  lambda c=None: _tel("noeud", "nom", cible=c),
    # RDMA : remplissage des liens de réplication. `rdma_pct` = (rx+tx) rapporté au débit NOMINAL
    # du lien — c'est la question « reste-t-il de la place ? », celle qu'on se pose avant de
    # déplacer un flux. rx/tx séparés pour savoir DANS QUEL SENS ça se remplit.
    "rdma_pct":   lambda c=None: _tel("noeud", "rdma_pct", " %", cible=c),
    "rdma_rx":    lambda c=None: _tel("noeud", "rdma_rx_gbps", " Gb/s", cible=c),
    "rdma_tx":    lambda c=None: _tel("noeud", "rdma_tx_gbps", " Gb/s", cible=c),
    "rdma_debit": lambda c=None: _tel("noeud", "rdma_rate_gbps", " Gb/s", cible=c),
    "rdma_liens": lambda c=None: _tel("noeud", "rdma_liens", cible=c),
    "cpu_orch":   lambda: _tel("orchestrateur", "cpu_pct", " %"),
    "ram_orch":   lambda: _tel("orchestrateur", "ram_pct", " %"),
    "disque_orch": lambda: _tel("orchestrateur", "disk_pct", " %"),
    "fps":        lambda: "%.1f" % (metrics.get("fps") or 0.0),
    "format":     _var_format,
    "entrees":    lambda: str(sum(1 for c in FLUX_CONFIG if (c.get("path") or "").strip())),
    "duree":      _var_uptime,
}}

def _src_format(cfg):
    w, h = cfg.get("in_w") or 0, cfg.get("in_h") or 0
    if not (w and h):
        return "—"
    sc = (cfg.get("in_scan") or "p").strip().lower()
    fps = str(cfg.get("in_fps") or "").strip()
    return "%dx%d%s%s" % (w, h, "i" if sc == "i" else "p", fps)

def _src_label(cfg, n):
    """Libellé de NIVEAU n de la source (table source_labels de l'orchestrateur, injectée par le
    hook au déploiement). Vide → repli sur le nom résolu de la fenêtre, pour qu'un niveau non
    renseigné n'efface pas l'étiquette à l'écran."""
    lb = cfg.get("labels") or {{}}
    v = str(lb.get(str(n)) or lb.get(n) or "").strip()
    return v or (cfg.get("name") or "")

# Variables liées à la SOURCE d'une fenêtre : elles n'ont de sens que dans un composant de
# MODÈLE (rendu par cellule). Dans un overlay de mur, elles rendent « — » : un mur n'a pas UNE
# source. Chaque entrée prend la cfg de la fenêtre.
_SRC_VARS = {{
    "src":            lambda c: c.get("name") or "",
    "src_flux":       lambda c: (c.get("path") or "").removeprefix("/dev/shm/"),
    "src_format":     _src_format,
    "src_fps":        lambda c: str(c.get("in_fps") or "—"),
    "src_scan":       lambda c: "entrelacé" if (c.get("in_scan") or "p") == "i" else "progressif",
    "src_colorimetrie": lambda c: str(c.get("in_colorimetry") or "—"),
    "src_projet":     lambda c: str(c.get("projet") or ""),
    "src_audio":      lambda c: (c.get("audio_path") or "").removeprefix("/dev/shm/") or "—",
}}
for _n in range(2, 10):
    _SRC_VARS["src_label%d" % _n] = (lambda n: (lambda c: _src_label(c, n)))(_n)

# `%nom%` ou `%nom:cible%` — la cible désigne un NŒUD par son nom. Sans elle, les variables
# d'infra parlent du nœud qui porte ce mur, ce qui est le cas d'usage courant ; avec elle, un mur
# de supervision peut afficher n'importe quel nœud du parc. Le nom d'un nœud comporte des tirets
# (`dl360-1`), d'où leur admission dans l'argument mais pas dans le nom de variable.
_VAR_RE = re.compile(r"%([a-zA-Z_][a-zA-Z0-9_]*)(?::([A-Za-z0-9_.\-]+))?%")

def _texts_with_vars():
    """Textes bruts susceptibles de contenir des variables (composants `text`/`umd` fixes des
    modèles + overlays texte). Filtre sur la présence d'un « % » : un mur sans variable ne paie
    ni parcours ni évaluation."""
    out = []
    for cfg in FLUX_CONFIG:
        if cfg.get("hidden"):
            continue
        for comp in (_tpl_comps(cfg) or ()):
            if not isinstance(comp, dict):
                continue
            if comp.get("type") in ("text", "umd") and "%" in str(comp.get("text") or ""):
                out.append((comp["text"], cfg))
    for ov in OVERLAYS:
        if ov.get("kind") == "text" and "%" in str(ov.get("text") or ""):
            out.append((ov["text"], None))
    return out

def _vars_signature():
    """Signature des textes à variables, une fois RENDUS. None = aucun texte à variable (on ne
    déclenche alors jamais de re-bake)."""
    raw = _texts_with_vars()
    if not raw:
        return None
    return tuple(_expand_vars(t, c) for t, c in raw)

def _expand_vars(txt, cfg=None):
    """Remplace les %variables% d'un texte. `cfg` = la fenêtre, pour les variables de SOURCE
    (absent sur un overlay de mur → elles rendent « — », un mur n'ayant pas UNE source).
    Jamais fatal : une variable qui lève rend « — », un nom inconnu est laissé tel quel (la faute
    de frappe reste visible à l'écran plutôt que de créer un trou)."""
    if not txt or "%" not in txt:
        return txt
    def _sub(m):
        k = m.group(1)
        arg = m.group(2)
        fn = _TEXT_VARS.get(k)
        if fn is not None:
            try:
                # Les variables d'INFRA acceptent une cible ; les autres l'ignorent. On tente
                # d'abord avec, on retombe sans : une variable qui n'en veut pas ne doit pas
                # échouer parce que quelqu'un en a écrit une.
                if arg:
                    try:
                        return str(fn(arg))
                    except TypeError:
                        pass
                return str(fn())
            except Exception:
                return "—"
        sf = _SRC_VARS.get(k)
        if sf is not None:
            if cfg is None:
                return "—"
            try:
                return str(sf(cfg))
            except Exception:
                return "—"
        return m.group(0)
    return _VAR_RE.sub(_sub, txt)

def _tpl_text_value(i, cfg, comp):
    """Texte d'un composant umd : nom résolu de la source (défaut), texte TSL ou texte fixe."""
    src = comp.get("text_source") or "name"
    if src == "tsl":
        return tsl_text.get(i, "") or ""
    if src == "fixed":
        return _expand_vars(comp.get("text") or "", cfg)
    return cfg.get("name", "") or ""

# Couleur de texte par dominante tally (option tally_text des umd).
_TPL_TALLY_TEXT_HEX = {{"red": "#ff5a5a", "green": "#78ff8c", "amber": "#ffc83c"}}

def _tpl_draw_tally(d, i, comp, rect):
    """Composant tally : lamp (pastille ronde), bar (pavé plein) ou border (cadre)."""
    slot = comp.get("slot") or "dominant"
    if slot == "L":
        st = tally_state.get(f"{{i}}_L", "off")
    elif slot == "R":
        st = tally_state.get(f"{{i}}_R", "off")
    else:
        st = _window_tally_dominant(i)
    rx, ry, rw, rh = rect
    shape = comp.get("shape") or "lamp"
    if shape == "border":
        col = _TALLY_BORDER_RGBA.get(st, (0, 0, 0, 0))
        if st == "off":
            col = _FRAME_NEUTRAL
        try:
            t = max(2, int(comp.get("thickness") or 4))
        except (TypeError, ValueError):
            t = 4
        _render_border_colored(d, rx, ry, rw, rh, col, min(t, rw // 2, rh // 2))
    elif shape == "bar":
        col = _TALLY_BORDER_RGBA.get(st, (0, 0, 0, 0))
        if st == "off":
            col = (50, 50, 56, 235)
        d.rectangle([rx, ry, rx + rw - 1, ry + rh - 1], fill=col)
    else:  # lamp
        fill, outline = _PILL_COLORS.get(st, _PILL_COLORS["off"])
        r = max(2, min(rw, rh) // 2 - 1)
        _render_pill(d, rx + rw // 2, ry + rh // 2, r, fill, outline)

def _tpl_draw_video_border(d, img, i, cfg, comp):
    """CADRE du composant vidéo — l'habillage de cadre historique (ex-frame_style global de
    mur) migré dans le MODÈLE (0.33.0). Dessiné sur le rectangle IMAGE réel (_video_rect :
    fit contain compris → le cadre épouse l'image, letterbox inclus), vers l'INTÉRIEUR
    (jamais de débord sur les cellules voisines). Styles de `border` :
      fixed      : couleur pleine (border_color), épaisseur border_w ;
      tally      : fin, teinté par la dominante tally (neutre au repos) ;
      classic    : cadre fin neutre (ex-UMD broadcast) ;
      stylized   : bezel « moniteur » arrondi sombre, lèvre interne claire ;
      viewfinder : équerres de viseur aux 4 coins (teintées tally) ;
      flat       : soulignement bas pleine largeur (teinté tally)."""
    mode = comp.get("border") or "none"
    if mode == "none":
        return
    g = _video_rect(cfg)
    vx, vy, vw, vh = g["vx"], g["vy"], g["vw"], g["vh"]
    if vw < 2 or vh < 2:
        return
    try:
        bw = max(1, min(24, int(comp.get("border_w") or 3)))
    except (TypeError, ValueError):
        bw = 3
    dom = _window_tally_dominant(i)
    tally_col = _TALLY_BORDER_RGBA[dom] if dom != "off" else None
    if mode == "fixed":
        col = _hex_rgb(comp.get("border_color"), (255, 255, 255)) + (255,)
        _render_border_colored(d, vx, vy, vw, vh, col, bw)
    elif mode == "tally":
        _render_border_colored(d, vx, vy, vw, vh, tally_col or _FRAME_NEUTRAL, bw)
    elif mode == "classic":
        _render_border_colored(d, vx, vy, vw, vh, _FRAME_NEUTRAL, max(2, bw))
    elif mode == "stylized":
        # Bezel « moniteur » : anneau sombre arrondi sur le pourtour de l'image, PERCÉ sur
        # une tuile RGBA séparée (fill alpha 0 = trou) puis composée — le trou laisse la
        # vidéo ET les composants déjà dessinés intacts (contrairement à un perçage in-situ).
        t = max(4, int(round(bw * 2.2)))
        if vw <= 2 * t + 4 or vh <= 2 * t + 4:
            return
        tile = Image.new("RGBA", (vw, vh), (0, 0, 0, 0))
        dd = ImageDraw.Draw(tile)
        dd.rounded_rectangle([0, 0, vw - 1, vh - 1], radius=max(3, t), fill=(46, 46, 53, 255))
        dd.rectangle([t, t, vw - 1 - t, vh - 1 - t], fill=(0, 0, 0, 0))
        dd.rectangle([t - 1, t - 1, vw - t, vh - t], outline=(98, 98, 108, 255))
        img.alpha_composite(tile, (vx, vy))
    elif mode == "viewfinder":
        # Équerres de viseur aux 4 coins (blanches au repos, couleur tally sinon).
        col = tally_col or (225, 225, 232, 255)
        bt = max(2, bw)
        arm = max(8, int(round(min(vw, vh) * 0.14)))
        # ★ Les équerres se posent À L'EXTÉRIEUR de l'image : elles CADRENT le PiP au lieu de
        # recouvrir des pixels de la source (une équerre posée sur l'image masque justement le
        # coin qu'on veut vérifier). On les décale de `bt` vers l'extérieur — avec ce décalage
        # les deux bras tombent entièrement dans la marge letterbox/pillarbox laissée par le fit
        # `contain`, jamais sur l'image.
        # Repli : s'il n'y a pas la place DANS LA CELLULE (image qui la remplit exactement), on
        # redessine à l'intérieur comme avant. Recouvrir un peu d'image reste préférable à
        # déborder sur la fenêtre voisine — c'est la règle du cadre (cf. docstring).
        # Décision PAR AXE : une image letterboxée a de la marge en haut/bas mais pas sur les
        # côtés, une pillarboxée l'inverse. Trancher en tout-ou-rien ferait retomber tout le
        # cadre à l'intérieur alors qu'un axe avait la place.
        _cx0, _cy0 = g["x"], g["y"]
        _cx1, _cy1 = g["x"] + g["w"] - 1, g["y"] + g["h"] - 1
        _ox = bt if (vx - bt >= _cx0 and vx + vw - 1 + bt <= _cx1) else 0
        _oy = bt if (vy - bt >= _cy0 and vy + vh - 1 + bt <= _cy1) else 0
        x0, y0 = vx - _ox, vy - _oy
        x1, y1 = vx + vw - 1 + _ox, vy + vh - 1 + _oy
        for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                               (x0, y1, 1, -1), (x1, y1, -1, -1)):
            ex, ey = cx + sx * arm, cy + sy * (bt - 1)
            d.rectangle([min(cx, ex), min(cy, ey), max(cx, ex), max(cy, ey)], fill=col)
            ex, ey = cx + sx * (bt - 1), cy + sy * arm
            d.rectangle([min(cx, ex), min(cy, ey), max(cx, ex), max(cy, ey)], fill=col)
    elif mode == "flat":
        # Soulignement bas pleine largeur de l'image (neutre au repos, teinté tally).
        t = max(2, bw)
        d.rectangle([vx, vy + vh - t, vx + vw - 1, vy + vh - 1],
                    fill=tally_col or _FLAT_NEUTRAL)

def _tpl_render_dynamic(d, img, i, cfg, comps):
    """Composants BAKÉS d'un modèle (umd / tally / text / format) — re-rendus sur tally_dirty
    (changement tally, texte TSL, ou transition d'état signal pour les conditions), jamais
    per-frame. Les composants per-frame (meters / anc / clock) ont leur propre machinerie.
    Un composant malformé est ignoré (jamais de trame perdue pour un modèle cassé)."""
    for comp in comps:
        if not isinstance(comp, dict):
            continue
        k = comp.get("type")
        if k not in ("umd", "tally", "text", "format", "video"):
            continue
        try:
            if not _comp_visible(i, cfg, comp):
                continue
            if k == "video":
                _tpl_draw_video_border(d, img, i, cfg, comp)
                continue
            rect = _comp_rect(cfg, comp)
            if k == "tally":
                _tpl_draw_tally(d, i, comp, rect)
                continue
            if k == "umd":
                txt = _tpl_text_value(i, cfg, comp)
                ov = _tpl_pseudo_ov(i, comp, rect, bg_default="#000000", bg_op_default=70)
                col_over = None
                dom = _window_tally_dominant(i)
                if _as_bool(comp.get("tally_bg"), False):
                    d.rectangle([rect[0], rect[1], rect[0] + rect[2] - 1, rect[1] + rect[3] - 1],
                                fill=_BAR_TINTS.get(dom, _BAR_TINTS["off"]))
                    ov["bg_color"] = ""
                if _as_bool(comp.get("tally_text"), False) and dom != "off":
                    col_over = _TPL_TALLY_TEXT_HEX.get(dom)
                _draw_text_overlay(d, ov, txt, color_override=col_over)
            elif k == "format":
                ov = _tpl_pseudo_ov(i, comp, rect, bg_default="#000000", bg_op_default=65)
                _draw_text_overlay(d, ov, _fmt_chip_txt(cfg, None))
            else:  # texte FIXE seulement. Un texte à VARIABLES est rendu en tuile per-frame
                # (cf. _tpl_var_text_ovs) : le laisser ici le dessinerait deux fois, et surtout sa
                # valeur changeante forcerait un re-bake de l'habillage PLEIN CADRE — 56 à 73 ms,
                # jusqu'à une fois par seconde avec un %cpu%.
                if "%" in str(comp.get("text") or ""):
                    continue
                ov = _tpl_pseudo_ov(i, comp, rect)
                _draw_text_overlay(d, ov, comp.get("text") or "")
        except Exception:
            continue

def _tpl_clock_ovs():
    """Pseudo-overlays HORLOGE des modèles de PiP affichés — injectés dans la machinerie des
    horloges (_dyn_overlays : cache par signature + tuiles YUV par bbox). tc_source = l'entrée
    de la tuile (une horloge ANC lit LE timecode de SA source)."""
    out = []
    for i, cfg in enumerate(FLUX_CONFIG):
        if cfg.get("hidden"):
            continue
        comps = _tpl_comps(cfg)
        if not comps:
            continue
        for comp in comps:
            if not isinstance(comp, dict) or comp.get("type") != "clock":
                continue
            try:
                if not _comp_visible(i, cfg, comp):
                    continue
                ov = _tpl_pseudo_ov(i, comp, _comp_rect(cfg, comp),
                                    bg_default="#000000", bg_op_default=60)
                ov.update({{"kind": "clock",
                            "clock_source": comp.get("clock_source") or "ptp",
                            # Fuseau PROPRE à cette horloge ; vide = fuseau du mur (cf. _civil_hms).
                            "tz": (comp.get("tz") or "").strip(),
                            "tc_source": i,
                            "show_hh": _as_bool(comp.get("show_hh", True)),
                            "show_mm": _as_bool(comp.get("show_mm", True)),
                            "show_ss": _as_bool(comp.get("show_ss", True)),
                            "show_ff": _as_bool(comp.get("show_ff", False)),
                            "offset_ms": comp.get("offset_ms") or 0,
                            "chrono_start": comp.get("chrono_start") or "00:00:00",
                            "chrono_running": _as_bool(comp.get("chrono_running", False))}})
                out.append(ov)
            except Exception:
                continue
    return out

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
    else:  # ptp : heure CIVILE (horloge nœud TAI − offset TAI→UTC) + offset signé, dans le fuseau
        # de CETTE horloge (`tz`) — vide = fuseau du mur. Cf. _civil_hms : conversion explicite par
        # zone, et non `time.localtime`, pour que deux horloges du même mur puissent afficher deux
        # villes. La part fractionnaire (champs FF) ne dépend pas du fuseau : tous les décalages
        # IANA sont des multiples de la minute.
        _civ = now - TAI_UTC_OFFSET_S + int(_overlay_get(ov, "offset_ms", 0)) / 1000.0
        _hh, _mm, _ss = _civil_hms(_civ, ov.get("tz"))
        val = _hh * 3600 + _mm * 60 + _ss + (_civ % 1.0)
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
    # Les textes à VARIABLES sont exclus : ils passent par les tuiles per-frame (cf.
    # _tpl_var_text_ovs). Les laisser ici les dessinerait DEUX fois — et surtout, c'est leur
    # présence dans l'habillage qui déclenchait un re-bake plein cadre à chaque changement.
    fg = [ov for ov in OVERLAYS if (not ov.get("hidden")) and ov.get("kind") in ("text", "image")
          and not (ov.get("kind") == "image" and ov.get("layer") == "background")
          and not (ov.get("kind") == "text" and "%" in str(ov.get("text") or ""))]
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

def _tpl_var_text_ovs():
    """Pseudo-overlays des TEXTES À VARIABLES des modèles de PiP — même machinerie que les
    horloges (tuile YUV par bbox, re-rendue au changement de valeur).

    ★ Pourquoi ils ne sont plus bakés dans l'habillage : un texte qui contient `%cpu%` ou `%ram%`
    change à CHAQUE évaluation, et le mécanisme de surveillance déclenchait alors un re-bake du
    chrome PLEIN CADRE — 56 à 73 ms mesurés, environ une fois par seconde. Avec le GIL, cela fige
    la boucle de composition pendant trois créneaux de 20 ms : 3 images perdues par seconde, sur
    tout mur portant un tel texte. Le diagnostic a coûté une soirée entière parce que les gains
    obtenus ailleurs (bandeau ANC, vignettes de frise) étaient noyés par ce marteau.
    En tuile, la même mise à jour ne coûte que sa propre bbox — quelques milliers de pixels.
    Les textes SANS variable restent bakés : ils ne changent jamais, et une tuile per-frame leur
    coûterait un blend pour rien."""
    out = []
    for i, cfg in enumerate(FLUX_CONFIG):
        if cfg.get("hidden"):
            continue
        for comp in (_tpl_comps(cfg) or ()):
            if (not isinstance(comp, dict) or comp.get("type") not in ("text", "umd")
                    or "%" not in str(comp.get("text") or "")):
                continue
            try:
                if not _comp_visible(i, cfg, comp):
                    continue
                ov = _tpl_pseudo_ov(i, comp, _comp_rect(cfg, comp))
                ov.update({{"kind": "vartext", "text": comp.get("text") or "", "src_idx": i}})
                out.append(ov)
            except Exception:
                continue
    return out

def _dyn_overlays():
    # Horloges (mur + modèles) ET textes à VARIABLES : tout ce dont la valeur change en cours de
    # route, donc tout ce qui n'a rien à faire dans l'habillage caché.
    return ([ov for ov in OVERLAYS
             if (not ov.get("hidden")) and (ov.get("kind") == "clock"
                                            or (ov.get("kind") == "text"
                                                and "%" in str(ov.get("text") or "")))]
            + _tpl_clock_ovs() + _tpl_var_text_ovs())

# Valeur RENDUE d'un texte à variables, plafonnée dans le temps. En tuile per-frame, `_dyn_text`
# est appelé à CHAQUE trame : sans ce plafond, `%cpu%` serait relu 50 fois par seconde et le texte
# sauterait sans arrêt à l'écran — un moniteur de ressources illisible. Le chemin baké imposait
# déjà cette limite (_VARS_CHECK_S) ; elle suit le texte dans sa nouvelle machinerie, sinon on
# remplace un problème de performance par un problème d'affichage.
_vartext_cache = {{}}    # id du pseudo-overlay → (t, texte rendu)

def _vartext_value(ov):
    _k = (ov.get("id") or "", ov.get("src_idx"), ov.get("text") or "")
    _t, _v = _vartext_cache.get(_k, (0.0, None))
    _now = time.monotonic()
    if _v is None or (_now - _t) >= _VARS_CHECK_S:
        _i = ov.get("src_idx")
        _cfg = FLUX_CONFIG[_i] if isinstance(_i, int) and 0 <= _i < len(FLUX_CONFIG) else None
        _v = _expand_vars(ov.get("text") or "", _cfg)
        _vartext_cache[_k] = (_now, _v)
    return _v

def _dyn_text(ov, now):
    if ov.get("kind") in ("vartext", "text"):
        return _vartext_value(ov)
    return _format_clock(ov, now)

def render_overlays_fg(now):
    """PER-FRAME : uniquement les horloges (valeur qui change à chaque trame). Le texte/les images
    fixes sont dans le chrome caché (render_overlays_fg_static)."""
    clk = _dyn_overlays()
    if not clk:
        return None
    img = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    for ov in clk:
        _draw_text_overlay(d, ov, _dyn_text(ov, now), color_override=_countdown_color(ov, now))
    return img   # RGBA — consolidation : converti une seule fois après alpha_composite

_pf_tile_cache = {{}}   # clé d'élément → ((valeur, couleur, géométrie), tuile) — cf. render_clock_tiles

def render_clock_tiles(now):
    """Horloges en TUILES YUV — UNE par horloge sur sa propre bbox (alignée chroma), comme
    render_meters. Évite la bbox-UNION quasi plein écran quand les horloges sont DISPERSÉES (le blend
    per-frame coûtait ∝ surface englobante : 4 horloges éparpillées → ¾ d'écran blendés à chaque trame
    alors que chaque horloge fait ~60 k px). Le rendu PIL + conversion YUV ne sont payés qu'au
    changement de valeur (gate appelant) ; seul le blend par petite bbox reste per-frame. Renvoie
    [(bx0, by0, bx1, by1, y, u, v, a, a2), ...] ou None."""
    clk = _dyn_overlays()
    if not clk:
        return None
    tiles = []
    _vus = set()
    for ov in clk:
        # ── CACHE PAR ÉLÉMENT ────────────────────────────────────────────────────────────────
        # Le groupe entier était re-rendu dès qu'UN de ses éléments changeait. Avec un texte à
        # variables dedans (`%cpu%`, qui bouge toutes les 2 s), les horloges — qui ne changent
        # qu'à la seconde — étaient redessinées avec lui : pic d'`ov_clock` mesuré à 20,4 ms,
        # pour une moyenne de 1,7. C'est la troisième fois ce soir qu'un cache trop grossier
        # fait payer tout le monde au rythme du plus rapide.
        # Bénéfice second : une tuile inchangée garde son IDENTITÉ, donc le cache VRAM la
        # reconnaît et ne la retéléverse pas.
        _val = _dyn_text(ov, now)
        _col = _countdown_color(ov, now)
        _cle = ov.get("id") or ("%s@%s" % (ov.get("kind"), ov.get("x")))
        _vus.add(_cle)
        _hit = _pf_tile_cache.get(_cle)
        if _hit is not None and _hit[0] == (_val, _col, ov.get("x"), ov.get("y"),
                                            ov.get("w"), ov.get("h")):
            tiles.append(_hit[1])
            continue
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
        ovs = dict(ov); ovs["x"] = x - bx0; ovs["y"] = y - by0   # même overlay, coords LOCALES à la tuile
        _draw_text_overlay(dd, ovs, _dyn_text(ov, now), color_override=_countdown_color(ov, now))
        oy, ou, ovv, oa, oa2 = rgba_to_yuv(tile)
        _t = (bx0, by0, bx1, by1, oy, ou, ovv, oa, oa2)
        _pf_tile_cache[_cle] = ((_val, _col, ov.get("x"), ov.get("y"),
                                 ov.get("w"), ov.get("h")), _t)
        tiles.append(_t)
    # Purge des éléments disparus (horloge retirée, cellule masquée) : sans ça le cache
    # grossirait à chaque édition du mur.
    for _k in [_k for _k in _pf_tile_cache if _k not in _vus]:
        _pf_tile_cache.pop(_k, None)
    return tiles or None

dyn_rgba     = None              # couche d'habillage des MODÈLES (cachée, re-bake sur dirty)
overlay_fg_rgba = render_overlays_fg_static()   # overlays texte/images fixes, bakés dans le chrome
_chrome_pre  = None              # opérandes de blend PRÉ-CALCULÉS du chrome (inv_a, src_a par plan) — chemin rapide
_chrome_dirty = True             # force la 1re composition du chrome (drapeau INTERNE au boulanger)

# ─── ★ BOULANGER DU CHROME (0.42.0) — le re-bake d'habillage sort de la trame ──────────────────
# Le churn est mort (0.39.2 : plus de re-bake sur keepalive TSL), mais un VRAI changement (bascule
# de tally, texte UMD qui change) recomposait encore l'habillage PLEIN CADRE *dans la trame* :
# PIL Image.new 1920×1080 + alpha_composite + getbbox + crop + rgba_to_yuv + opérandes ≈ 25 ms,
# dans un budget de 20 ms → UNE TRAME PERDUE À CHAQUE BASCULE DE TALLY. Même arithmétique que les
# frises (0.40.0), donc même remède, éprouvé : un THREAD BOULANGER fabrique l'habillage HORS TRAME
# et le publie PRÊT ; la boucle de compo ne fait plus que ramasser (et, sur GPU, uploader les
# opérandes — quelques centaines de µs).
#   • Publication ATOMIQUE : une affectation d'un TUPLE À UN ÉLÉMENT (le payload, ou None quand il
#     n'y a pas de chrome) → la boucle compare l'IDENTITÉ de l'objet ; elle ne peut jamais voir une
#     publication à moitié écrite. Un seul écrivain par structure (le boulanger).
#   • Le boulanger travaille en NUMPY HÔTE de bout en bout (y compris le PRÉ-BLEND du canvas de
#     base) : aucun appel cupy hors du thread de compo → pas de question de stream/contexte GPU.
#   • ⚠ Le boulanger ne prend PAS `state_lock` pendant ses rendus : la boucle de compo le prend à
#     CHAQUE trame (ensure_input) — un boulanger qui le garderait 25 ms rendrait tout l'exercice
#     vain. Il baisse donc les drapeaux AVANT de rendre : une mutation concurrente les relève et
#     déclenche une nouvelle passe → convergence garantie (au pire un habillage intermédiaire vit
#     10 ms). Les lectures sont des `dict.get` / itérations de liste (atomiques sous le GIL).
#   • Latence : un changement de tally apparaît à la trame suivante (poll 5 ms) — imperceptible.
_chrome_pub = None               # (opérandes HÔTE du chrome | None,) — publié par le boulanger
_bg_pub     = None               # ((base_y, base_u, base_v) HÔTE pré-blendés | None,) — idem
_chrome_bake = {{"last": 0.0, "max": 0.0}}   # coût UNITAIRE d'une passe de boulange (HORS trame)
_chrome_bake_ctr = [0]
_statuses_pub = ()               # signature des statuts de tuile, publiée par la boucle de compo
# Suivi des VARIABLES DE TEXTE (%cpu%, %heure%…) : les textes sont BAKÉS dans l'habillage, re-rendu
# seulement sur tally_dirty. Sans surveillance, une variable resterait figée à sa valeur du
# déploiement — l'utilisateur croirait la fonction cassée. On ré-évalue à intervalle borné et on
# ne re-bake QUE si la valeur RENDUE a changé : une horloge %heure% coûte donc un bake par minute,
# un %cpu% un bake par échantillon, et un texte sans variable ne coûte RIEN.
_VARS_CHECK_S  = 2.0
_vars_last_ts  = 0.0
_vars_last_sig = None
_bake_wake = threading.Event()   # réveil du boulanger (changement de statut vu par la compo)
_BAKE_BAND_H = 120               # hauteur de bande du compositing PIL (granularité de relâche du GIL)

def _chrome_bake_pass():
    """UNE passe du boulanger : re-bake des couches CACHÉES sales, puis publication des opérandes
    HÔTE prêts à blender. ★ APPELÉE PAR LE THREAD BOULANGER SEULEMENT ★ (sauf l'amorce au boot)."""
    global overlay_fg_rgba, dyn_rgba, _info_layer, _info_sig, _chrome_pub, _bg_pub, _chrome_dirty
    _t0 = time.time_ns()
    _did = False

    # (a) FOND (layer=background) + overlays statiques : couche cachée. Le canvas de base est
    # PRÉ-BLENDÉ ici, hôte, hors trame — la boucle par-trame n'en fait plus qu'une copie.
    if overlay_dirty.is_set():
        overlay_dirty.clear()          # AVANT le rendu (cf. convergence ci-dessus)
        _bake_ctr["bg"] += 1
        _bg_rgba = render_overlays_bg()
        overlay_fg_rgba = render_overlays_fg_static()
        if _bg_rgba is not None:
            _oby, _obu, _obv, _oba, _oba2 = rgba_to_yuv(_bg_rgba)
            _bg_pub = ((blend(np.zeros((OUT_HEIGHT, OUT_WIDTH), dtype=_NP_DT), _oby, _oba),
                        blend(np.full((OUT_HEIGHT // _CH, OUT_WIDTH // _CW), _NEUTRAL, dtype=_NP_DT), _obu, _oba2),
                        blend(np.full((OUT_HEIGHT // _CH, OUT_WIDTH // _CW), _NEUTRAL, dtype=_NP_DT), _obv, _oba2)),)
        else:
            _bg_pub = (None,)
        _chrome_dirty = True; _did = True

    # (b) géométrie / tally / statuts → couches du chrome.
    if geom_dirty.is_set():
        geom_dirty.clear(); tally_dirty.set(); _info_sig = None
        _bake_ctr["geom"] += 1; _chrome_dirty = True
    if tally_dirty.is_set():
        tally_dirty.clear()
        dyn_rgba = render_dynamic()                      # composants bakés des modèles (géométrie incluse)
        overlay_fg_rgba = render_overlays_fg_static()    # texte tally-réactif (couleur on/off)
        _bake_ctr["tally"] += 1; _chrome_dirty = True
    _sig = _statuses_pub
    if _sig != _info_sig:
        _info_layer = render_info(list(_sig)) if _sig else None
        _info_sig = _sig
        _bake_ctr["info"] += 1; _chrome_dirty = True

    # (c) chrome consolidé (z-ordre info < dynamique < statique) → opérandes de blend HÔTE.
    if _chrome_dirty:
        _bake_ctr["chrome"] += 1
        _ch = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
        for _lyr in (_info_layer, dyn_rgba, overlay_fg_rgba):   # z-ordre conservé
            if _lyr is not None:
                # ★ COMPOSITION PAR BANDES + relâche explicite du GIL entre chaque bande.
                # `alpha_composite` plein cadre est UN appel C qui GARDE LE GIL du début à la fin
                # (Pillow n'y fait pas de ImagingSection) : 1920×1080×3 couches = plusieurs ms
                # pendant lesquelles la boucle de compo, même prioritaire, ne peut PAS se réveiller
                # → tick en retard → slot de grille raté. Sortir le bake du thread ne suffit donc
                # pas : il faut aussi qu'il ne SÉQUESTRE pas l'interpréteur. Mesuré : nice(10) seul
                # ne corrigeait rien (un thread niché qui tient le GIL bloque quand même) — c'est
                # bien la granularité de l'appel C qui compte. Bandes de 120 lignes → GIL rendu
                # ~9 fois par couche, hold ≈ 0,3 ms.
                for _yb in range(0, OUT_HEIGHT, _BAKE_BAND_H):
                    _yb1 = min(OUT_HEIGHT, _yb + _BAKE_BAND_H)
                    _ch.alpha_composite(_lyr.crop((0, _yb, OUT_WIDTH, _yb1)), (0, _yb))
                    time.sleep(0)     # point de commutation : la compo peut prendre le GIL ICI
        # Bornage à la BBOX réelle (chroma-alignée) : chrome épars → on ne blende pas tout l'écran.
        # Chrome entièrement transparent → publication None → blend totalement sauté (cas assembleur).
        _cbb = _ch.getbbox()
        if _cbb is None:
            _chrome_pub = (None,)
        else:
            bx0, by0, bx1, by1 = _cbb
            bx0 -= bx0 % _CW; by0 -= by0 % _CH
            if bx1 % _CW: bx1 = min(OUT_WIDTH, bx1 + (_CW - bx1 % _CW))
            if by1 % _CH: by1 = min(OUT_HEIGHT, by1 + (_CH - by1 % _CH))
            _cy, _cu, _cv, _ca, _ca2 = rgba_to_yuv(_ch.crop((bx0, by0, bx1, by1)))
            # inv_a=(255−α) et src_a=src·α par plan — arithmétique IDENTIQUE à l'ancien chemin,
            # simplement calculée hors trame ; la boucle ne fait plus que l'upload (_to_xp).
            _chrome_pub = ((bx0, by0, bx1, by1,
                            255 - _ca.astype(_ACC),  _cy.astype(_ACC) * _ca,
                            255 - _ca2.astype(_ACC), _cu.astype(_ACC) * _ca2, _cv.astype(_ACC) * _ca2),)
        _chrome_dirty = False; _did = True

    if _did:
        _ms = (time.time_ns() - _t0) / 1e6
        _chrome_bake["last"] = round(_ms, 2)
        if _ms > _chrome_bake["max"]:
            _chrome_bake["max"] = round(_ms, 2)
        _chrome_bake_ctr[0] += 1
    return _did

_bake_errs = [0.0]

# ─── Séparation des cœurs : la composition d'un côté, les boulangers de l'autre ──────────────
# `nice` ne suffit pas. Il arbitre le CPU, pas la bande passante mémoire, et surtout il n'empêche
# pas les deux fils d'atterrir sur le MÊME cœur : le conteneur est épinglé sur N cœurs et tous ses
# threads y flottent librement. Mesuré le 2026-08-08 : un boulangeage de frise coûte jusqu'à 44 ms
# et les pics de `own` de la compo valent exactement ça — le fil de compo est bien stallé pendant
# qu'une passe s'exécute, alors qu'elle est censée être parallèle. Or le shard n'utilise que
# 0,83 cœur sur 3 : ce n'est pas la puissance qui manque, c'est l'isolement.
# ⛔ ESSAYÉ, ÉCARTÉ (2026-08-08) — désactivé par défaut, `cpu_split` pour le rejouer.
# Réserver le premier cœur du cpuset à la composition et reléguer les boulangers sur les autres a
# DÉGRADÉ le mur : 39,2 fps contre 43,7, `own` 17,5 contre 15,5, et surtout des pics INCHANGÉS à
# 45-50 ms. Deux enseignements :
#   • confiner la composition sur UN cœur lui fait perdre ce qu'elle tirait des trois (les kernels
#     numpy/mvk et les threads internes de cupy s'y répartissaient) ;
#   • les pics n'ayant pas bougé d'un millimètre, la cause du stall n'est PAS la concurrence CPU.
# Avec le nice (sans effet) et le boulanger dédié (pire), c'est le troisième essai de parallélisme
# qui échoue sans déplacer les pics. L'explication qui reste, et qui les couvre tous les trois :
# le GIL. Un boulangeage PIL de 40 ms qui ne relâche pas le verrou global bloque le fil de
# composition quel que soit son cœur et quelle que soit sa priorité. Aucun épinglage n'y peut rien
# — il faudrait un PROCESSUS séparé, ou rendre la passe bon marché (ce qu'a fait l'atlas de
# glyphes pour le bandeau ANC : 8,21 → 1,37 ms, et ce gain-là, lui, a tenu).
# L'affinité est PAR THREAD sous Linux et s'HÉRITE à la création : chaque boulanger pose donc la
# sienne au démarrage, et la compo pose la sienne APRÈS le lancement des threads — sinon ils
# naîtraient tous confinés sur le cœur de la compo, le contraire du but.
_CPUS = []
try:
    _CPUS = sorted(os.sched_getaffinity(0))
except Exception:                                                          # noqa: BLE001
    _CPUS = []
_CPU_SPLIT = len(_CPUS) >= 2 and _as_bool(CONFIG.get("cpu_split", False), False)

def _nice_baker():
    """Le boulanger est un thread SECONDAIRE : sous contention, la boucle de composition doit
    gagner. Sous Linux, `nice` est PAR THREAD (setpriority(PRIO_PROCESS, 0) = thread courant) →
    on peut dé-prioriser le boulanger sans toucher à la compo. Sans ça, sur un nœud sans cœur
    libre, les 40-55 ms de PIL du boulanger volent le cœur (ou le GIL) au tick de la compo, qui
    se réveille en retard et rate son slot de grille — le pic serait sorti de la trame pour
    revenir par la fenêtre. Best-effort (échec silencieux si la plateforme refuse).

    Pose AUSSI l'affinité : tous les cœurs SAUF le premier, réservé à la composition."""
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
    except Exception:
        pass
    if _CPU_SPLIT:
        try:
            os.sched_setaffinity(0, set(_CPUS[1:]))
        except Exception:                                                  # noqa: BLE001
            pass

def _chrome_baker_loop():
    """Thread BOULANGER de l'habillage. Sérialisé (une passe à la fois) : à quelques bakes/s son
    taux d'occupation est de quelques pour cent."""
    _nice_baker()
    while True:
        try:
            _chrome_bake_pass()
        except Exception as _e:
            # ⛔ jamais d'échec muet : on log (throttlé) et on RE-LÈVE le drapeau → nouvelle tentative.
            _chrome_dirty_retry = time.time()
            if _chrome_dirty_retry - _bake_errs[0] > 5.0:
                _bake_errs[0] = _chrome_dirty_retry
                log(f"multiview: boulanger chrome — échec de re-bake : {{_e!r}}", "warning")
            tally_dirty.set()
            time.sleep(0.05)
        _bake_wake.wait(0.005)
        _bake_wake.clear()
# Cache de la couche PER-FRAME des HORLOGES (les VU-mètres ont leur propre chemin par tuiles, jamais
# caché) : YUV+bbox réutilisés tant que la valeur affichée ne change pas (re-render+convert
# DÉTERMINISTE, 1×/s sans le champ images).
_pf_cache_sig = None
_pf_tiles     = None   # tuiles YUV des horloges (une par horloge) — recalculées au changement de valeur
_clk_cache_sig = None  # signature des HORLOGES (valeur affichée + couleur de compte à rebours)
_anc_cache_sig = None  # signature des BANDEAUX ANC (texte décodé + alignement)
_clk_tiles    = []     # tuiles des horloges, conservées tant que leur signature ne bouge pas
_anc_tiles    = []     # tuiles des bandeaux ANC, idem

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
_frame_neuve = False   # trame courante : au moins une entrée a avancé (cf. `fps_content`)
# i → nb de reconnexions CONSÉCUTIVES sans lecture fraîche derrière. Remis à 0 dès qu'une lecture
# fraîche revient. ≥ 2 ⇒ le close+GC+reopen n'a pas suffi (il reste une référence sur la génération
# morte DANS NOTRE Instance) ⇒ ensure_input escalade sur une Instance MXL dédiée à cette entrée.
_stale_drops = {{}}

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
            _close_source(cur)
            with state_lock:
                if i < len(sources): sources[i] = None
        return None
    if cur is not None and cur.get("path") == wanted:
        return cur
    if cur is not None:
        _close_source(cur)
        with state_lock:
            if i < len(sources): sources[i] = None
    # ESCALADE : au 2ᵉ décrochage consécutif de la MÊME entrée, on ouvre sur une Instance dédiée
    # — cache vierge, donc résolution du flux SUR DISQUE (banc : seul chemin qui récupère quand le
    # GC est bloqué par une référence résiduelle de notre propre Instance).
    src = open_source({{"path": wanted, "in_w": cfg_iw, "in_h": cfg_ih}},
                      own_instance=(_stale_drops.get(i, 0) >= 2))
    if src is not None:
        src["path"] = wanted
        with state_lock:
            if i < len(sources): sources[i] = src
    return src

# ─── LIBÉRATION DIFFÉRÉE DES MAPPINGS (course mortelle, corrigée le 2026-08-06) ─────────────
#
# La boucle de composition lit les plans source comme des VUES numpy sur le mapping partagé d'un
# flux MXL (`tv[1][r0:r1]`, `tv[2][rc][:, tv[5]]`…), SANS prendre `state_lock` — elle ne peut pas,
# la cadence s'effondrerait. Or trois autres threads libéraient ces mappings sous ses pieds :
#   • l'échantillonneur des frises (`_vh_close`) : ferme le Reader PUIS collecte le ring ;
#   • le gestionnaire HTTP (`/reconfigure`) : ferme les Readers dont la source a changé ;
#   • les chemins de réouverture d'un flux périmé.
# Une zone démappée donne un SIGSEGV, pas un SIGBUS : le handler SIGBUS existant ne la couvre pas.
# Le mur mourait ainsi 34 fois en une journée. Les 5 traces `faulthandler` désignent toutes le
# thread de composition dans `_gather_band`, et l'une d'elles capture les DEUX bouts de la course
# dans le même dump (compose dans `_gather_band` ; l'échantillonneur dans `garbage_collect`).
#
# Règle : un mapping n'est libéré que DEPUIS LE THREAD DE COMPOSITION, en début de trame, seul
# instant où l'on sait qu'aucune vue n'est vivante. Les autres threads mettent en file.
# ⚠ On ne diffère PAS les appels VENANT du thread de composition : `ensure_input` ferme puis
# ROUVRE dans la foulée, et son `garbage_collect()` est justement ce qui la rattache à la
# génération vivante — le différer casserait la reconnexion.
_thread_compo = None                  # ident du thread de composition (posé avant la boucle)
_a_liberer = []                       # sources en attente de fermeture
_a_liberer_lock = threading.Lock()
_gc_demande = False                   # un garbage_collect a été demandé par un autre thread


def _sur_thread_compo():
    # `None` = boucle de composition pas encore armée (démarrage) : aucune vue ne peut être
    # vivante, on libère immédiatement. Différer là serait un piège : les réouvertures du
    # démarrage rateraient la génération vivante du flux.
    return _thread_compo is None or threading.get_ident() == _thread_compo


def _fermer_source_maintenant(src):
    """Ferme le Reader d'une source ET son Instance dédiée s'il en a une (sinon on fuit une
    Instance MXL par escalade — et la génération qu'elle référence ne serait jamais collectée)."""
    try: src["reader"].close()
    except Exception: pass
    own = src.get("own_inst")
    if own is not None:
        try: own.close()
        except Exception: pass


def _close_source(src):
    """Ferme une source — IMMÉDIATEMENT depuis le thread de composition, en DIFFÉRÉ sinon."""
    if _sur_thread_compo():
        _fermer_source_maintenant(src)
        return
    with _a_liberer_lock:
        _a_liberer.append(src)


def _gc_mxl():
    """`inst.garbage_collect()` — global au ring MXL, donc encore plus dangereux qu'une
    fermeture : il purge des GÉNÉRATIONS entières. Même règle que `_close_source`."""
    global _gc_demande
    if _sur_thread_compo():
        try: inst.garbage_collect()
        except Exception: pass
        return
    with _a_liberer_lock:
        _gc_demande = True


def _liberer_differe():
    """Purge la file — À APPELER EN DÉBUT DE TRAME, avant toute construction de vue."""
    global _gc_demande
    with _a_liberer_lock:
        lot = _a_liberer[:]
        del _a_liberer[:]
        gc = _gc_demande
        _gc_demande = False
    for src in lot:
        _fermer_source_maintenant(src)
    if lot or gc:
        try: inst.garbage_collect()
        except Exception: pass
        if len(lot) > 8:
            log("multiview: %d source(s) libérée(s) en différé" % len(lot), "debug")


def _drop_input(i, rd, raison=""):
    """Ferme le Reader de l'entrée i et oublie la source (→ ensure_input la ROUVRE à la frame
    suivante sur la génération courante du flux). Utilisé pour reconnecter un Reader périmé
    (flux amont recréé sous le même nom) que le cache de ensure_input ne rouvrirait jamais."""
    with state_lock:
        _cur = sources[i] if i < len(sources) else None
    _close_source(_cur if isinstance(_cur, dict) else {{"reader": rd}})
    # GC OBLIGATOIRE entre close et réouverture (même parade que le moteur, tx_reopen_if_stale) :
    # le flux périmé reste résolvable PAR NOM tant qu'il n'est pas collecté — sans GC, la
    # réouverture retombe sur L'ORPHELIN qu'on vient de lâcher (mesuré : boucle drop/reopen à
    # ½ cadence sur un shard dont le proxy amont avait été recréé).
    try: _gc_mxl()
    except Exception: pass
    with state_lock:
        if i < len(sources): sources[i] = None
    # TRACE OBLIGATOIRE (niveau « warning » : passe même en log_level=info, et ce n'est PAS une
    # métrique — c'est un événement rare qui signe une recréation de flux amont). Sans elle on ne
    # peut pas distinguer « jamais déclenché » de « boucle sans effet » : c'est exactement ce qui a
    # coûté le plus de temps sur l'incident du 2026-07-26 (shards totalement muets).
    _n = _stale_drops.get(i, 0) + 1
    _stale_drops[i] = _n
    # Source réellement absente/arrêtée → on retente indéfiniment : on ne garde en « warning » que
    # les 3 premières tentatives puis une ligne par centaine (RÈGLE 3 : pas de rafale au journal).
    log("multiview: entrée %d (%s) — Reader PÉRIMÉ%s, reconnexion (tentative %d)%s"
        % (i, _cur.get("path", "?") if isinstance(_cur, dict) else "?",
           (" : " + raison) if raison else "", _n,
           " → Instance DÉDIÉE" if _n >= 2 else ""),
        "warning" if (_n <= 3 or _n % 100 == 0) else "debug")
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
    def _do_telemetry(self):
        """Télémétrie du NŒUD et du CONTRÔLEUR, poussée par l'orchestrateur. Remplace le cache en
        bloc : une poussée partielle vaut mieux qu'un panachage de valeurs d'âges différents."""
        b = self._json()
        if not isinstance(b, dict):
            self.send_response(400); self.end_headers(); return
        _TELEMETRIE.clear()
        # `noeuds` (au pluriel) porte TOUS les nœuds nommés : c'est lui qui permet à un mur de
        # supervision d'afficher une autre machine que la sienne (`%cpu_noeud:dl360-1%`). L'oublier
        # ici faisait silencieusement retomber tout ciblage sur « — ».
        for _sec in ("noeud", "orchestrateur", "noeuds"):
            if isinstance(b.get(_sec), dict):
                _TELEMETRIE[_sec] = b[_sec]
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.end_headers(); self.wfile.write(b'{{"ok": true}}')

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
        if self.path == "/clock_tz":
            return self._do_clock_tz()
        if self.path == "/overlay_text":
            return self._do_overlay_text()
        if self.path == "/log_level":
            return self._do_log_level()
        if self.path == "/telemetry":
            return self._do_telemetry()
        if self.path != "/input":
            self.send_response(404); self.end_headers(); return
        b = self._json()
        # Ports des BLOCS D'HISTORIQUE de mur (page Câbles) : vidéo (`vh_idx` → `path`) et audio
        # (`ah_idx` → `audio_path`). Clés d'entrée DISTINCTES de `idx`/`block_idx` (sinon un câblage
        # de bloc écraserait la fenêtre du même indice numérique). Les rings d'échantillonnage sont
        # keyés par NOM DE FLUX → le simple changement de source suffit (le thread ferme l'ancien).
        for _hk, _hlist, _hfield in (("vh_idx", VHIST_BLOCKS, "path"),
                                     ("ah_idx", AHIST_BLOCKS, "audio_path")):
            if _hk in b:
                try: hidx = int(b.get(_hk))
                except Exception:
                    self.send_response(400); self.end_headers(); return
                shm = (b.get("shm") or "").strip()
                ok = False
                with state_lock:
                    if 0 <= hidx < len(_hlist):
                        _hlist[hidx][_hfield] = ("/dev/shm/" + shm) if shm else ""
                        ok = True
                if ok:
                    _hist_cache.clear()
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({{"ok": ok}}).encode())
                return
        # Port AUDIO d'un bloc VU-mètres de MUR (câblage page Câbles, `wiring.consumes`
        # `from_list: meter_blocks`, `input_key: block_idx` — DISTINCT de `idx` des fenêtres,
        # sinon un câblage de bloc écraserait l'entrée FLUX_CONFIG du même indice numérique).
        if "block_idx" in b:
            try: bidx = int(b.get("block_idx"))
            except Exception:
                self.send_response(400); self.end_headers(); return
            shm = (b.get("shm") or "").strip()
            path = ("/dev/shm/" + shm) if shm else ""
            ok = False
            with state_lock:
                if 0 <= bidx < len(METER_BLOCKS):
                    METER_BLOCKS[bidx]["audio_path"] = path; ok = True
            if ok:
                for k in list(audio_states):
                    if isinstance(k, tuple) and len(k) == 2 and k[0] == ("mb", bidx):
                        st = audio_states.pop(k, None)
                        try:
                            if st and st.get("ar"):
                                st["ar"].close()
                        except Exception:
                            pass
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps({{"ok": ok}}).encode())
            return
        try: idx = int(b.get("idx"))
        except Exception:
            self.send_response(400); self.end_headers(); return
        shm = (b.get("shm") or "").strip()
        path = ("/dev/shm/" + shm) if shm else ""
        ok = False
        # essence `data` = port ANC de l'entrée (un flux ANC par entrée vidéo). Le lecteur ANC
        # est rouvert tout seul au prochain accès (_anc_packets compare `for_path`).
        _ess = b.get("essence") or "video"
        if _ess == "data":
            with state_lock:
                if 0 <= idx < len(FLUX_CONFIG):
                    FLUX_CONFIG[idx]["anc_path"] = path; ok = True
        elif _ess == "audio":
            # Port AUDIO câblé de l'entrée (VU-mètres) : les états audio de la tuile sont
            # PURGÉS (clé legacy i + clés (i, flux)) → réouverture sur le nouveau nom au
            # prochain rendu (le nom est capturé à l'ouverture du state).
            with state_lock:
                if 0 <= idx < len(FLUX_CONFIG):
                    FLUX_CONFIG[idx]["audio_path"] = path; ok = True
            if ok:
                for k in list(audio_states):
                    if k == idx or (isinstance(k, tuple) and k and k[0] == idx):
                        st = audio_states.pop(k, None)
                        try:
                            if st and st.get("ar"):
                                st["ar"].close()
                        except Exception:
                            pass
        else:
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
                for k in ("x", "y", "w", "h", "tsl_index", "meter_channels", "meter_opacity",
                          "anc_opacity") + ANC_FIELDS:
                    if k in b and b[k] is not None:
                        try: cfg[k] = int(b[k])
                        except (TypeError, ValueError): pass
                # anc_position est une CHAÎNE (top/bottom) — avant 0.29.0 elle passait dans la
                # coercition int ci-dessus et n'était donc JAMAIS appliquée à chaud.
                for k in ("name", "meter_position", "meter_scale", "anc_position"):
                    if k in b and b[k] is not None:
                        cfg[k] = str(b[k])
                # audio_path : sélecteur de source audio du composer (3 états, cf. _audio_name_for
                # / AUDIO_PATH_NONE) — même effet que le câblage page Câbles (essence=audio sur
                # /input), donc même purge des états audio ouverts (le nom résolu peut changer).
                if "audio_path" in b and b["audio_path"] is not None:
                    cfg["audio_path"] = str(b["audio_path"])
                    for k in list(audio_states):
                        if k == idx or (isinstance(k, tuple) and k and k[0] == idx):
                            st = audio_states.pop(k, None)
                            try:
                                if st and st.get("ar"):
                                    st["ar"].close()
                            except Exception:
                                pass
                for k in ("show_label", "show_tally", "meter_inside", "hidden", "label_proportional"):
                    if k in b and b[k] is not None:
                        cfg[k] = _as_bool(b[k])
                # Modèle de PiP : dict {{name, components}} appliqué à chaud, null/{{}} = retour
                # à l'héritage (défaut du mur, sinon modèle « Classique » généré).
                if "template" in b:
                    t = b.get("template")
                    if isinstance(t, dict) and t.get("components"):
                        cfg["template"] = t
                    else:
                        cfg.pop("template", None)
                ok = True
                geom_dirty.set()
                tally_dirty.set()
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({{"ok": ok}}).encode())
    def _do_style(self):
        # Params globaux visuels restants : default_template (modèle de PiP par défaut du mur),
        # show_no_signal, freeze_detect_s, show_proxy. L'habillage de mur historique
        # (border_w/overlay_below/label_size/frame_style/show_format) vit dans les MODÈLES.
        b = self._json()
        with state_lock:
            global SHOW_NO_SIGNAL, FREEZE_DETECT_S, SHOW_PROXY, DEFAULT_TEMPLATE
            # ★ IDEMPOTENT : le tissu pousse /style à chaque reconfiguration d'assembleur, avec la
            # même valeur qu'avant. Armer geom+tally sans changement = re-bake du chrome plein
            # cadre (≈ 25 ms mesurées) → une trame lente → le TX ré-émet le grain précédent.
            _av_style = (DEFAULT_TEMPLATE, SHOW_NO_SIGNAL, FREEZE_DETECT_S, SHOW_PROXY)
            if "default_template" in b:
                # Modèle de PiP par défaut du MUR (héritage) : dict {{name, components}} ou null.
                t = b.get("default_template")
                DEFAULT_TEMPLATE = t if (isinstance(t, dict) and t.get("components")) else None
            if "show_no_signal" in b:
                SHOW_NO_SIGNAL = _as_bool(b["show_no_signal"])
            if "freeze_detect_s" in b:
                try: FREEZE_DETECT_S = max(0.0, float(b["freeze_detect_s"]))
                except (TypeError, ValueError): pass
            if "show_proxy" in b:
                SHOW_PROXY = _as_bool(b["show_proxy"])
            if (DEFAULT_TEMPLATE, SHOW_NO_SIGNAL, FREEZE_DETECT_S, SHOW_PROXY) != _av_style:
                geom_dirty.set()
                tally_dirty.set()
        self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({{"ok": True}}).encode())
    def _do_log_level(self):
        # Verbosité À CHAUD (pas de redéploiement) : passer un mur en `debug` pendant un
        # incident, puis le remettre en `warning`. Le niveau persistant reste le champ
        # `log_level` du config_schema (celui-ci est volatil, perdu au redéploiement).
        b = self._json()
        ok = set_log_level(b.get("level") or b.get("log_level"))
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({{"ok": ok, "log_level": LOG_LEVEL}}).encode())
    def _do_reconfigure(self):
        # Remplacement atomique de toute la liste de sources + géométrie.
        # Permet l'ajout/suppression de fenêtres à chaud depuis l'éditeur.
        b = self._json()
        # Polices de la bibliothèque (clés `lib:*`) : le hot-apply peut introduire une police que
        # le conteneur n'a pas encore (l'orchestrateur ne repousse le script qu'au déploiement) →
        # on rematérialise ce que la charge utile embarque AVANT d'appliquer la nouvelle config,
        # sinon le texte retomberait silencieusement sur DejaVu jusqu'au prochain redéploiement.
        # ★ IDEMPOTENT (même motif que les frises/blocs plus bas) : le tissu ré-embarque la
        # bibliothèque de polices à CHAQUE poussée. Re-matérialiser à l'identique purgeait le cache
        # de polices et armait tally+overlay_dirty → RECUISSON plein cadre pour rien. On ne bouge
        # que si la bibliothèque a réellement changé.
        if b.get("font_library") and b["font_library"] != (CONFIG.get("font_library") or []):
            CONFIG["font_library"] = b["font_library"]
            _materialize_font_library()
            _ofont_cache.clear()
            tally_dirty.set()
            overlay_dirty.set()
        new_fc = b.get("flux_config") or []
        # Blocs VU-mètres du MUR : même remplacement atomique (ajout/suppression/réordonnement à
        # chaud depuis le composer). On purge TOUS les états audio de blocs (clé ("mb", j)) —
        # les indices peuvent avoir bougé (réordonnement/suppression), les rouvrir est cheap.
        new_mb = b.get("meter_blocks") or []
        # Blocs d'HISTORIQUE (vidéo/audio) : même remplacement atomique. Les états d'échantillonnage
        # (_vh/_ah) sont keyés par NOM DE FLUX, pas par indice → ils survivent au réordonnement ; les
        # threads d'échantillonnage ferment d'eux-mêmes ceux qui ne sont plus demandés. Seul le cache
        # de TUILES (keyé par indice de bloc) est purgé.
        # ★ IDEMPOTENT : ne purger le cache QUE si les blocs ont RÉELLEMENT changé. Le tissu
        # repousse la config au mur toutes les ~30 s même quand rien ne bouge ; purger à chaque
        # fois faisait DISPARAÎTRE les frises pendant la ou les trames que met le boulanger à les
        # refabriquer → CLIGNOTEMENT périodique du bas de l'image, observé en prod (mur 333 : les
        # 3 frises à 0 pendant 1 trame, toutes les ~35 s, mesuré sur le flux de sortie décodé).
        # ★ PERF : le tissu repousse cette config au mur toutes les ~30 s MÊME sans changement.
        # On accumule ici si QUELQUE CHOSE a réellement bougé (frises, blocs VU, flux) et on
        # n'arme geom_dirty/tally_dirty (→ re-bake chrome PLEIN CADRE ≈ 25 ms) qu'à ce prix —
        # sinon chaque poussée identique re-bakait l'habillage pour rien (churn CPU périodique).
        _reconf_changed = False
        if "video_history_blocks" in b or "audio_history_blocks" in b:
            _nv = b.get("video_history_blocks") or []
            _na = b.get("audio_history_blocks") or []
            _change = (_nv != list(VHIST_BLOCKS)) or (_na != list(AHIST_BLOCKS))
            if _change:
                _reconf_changed = True
                with state_lock:
                    VHIST_BLOCKS[:] = _nv
                    AHIST_BLOCKS[:] = _na
                _hist_cache.clear()
        # ★ IDEMPOTENT (même raison que les frises) : rouvrir les Readers audio des blocs VU à chaque
        # poussée du tissu ferait retomber les VU-mètres à zéro le temps de la réouverture. On ne
        # touche à rien si la liste est identique.
        if new_mb != list(METER_BLOCKS):
            _reconf_changed = True
            with state_lock:
                METER_BLOCKS[:] = new_mb
                for k in list(audio_states):
                    if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], tuple) and k[0] and k[0][0] == "mb":
                        st = audio_states.pop(k, None)
                        try:
                            if st and st.get("ar"):
                                st["ar"].close()
                        except Exception:
                            pass
        with state_lock:
            # PRÉSERVATION des Readers inchangés : un ajout/déplacement/retrait de PiP ne doit PAS
            # figer les AUTRES tuiles. L'ancien code fermait TOUS les Readers + remettait sources à
            # None → chaque tuile inchangée se figeait le temps de la ré-ouverture (et ne récupérait
            # pas toujours seule → re-câblage manuel). On ne ferme désormais QUE les Readers dont la
            # source (chemin câblé) a changé ou qui disparaissent ; les tuiles inchangées gardent
            # leur Reader VIVANT. Les indices de banque sont stables (0.9.0) → appariement par index.
            # Changement réel du flux (positions/tailles/sources/labels) → re-bake légitime.
            _fc_changed = (new_fc != list(FLUX_CONFIG))
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
                    _close_source(src)
            FLUX_CONFIG[:] = new_fc
            mv_state["inputs"][:] = new_inputs
            sources[:] = new_sources
            _in_track.clear(); _stale_drops.clear()
            _stale_since.clear()
            if _fc_changed or _reconf_changed:
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
            # ★ IDEMPOTENT : `overlay_dirty` déclenche le re-bake de la couche de FOND + des
            # overlays statiques — le plus cher de tous (canvas RGBA plein cadre → rgba_to_yuv →
            # blends). Le tissu repousse la liste d'overlays à chaque reconfiguration d'assembleur,
            # or elle n'a le plus souvent pas bougé : on ne salit que sur changement réel.
            _ov_changed = (new_ov != list(OVERLAYS))
            OVERLAYS[:] = new_ov
            live = {{ov.get("id") for ov in new_ov}}
            for cid in list(_overlay_img_cache):
                if cid not in live:
                    _overlay_img_cache.pop(cid, None)
            for cid in list(_chrono_state):
                if cid not in live:
                    _chrono_state.pop(cid, None)
            if _ov_changed:
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
    def _do_clock_tz(self):
        """Fuseau d'UNE horloge, à chaud : {{id, tz}}. `tz` vide = revenir au fuseau du mur.
        Cible les overlays GLOBAUX et les composants `clock` des MODÈLES de PiP (mêmes ids que
        /chronos). Un fuseau inconnu de l'image est REFUSÉ ici (400) plutôt qu'accepté puis replié
        en silence sur le mur — l'appelant (macro, UI) doit savoir que son réglage n'a pas pris."""
        b = self._json()
        cid = str(b.get("id") or "")
        tzn = (b.get("tz") or "").strip()
        if not cid:
            self.send_response(400); self.end_headers(); return
        if tzn and _zone(tzn) is None:
            self.send_response(400); self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({{"ok": False, "error": "fuseau inconnu : %s" % tzn}}).encode())
            return
        n = 0
        with state_lock:
            for ov in OVERLAYS:
                if str(ov.get("id") or "") == cid and ov.get("kind") == "clock":
                    ov["tz"] = tzn; n += 1
            for cfg in FLUX_CONFIG:
                for comp in (_tpl_comps(cfg) or ()):
                    if (isinstance(comp, dict) and comp.get("type") == "clock"
                            and str(comp.get("id") or "") == cid):
                        comp["tz"] = tzn; n += 1
        # Pas de purge de cache à faire : les tuiles d'horloge sont cachées par SIGNATURE = les
        # chaînes formatées (_pf_sig). Changer de fuseau change l'heure affichée, donc la
        # signature, donc le rendu est refait à la trame suivante.
        self.send_response(200 if n else 404)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({{"ok": bool(n), "applied": n}}).encode())

    def _do_overlay_text(self):
        # Change le TEXTE d'un overlay texte (par id) via la couche overlay_central (même
        # mécanisme que le push TSL). Pilotable par macro/déclencheur : {{id, text}}.
        b = self._json()
        oid = str(b.get("id") or "")
        if not oid:
            self.send_response(400); self.end_headers(); return
        with state_lock:
            overlay_central[oid] = {{"text": str(b.get("text") or ""), "active": True}}
        overlay_dirty.set()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({{"ok": True}}).encode())
    def do_GET(self):
        # Listes pour les sélecteurs d'action (horloges / champs texte) : {{items:[{{value,label}}]}}.
        # Fuseaux DISPONIBLES DANS L'IMAGE (tzdata réellement présente), pas une liste codée en
        # dur : c'est la seule qui garantit qu'un choix de l'utilisateur sera applicable. Le mur
        # est renvoyé à part pour que l'UI puisse étiqueter l'option « hérite du mur ».
        if self.path == "/timezones":
            try:
                from zoneinfo import available_timezones
                _zs = sorted(available_timezones())
            except Exception as _e:
                log("liste des fuseaux indisponible (%s)" % _e, "warning")
                _zs = []
            _b = json.dumps({{"wall_tz": _TZ_NAME,
                              "items": [{{"value": "", "label": "Fuseau du mur (%s)"
                                          % (_TZ_NAME or "système")}}]
                                       + [{{"value": z, "label": z}} for z in _zs]}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(_b); return
        if self.path in ("/chronos", "/texts"):
            want = "clock" if self.path == "/chronos" else "text"
            with state_lock:
                items = [{{"value": ov.get("id"), "label": ov.get("name") or ov.get("id")}}
                         for ov in OVERLAYS if ov.get("kind") == want and ov.get("id")]
            _b = json.dumps({{"items": items}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(_b); return
        # /state : entrées câblées + LISTE DES FENÊTRES (PiP) avec leurs params courants +
        # OVERLAYS (texte/horloge) + bornes (caps) → l'éditeur de macros expose chaque
        # élément. Handler rare (pas la boucle de rendu) ; snapshot sous state_lock.
        with state_lock:
            inp = list(mv_state["inputs"])
            wins = [dict(c) for c in FLUX_CONFIG]
            ovs = [{{"id": ov.get("id"), "kind": ov.get("kind"),
                     "label": ov.get("name") or ov.get("id"),
                     "text": overlay_central.get(str(ov.get("id") or ""), {{}}).get("text")
                             or ov.get("text") or ""}}
                   for ov in OVERLAYS if ov.get("id")]
        payload = {{
            "inputs":    inp,
            "mxl_lib":   _mxl_lib_state(),
            "log_level": LOG_LEVEL,   # lisible en CONDITION de macro (« si le mur est en debug »)
            "n_windows": len(wins),
            "canvas":    [OUT_WIDTH, OUT_HEIGHT],
            "windows":   wins,
            "overlays":  ovs,
            # Bornes des champs de fenêtre (l'UI lit caps, rien en dur) : pixels bornés au
            # canvas de sortie, opacités en %, index TSL entier.
            "caps": {{"window_fields": {{
                "x": [0, OUT_WIDTH, 0],          "y": [0, OUT_HEIGHT, 0],
                "w": [2, OUT_WIDTH, OUT_WIDTH],  "h": [2, OUT_HEIGHT, OUT_HEIGHT],
                "meter_opacity": [0, 100, 100],  "anc_opacity": [0, 100, 100],
                "tsl_index": [0, 255, 0],
            }}}},
        }}
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps(payload).encode())
    def log_message(self, *a): pass

threading.Thread(
    target=lambda: HTTPServer(("0.0.0.0", 8082), MvControlHandler).serve_forever(),
    daemon=True).start()

# Injection de proxies pyramide À CHAUD : scrute /dev/shm en continu (cf. _proxy_scan_loop).
threading.Thread(target=_proxy_scan_loop, daemon=True).start()

# Historiques vidéo/audio : échantillonnage HORS de la boucle de mix (threads dédiés). Aucune unité
# configurée → les boucles ne font que dormir (coût strictement nul sur les murs existants).
threading.Thread(target=_vhist_loop, daemon=True).start()
threading.Thread(target=_ahist_loop, daemon=True).start()
# ★ BOULANGER des frises (0.40.0) : la RECOMPOSITION (6-27 ms) sort de la boucle de composition —
# elle ne fait plus tomber de trame. Voir le bloc de commentaire sur `_hist_want`.
threading.Thread(target=_hist_baker_loop, daemon=True).start()
# ★ BOULANGER de l'HABILLAGE (0.42.0) : le re-bake du chrome (≈25 ms plein cadre) sort lui aussi de
# la trame. Amorce SYNCHRONE d'abord — la 1re trame doit sortir habillée, pas nue.
try:
    _chrome_bake_pass()
except Exception as _e:
    log(f"multiview: amorce du chrome échouée ({{_e!r}}) — le boulanger réessaiera", "warning")
threading.Thread(target=_chrome_baker_loop, daemon=True).start()
# Ressources (%cpu%, %ram%) : relevées hors trame, cf. _var_sample_loop.
threading.Thread(target=_var_sample_loop, daemon=True).start()

# Mode tranche : le flowDef porte slice_height → libmxl publie le grain en N tranches égales
# (commit progressif). Writer forwarde **flow_kw à build_flow_def. Sans slice : inchangé (1 tranche).
out_writer = bobimxl.Writer(inst, SHM_OUT_NAME, OUTPUT_W, OUTPUT_H, CHROMA, BIT_DEPTH,
                            _FN, _FD, interlace=IL_MODE,
                            **({{"slice_height": SLICE_LINES}} if SLICE_ON else {{}}))
if INTERLACED:
    # Le mur COMPOSE toujours une trame PLEINE (OUTPUT_H) : tout le rendu (tuiles, habillage,
    # VU, frises) est inchangé. Seule l'ÉCRITURE change : la trame est découpée en 2 champs.
    log(f"multiview: sortie ENTRELACÉE {{OUTPUT_W}}x{{OUTPUT_H}}{{FIELD_ORDER}} — "
        f"{{_FN}}/{{_FD}} trames/s, 2 grains-champs de {{OUT_FIELD_SIZE}} o par trame", "info")
if SLICE_ON:
    log(f"multiview: MODE TRANCHE actif — {{SLICE_LINES}} lignes/bande "
        f"({{OUT_HEIGHT // SLICE_LINES}} tranches/trame)", "info")
    metrics["slice_mode"] = True
out_frame_index = 0
start_time = time.time()
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
    décalée d'1 image) — la sortie n'est JAMAIS bloquée par une source.
    GPU + gpu_batch_bands>1 (micro-batch, GO-MEGA-135) : mêmes attentes/budgets/backoff/ciblage
    fi_out au grain SLICE_LINES, mais H2D/kernels/D2H opèrent par LOT de grp bandes, avec D2H
    recouvert (stream dédié) et commits toujours au grain SLICE_LINES (phase +1 lot)."""
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
    # MICRO-BATCH GPU (§b var.2, verdict GO-MEGA-135) : grp = bandes MXL par LOT GPU. Les attentes
    # get_slice et les commits restent au grain SLICE_LINES ; seuls H2D/kernels/D2H grossissent au
    # lot (8 lancements/trame à 1080p/36 l/k=4, au lieu de 30 — le coût de lancement dominait).
    grp = GPU_BATCH_BANDS if GPU_SLICE else 1
    if GPU_SLICE and grp <= 1:
        # Buffers de bande de SORTIE épinglés (persistants) : D2H par bande via .get(out=épinglé)
        # — un .get direct dans le grain (mmap non épinglé) serait ~4× plus lent (banc Phase 0).
        if _slgpu["hy"] is None:
            def _mkpin(h, w):
                return np.frombuffer(cp.cuda.alloc_pinned_memory(h * w * _BPS),
                                     dtype=_NP_DT, count=h * w).reshape(h, w)
            _slgpu["hy"] = _mkpin(SLICE_LINES, OUT_WIDTH)
            _slgpu["hu"] = _mkpin(SLICE_LINES // _CH, OUT_WIDTH // _CW)
            _slgpu["hv"] = _mkpin(SLICE_LINES // _CH, OUT_WIDTH // _CW)
    if GPU_SLICE and grp > 1 and (_slgpu["hb"] is None or _slgpu["hb_grp"] != grp):
        # Buffers de LOT épinglés ×2 (double-buffer) + stream D2H DÉDIÉ. non_blocking=True : pas de
        # synchronisation implicite avec le stream par défaut → le D2H async du lot j (copy engine)
        # RECOUVRE réellement les kernels de compose du lot j+1 (M2 « recouvert » du banc gate).
        def _mkpinb(h, w):
            return np.frombuffer(cp.cuda.alloc_pinned_memory(h * w * _BPS),
                                 dtype=_NP_DT, count=h * w).reshape(h, w)
        _slgpu["hb"] = [(_mkpinb(grp * SLICE_LINES, OUT_WIDTH),
                         _mkpinb(grp * SLICE_LINES // _CH, OUT_WIDTH // _CW),
                         _mkpinb(grp * SLICE_LINES // _CH, OUT_WIDTH // _CW)) for _ in (0, 1)]
        _slgpu["hb_grp"] = grp
        _slgpu["stream"] = cp.cuda.Stream(non_blocking=True)
    _d2h = [None]   # D2H de lot EN VOL : (event, bande0, bande1, (hy,hu,hv)) — recouvrement prof. 1
    def _flush_d2h():
        """Publie le lot précédent : attend son D2H async (recouvert par le compose du lot courant),
        copie pinned→grain puis commit PROGRESSIF au grain SLICE_LINES — validSlices avance de bande
        en bande dès que les lignes sont dans le shm (le réveil aval reste FIN ; seule la PHASE de
        commit grossit d'un lot, coût assumé du recouvrement — TISSU_SLICE_GPU.md §c)."""
        if _d2h[0] is None:
            return
        _ev, _ka, _kb, (_hy, _hu, _hv) = _d2h[0]
        _d2h[0] = None
        _ev.synchronize()
        for _k in range(_ka, _kb):
            _r = (_k - _ka) * SLICE_LINES
            _a = _k * SLICE_LINES; _b = _a + SLICE_LINES
            ov_y[_a:_b] = _hy[_r:_r + SLICE_LINES]
            ov_u[_a//_CH:_b//_CH] = _hu[_r//_CH:_r//_CH + SLICE_LINES//_CH]
            ov_v[_a//_CH:_b//_CH] = _hv[_r//_CH:_r//_CH + SLICE_LINES//_CH]
            out_writer.commit(gi_o, valid_slices=None if _k == nb - 1 else _k + 1)
    # Blends IN-PLACE : GPU → kernel FUSIONNÉ écrivant directement dans la vue canvas (évite le
    # kernel de recopie du setitem — à N lots × plans, le coût de lancement comptait double) ;
    # CPU → strictement équivalent à l'assignation `vue[...] = blend(...)` (mêmes octets).
    def _bl_pre(dv, ia, sa):
        if GPU_SLICE:
            _blend_pre_k(dv, ia, sa, dv)
        else:
            _blend_pre_into(dv, ia, sa)   # mvk 1 passe, sinon strictement l'ancien code
    def _bl(dv, s, aa):
        if GPU_SLICE:
            _blend_k(dv, s, aa, dv)
        else:
            _blend_into(dv, s, aa)
    if GPU_SLICE:
        # Opérandes VU/horloges → VRAM UNE fois par trame (le chrome pré-calculé y réside déjà,
        # cf. _to_xp au bake) : les blends par bande restent alors 100 % en VRAM.
        if meter_tiles:
            meter_tiles = [t[:4] + tuple(_to_xp(p) for p in t[4:]) for t in meter_tiles]
        if pf_tiles:
            pf_tiles = [t[:4] + tuple(_to_xp(p) for p in t[4:]) for t in pf_tiles]
    _deadl = time.monotonic() + FRAME_INTERVAL * 1.5   # garde-fou GLOBAL (borne dure de la trame)
    _sl_dbg = [len(batch), -1, 0, 0, 0]   # [tuiles, valid0, attentes, replis, dormantes]
    # état mutable par tuile : plans (rebindés au repli grain complet), tranches valides, colonnes.
    # FAST-PATH ratio ENTIER (grilles 2×2/3×3…) : vues STRIDÉES pré-calculées (comme resize_plane)
    # → placement de bande = simple slice-assign (~8 µs vs ~120 µs np.ix_ 2D, mesuré : le np.ix_
    # par bande×tuile coûtait ~15 ms/trame et faisait dériver le compose d'une demi-trame).
    # Éléments [6..10] : plans SOURCE de base + pas de décimation — consommés par le placement
    # fusionné mvk (_mvk_band, indices absolus) ; les index [0..5] historiques sont inchangés
    # (repli _gather_band). cxi/cci en int32 (contrat mvk_place ; identique en fancy-indexing).
    # Éléments [11..12] : facteurs de réduction du plan CHROMA (le filtre box réduit la tranche
    # source elle-même, il ne peut pas réutiliser une vue stridée pré-calculée — et les facteurs
    # chroma ne se déduisent pas des facteurs luma quand la géométrie n'est pas divisible).
    def _tile_views(sy, su, sv, in_h, in_w, vh, vw):
        if vh > 0 and vw > 0 and in_h % vh == 0 and in_w % vw == 0:
            _sy, _sx = in_h // vh, in_w // vw
            _cfy = max(1, (in_h // _CH) // max(1, vh // _CH))
            _cfx = max(1, (in_w // _CW) // max(1, vw // _CW))
            return ("v", sy[::_sy, ::_sx], su[::_sy, ::_sx], sv[::_sy, ::_sx], None, None,
                    sy, su, sv, _sy, _sx, _cfy, _cfx)
        cxi = ((np.arange(vw) * in_w) // vw).astype(np.int32)
        cci = ((np.arange(vw // _CW) * (in_w // _CW)) // (vw // _CW)).astype(np.int32)
        return ("g", sy, su, sv, cxi, cci, sy, su, sv, 0, 0, 0, 0)
    def _gather_band(tv, in_h, vh, vy, a, b):
        """Rangées SOURCE (hôte) des lignes de sortie [a, b) d'une tuile : vue STRIDÉE (ratio
        entier, ~8 µs) sinon gather chaîné lignes→colonnes (~33 µs — np.ix_ 2D mesuré à ~120 µs,
        prohibitif à bande×tuile). Mapping nearest IDENTIQUE ligne à ligne quelle que soit la
        plage → coalescer [a, b) au LOT donne les mêmes octets que bande par bande."""
        r0 = a - vy; r1 = r0 + (b - a)
        ca0, cb0 = a // _CH, b // _CH
        _bu = _bv = None
        if tv[0] == "v":
            if _BOX:
                # box : la bande de sortie [r0, r1) consomme les lignes source [r0·fy, r1·fy).
                # On réduit CETTE tranche — pas le plan entier — donc le mode tranche garde son
                # intérêt (l'aval démarre sur la 1ʳᵉ bande) et le coût reste proportionnel.
                _fy, _fx = tv[9], tv[10]
                _by = _box_reduce(tv[6][r0 * _fy:r1 * _fy], _fy, _fx)
                if cb0 > ca0:
                    rc0 = r0 // _CH
                    _cfy, _cfx = tv[11], tv[12]
                    _nc = cb0 - ca0
                    _bu = _box_reduce(tv[7][rc0 * _cfy:(rc0 + _nc) * _cfy], _cfy, _cfx)
                    _bv = _box_reduce(tv[8][rc0 * _cfy:(rc0 + _nc) * _cfy], _cfy, _cfx)
                return _by, _bu, _bv, ca0, cb0
            _by = tv[1][r0:r1]
            if cb0 > ca0:
                rc0 = r0 // _CH
                _bu = tv[2][rc0:rc0 + (cb0 - ca0)]
                _bv = tv[3][rc0:rc0 + (cb0 - ca0)]
        else:
            ry = ((np.arange(r0, r1) * in_h) // vh)
            _by = tv[1][ry][:, tv[4]]
            if cb0 > ca0:
                rc = ((np.arange(r0 // _CH, r0 // _CH + (cb0 - ca0)) * (in_h // _CH)) // (vh // _CH))
                _bu = tv[2][rc][:, tv[5]]
                _bv = tv[3][rc][:, tv[5]]
        return _by, _bu, _bv, ca0, cb0
    def _mvk_band(tv, in_h, vh, vy, a, b, vx, vw, cx0):
        """Placement de bande FUSIONNÉ (mvk_place : plan source → canvas en 1 passe, plus de
        gather intermédiaire + assignation). Indices SOURCE ABSOLUS = mêmes formules que
        _gather_band/_tile_views (bit-exact). False → repli gather+assign (l'échec d'un plan
        fait ré-écrire toute la bande de la tuile par le repli : sûr)."""
        r0 = a - vy; r1 = r0 + (b - a)
        ca0, cb0 = a // _CH, b // _CH
        cw = vw // _CW
        if tv[0] == "v":
            sr, sc = tv[9], tv[10]
            ry = (np.arange(r0, r1) * sr).astype(np.int32)
            if not bobimxl.mvk_place_into(cy[a:b, vx:vx + vw], tv[6], ry, col0=0, col_step=sc):
                return False
            if cb0 > ca0:
                rc0 = r0 // _CH
                rc = (np.arange(rc0, rc0 + (cb0 - ca0)) * sr).astype(np.int32)
                return bool(
                    bobimxl.mvk_place_into(cu[ca0:cb0, cx0:cx0 + cw], tv[7], rc, col0=0, col_step=sc)
                    and bobimxl.mvk_place_into(cv[ca0:cb0, cx0:cx0 + cw], tv[8], rc, col0=0, col_step=sc))
            return True
        ry = ((np.arange(r0, r1) * in_h) // vh).astype(np.int32)
        if not bobimxl.mvk_place_into(cy[a:b, vx:vx + vw], tv[6], ry, col_idx=tv[4]):
            return False
        if cb0 > ca0:
            rc = ((np.arange(r0 // _CH, r0 // _CH + (cb0 - ca0)) * (in_h // _CH))
                  // (vh // _CH)).astype(np.int32)
            return bool(
                bobimxl.mvk_place_into(cu[ca0:cb0, cx0:cx0 + cw], tv[7], rc, col_idx=tv[5])
                and bobimxl.mvk_place_into(cv[ca0:cb0, cx0:cx0 + cw], tv[8], rc, col_idx=tv[5]))
        return True
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
    _bstage = []   # GPU SLICE : bandes source du LOT en cours → 1 H2D groupé au lot complet
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
            if _BOX and tv[0] == "v":
                # ★ En box, la bande n'a pas besoin de la SEULE ligne échantillonnée mais de TOUT
                # le bloc : fy−1 lignes source de plus. Sans ce décalage, la dernière ligne de
                # chaque bande serait moyennée avec des lignes que le producteur n'a pas encore
                # écrites — bandes souillées, et d'autant plus visibles que la grille est fine.
                need_row = (b - vy) * tv[9] - 1
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
            # placement : en micro-batch le gather est fait AU LOT complet (coalescé, plus bas) —
            # ici on n'a fait QUE les attentes get_slice (protocole au grain SLICE_LINES inchangé).
            if GPU_SLICE and grp > 1:
                continue
            # Les rangées source de la bande (_by/_bu/_bv, HÔTE, _gather_band) sont soit posées
            # directement dans le canvas (CPU, copie = l'assignation, octet-identique à avant),
            # soit mises en attente pour l'upload H2D GROUPÉ (GPU SLICE k=1, _gpu_place_band).
            _cx0 = vx // _CW
            if GPU_SLICE:
                _by, _bu, _bv, ca0, cb0 = _gather_band(tv, in_h, vh, vy, a, b)
                _bstage.append((_by, _bu, _bv, a, b, ca0, cb0, vx, vw, _cx0, 1, 1))
            # `not _BOX` : _mvk_band est un gather nearest fusionné — même raison que
            # _mvk_place_plane, il ne peut pas servir sous filtre box.
            elif not (_MVK and not _BOX and _mvk_band(tv, in_h, vh, vy, a, b, vx, vw, _cx0)):
                _by, _bu, _bv, ca0, cb0 = _gather_band(tv, in_h, vh, vy, a, b)
                cy[a:b, vx:vx + vw] = _by
                if _bu is not None:
                    cu[ca0:cb0, _cx0:_cx0 + vw//_CW] = _bu
                    cv[ca0:cb0, _cx0:_cx0 + vw//_CW] = _bv
        # MICRO-BATCH : tant que le LOT n'est pas complet, on continue d'ACCUMULER (les attentes
        # get_slice ci-dessus restent au grain SLICE_LINES — le protocole amont ne change pas) ;
        # H2D/kernels/blends/D2H/habillage sont faits au LOT complet (dernier lot partiel inclus).
        if GPU_SLICE and (k + 1) % grp != 0 and k + 1 < nb:
            continue
        B0 = (k - k % grp) * SLICE_LINES     # 1ʳᵉ ligne du lot courant ([B0, b1) = grp bandes max)
        if GPU_SLICE and grp > 1:
            # COALESCENCE au lot : rangées source de TOUT [B0, b1) par tuile → 1 gather hôte + 1
            # slice-assign VRAM par tuile/plan/LOT (pas par bande : 180 lancements de placement
            # par trame au grain 36 l — mesuré ~7 ms de coût de lancement seul, banc micro-batch).
            # Fast-path ratio entier : rangées décimées PLEINE LARGEUR (copies hôte contiguës par
            # rangée) + décimation COLONNE en VRAM (csx/csc) — le gather colonne-stridé hôte
            # coûtait ~5,6 ms/trame. Octet-identique (mêmes indices nearest, appliqués en VRAM).
            # NB repli : une tuile rebasculée sur son grain complet PENDANT le lot voit tout
            # [B0, b1) resservi depuis ce grain (frontière de tearing au lot, pas à la bande).
            for _t2 in st:
                _vy2, _vh2, _vx2, _vw2 = _t2[8], _t2[9], _t2[10], _t2[11]
                a = max(_vy2, B0); b = min(_vy2 + _vh2, b1)
                if a >= b:
                    continue
                _tv2 = _t2[15]
                if _tv2[0] == "v" and _BOX:
                    # box : le fast-path « rangées pleine largeur + décimation COLONNE en VRAM »
                    # ne s'applique pas à une moyenne (csx/csc = 1). On réduit sur l'hôte via le
                    # MÊME chemin que la bande simple et on uploade la tuile déjà réduite — moins
                    # d'octets à transférer, en compensation partielle du coût de la moyenne.
                    _by, _bu, _bv, ca0, cb0 = _gather_band(_tv2, _t2[6], _vh2, _vy2, a, b)
                    _bstage.append((_by, _bu, _bv, a, b, ca0, cb0, _vx2, _vw2, _vx2 // _CW, 1, 1))
                elif _tv2[0] == "v":
                    _sty = _t2[6] // _vh2; _stx = _t2[7] // _vw2
                    r0 = a - _vy2; r1 = r0 + (b - a)
                    _by = _t2[3][::_sty][r0:r1]            # rangées décimées, colonnes PLEINES
                    ca0, cb0 = a // _CH, b // _CH
                    _bu = _bv = None
                    if cb0 > ca0:
                        rc0 = r0 // _CH
                        _bu = _t2[4][::_sty][rc0:rc0 + (cb0 - ca0)]
                        _bv = _t2[5][::_sty][rc0:rc0 + (cb0 - ca0)]
                    _bstage.append((_by, _bu, _bv, a, b, ca0, cb0, _vx2, _vw2, _vx2 // _CW,
                                    _stx, _stx))
                else:
                    _by, _bu, _bv, ca0, cb0 = _gather_band(_tv2, _t2[6], _vh2, _vy2, a, b)
                    _bstage.append((_by, _bu, _bv, a, b, ca0, cb0, _vx2, _vw2, _vx2 // _CW, 1, 1))
        # GPU SLICE, ancre (a)+(b) : les bandes d'ENTRÉE du lot sont complètes (attentes faites)
        # → 1 H2D groupé épinglé DU LOT + placement VRAM, puis blends du lot en VRAM (ci-dessous).
        if GPU_SLICE and _bstage:
            _gpu_place_band(cy, cu, cv, _bstage)
            _bstage = []
        # habillage intersectant le lot : 1) chrome statique pré-calculé (bbox)
        if chrome_pre is not None:
            bx0, by0, bx1, by1, _piY, _saY, _piC, _saU, _saV = chrome_pre
            a = max(by0, B0); b = min(by1, b1)
            if a < b:
                l0 = a - by0; l1 = b - by0
                _bl_pre(cy[a:b, bx0:bx1], _piY[l0:l1], _saY[l0:l1])
                ca0, cb0 = a // _CH, b // _CH
                lc0 = ca0 - by0 // _CH; lc1 = lc0 + (cb0 - ca0)
                cx0, cx1 = bx0 // _CW, bx1 // _CW
                if cb0 > ca0:
                    _bl_pre(cu[ca0:cb0, cx0:cx1], _piC[lc0:lc1], _saU[lc0:lc1])
                    _bl_pre(cv[ca0:cb0, cx0:cx1], _piC[lc0:lc1], _saV[lc0:lc1])
        # 2) VU-mètres puis 3) horloges (z-ordre conservé), bornés aux lignes du lot
        for _tiles in (meter_tiles, pf_tiles):
            if not _tiles:
                continue
            for (bx0, by0, bx1, by1, _oy, _ou, _ovv, _oa, _oa2) in _tiles:
                a = max(by0, B0); b = min(by1, b1)
                if a >= b:
                    continue
                l0 = a - by0; l1 = b - by0
                _bl(cy[a:b, bx0:bx1], _oy[l0:l1], _oa[l0:l1])
                ca0, cb0 = a // _CH, b // _CH
                lc0 = ca0 - by0 // _CH; lc1 = lc0 + (cb0 - ca0)
                cx0, cx1 = bx0 // _CW, bx1 // _CW
                if cb0 > ca0:
                    _bl(cu[ca0:cb0, cx0:cx1], _ou[lc0:lc1], _oa2[lc0:lc1])
                    _bl(cv[ca0:cb0, cx0:cx1], _ovv[lc0:lc1], _oa2[lc0:lc1])
        # publication du lot : copie canvas→grain + commit PROGRESSIF (réveille l'aval)
        if GPU_SLICE and grp > 1:
            # Ancre (c) GREFFÉE (M2 « recouvert » du banc gate, ~0,44 ms gagnés à 135 l) : le D2H
            # du lot j part ASYNC sur le stream dédié après les kernels du lot (event) ; il est
            # attendu/publié/commité PENDANT le compose du lot j+1 (_flush_d2h ci-dessous au tour
            # suivant, ou après la boucle pour le dernier lot) → D2H masqué, au prix d'UN lot de
            # phase de commit. Le .get direct dans le grain reste proscrit (pageable, Phase 0).
            _hby, _hbu, _hbv = _slgpu["hb"][(k // grp) % 2]
            _sd = _slgpu["stream"]
            _evc = cp.cuda.Event(block=False, disable_timing=True)
            _evc.record()                      # fin des kernels du lot (stream par défaut)
            _sd.wait_event(_evc)
            _flush_d2h()                       # publie le lot j-1 (son D2H a recouvert CE compose)
            _rows = b1 - B0; _wc = OUT_WIDTH // _CW
            _dth = cp.cuda.runtime.memcpyDeviceToHost
            cp.cuda.runtime.memcpyAsync(_hby.ctypes.data,
                                        int(cy.data.ptr) + B0 * OUT_WIDTH * _BPS,
                                        _rows * OUT_WIDTH * _BPS, _dth, _sd.ptr)
            cp.cuda.runtime.memcpyAsync(_hbu.ctypes.data,
                                        int(cu.data.ptr) + (B0 // _CH) * _wc * _BPS,
                                        (_rows // _CH) * _wc * _BPS, _dth, _sd.ptr)
            cp.cuda.runtime.memcpyAsync(_hbv.ctypes.data,
                                        int(cv.data.ptr) + (B0 // _CH) * _wc * _BPS,
                                        (_rows // _CH) * _wc * _BPS, _dth, _sd.ptr)
            _evd = cp.cuda.Event(block=False, disable_timing=True)
            _evd.record(_sd)
            _d2h[0] = (_evd, B0 // SLICE_LINES, k + 1, (_hby, _hbu, _hbv))
        elif GPU_SLICE:
            # Ancre (c), squelette k=1 : D2H PAR BANDE via .get(out=épinglé) puis copie vers la
            # vue grain — synchrone (le commit exige les octets posés).
            cy[b0:b1].get(out=_slgpu["hy"])
            ov_y[b0:b1] = _slgpu["hy"]
            if b1 // _CH > b0 // _CH:
                cu[b0//_CH:b1//_CH].get(out=_slgpu["hu"])
                cv[b0//_CH:b1//_CH].get(out=_slgpu["hv"])
                ov_u[b0//_CH:b1//_CH] = _slgpu["hu"]
                ov_v[b0//_CH:b1//_CH] = _slgpu["hv"]
            out_writer.commit(gi_o, valid_slices=None if k == nb - 1 else k + 1)
        else:
            ov_y[b0:b1] = cy[b0:b1]
            if b1 // _CH > b0 // _CH:
                ov_u[b0//_CH:b1//_CH] = cu[b0//_CH:b1//_CH]
                ov_v[b0//_CH:b1//_CH] = cv[b0//_CH:b1//_CH]
            out_writer.commit(gi_o, valid_slices=None if k == nb - 1 else k + 1)
    if GPU_SLICE and grp > 1:
        _flush_d2h()                           # dernier lot (commit final validSlices=None dedans)
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
# Agrégat FRAME LENTE sur fenêtre glissante d'1 min (cf. règle 3 du bloc « Niveau de log ») :
# les frames lentes sont un PRÉCURSEUR d'incident → visibles au niveau par défaut, mais en UNE
# ligne de synthèse par minute au lieu d'une rafale. `seg` = histogramme du segment dominant.
_slow_acc = {{"n": 0, "worst_own": 0.0, "worst_tick": 0.0, "t0": time.time(),
              "seg": {{"tick": 0, "inputs": 0, "overlays": 0, "output": 0, "waited": 0}}}}
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
# Âge MAXIMAL toléré de la dernière écriture producteur (now_tai − lastWriteTime) avant de
# considérer le Reader décroché et de le reconnecter. Critère INDÉPENDANT du format (contrairement
# au retard de tête sur la grille FLOW, qui ne vaut qu'en mode tranche progressif) — cf. le
# garde-fou de la boucle de composition. 5 s = ~250 trames à 50 Hz : au-delà, plus aucune source
# vivante n'est plausible, et une source réellement arrêtée ne perd rien à être rouverte.
STALE_REOPEN_MS = 5000.0

# GC CPython DISCIPLINÉ (chantier tissu slice — grain tardif de l'assembleur) : le collect gen2
# AUTOMATIQUE tombe N'IMPORTE OÙ dans le cycle (mesuré au banc dl360-1 : pause strictement
# périodique ~34 s → le grain de sortie sort +1 epoch en retard, phase +25 ms, le TX aval rate
# sa fenêtre → compteur late / trou d'1 trame quand le rejeu ne couvre pas). Remède standard :
# on COUPE le déclenchement automatique et on collecte MANUELLEMENT au point sûr (fin de cycle,
# dernière bande committée, temps mort avant le tick suivant) : gen0/gen1 chaque trame (sub-ms),
# gen2 cadencé (~5 s, durée mesurée → métrique gc_full_ms). gc.freeze() sort le tas de démarrage
# (modules, fonts, config…) du scan gen2 → collect court ; les objets gelés ne sont plus jamais
# collectés (OK : quasi tout est pérenne ; les Readers rouverts au churn ne sont pas cycliques,
# libérés par refcount). NB : RIEN À VOIR avec le GC du ring MXL (flux
# orphelins), qu'on ne touche pas.
gc.collect(2)
gc.freeze()
gc.disable()
_gc_frames = 0
_gc_full_every = max(1, int(round(5.0 / FRAME_INTERVAL)))   # gen2 ~toutes les 5 s
_gc_last_full_ms = 0.0
_gc_max_full_ms = 0.0

# ─── Chien de garde : QUI tient la trame quand elle dépasse ──────────────────
# Toute la difficulté de cette chasse est là : `compose_peak_ms` dit QUEL ÉTAGE pique, jamais
# quelle ligne. J'ai dépensé quatre hypothèses là-dessus, trois fausses. Un fil échantillonne donc
# la pile de la boucle de composition PENDANT qu'une trame dépasse — pas après, où l'on ne verrait
# que la fin du travail.
# Coût : un réveil toutes les 2 ms qui compare deux entiers. La capture elle-même n'a lieu qu'au
# franchissement du seuil, et UNE seule fois par trame (on attend le compteur de trame suivant),
# sinon une trame de 40 ms produirait vingt piles identiques.
_watch = {{"t0": 0, "fi": 0, "seuil_ms": 22.0}}
_piles = {{}}      # signature de pile → nombre de fois observée pendant un dépassement

def _chien_de_garde():
    import traceback
    vue = -1
    while True:
        time.sleep(0.002)
        t0 = _watch["t0"]; fi = _watch["fi"]
        if not t0 or fi == vue:
            continue
        if (time.time_ns() - t0) / 1e6 < _watch["seuil_ms"]:
            continue
        vue = fi                      # une capture par trame, pas une par réveil
        try:
            cadre = sys._current_frames().get(_thread_compo)
            if cadre is None:
                continue
            # On garde les quelques appels les plus INTERNES : c'est là qu'est le temps. Les
            # niveaux supérieurs sont toujours les mêmes (la boucle), ils n'apprennent rien.
            pile = traceback.extract_stack(cadre)[-4:]
            sig = " < ".join("%s:%d" % (f.name, f.lineno) for f in reversed(pile))
            _piles[sig] = _piles.get(sig, 0) + 1
        except Exception:                                                  # noqa: BLE001
            pass

threading.Thread(target=_chien_de_garde, daemon=True).start()

_thread_compo = threading.get_ident()   # à partir d'ici, ce thread SEUL libère les mappings
# Affinité de la COMPOSITION : le premier cœur du cpuset, pour elle seule. Posée ICI, après le
# lancement des boulangers — l'affinité s'hérite à la création, la poser plus tôt les confinerait
# tous sur ce même cœur.
if _CPU_SPLIT:
    try:
        os.sched_setaffinity(0, {{_CPUS[0]}})
        log("multiview: composition épinglée sur le cœur %d, boulangers sur %s"
            % (_CPUS[0], _CPUS[1:]), "info")
    except Exception as _e:                                                # noqa: BLE001
        log("multiview: séparation des cœurs impossible (%s)" % _e, "warning")

while True:
    # Fermetures/GC mis en file par les autres threads : on les exécute ICI, avant toute
    # construction de vue sur un mapping (cf. LIBÉRATION DIFFÉRÉE). Coût nul quand la file
    # est vide, c'est-à-dire à la quasi-totalité des trames.
    _liberer_differe()
    now = time.time()
    now_m = time.monotonic()   # horloge MONOTONE pour les durées (détection freeze) — insensible aux sauts de CLOCK_REALTIME (genlock garde time.time())
    if _bus_error.is_set():
        _bus_error.clear()
        with state_lock:
            for _s in sources:
                if _s is not None:
                    _close_source(_s)
            sources[:] = [None] * len(sources)
            _in_track.clear(); _stale_drops.clear()
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
    _watch["t0"] = ts_cycle_start; _watch["fi"] += 1   # chien de garde : cette trame commence ICI
    # Diagnostic grain tardif (chantier tissu slice) : retard du TICK lui-même (la boucle a raté
    # sa grille — la pause est ARRIVÉE AVANT le cycle) vs frame LENTE (le cycle a coûté trop cher
    # — la pause est DANS le cycle). Journalisé plus bas (FRAME LENTE, throttlé).
    _tick_late_ms = (ts_cycle_start / 1e9 - next_frame_time) * 1000.0 if GENLOCK else 0.0
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
    _woven = 0              # tuiles DÉSENTRELACÉES par weave cette frame (exposé sur :8080 —
                            # une source déclarée 1080i qui reste à 0 signale un repli bob muet)

    with state_lock:
        _fc = list(FLUX_CONFIG)   # snapshot stable pour cette frame

    # Images de fond (layer=background) : sous la vidéo. Couche cachée, PRÉ-BLENDÉE dans un canvas de
    # base `_base_*` — la boucle par-trame n'en fait qu'une COPIE (≈1 ms) au lieu de re-blender plein
    # écran 3 plans à chaque trame (≈30 ms). ★ 0.42.0 : le RE-BAKE (PIL + rgba_to_yuv + pré-blend,
    # ≈13 ms plein cadre) est fait par le BOULANGER, hors trame. Il ne reste ici que le RAMASSAGE de
    # la publication : sur GPU un upload des 3 plans (~0,3 ms), en CPU rien du tout (_to_xp = identité).
    if _bg_pub is not _bg_seen:
        _ts_bg0 = time.time_ns()
        _bg_seen = _bg_pub
        _bgh = _bg_seen[0]
        if _bgh is None:
            _base_y = _base_u = _base_v = None
        else:
            _base_y = _to_xp(_bgh[0]); _base_u = _to_xp(_bgh[1]); _base_v = _to_xp(_bgh[2])
        _t_ov_bg.push((time.time_ns() - _ts_bg0) / 1e6)

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
            _tile_status.pop(i, None)
            continue   # entrée de la banque non affichée : source câblée conservée, pas de rendu
        # Modèle de PiP SANS composant vidéo : rien à lire ni à poser (cellule d'habillage pur,
        # composants rendus par le chrome/les tuiles per-frame).
        if _tpl_comps(cfg) is not None and _tpl_video_comp(cfg) is None:
            _tile_status[i] = ""
            continue
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
            _tile_status[i] = "nosignal"
            if SHOW_NO_SIGNAL:
                _statuses.append((i, "nosignal", "", None))
            continue

        try:
            rd      = src["reader"]
            in_w    = src["in_w"]
            in_h    = src["in_h"]
            frame_size = src["frame_size"]
            # GÉNÉRATION PÉRIMÉE — DÉTECTION INDÉPENDANTE DU FORMAT. Le producteur amont recrée son
            # flux SOUS LE MÊME NOM (changement de source d'un slot RX, redeploy…) : notre Reader
            # reste collé à la génération morte, dont les grains restent LISIBLES → ni « got is
            # None », ni SIGBUS, et l'index figé ne suffit pas (parité entrelacée, grille FLOW).
            # Le seul critère fiable ET sanctionné par la spec MXL est `lastWriteTime` : sur un
            # lecteur décroché il ne bouge plus pendant que now_tai() avance (mesuré en prod :
            # 3 h 20 d'âge sur un shard, tuile figée à vie). Vaut pour TOUS les flux — progressif
            # ou entrelacé, mode tranche ou non. `lw = 0` = info indisponible (producteur qui ne
            # maintient pas lastWriteTime, ex. audio) → on laisse les garde-fous historiques.
            _lw = rd.last_write_time()
            _age_ms = (bobimxl.now_tai() - _lw) / 1e6 if _lw else 0.0
            if _lw and _age_ms > STALE_REOPEN_MS:
                _drop_input(i, rd, "aucune écriture depuis %.1f s" % (_age_ms / 1000.0))
                canvas_y[vy:vy+vh, video_x:video_x+video_w] = 0
                _tile_status[i] = "nosignal"
                if SHOW_NO_SIGNAL:
                    _statuses.append((i, "nosignal", "", None))
                continue
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
                        _drop_input(i, rd, "tête figée %d, %d trames de retard sur la grille"
                                    % (_hi, _fi_out - _hi))
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
            # ENTRELACÉ. Deux traitements possibles, cf. INTERLACE_MODE.
            # • WEAVE (défaut) : on retisse les DEUX champs de la dernière trame COMPLÈTE →
            #   résolution verticale PLEINE. Appariement : une trame = (index PAIR 2k = champ
            #   haut, index IMPAIR 2k+1 = champ bas). Si le dernier grain est IMPAIR, la paire
            #   (n−1, n) est entièrement commitée. S'il est PAIR, le champ bas n+1 n'est pas
            #   encore écrit : on recule d'une trame sur (n−2, n−1), commitée elle aussi. On
            #   n'attend donc JAMAIS, et on n'affiche jamais une demi-trame — au prix d'au plus
            #   un champ de latence. Paire indisponible (démarrage, ring court) → repli bob.
            # • BOB : on VERROUILLE la parité sur le champ HAUT (index pair). Sans ce verrou,
            #   get_latest rend alternativement top/bottom (50/s) → une fois scalés à la tuile les
            #   deux champs ont un décalage vertical d'½ ligne → SCINTILLEMENT des bords
            #   horizontaux. Un seul champ = bob progressif STABLE à la cadence trame (25/s).
            _wv = None
            if got is not None and src.get("interlaced"):
                if INTERLACE_MODE == "weave":
                    _n = got[0]
                    _it, _ib = ((_n - 1, _n) if (_n % 2 == 1) else (_n - 2, _n - 1))
                    if _it >= 0:
                        _gt = rd.get(_it)
                        _gb = rd.get(_ib)
                        if _gt is not None and _gb is not None:
                            _wv = (_gt[2], _gb[2])
                            got = _gt
                if _wv is None and (got[0] % 2 == 1):
                    _gt = rd.get(got[0] - 1)    # champ haut apparié (déjà commité → retour immédiat)
                    if _gt is not None:
                        got = _gt
            if got is None:        # flux ouvert mais aucun grain lisible (vide OU Reader périmé)
                canvas_y[vy:vy+vh, video_x:video_x+video_w] = 0
                _tile_status[i] = "nosignal"
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
                    _drop_input(i, rd, "aucun grain lisible depuis %.1f s" % (now_m - _t0))
                continue
            _stale_since.pop(i, None)   # lecture réussie → réarme le compteur de péremption
            _stale_drops.pop(i, None)   # … et l'escalade Instance dédiée (reconnexion réussie)
            fi = got[0]; src_view = got[2]
            # TRANSIT (arrivée) = âge de la trame d'entrée (now_tai − dernière écriture producteur),
            # déjà mesuré plus haut pour le garde-fou de péremption (une seule lecture par trame).
            ts_in_per_input[src.get("path", cfg["path"])] = _age_ms if _lw else None
            # Suivi freeze : t = dernier instant où l'index de grain a avancé.
            tr = _in_track.get(i)
            if tr is None or tr.get("path") != src["path"]:
                tr = {{"path": src["path"], "fi": fi, "t": now_m}}
                _in_track[i] = tr
            elif fi != tr["fi"]:
                tr["fi"] = fi; tr["t"] = now_m
                _frame_neuve = True     # au moins une entrée a AVANCÉ → cette composition relaie
                                        # de la matière neuve (cf. `fps_content`)
            _st = "freeze" if (FREEZE_DETECT_S > 0 and now_m - tr["t"] > FREEZE_DETECT_S) else ""
            _tile_status[i] = _st
            if _st or _proxy_chip:
                _statuses.append((i, _st, "", _proxy_chip))
            # Freeze PROLONGÉ : même cause racine que le « No Signal » (Reader périmé sur flux amont
            # recréé) — l'ancien ring peut renvoyer indéfiniment son dernier grain (index figé) au
            # lieu de None. Au-delà de FREEZE_DETECT_S + REOPEN_STALE_S, on reconnecte le Reader.
            # Sans danger pour une source légitimement statique : elle relit simplement son dernier
            # grain après ré-ouverture. (drop = continue : la tuile garde sa dernière trame.)
            if FREEZE_DETECT_S > 0 and now_m - tr["t"] > FREEZE_DETECT_S + REOPEN_STALE_S:
                _drop_input(i, rd, "image figée depuis %.1f s" % (now_m - tr["t"]))
                continue

            _yb  = in_w * in_h * _BPS               # octets du plan Y
            _uvb = (in_w // _CW) * (in_h // _CH) * _BPS   # octets d'un plan chroma
            if SLICE_ON and _wv is None:
                # MODE TRANCHE : vues ZÉRO-COPIE sur le grain (les lignes au-delà de validSlices ne
                # sont PAS lues ici — _compose_bands attend chaque bande avant de la toucher ; le
                # handler SIGBUS couvre la recréation amont, comme pour les mmaps historiques).
                src_y = src_view[:_yb].view(_NP_DT).reshape(in_h, in_w)
                src_u = src_view[_yb:_yb + _uvb].view(_NP_DT).reshape(in_h//_CH, in_w//_CW)
                src_v = src_view[_yb + _uvb:_yb + 2 * _uvb].view(_NP_DT).reshape(in_h//_CH, in_w//_CW)
                _slice_batch.append((i, rd, fi, got[1], src_y, src_u, src_v, in_h, in_w,
                                     vy, vh, video_x, video_w))
                continue
            if _wv is not None:
                # ENTRELACÉ/WEAVE : trame PLEINE retissée (2·in_h lignes) — c'est elle, et non un
                # champ, qui part au placement. Tout l'aval (resize_plane, filtre box, géométrie
                # `contain`) travaille dès lors sur la vraie hauteur d'image : le facteur de
                # réduction vertical double, donc le box moyenne des lignes des DEUX champs.
                src_y, src_u, src_v = _weave_fields(_wv[0], _wv[1], in_w, in_h)
                _woven += 1
            else:
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
            _tile_status[i] = "nosignal"
            if SHOW_NO_SIGNAL:
                _statuses.append((i, "nosignal", "", None))

    # Conditions des modèles de PiP liées au SIGNAL (no_signal/freeze/signal_ok) : une transition
    # d'état re-bake le chrome via tally_dirty (même mécanique que le tally — coût payé à la
    # transition seulement, jamais per-frame).
    if _tile_status != _tpl_status_prev:
        _tpl_status_prev = dict(_tile_status)
        for _cfgc in _fc:
            _cc = _tpl_comps(_cfgc)
            if _cc and any(isinstance(_c, dict) and (_c.get("when") or "always") in _TPL_SIGNAL_CONDS
                           for _c in _cc):
                tally_dirty.set()
                break

    # Placement des tuiles vidéo collectées : 1 upload GPU groupé épinglé (ou resize numpy direct en CPU).
    _ts_place0 = time.time_ns()   # banc : frontière moisson / placement (cf. _t_in_read/_t_in_place)
    _place_batch(canvas_y, canvas_u, canvas_v, _gpu_batch)
    _t_in_place.push((time.time_ns() - _ts_place0) / 1e6)
    _t_in_read.push((_ts_place0 - ts_cycle_start) / 1e6)

    # Publication du monitoring proxy de cette frame (swap de référence = atomique pour le lecteur).
    _proxy_usage_latest = _pu
    _proxy_read_latest = sorted(set(_pread))
    metrics["interlace_woven_tiles"] = _woven

    _t_after_inputs = time.time_ns()   # profiling : fin des entrées vidéo (lecture+resize+blend tuiles)

    try:
        # ★ 0.42.0 : le RE-BAKE du chrome (PIL plein cadre + conversion + opérandes ≈ 25 ms) est fait
        # par le BOULANGER, HORS TRAME. Ici on ne fait plus que RAMASSER la publication (comparaison
        # d'identité) et, sur GPU, uploader les opérandes de la bbox (~0,3 ms) pour que le blend_pre
        # par-trame reste 100 % en VRAM. En CPU, _to_xp = identité → coût strictement nul.
        # Les statuts de tuile (couche `info`) sont PUBLIÉS au boulanger, qui les rend hors trame.
        _ts_bake0 = time.time_ns()
        _sig = tuple(_statuses)
        if _sig != _statuses_pub:
            _statuses_pub = _sig
            _bake_wake.set()
        # Variables de texte : ré-évaluation bornée, re-bake sur CHANGEMENT de rendu seulement.
        if now - _vars_last_ts >= _VARS_CHECK_S:
            _vars_last_ts = now
            try:
                _vsig = _vars_signature()
            except Exception:
                _vsig = _vars_last_sig
            if _vsig != _vars_last_sig:
                # Plus de `tally_dirty` ici : les textes à variables sont désormais des TUILES
                # per-frame, dont la signature (`_pf_sig`) déclenche seule leur re-rendu. Remettre
                # l'habillage plein cadre en cause à chaque changement de %cpu% coûtait 56-73 ms,
                # près d'une fois par seconde. On garde la surveillance pour la trace seule.
                _vars_last_sig = _vsig
        if _chrome_pub is not _chrome_seen:
            _chrome_seen = _chrome_pub
            _cph = _chrome_seen[0]
            if _cph is None:
                _chrome_pre = None      # chrome entièrement transparent → blend sauté (cas assembleur)
            else:
                _chrome_pre = (_cph[0], _cph[1], _cph[2], _cph[3],
                               _to_xp(_cph[4]), _to_xp(_cph[5]),
                               _to_xp(_cph[6]), _to_xp(_cph[7]), _to_xp(_cph[8]))
            # Diagnostic : la zone RÉELLEMENT habillée, telle qu'elle est blendée. `null` = aucun
            # habillage (légitime sur un assembleur, ANORMAL sur un mur à labels/tally → le
            # boulanger n'a rien publié). Rend visible une panne d'habillage sans capture d'écran.
            metrics["chrome_bbox"] = None if _cph is None else [_cph[0], _cph[1], _cph[2], _cph[3]]

        # Habillage = chrome STATIQUE (caché) + VU-mètres (tuiles per-frame) + horloges (cachées).
        _ts_ov0 = time.time_ns()
        _t_ov_bake.push((_ts_ov0 - _ts_bake0) / 1e6)
        # VU-mètres : rendus + convertis en TUILES locales (une bbox par meter, cf. render_meters).
        # Inhérents per-frame (le niveau change à chaque trame) → jamais cachés, mais chaque tuile ne
        # couvre que la bande VU d'une cellule (≪ bbox-union quasi plein écran de l'ancien chemin).
        _meter_tiles = render_meters(now)
        _ts_ovm = time.time_ns()   # banc : fin rendu VU-mètres
        # Historiques vidéo/audio : tuiles CACHÉES (recomposées à l'arrivée d'une vignette / au
        # changement de colonne — cf. render_history_tiles) ; par trame, seul le blend est payé.
        _hist_tiles = render_history_tiles(now)
        _ts_ovh = time.time_ns()   # banc : fin rendu frises d'historique
        # Couche PER-FRAME des overlays fg = HORLOGES, en TUILES (une bbox par horloge, cf. render_meters
        # ; texte/images fixes bakés dans le chrome). Rendu PIL + conversion YUV refaits SEULEMENT au
        # changement de VALEUR (signature = chaînes formatées, déterministe → 1×/s sans le champ images
        # FF) ; le blend, lui, se fait par PETITE bbox d'horloge à chaque trame (au lieu de la bbox-UNION
        # quasi plein écran quand les horloges sont dispersées → gros gain de blend per-frame).
        # Les bandeaux ANC des cellules (opt-in) partagent cette machinerie : même cache par
        # signature, même blend par petite bbox. Aucune cellule cochée → coût strictement nul.
        # SIGNATURES SÉPARÉES horloges / ANC. Une signature commune faisait qu'un bandeau de
        # timecode, qui change 25 fois par seconde, re-dessinait AUSSI les horloges — qui, elles,
        # ne changent qu'une fois par seconde. Deux conséquences par trame : le rendu PIL+YUV des
        # horloges payé pour rien, et leurs tuiles reconstruites donc perdues par le cache VRAM
        # (qui compare l'identité). Mesuré sur le shard 917 : ~1,2 à 2,4 ms de rendu d'horloges
        # jetées à chaque trame. ⚠ Les horloges d'un mur peuvent venir des overlays globaux OU des
        # MODÈLES DE PIP (`_tpl_clock_ovs`) — un mur dont la liste `overlays` est vide en a quand
        # même, c'est ce qui m'a fait conclure à tort qu'il n'y en avait pas ici.
        _ts_sig0 = time.time_ns()
        _dyn_ovs = _dyn_overlays()
        _clk_sig = tuple((_dyn_text(ov, now), _countdown_color(ov, now)) for ov in _dyn_ovs)
        # Valeurs RENDUES des textes dynamiques (horloges et textes à variables) — diagnostic.
        # Sans ça, une variable qui rend « — » se voit à l'écran mais nulle part dans les
        # métriques : il faut une capture d'image pour savoir si le mur reçoit sa télémétrie.
        metrics["dyn_text"] = {{(ov.get("id") or ov.get("kind") or "?"): v[0]
                               for ov, v in zip(_dyn_ovs, _clk_sig)}}
        _a_sig = _anc_sig()
        _t_pf_sig.push((time.time_ns() - _ts_sig0) / 1e6)
        _pf_neuf = False
        if _clk_sig != _clk_cache_sig:
            _clk_cache_sig = _clk_sig
            _clk_tiles = render_clock_tiles(now) or []
            _pf_neuf = True
        if _a_sig != _anc_cache_sig:
            _anc_cache_sig = _a_sig
            _anc_tiles = render_anc_tiles(now) or []
            _pf_neuf = True
        if _pf_neuf:
            # Concaténation refaite seulement quand l'un des deux groupes a bougé : les TUILES de
            # l'autre restent les MÊMES objets, donc leur copie VRAM est réutilisée.
            _pf_tiles = (_clk_tiles + _anc_tiles) or None
        _ts_ov1 = time.time_ns()   # fin rendu PIL + conversion YUV (tuiles VU + horloges AU CHANGEMENT)
        # Banc perf (docs/chantiers/MULTIVIEW_BENCH.md) : ventilation FINE de `ov_render` en ses trois producteurs
        # de tuiles per-frame — VU-mètres / frises d'historique / horloges+ANC. Sans ça, `ov_render`
        # est un agrégat opaque et on ne peut pas chiffrer une frise isolément.
        _t_ov_meters.push((_ts_ovm - _ts_ov0) / 1e6)
        _t_ov_hist.push((_ts_ovh - _ts_ovm) / 1e6)
        _t_ov_clock.push((_ts_ov1 - _ts_ovh) / 1e6)
        # 1) Chrome statique : blendé via opérandes pré-calculés sur sa BBOX (SANS conversion). Borné à
        # la zone réellement habillée → un assembleur sans chrome (bbox None) ne paie RIEN ici.
        # MODE TRANCHE : blends + écriture faits BANDE PAR BANDE dans _compose_bands (plus bas).
        if _chrome_pre is not None and not SLICE_ON:
            bx0, by0, bx1, by1, _piY, _saY, _piC, _saU, _saV = _chrome_pre
            _blend_pre_into(canvas_y[by0:by1, bx0:bx1], _piY, _saY)
            cy0, cy1, cx0, cx1 = by0 // _CH, by1 // _CH, bx0 // _CW, bx1 // _CW
            _blend_pre_into(canvas_u[cy0:cy1, cx0:cx1], _piC, _saU)
            _blend_pre_into(canvas_v[cy0:cy1, cx0:cx1], _piC, _saV)
        _ts_ov2 = time.time_ns()   # fin du blend chrome
        # 2) VU-mètres puis 3) horloges : blend de chaque TUILE sur sa propre bbox (z-ordre conservé :
        # meters sous les horloges fg). Chaque tuile ne couvre que son petit rectangle local.
        _px_blend = 0   # pixels Y réellement recouverts par les tuiles de cette trame
        _up_ns = 0      # ns passés à TÉLÉVERSER les tuiles vers le backend (identité en CPU)
        if not SLICE_ON:
            _n_meters.push(len(_meter_tiles or ()))
            _n_hist.push(len(_hist_tiles or ()))
            _n_clock.push(len(_pf_tiles or ()))
        # `_cache` dit si la CATÉGORIE est stable entre deux trames : les frises et les horloges le
        # sont (re-dessinées au changement de valeur), les VU-mètres non (le niveau bouge à chaque
        # trame). Z-ordre inchangé : meters sous les horloges fg.
        for _cache, _src_tiles in (() if SLICE_ON else ((False, _meter_tiles), (True, _hist_tiles),
                                                        (True, _pf_tiles))):
            if not _src_tiles:
                continue
            for _tuile in _src_tiles:
                # Téléversement au backend pour le blend en VRAM (_to_xp = identité en CPU), en
                # réutilisant la copie VRAM des tuiles qui n'ont pas changé.
                _tsu = time.time_ns()
                if _cache:
                    (bx0, by0, bx1, by1, _oy, _ou, _ov, _oa, _oa2) = _tuile_dev(_tuile)
                else:
                    # Tuile qui change à chaque trame : un seul transfert épinglé au lieu de cinq
                    # copies pageables synchrones (cf. _tuile_vram). Repli COMPTÉ sur l'ancien
                    # chemin — une tuile qui disparaîtrait en silence serait pire que lente.
                    _v = _tuile_vram(_tuile) if GPU else None
                    if _v is not None:
                        (bx0, by0, bx1, by1, _oy, _ou, _ov, _oa, _oa2) = _v
                    else:
                        (bx0, by0, bx1, by1, _oy, _ou, _ov, _oa, _oa2) = _tuile
                        _oy, _ou, _ov, _oa, _oa2 = (_to_xp(_oy), _to_xp(_ou), _to_xp(_ov),
                                                    _to_xp(_oa), _to_xp(_oa2))
                _up_ns += time.time_ns() - _tsu
                _px_blend += (bx1 - bx0) * (by1 - by0)
                _blend_into(canvas_y[by0:by1, bx0:bx1], _oy, _oa)
                cy0, cy1, cx0, cx1 = by0 // _CH, by1 // _CH, bx0 // _CW, bx1 // _CW
                _blend_into(canvas_u[cy0:cy1, cx0:cx1], _ou, _oa2)
                _blend_into(canvas_v[cy0:cy1, cx0:cx1], _ov, _oa2)

        _tuiles_dev_evict()
        _n_px_blend.push(_px_blend); _t_ov_upload.push(_up_ns / 1e6)
        _t_after_overlays = time.time_ns()   # profiling : ov_convert=blend chrome, ov_blend=blend VU+horloges (bbox)
        _t_ov_render.push((_ts_ov1 - _ts_ov0) / 1e6)
        _t_ov_convert.push((_ts_ov2 - _ts_ov1) / 1e6)
        _t_ov_blend.push((_t_after_overlays - _ts_ov2) / 1e6)
        if SLICE_ON:
            # MODE TRANCHE : placement des tuiles + habillage + écriture sortie BANDE PAR BANDE,
            # avec commit MXL progressif (l'aval démarre sur la 1ʳᵉ bande). CPU/paysage (gaté).
            # MODE TRANCHE : les historiques passent par le même canal que les horloges (tuiles
            # per-frame blendées bande par bande) — concaténés à _pf_tiles.
            _sl_waited = _compose_bands(canvas_y, canvas_u, canvas_v, _slice_batch,
                                        _chrome_pre, _meter_tiles,
                                        ((_pf_tiles or []) + (_hist_tiles or [])) or None,
                                        fi_out=_fi_out) or 0
        else:
            if _PORTRAIT:                        # compose en portrait → tourne 90° vers la trame paysage
                canvas_y, canvas_u, canvas_v = _rotate_out(canvas_y, canvas_u, canvas_v)
            if INTERLACED:
                # SORTIE CHAMP-NATIVE : la trame composée (PLEINE) est découpée en 2 CHAMPS, écrits
                # dans 2 grains distincts — c'est ce que libmxl alloue (grain = ½ hauteur) et ce que
                # l'aval champ-natif attend (moteur 2110 : `reader_field`, index = trame×2 + parité).
                # Champ de parité p = lignes p::2 des TROIS plans : la chroma suit le même découpage
                # (4:2:2 → champ chroma = ½ largeur × ½ hauteur de trame ; 4:2:0 → ¼) — exactement le
                # `full[k][fld::2]` du générateur de mire du moteur, la seule référence qui fait foi.
                # Index PAIR = lignes PAIRES = champ HAUT ; l'ordre de champ (tff/bff) est DÉCLARÉ
                # dans le flowDef et n'inverse pas cette correspondance (il dit lequel part d'abord).
                _fbase = out_writer.next_index()
                for _p in (0, 1):
                    _fld = xp.concatenate([canvas_y[_p::2].ravel(),
                                           canvas_u[_p::2].ravel(),
                                           canvas_v[_p::2].ravel()])
                    _gidx, _gi_o, _vw_o = out_writer.open_grain(index=_fbase * 2 + _p)
                    _vw_o[:OUT_FIELD_SIZE] = _out_host(_fld).view(np.uint8)
                    out_writer.commit(_gi_o)
            else:
                # Grain de sortie MXL (zéro-copie : vue uint8 de l'array _NP_DT). Index implicite :
                # open_grain() sans argument appelle bobimxl.Writer.next_index (grille TAI).
                # Les trois plans sont déversés DIRECTEMENT dans le tampon épinglé, sans passer par
                # une trame contiguë en VRAM : l'allocation et la recopie plein cadre que coûtait
                # `xp.concatenate` disparaissent (cf. _out_host_plans).
                _gidx, _gi_o, _vw_o = out_writer.open_grain()
                _vw_o[:OUT_FRAME_SIZE] = _out_host_plans(
                    (canvas_y, canvas_u, canvas_v)).view(np.uint8)
                out_writer.commit(_gi_o)
        ts_out = time.time_ns()
        # Détail du compositing : entrées (lecture+resize+blend tuiles) / habillage / assemblage sortie.
        _t_inputs.push((_t_after_inputs - ts_cycle_start) / 1e6)
        _t_overlays.push((_t_after_overlays - _t_after_inputs) / 1e6)
        _t_output.push((ts_out - _t_after_overlays) / 1e6)
        _bake_ctr["frames"] += 1
        _rate_total["frames"] += 1     # compteur MONOTONE des trames RÉELLEMENT composées+émises
        if _frame_neuve:
            _rate_total["neuves"] += 1
        _frame_neuve = False
    except Exception as _e:
        # Garde-fou : une exception transitoire (rendu overlay/chrome, écriture sortie…) ne doit
        # PAS tuer le process. On saute cette frame (la sortie garde sa dernière image via le ring)
        # et la cadence avance normalement ci-dessous. Log throttlé pour ne pas inonder.
        _nowe = time.time()
        if _nowe - _last_frame_err > 5.0:
            log(f"multiview: erreur de rendu ignorée (frame {{out_frame_index}}) : {{_e}}", "warning")
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
    _watch["t0"] = 0        # FIN de trame : au-delà, la boucle ATTEND son créneau et ce temps
                            # n'est pas du travail. Sans cette remise à zéro, le chien de garde
                            # capturait `time.sleep` — 37 % de faux positifs au premier essai.
    own_lat.push((ts_out - ts_cycle_start - _sl_waited) / 1e6)
    # FRAME LENTE = PRÉCURSEUR d'incident : reste visible au niveau par défaut (`info`) — mais
    # AGRÉGÉE (règle 3 du bloc « Niveau de log »). L'ancienne version sortait jusqu'à 1 ligne/s
    # en régime dégradé (spam de rafale, la rétention du journal y passait) ; on accumule
    # désormais sur une fenêtre d'1 min et on émet UNE ligne : combien, la pire, et le segment
    # DOMINANT (tick raté avant le cycle / gather des entrées / habillage / écriture sortie).
    # Le détail par occurrence reste disponible en `debug`.
    _own_ms_dbg = (ts_out - ts_cycle_start - _sl_waited) / 1e6
    if _own_ms_dbg > 15.0 or _tick_late_ms > 10.0:
        try:
            _seg_in = (_t_after_inputs - ts_cycle_start) / 1e6
            _seg_ov = (_t_after_overlays - _t_after_inputs) / 1e6
            _seg_out = (ts_out - _t_after_overlays) / 1e6
        except Exception:
            _seg_in = _seg_ov = _seg_out = -1.0
        _slow_acc["n"] += 1
        if _own_ms_dbg > _slow_acc["worst_own"]:
            _slow_acc["worst_own"] = _own_ms_dbg
        if _tick_late_ms > _slow_acc["worst_tick"]:
            _slow_acc["worst_tick"] = _tick_late_ms
        # Segment dominant de CETTE occurrence (le plus cher des quatre) → histogramme de la fenêtre.
        _segs = (("tick", _tick_late_ms), ("inputs", _seg_in), ("overlays", _seg_ov),
                 ("output", _seg_out), ("waited", _sl_waited / 1e6))
        _slow_acc["seg"][max(_segs, key=lambda kv: kv[1])[0]] += 1
        log(f"multiview: FRAME LENTE own={{_own_ms_dbg:.1f}}ms tick_late={{_tick_late_ms:.1f}}ms "
            f"inputs={{_seg_in:.1f}} overlays={{_seg_ov:.1f}} output={{_seg_out:.1f}} "
            f"waited={{_sl_waited / 1e6:.1f}} fi={{out_frame_index}}", "debug")
    # Émission de la ligne AGRÉGÉE : au plus 1/min, et UNIQUEMENT si la fenêtre a vu des frames
    # lentes (un mur sain n'écrit RIEN). Compteur cumulé aussi exposé sur :8080 (métrique).
    _nowd = time.time()
    if _slow_acc["n"] and _nowd - _slow_acc["t0"] >= 60.0:
        _dom = max(_slow_acc["seg"].items(), key=lambda kv: kv[1])
        log(f"multiview: {{_slow_acc['n']}} frame(s) lente(s) sur la dernière minute — "
            f"pire own={{_slow_acc['worst_own']:.1f}}ms tick_late={{_slow_acc['worst_tick']:.1f}}ms, "
            f"segment dominant = {{_dom[0]}} ({{_dom[1]}}×)", "info")
        metrics["slow_frames_total"] = metrics.get("slow_frames_total", 0) + _slow_acc["n"]
        _slow_acc.update({{"n": 0, "worst_own": 0.0, "worst_tick": 0.0, "t0": _nowd}})
        for _k in _slow_acc["seg"]:
            _slow_acc["seg"][_k] = 0
    elif _nowd - _slow_acc["t0"] >= 60.0:
        _slow_acc["t0"] = _nowd
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
                _nft = _grid_next(time.time(), FRAME_INTERVAL)
                # ★ VRAIES TRAMES PERDUES : le recale saute les slots de grille compris entre le tick
                # attendu et le prochain. C'est LE signal de santé (« ai-je perdu des images ? ») —
                # ne plus le déduire d'un fps bruité. 0 = le mur tient sa grille.
                _miss = int(round((_nft - next_frame_time) / FRAME_INTERVAL))
                if _miss > 0:
                    _rate_total["missed"] += _miss
                next_frame_time = _nft
    else:
        next_frame_time = start_time + (out_frame_index * FRAME_INTERVAL)
    if out_frame_index % 25 == 0:
        # Échantillon de cadence (compteurs monotones) : le fps LUI-MÊME est calculé sur une fenêtre
        # de TEMPS au scrape (cf. _update_rate_metrics) — plus de fenêtre à nombre de trames FIXE,
        # dont la durée variable faisait « chuter » fps à 47-48 sans qu'aucune trame ne soit perdue.
        _rate_sample()
        _update_rate_metrics()
        _refresh_lat_metrics()
        _sd = globals().get('_sl_dbg')
        if _sd:
            # Observabilité recette (docs/chantiers/TISSU_SLICE.md) : compteurs slice promus en métriques :8080.
            metrics["slice"] = {{"tiles": _sd[0], "valid0": _sd[1], "waits": _sd[2],
                               "fallbacks": _sd[3], "dormant": _sd[4],
                               "backoff": len(_sl_backoff)}}
        # ⚠ MÉTRIQUE, PAS UN ÉVÉNEMENT (règle 2 du bloc « Niveau de log ») : fps, trames
        # perdues et compteurs slice sont DÉJÀ publiés sur :8080 et échantillonnés par
        # l'orchestrateur. Cette ligne (2/s à 50 fps) ne faisait que dupliquer la mesure et
        # BRÛLER la fenêtre de rétention du journal — c'est elle qui produisait les 24 Mo
        # mesurés au banc, purgeant les lignes utiles anciennes. Reléguée en `debug`.
        log(f"Mix frame {{out_frame_index}} — {{metrics['fps']}} {{metrics['fps_unit']}}/s"
            + (f" ({{metrics['frames_per_s']}} trames/s)" if INTERLACED else "")
            + (f" [perdues: {{_rate_total['missed']}}]" if _rate_total["missed"] else "")
            + (f" [slice: tuiles={{_sd[0]}} valid0={{_sd[1]}} waits={{_sd[2]}} replis={{_sd[3]}} dorm={{_sd[4]}}]" if _sd else ""),
            "debug")
    # Point sûr GC (cf. bloc gc.disable() avant la boucle) : la dernière bande de la trame est
    # committée (l'aval a déjà tout), on est dans le temps mort avant le tick suivant. gen0+gen1
    # chaque trame ; gen2 cadencé et MESURÉ (gc_full_ms sur :8080 — recette : doit rester à
    # quelques ms grâce au freeze, sinon investiguer la croissance du tas).
    _gc_frames += 1
    if _gc_frames % _gc_full_every == 0:
        _t_gc = time.monotonic_ns()
        gc.collect(2)
        _gc_last_full_ms = (time.monotonic_ns() - _t_gc) / 1e6
        if _gc_last_full_ms > _gc_max_full_ms:
            _gc_max_full_ms = _gc_last_full_ms
        metrics["gc_full_ms"] = {{"last": round(_gc_last_full_ms, 2),
                                 "max": round(_gc_max_full_ms, 2)}}
    else:
        gc.collect(1)
