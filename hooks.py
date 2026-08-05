# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 BOBI SAS, France
# Auteur : Cyril Mazouer, pour le compte de BOBI SAS
# Distribué sous licence GNU GPL v3 (ou ultérieure) ; voir le fichier LICENSE.

import re


def _derive_essence_shm(video_shm, essence):
    """`mire1_0` (ou `/dev/shm/mire1_0`) → `mire1_audio_0` / `mire1_anc_0`. None si pas de _<n>
    final. MÊME dérivation que script.py:_derive_audio_name → le nom matche la sortie du producteur
    (2110_io : `<hôte>_audio_<i>` / `<hôte>_anc_<i>`) donc l'arête se trace sur la page Câbles."""
    name = (video_shm or "")
    if name.startswith("/dev/shm/"):
        name = name[len("/dev/shm/"):]
    m = re.match(r"(.+?)_(\d+)$", name)
    if not m:
        return None
    return "%s_%s_%s" % (m.group(1), essence, m.group(2))


def topology_ports(hostname, params, ctx):
    """Ports topologie (page Câbles). Réutilise derive_wiring pour les ports VIDÉO (parité TOTALE
    avec le câblage, inchangé) puis AJOUTE, par entrée vidéo câblée, les ports AUDIO (lu pour les
    VU-mètres) et ANC (data) DÉRIVÉS du nom de la source, GROUPÉS sous l'entrée. Ports dérivés =
    INFORMATIFS : ils suivent automatiquement la source vidéo (le câblage reste piloté par la
    vidéo via derive_wiring) → marqués `derived` (non câblables séparément)."""
    from app import plugins as _pl
    w = _pl.derive_wiring("multiview", hostname, params)
    adapts = bool((_pl.get("multiview") or {}).get("adapts_input"))
    produces = []
    for prod in (w.get("produces") or []):
        if not prod.get("shm"):
            continue
        pp = {"shm": prod["shm"], "kind": prod.get("essence") or "video"}
        if prod.get("label"):
            pp["label"] = prod["label"]
        if prod.get("format"):
            pp["format"] = prod["format"]
        produces.append(pp)
    # Ordonné PAR TYPE DE SIGNAL (toutes les vidéos, puis tous les audios, puis tous les ANC) —
    # harmonisé avec les autres conteneurs (ex. 2110_io). Le champ `group` (in<slot>) garde
    # l'association visuelle vidéo↔audio↔ANC d'une même entrée. Les ports AUDIO et ANC sont
    # désormais de VRAIS ports câblables (wiring audio_path / anc_path) ; quand ils ne sont pas
    # câblés mais que la vidéo du même slot l'est, on affiche le flux DÉRIVÉ du nom de la source
    # (repli du moteur, marqué `derived` = informatif) plutôt qu'un port déconnecté.
    vids, auds, ancs = [], [], []
    video_shm_by_slot = {}
    for spec in (w.get("consumes") or []):
        ess = spec.get("essence") or "video"
        slot = spec.get("slot")
        shm = spec.get("shm") or ""
        grp = "in%s" % (slot if slot is not None else len(vids))
        lbl = spec.get("label") or ""
        # Le `name` d'une fenêtre en libellé TSL (label_source == "protocol") est un PLACEHOLDER
        # d'ÉDITEUR : « (TSL #<idx>) », posé par computeDisplayName() parce que le texte réel n'est
        # connu qu'au runtime. En central l'index n'est même pas saisi (il est déduit de la source,
        # cf. services/tsl) → l'éditeur écrit « (TSL #?) ». Le laisser remonter au wiring le fait
        # afficher tel quel comme NOM DE PORT sur la page Câbles, où il ne veut rien dire. Le port
        # reprend donc son identité de slot ; la source, elle, est nommée à côté.
        if ess == "video" and spec.get("from_list") == "flux_config" and slot is not None:
            _fx = params.get("flux_config") or []
            _e = _fx[slot] if 0 <= slot < len(_fx) and isinstance(_fx[slot], dict) else {}
            if (_e.get("label_source") or "hostname") == "protocol":
                lbl = "Entrée %d" % (slot + 1)
        port = {"kind": ess, "group": grp}
        if slot is not None:
            port["slot"] = slot
        if lbl:
            port["label"] = lbl
        if spec.get("format") and not adapts:
            port["format"] = spec["format"]
        if shm:
            port["shm"] = shm
        else:
            port["shm"] = ""; port["disconnected"] = True
        if ess == "video":
            video_shm_by_slot[slot] = shm
            vids.append(port)
            continue
        # Port audio/ANC non câblé + vidéo du slot câblée → repli dérivé du moteur (informatif).
        if port.get("disconnected"):
            vshm = video_shm_by_slot.get(slot) or ""
            dn = _derive_essence_shm(vshm, "audio" if ess == "audio" else "anc") if vshm else None
            if dn:
                port = {"kind": ess, "group": grp, "shm": dn, "derived": True}
                if slot is not None:
                    port["slot"] = slot
                if lbl:
                    port["label"] = lbl
        (auds if ess == "audio" else ancs).append(port)
    return {"produces": produces, "consumes": vids + auds + ancs}


def _parse_video_formats(raw):
    """Parse la chaîne video_formats des Settings → liste de dicts."""
    result = []
    for line in (raw or "").splitlines():
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 5:
            continue
        try:
            result.append({
                "label": parts[0],
                "width": int(parts[1]),
                "height": int(parts[2]),
                "fps":    float(parts[3]),
                "scan":   parts[4],
            })
        except (ValueError, IndexError):
            pass
    return result


def before_deploy(params, context):
    """Normalise les params multiview avant déploiement :
    - shm_out dérivé du hostname si encore à la valeur par défaut
    - résolution/fps/scan issus du format vidéo par défaut des Réglages
      (appliqués seulement si encore aux valeurs par défaut du plugin)
    """
    hostname = context.get("hostname") or params.get("hostname") or ""
    settings = context.get("settings") or {}

    params = dict(params)

    # 1. SHM de sortie : hostname si valeur générique
    if not params.get("shm_out") or params.get("shm_out") == "mxl_mix":
        if hostname:
            params["shm_out"] = hostname

    # 2. Résolution par défaut : lire video_format_default des Réglages
    #    Appliqué seulement si la résolution est encore à la valeur par défaut du plugin
    DEFAULT_W, DEFAULT_H, DEFAULT_FPS = 1280, 720, 25
    is_default_res = (
        int(params.get("out_width")  or 0) == DEFAULT_W and
        int(params.get("out_height") or 0) == DEFAULT_H and
        float(params.get("fps") or 0) == DEFAULT_FPS
    )
    if is_default_res:
        fmt_label = settings.get("video_format_default") or ""
        fmt_raw   = settings.get("video_formats") or ""
        if fmt_label and fmt_raw:
            fmts = _parse_video_formats(fmt_raw)
            fmt  = next((f for f in fmts if f["label"] == fmt_label), None)
            if fmt:
                params["out_width"]  = fmt["width"]
                params["out_height"] = fmt["height"]
                params["fps"]        = fmt["fps"]
                params["scan"]       = fmt.get("scan", "p")

    # 2bis. FILET ports suiveurs (audio_path/anc_path, posés par la page Câbles) : un front
    #    qui reconstruit flux_config sans ces clés (classe de bug déjà vue : flags ANC avant
    #    0.29.0, puis les PORTS eux-mêmes jusqu'à multiview.js 0.32.1) les perdait au
    #    redéploiement → VU en ABSENCE / ANC vide dès que la dérivation _N est impossible.
    #    Ré-hydratation depuis la config PERSISTÉE, STRICTEMENT bornée : uniquement si la clé
    #    est ABSENTE de l'entrée postée (une valeur posée — y compris "" = décâblage
    #    VOLONTAIRE, ou un autre flux choisi exprès — n'est JAMAIS remplacée), et uniquement
    #    si le path VIDÉO de l'entrée n'a pas changé (nouvelle source = suiveurs re-dérivés
    #    par le câblage, pas par ce filet).
    try:
        _vmid = context.get("vmid")
        if _vmid:
            import json as _json
            from app.database import db_get_container as _dgc
            _row = _dgc(_vmid) or {}
            _dc = _row.get("deploy_config")
            _dc = _json.loads(_dc) if isinstance(_dc, str) else (_dc or {})
            if _dc.get("type") == "multiview":
                _old = (_dc.get("params") or {}).get("flux_config") or []
                _new = params.get("flux_config") or []
                for _i, _e in enumerate(_new):
                    if not isinstance(_e, dict) or _i >= len(_old):
                        continue
                    _o = _old[_i] if isinstance(_old[_i], dict) else {}
                    if (_e.get("path") or "") != (_o.get("path") or ""):
                        continue
                    for _k in ("audio_path", "anc_path"):
                        if _k not in _e and _k in _o:
                            _e[_k] = _o[_k]
    except Exception:
        pass   # filet best-effort : ne jamais bloquer un déploiement

    # 3. Banque d'entrées : flux_config paddée à max_inputs entrées à indices STABLES
    #    (slot de câblage = tally = Ember). Les entrées au-delà des PiP affichés sont
    #    masquées (hidden) mais restent câblables — supprimer un PiP ne coupe plus la
    #    source. Migration implicite des anciens configs (liste courte → paddée).
    #    On ne tronque JAMAIS (une entrée câblée au-delà de max_inputs survit).
    params["flux_config"] = pad_input_bank(params)

    # 4. Overlays (texte/horloge/image) : objets purement visuels, non câblés. On garantit
    #    juste un id stable et un kind (le résolveur image base64 est dans deploy.py).
    params["overlays"] = normalize_overlays(params.get("overlays"))

    # 5. Filet anti-régression : repli CPU forcé via le réglage global `multiview_force_cpu`
    #    (le script lit le param `force_cpu` → xp=numpy même sur nœud GPU). Un param explicite
    #    déjà posé (par-conteneur) reste prioritaire ; sinon on applique le réglage global.
    if "force_cpu" not in params:
        _fc = str(settings.get("multiview_force_cpu") or "").strip().lower()
        if _fc in ("1", "true", "yes", "on"):
            params["force_cpu"] = True

    # 6. Heure CIVILE pour les horloges « PTP » (overlays + composants clock des modèles de
    #    PiP) : fuseau du contrôleur (`tz` — images runtime en UTC) + `tai_utc_offset_s`
    #    (l'horloge des nœuds est sur l'échelle PTP/TAI, cf. docs/reference/PTP_CLOCK.md — offset MESURÉ
    #    contre le contrôleur, jamais 37 figé). Ré-évalué à chaque déploiement.
    #    ★ Le `tz` EXPLICITE du mur PRIME. `params.update()` écrasait sans condition la valeur
    #    saisie par l'utilisateur à CHAQUE déploiement : le réglage semblait accepté puis
    #    revenait au fuseau du système au premier redéploiement. Un mur dont `tz` est vide suit
    #    le réglage global, ce qui reste le défaut voulu.
    # Variables de texte de SOURCE : les niveaux de libellé (label_2..label_9) et le projet
    # vivent dans `source_labels`, côté orchestrateur, keyés par shm. Le conteneur ne peut pas
    # les lire — on les embarque PAR FENÊTRE au déploiement. Rien n'est écrasé : on ne pose que
    # les clés dérivées (`labels`, `projet`), jamais le `name` choisi par l'utilisateur.
    try:
        from app.database import db_get_source_labels
        _lbl = {}
        for _row in (db_get_source_labels() or []):
            _shm = (_row.get("shm") or "").strip()
            if _shm:
                _lbl[_shm] = _row
        _fx = params.get("flux_config")
        if isinstance(_fx, list) and _lbl:
            _out = []
            for _e in _fx:
                if not isinstance(_e, dict):
                    _out.append(_e); continue
                _e = dict(_e)
                _shm = (_e.get("path") or "").removeprefix("/dev/shm/")
                _r = _lbl.get(_shm)
                if _r:
                    _e["labels"] = {str(_n): (_r.get("label_%d" % _n) or "")
                                    for _n in range(2, 10)}
                    _e["projet"] = _r.get("projet") or ""
                _out.append(_e)
            params["flux_config"] = _out
    except Exception:
        pass

    # Variables de texte %systeme% / %noeud% : le conteneur ne peut PAS les connaître (branding
    # et table des nœuds vivent côté orchestrateur). Injectées ici, comme le fuseau.
    try:
        from app import settings as _st
        params["system_name"] = (_st.get("brand_system_name") or "").strip()
    except Exception:
        pass
    try:
        from app.database import db_get_container, db_get_nodes
        _c = db_get_container(context.get("vmid")) or {}
        _nid = _c.get("node_id")
        if _nid:
            for _n in (db_get_nodes() or []):
                if _n.get("id") == _nid:
                    params["node_name"] = _n.get("name") or ""
                    break
    except Exception:
        pass
    try:
        from app.ptp import civil_clock_params
        _civ = civil_clock_params(context.get("vmid")) or {}
        if (params.get("tz") or "").strip():
            _civ.pop("tz", None)
        params.update(_civ)
    except Exception:
        pass

    return params


def normalize_overlays(ov_list):
    """Garantit id stable + kind pour chaque overlay. Idempotent."""
    out = []
    for i, ov in enumerate(ov_list or []):
        if not isinstance(ov, dict):
            continue
        ov = dict(ov)
        if not ov.get("id"):
            ov["id"] = "ov%d" % i
        if not ov.get("kind"):
            ov["kind"] = "text"
        out.append(ov)
    return out


def pad_input_bank(params):
    """flux_config complétée à max_inputs entrées (masquées, sans source). Idempotent."""
    try:
        max_inputs = int(params.get("max_inputs") or 0)
    except (TypeError, ValueError):
        max_inputs = 0
    fc = [dict(f) for f in (params.get("flux_config") or []) if isinstance(f, dict)]
    out_w = int(params.get("out_width") or 1280)
    out_h = int(params.get("out_height") or 720)
    while len(fc) < max_inputs:
        fc.append({
            "path": "", "hidden": True, "name": "", "label_source": "hostname",
            "in_w": 0, "in_h": 0,
            "x": 0, "y": 0, "w": (out_w // 2) & ~1, "h": (out_h // 2) & ~1,
            "show_label": True, "show_tally": False, "tsl_index": 0,
            "label_col": 0, "tally_level": 0, "tally_red": False, "tally_green": False,
            "meter_channels": 0, "meter_position": "right", "meter_inside": False,
            "meter_opacity": 70, "meter_scale": "dbfs",
        })
    return fc


def produced_flow_count(params, ctx):
    return 1


def ember_clear_slot(slot_type, slot_idx, params, context):
    """Efface le câblage Ember+ d'une fenêtre multiview (flux_config[i].path → vide)."""
    if slot_type != "multiview_window":
        return None
    flux = [dict(f) for f in (params.get("flux_config") or [])]
    if not (0 <= slot_idx < len(flux)):
        return None
    flux[slot_idx]["path"] = ""
    params["flux_config"] = flux
    return {"params": params, "body": {"idx": slot_idx, "shm": ""}}


def ember_sources(params, context):
    """Sortie composite — source du routing Ember+."""
    shm_out = params.get("shm_out", "")
    if not shm_out:
        return []
    return [{"label": context.get("hostname", ""), "shm": shm_out}]


def ember_targets(params, context):
    """Fenêtres d'entrée — cibles du routing Ember+."""
    hn = context.get("hostname", "")
    return [
        {
            "label": f"{hn}/{f.get('name') or f'win{i}'}",
            "slot_type": "multiview_window",
            "slot_idx": i,
            "shm": f.get("path", ""),
        }
        for i, f in enumerate(params.get("flux_config") or [])
    ]
