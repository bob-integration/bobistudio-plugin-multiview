# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 BOBI SAS, France
# Auteur : Cyril Mazouer, pour le compte de BOBI SAS
# Distribué sous licence GNU GPL v3 (ou ultérieure) ; voir le fichier LICENSE.

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
            "label_col": 0, "tally_l_level": 0, "tally_r_level": 1,
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
