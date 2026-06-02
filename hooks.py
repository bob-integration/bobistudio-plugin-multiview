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

    return params


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
