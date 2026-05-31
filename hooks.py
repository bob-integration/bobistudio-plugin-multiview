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
