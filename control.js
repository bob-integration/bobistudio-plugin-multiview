// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 BOBI SAS, France
// Auteur : Cyril Mazouer, pour le compte de BOBI SAS
// Distribué sous licence GNU GPL v3 (ou ultérieure) ; voir le fichier LICENSE.

// Multiview — UI de contrôle embarquée (plugin). Thin wrapper : le HTML complet du
// composer est dans control.html ; multiview.js est chargé par le shell Traitements
// directement (script tag statique) pour éviter les problèmes de timing/cache.
'use strict';
window.MXLPlugins = window.MXLPlugins || {};
window.MXLPlugins.multiview = {
    // Modèle par-instance du shell générique Traitements : le shell injecte control.html
    // dans le mount puis appelle mount(el, vmid) pour l'instance sélectionnée. La sidebar
    // (liste + mini-aperçu + badge version) est gérée par le shell via listPreview ci-dessous.
    mount(el, vmid, ctx) {
        // multiview.js déjà chargé par le shell → appel direct.
        // i18n : traduit le HTML statique injecté (data-i18n*) avant le premier rendu.
        if (typeof mwApplyI18n === 'function') mwApplyI18n(el);
        if (vmid != null && typeof chargerMw === 'function') chargerMw(vmid);
        if (typeof rafraichirListeLayouts === 'function') rafraichirListeLayouts();
        window.addEventListener('resize', this._onResize);
    },

    _onResize() {
        if (typeof resizeCanvas === 'function' && document.getElementById('ed_canvas')) {
            resizeCanvas();
            if (typeof dessiner === 'function') dessiner();
        }
    },

    // Mini-aperçu (schéma des fenêtres) pour une instance dans la sidebar générique.
    async listPreview(canvas, inst) {
        if (!canvas || !inst) return;
        try {
            const c = await (await fetch('/api/containers/' + inst.vmid + '/config')).json();
            const dc = c.deploy_config ? JSON.parse(c.deploy_config) : null;
            if (dc && dc.params && typeof drawMiniPreview === 'function') {
                drawMiniPreview(canvas, dc.params);
            }
        } catch (e) {}
    },

    unmount() {
        window.removeEventListener('resize', this._onResize);
    }
};
