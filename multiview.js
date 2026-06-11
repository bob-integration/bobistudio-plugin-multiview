// ─── État global ──────────────────────────────────────────────

let allContainers = [];
let videoSources = [];   // sorties vidéo individuelles de la flotte (cf. loadVideoSources)
let editorVmid    = null;
let editorParams  = null;   // {flux_config, shm_out, out_width, out_height, border_w, border_color, overlay_below, max_inputs}
let selectedIdxs  = [];     // multi-select : le dernier est le primary (référence pour align/match-size)
let dragMode      = null;   // 'move' | 'resize'
let dragStart     = null;
let dragOrigRect  = null;
let snapEnabled   = true;
let snapGuides    = [];     // [{type:'v'|'h', pos:number}] dessinées pendant le drag

const HANDLE_SIZE = 10;
const SNAP_PX     = 8;      // distance de snap (en coords canvas)

function primaryIdx() {
    return selectedIdxs.length ? selectedIdxs[selectedIdxs.length - 1] : -1;
}
function isSelected(i) { return selectedIdxs.includes(i); }
const COLORS = [
    'oklch(0.56 0.15 248)',
    'oklch(0.52 0.14 145)',
    'oklch(0.60 0.13 62)',
    'oklch(0.50 0.15 298)',
    'oklch(0.54 0.13 196)',
    'oklch(0.52 0.14 22)',
    'oklch(0.57 0.11 162)',
    'oklch(0.61 0.12 86)',
];

const LABEL_SOURCES = [
    {value: 'hostname', label: 'Nom de la source'},
    {value: 'mxl_path', label: 'Chemin MXL'},
    {value: 'protocol', label: 'Protocole (TSL 5.0)'}
];

// Nom de la source (hostname container) — toujours, indépendant de label_source.
// Sert d'identifiant dans le tableau / la liste.
// Match un shm individuel (ex "Mire_0") à un container dont le shm_out peut
// être agrégé ("Mire_0 · Mire_audio_0") ou en plage ("Mire_0..3").
function _containerForShm(shm) {
    if (!shm) return null;
    return allContainers.find(c => {
        if (!c.shm_out) return false;
        const so = String(c.shm_out);
        if (so === shm) return true;
        if (so.split(' · ').includes(shm)) return true;
        // plage Mire_0..3 → matche Mire_0..Mire_3
        for (const part of so.split(' · ')) {
            const m = part.match(/^(.+)_(\d+)\.\.(\d+)$/);
            if (m) {
                const base = m[1], a = +m[2], b = +m[3];
                const m2 = shm.match(/^(.+)_(\d+)$/);
                if (m2 && m2[1] === base) {
                    const i = +m2[2];
                    if (i >= a && i <= b) return true;
                }
            }
        }
        return false;
    });
}

function sourceHostname(f) {
    const shm = (f.path || '').replace(/^\/dev\/shm\//, '');
    if (!shm) return '';
    const c = _containerForShm(shm);
    return c ? c.hostname : shm;
}

// Calcule le texte affiché pour une entrée selon son label_source
function computeDisplayName(f) {
    const ls = f.label_source || 'hostname';
    if (ls === 'mxl_path') {
        // shm name (sans /dev/shm/)
        return (f.path || '').replace(/^\/dev\/shm\//, '');
    }
    if (ls === 'protocol') {
        // En éditeur, on n'a pas la valeur live ; on affiche un placeholder.
        // Le container substituera le texte TSL reçu au runtime.
        return `(TSL #${f.tsl_index || '?'})`;
    }
    // hostname : lookup container par shm_out (matching flexible cf _containerForShm)
    const shm = (f.path || '').replace(/^\/dev\/shm\//, '');
    const c = _containerForShm(shm);
    return c ? c.hostname : shm;
}


// ─── Mini-preview (canvas réduit, sans label/tally) ──────────

function drawMiniPreview(canvas, params) {
    if (!canvas || !params) return;
    const ctx = canvas.getContext('2d');
    const ow = params.out_width  || 1280;
    const oh = params.out_height || 720;
    // Canvas redimensionné pour conserver le ratio
    const cw = canvas.clientWidth || 200;
    const ch = Math.round(cw * oh / ow);
    canvas.width  = cw;
    canvas.height = ch;
    const sx = cw / ow;
    const sy = ch / oh;

    const _cs = getComputedStyle(document.documentElement);
    ctx.fillStyle = _cs.getPropertyValue('--canvas-bg').trim() || '#0d1117';
    ctx.fillRect(0, 0, cw, ch);

    const borderW     = (params.border_w || 0) * Math.min(sx, sy);
    const borderColor = params.border_color || '#ffffff';

    (params.flux_config || []).forEach((f, i) => {
        const x = f.x * sx, y = f.y * sy;
        const w = f.w * sx, h = f.h * sy;
        ctx.globalAlpha = 0.67;
        ctx.fillStyle   = COLORS[i % COLORS.length];
        ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = 1;
        if (borderW > 0.5) {
            ctx.strokeStyle = borderColor;
            ctx.lineWidth = Math.max(1, borderW);
            ctx.strokeRect(x + borderW/2, y + borderW/2,
                           Math.max(1, w - borderW), Math.max(1, h - borderW));
        }
    });
}

// ─── Containers (sources pour l'éditeur) ──────────────────────
// allContainers alimente les listes de sources de renderEditor() + _containerForShm().
// La liste d'instances multiview elle-même est rendue par le shell générique
// (sidebar tp-list + mini-aperçu via MXLPlugins.multiview.listPreview).
async function loadAllContainers() {
    try { allContainers = await (await fetch('/api/containers')).json(); }
    catch(e) { allContainers = []; }
}

// Sorties VIDÉO individuelles de toute la flotte (produces[] structuré : un container
// multi-sorties — mixer PGM/CLEAN/PVW, receiver multi-flux — expose chaque sortie
// séparément, avec son label). Bien plus fiable que de re-parser la chaîne composite
// shm_out. /api/sources = dérivation DB seule, sans le coût de /api/home/summary
// (PTP, sondes live) — l'éditeur s'ouvre vite même quand l'hôte PTP répond lentement.
async function loadVideoSources() {
    try {
        const srcs = await (await fetch('/api/sources?kind=video')).json();
        videoSources = (srcs || []).map(s => ({
            vmid: s.vmid, hostname: s.hostname || ('mxl' + s.vmid),
            shm: s.shm, label: s.label || '' }));
    } catch(e) { videoSources = []; }
}

async function chargerMw(vmid) {
    await Promise.all([loadAllContainers(), loadVideoSources()]);
    const r = await fetch('/api/containers/' + vmid + '/config');
    const c = await r.json();
    let dc = null;
    try { dc = c.deploy_config ? JSON.parse(c.deploy_config) : null; } catch(e) {}
    if (!dc || dc.type !== 'multiview') {
        mwFlash('Container sans config multiview.');
        return;
    }
    editorVmid   = vmid;
    editorParams = Object.assign({
        flux_config: [],
        shm_out: 'mxl_mix',
        out_width: 1280,
        out_height: 720,
        border_w: 0,
        border_color: '#ffffff',
        overlay_below: false,
        label_size: 14,
        frame_style: 'none',
        max_inputs: 4,
        genlock: true,
        tsl_port: 4801
    }, dc.params || {});
    // Couleurs locales pour le rendu (non sauvegardé)
    editorParams.flux_config = (editorParams.flux_config || []).map((f, i) => Object.assign({
        color: COLORS[i % COLORS.length],
        ratio: (f.in_w && f.in_h) ? f.in_w / f.in_h : 16/9
    }, f));
    selectedIdxs = [];
    renderEditor(c.hostname);
}

// ─── Éditeur (panneau droit) ──────────────────────────────────

function renderEditor(hostname) {
    const el = document.getElementById('mw-editor');
    el.classList.remove('empty');
    document.getElementById('mw-editor-empty').hidden = true;
    document.getElementById('mw-editor-form').hidden  = false;

    const p = editorParams;

    document.getElementById('ed_hostname').textContent = hostname || '';
    document.getElementById('ed_vmid').textContent     = editorVmid ?? '';

    document.getElementById('ed_max').value             = p.max_inputs;
    document.getElementById('ed_border_w').value        = p.border_w;
    document.getElementById('ed_border_color').value    = p.border_color;
    document.getElementById('ed_label_size').value      = p.label_size || 14;
    document.getElementById('ed_frame_style').value     = p.frame_style || 'none';
    document.getElementById('ed_overlay_below').checked = !!p.overlay_below;
    document.getElementById('ed_genlock').checked       = p.genlock !== false;
    document.getElementById('ed_tsl_port').value        = p.tsl_port ?? 0;
    document.getElementById('ed_snap').checked          = snapEnabled;
    document.getElementById('ed_paste_btn').disabled    = !reglagesClipboard;

    resizeCanvas();
    dessiner();
}

// ─── Handlers globaux ────────────────────────────────────────

function onGlobalChange() {
    editorParams.border_w      = parseInt(document.getElementById('ed_border_w').value) || 0;
    editorParams.border_color  = document.getElementById('ed_border_color').value;
    editorParams.overlay_below = document.getElementById('ed_overlay_below').checked;
    editorParams.label_size    = Math.max(6, parseInt(document.getElementById('ed_label_size').value) || 14);
    { const fs = document.getElementById('ed_frame_style'); if (fs) editorParams.frame_style = fs.value; }
    dessiner();
    hotApplyStyle();
}

function resizeCanvas() {
    const canvas = document.getElementById('ed_canvas');
    if (!canvas) return;
    const w = editorParams.out_width;
    const h = editorParams.out_height;
    canvas.width  = w;
    canvas.height = h;
    // Le canvas occupe toute la largeur de l'éditeur, conserve son ratio
    canvas.style.width   = '100%';
    canvas.style.height  = 'auto';
    canvas.style.maxWidth = w + 'px'; // pas plus grand que la résolution native
}

// ─── Ajouter / supprimer une entrée ──────────────────────────

function ajouterEntree() {
    if (editorParams.flux_config.length >= editorParams.max_inputs) {
        mwFlash(`Limite max_inputs = ${editorParams.max_inputs} atteinte.`);
        return;
    }
    const idx   = editorParams.flux_config.length;
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;
    const win_w = Math.round(out_w / 2);
    const win_h = Math.round(out_h / 2);
    const cols  = Math.max(1, Math.floor(out_w / win_w));
    const x     = (idx % cols) * win_w;
    const y     = Math.floor(idx / cols) * win_h;

    editorParams.flux_config.push({
        path: '',
        label_source: 'hostname',
        in_w: 0, in_h: 0,
        x, y,
        w: win_w,
        h: win_h,
        ratio: 16/9,
        color: COLORS[idx % COLORS.length],
        show_label: true,
        show_tally: false,
        tsl_index: 0,
        // Peak meters
        meter_channels: 0,           // 0 = désactivé ; sinon 2/4/6/8
        meter_position: 'right',     // left | right
        meter_inside: false,         // false = à côté (réduit la vidéo), true = overlay
        meter_opacity: 70,           // 10..100 (utilisé si inside=true)
        meter_scale: 'dbfs',         // dbfs | ppm (EBU)
    });
    selectedIdxs = [idx];
    dessiner();
    hotApplyFull();
}

function supprimerEntreeSelectionnee() {
    if (selectedIdxs.length === 0) return;
    // Supprime du dernier au premier pour préserver les indices
    [...selectedIdxs].sort((a, b) => b - a).forEach(i => {
        editorParams.flux_config.splice(i, 1);
    });
    selectedIdxs = [];
    dessiner();
    hotApplyFull();
}

// ─── Panneau d'édition d'une entrée ──────────────────────────

function refreshEntryPanel() {
    const panel = document.getElementById('ed_entry_panel');
    if (!panel) return;
    const primary = primaryIdx();
    if (primary < 0) {
        panel.hidden = true;
        return;
    }
    const f = editorParams.flux_config[primary];
    panel.hidden = false;
    document.getElementById('ed_label_source').value = f.label_source || 'hostname';
    document.getElementById('ed_show_label').checked = !!f.show_label;
    document.getElementById('ed_show_tally').checked = !!f.show_tally;
    document.getElementById('ed_tsl_index').value    = f.tsl_index || 0;
    document.getElementById('ed_x').value = f.x;
    document.getElementById('ed_y').value = f.y;
    document.getElementById('ed_w').value = f.w;
    document.getElementById('ed_h').value = f.h;
    // Peak meters
    document.getElementById('ed_meter_channels').value = String(f.meter_channels ?? 0);
    document.getElementById('ed_meter_position').value = f.meter_position || 'right';
    document.getElementById('ed_meter_inside').value   = (f.meter_inside ? '1' : '0');
    document.getElementById('ed_meter_opacity').value  = f.meter_opacity ?? 70;
    document.getElementById('ed_meter_scale').value    = f.meter_scale || 'dbfs';

    // Re-peuple la dropdown source pour cette entrée
    const pathSel = document.getElementById('ed_path');
    // Une <option> par sortie vidéo individuelle (libellé hostname → label (shm)).
    const opts = videoSources.map(s => {
        const txt = s.label ? `${s.hostname} → ${s.label} (${s.shm})`
                            : `${s.hostname} → ${s.shm}`;
        return `<option value="/dev/shm/${s.shm}">${txt}</option>`;
    });
    // PiP vide : option explicite « aucune source » en tête (path = '').
    opts.unshift('<option value="">— aucune source —</option>');
    // Inclure le path actuel même si introuvable dans la liste (container détruit p.ex.).
    if (f.path && !opts.some(o => o.includes(`value="${f.path}"`))) {
        opts.splice(1, 0, `<option value="${f.path}">${f.path}</option>`);
    }
    pathSel.innerHTML = opts.join('');
    pathSel.value = f.path || '';
}

function onEntryChange() {
    const primary = primaryIdx();
    if (primary < 0) return;
    const f = editorParams.flux_config[primary];
    f.label_source = document.getElementById('ed_label_source').value || 'hostname';
    f.path         = document.getElementById('ed_path').value;
    f.show_label   = document.getElementById('ed_show_label').checked;
    f.show_tally   = document.getElementById('ed_show_tally').checked;
    f.tsl_index    = parseInt(document.getElementById('ed_tsl_index').value) || 0;
    f.meter_channels = parseInt(document.getElementById('ed_meter_channels').value) || 0;
    f.meter_position = document.getElementById('ed_meter_position').value || 'right';
    f.meter_inside   = document.getElementById('ed_meter_inside').value === '1';
    f.meter_opacity  = Math.max(10, Math.min(100, parseInt(document.getElementById('ed_meter_opacity').value) || 70));
    f.meter_scale    = document.getElementById('ed_meter_scale').value || 'dbfs';
    dessiner();
    hotApplyWindow(primary);
}

function onEntryGeomChange() {
    const primary = primaryIdx();
    if (primary < 0) return;
    const f = editorParams.flux_config[primary];
    f.x = parseInt(document.getElementById('ed_x').value) || 0;
    f.y = parseInt(document.getElementById('ed_y').value) || 0;
    let nw = parseInt(document.getElementById('ed_w').value) || 64;
    let nh = parseInt(document.getElementById('ed_h').value) || 64;
    f.w = nw % 2 === 0 ? nw : nw - 1;
    f.h = nh % 2 === 0 ? nh : nh - 1;
    dessiner();
    hotApplyWindow(primary);
}

// ─── Copier / coller les réglages d'une fenêtre ──────────────
// Copie le format + l'apparence de la fenêtre primaire, SANS la position (x/y)
// ni la source (path/in_w/in_h/ratio/color), puis colle dans toutes les fenêtres
// sélectionnées en une fois.
const COPY_FIELDS = ['w', 'h', 'label_source', 'show_label', 'show_tally', 'tsl_index',
    'meter_channels', 'meter_position', 'meter_inside', 'meter_opacity', 'meter_scale'];
let reglagesClipboard = null;

function mwFlash(msg) {
    const t = document.getElementById('mw-toast');
    if (!t) return;
    t.textContent = msg;
    t.classList.add('visible');
    clearTimeout(mwFlash._t);
    mwFlash._t = setTimeout(() => t.classList.remove('visible'), 2200);
}

function copierReglagesFenetre() {
    const primary = primaryIdx();
    if (!editorParams || primary < 0) { mwFlash('Sélectionnez une fenêtre à copier'); return; }
    const f = editorParams.flux_config[primary];
    reglagesClipboard = {};
    COPY_FIELDS.forEach(k => { if (f[k] !== undefined) reglagesClipboard[k] = f[k]; });
    const btn = document.getElementById('ed_paste_btn');
    if (btn) btn.disabled = false;
    mwFlash('Réglages copiés (hors position) — sélectionnez les fenêtres cibles puis Coller');
}

function collerReglagesFenetre() {
    if (!editorParams || !reglagesClipboard) { mwFlash('Rien à coller — copiez d\'abord une fenêtre'); return; }
    if (selectedIdxs.length === 0) { mwFlash('Sélectionnez une ou plusieurs fenêtres cibles'); return; }
    selectedIdxs.forEach(i => {
        const f = editorParams.flux_config[i];
        if (f) Object.assign(f, reglagesClipboard);
    });
    dessiner();
    mwFlash(`Réglages collés dans ${selectedIdxs.length} fenêtre(s)`);
}

// ─── Dessin ──────────────────────────────────────────────────

function dessiner() {
    renderEntryTable();
    refreshEntryPanel();

    const canvas = document.getElementById('ed_canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;

    const _dcs = getComputedStyle(document.documentElement);
    const canvasBg   = _dcs.getPropertyValue('--canvas-bg').trim()   || '#0d1117';
    const gridColor  = _dcs.getPropertyValue('--border-soft').trim() || '#21262d';
    const mutedColor = _dcs.getPropertyValue('--text-muted').trim()  || '#8b949e';

    ctx.fillStyle = canvasBg;
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 64) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = 0; y < h; y += 64) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

    const borderW      = editorParams.border_w || 0;
    const borderColor  = editorParams.border_color || '#ffffff';
    const overlayBelow = !!editorParams.overlay_below;
    const labelSize    = Math.max(6, editorParams.label_size || 14);
    // Formules identiques à multiview.py
    const BAR_H      = Math.max(14, Math.round(labelSize * 2));
    const TALLY_SIZE = Math.max(8,  Math.round(labelSize * 1.4));
    const TALLY_PAD  = Math.max(2,  Math.round(labelSize * 0.35));

    const primary = primaryIdx();
    editorParams.flux_config.forEach((f, i) => {
        const sel = isSelected(i);
        const isPrimary = i === primary;
        const barOn = f.show_label || f.show_tally;
        const videoH = (overlayBelow && barOn) ? Math.max(2, f.h - BAR_H) : f.h;

        ctx.globalAlpha = sel ? 0.8 : 0.4;
        ctx.fillStyle   = f.color;
        ctx.fillRect(f.x, f.y, f.w, videoH);
        ctx.globalAlpha = 1;

        if (borderW > 0) {
            ctx.strokeStyle = borderColor;
            ctx.lineWidth   = borderW;
            ctx.strokeRect(f.x + borderW/2, f.y + borderW/2,
                           f.w - borderW, f.h - borderW);
        }

        ctx.strokeStyle = isPrimary ? '#ffffff' : (sel ? '#58a6ff' : f.color);
        ctx.lineWidth   = sel ? 2 : 1;
        ctx.setLineDash(isPrimary ? [6, 4] : (sel ? [3, 3] : []));
        ctx.strokeRect(f.x, f.y, f.w, f.h);
        ctx.setLineDash([]);

        if (barOn) {
            const barTop = f.y + f.h - BAR_H;
            ctx.fillStyle = 'rgba(0,0,0,0.7)';
            ctx.fillRect(f.x, barTop, f.w, BAR_H);

            let textL = f.x, textR = f.x + f.w;

            if (f.show_tally) {
                const ty = barTop + (BAR_H - TALLY_SIZE) / 2;
                ctx.fillStyle = 'rgba(128,128,128,0.7)';
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1;
                ctx.fillRect(f.x + TALLY_PAD, ty, TALLY_SIZE, TALLY_SIZE);
                ctx.strokeRect(f.x + TALLY_PAD, ty, TALLY_SIZE, TALLY_SIZE);
                ctx.fillRect(f.x + f.w - TALLY_PAD - TALLY_SIZE, ty, TALLY_SIZE, TALLY_SIZE);
                ctx.strokeRect(f.x + f.w - TALLY_PAD - TALLY_SIZE, ty, TALLY_SIZE, TALLY_SIZE);
                textL += TALLY_PAD + TALLY_SIZE + 4;
                textR -= TALLY_PAD + TALLY_SIZE + 4;
            }

            if (f.show_label) {
                ctx.fillStyle = '#ffffff';
                ctx.font = `bold ${labelSize}px monospace`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(computeDisplayName(f), (textL + textR) / 2, barTop + BAR_H / 2);
                ctx.textAlign = 'start';
                ctx.textBaseline = 'alphabetic';
            }
        }

        // Peak meter preview (statique : 6 dB de niveau pour donner l'idée)
        if (f.meter_channels && f.meter_channels > 0) {
            const N = f.meter_channels;
            const BAR_W = 5;
            const GAP   = 1;
            const TICK_W = 14;
            const meterW = TICK_W + N * BAR_W + (N - 1) * GAP;
            const meterH = videoH - 4;
            let mx;
            if (f.meter_position === 'left') {
                mx = f.x + 2;
            } else {
                mx = f.x + f.w - meterW - 2;
            }
            const my = f.y + 2;
            const alpha = f.meter_inside ? (f.meter_opacity / 100) : 0.95;
            // Fond du meter
            ctx.fillStyle = `rgba(0,0,0,${alpha})`;
            ctx.fillRect(mx, my, meterW, meterH);
            // Barres : dégradé vert→jaune→rouge, hauteur ~-12 dB ici (mock)
            for (let ch = 0; ch < N; ch++) {
                const bx = mx + TICK_W + ch * (BAR_W + GAP);
                const lvl = 0.7;  // mock : 70% de la hauteur
                const lh  = Math.round(meterH * lvl);
                // Zone verte
                ctx.fillStyle = `rgba(80,200,80,${alpha})`;
                ctx.fillRect(bx, my + meterH - lh, BAR_W, Math.min(lh, Math.round(meterH * 0.6)));
                if (lvl > 0.6) {
                    ctx.fillStyle = `rgba(230,180,40,${alpha})`;
                    const yh = Math.round(meterH * (lvl - 0.6));
                    ctx.fillRect(bx, my + meterH - Math.round(meterH * 0.6) - yh, BAR_W, yh);
                }
            }
            // Échelle text (juste un repère "0" en haut)
            ctx.fillStyle = `rgba(220,220,220,${alpha})`;
            ctx.font = '8px monospace';
            ctx.fillText(f.meter_scale === 'ppm' ? '+12' : '0', mx + 1, my + 8);
            ctx.fillText(f.meter_scale === 'ppm' ? '-12' : '-60', mx + 1, my + meterH - 1);
        }

        // Badge numéro (centre du slot)
        const badgeR = Math.min(28, Math.max(14, Math.min(f.w, f.h) / 6));
        const cx = f.x + f.w / 2;
        const cy = f.y + (videoH) / 2;
        ctx.fillStyle = isPrimary ? '#ffffff' : (sel ? '#58a6ff' : f.color);
        ctx.beginPath();
        ctx.arc(cx, cy, badgeR, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#0d1117';
        ctx.font = `bold ${Math.round(badgeR * 1.1)}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(i + 1), cx, cy);
        ctx.textAlign = 'start';
        ctx.textBaseline = 'alphabetic';

        ctx.fillStyle = mutedColor;
        ctx.font = '11px monospace';
        ctx.fillText(`${f.w}×${f.h}`, f.x + 4, f.y + 14);

        if (isPrimary) {
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(f.x + f.w - HANDLE_SIZE, f.y + f.h - HANDLE_SIZE,
                         HANDLE_SIZE, HANDLE_SIZE);
        }
    });

    // Lignes guides de snap (pendant le drag)
    if (dragMode && snapGuides.length) {
        ctx.strokeStyle = '#f97316';
        ctx.lineWidth   = 1;
        ctx.setLineDash([4, 4]);
        snapGuides.forEach(g => {
            ctx.beginPath();
            if (g.type === 'v') { ctx.moveTo(g.pos, 0); ctx.lineTo(g.pos, h); }
            else                { ctx.moveTo(0, g.pos); ctx.lineTo(w, g.pos); }
            ctx.stroke();
        });
        ctx.setLineDash([]);
    }

    ctx.fillStyle = mutedColor;
    ctx.font = '11px monospace';
    ctx.fillText(`${w} × ${h}`, 8, h - 8);
}

function renderEntryTable() {
    const tbody = document.getElementById('ed_tbody');
    if (!tbody) return;
    const primary = primaryIdx();
    tbody.innerHTML = editorParams.flux_config.map((f, i) => {
        const sel = isSelected(i);
        const isP = i === primary;
        const cls = isP ? 'is-primary' : (sel ? 'is-selected' : '');
        return `
        <tr class="mw-entry-row ${cls}" onclick="selectEntry(${i}, event)">
            <td><span class="mw-entry-color" style="background:${f.color}"></span>${i + 1}</td>
            <td>${sourceHostname(f) || '—'}</td>
            <td class="mw-entry-path">${f.path}</td>
            <td>${f.x}, ${f.y}</td>
            <td>${f.w}×${f.h}</td>
        </tr>`;
    }).join('');
}

function selectEntry(i, ev) {
    toggleSelection(i, ev && ev.shiftKey);
    dessiner();
}

// ─── Souris (canvas) ─────────────────────────────────────────

function getCanvasPos(e) {
    const canvas = document.getElementById('ed_canvas');
    const rect = canvas.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left) * (canvas.width  / rect.width),
        y: (e.clientY - rect.top)  * (canvas.height / rect.height)
    };
}

function canvasMouseDown(e) {
    const pos = getCanvasPos(e);
    const primary = primaryIdx();
    for (let i = editorParams.flux_config.length - 1; i >= 0; i--) {
        const f = editorParams.flux_config[i];
        if (i === primary &&
            pos.x >= f.x + f.w - HANDLE_SIZE && pos.y >= f.y + f.h - HANDLE_SIZE) {
            dragMode = 'resize'; dragStart = pos; dragOrigRect = {...f};
            return;
        }
        if (pos.x >= f.x && pos.x <= f.x + f.w &&
            pos.y >= f.y && pos.y <= f.y + f.h) {
            toggleSelection(i, e.shiftKey);
            dragMode = 'move'; dragStart = pos; dragOrigRect = {...editorParams.flux_config[primaryIdx()]};
            dessiner();
            return;
        }
    }
    if (!e.shiftKey) selectedIdxs = [];
    dragMode = null;
    dessiner();
}

function toggleSelection(i, additive) {
    if (additive) {
        const at = selectedIdxs.indexOf(i);
        if (at >= 0) selectedIdxs.splice(at, 1);
        else selectedIdxs.push(i);
    } else if (!selectedIdxs.includes(i)) {
        selectedIdxs = [i];
    } else {
        // Réordonne pour que i devienne le primary
        selectedIdxs = selectedIdxs.filter(x => x !== i);
        selectedIdxs.push(i);
    }
}

function canvasMouseMove(e) {
    const primary = primaryIdx();
    if (!dragMode || primary < 0) return;
    const pos = getCanvasPos(e);
    const dx = pos.x - dragStart.x;
    const dy = pos.y - dragStart.y;
    const f = editorParams.flux_config[primary];
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;

    if (dragMode === 'move') {
        let nx = Math.max(0, Math.min(out_w - f.w, Math.round(dragOrigRect.x + dx)));
        let ny = Math.max(0, Math.min(out_h - f.h, Math.round(dragOrigRect.y + dy)));
        if (snapEnabled) {
            const snapped = computeSnap(primary, nx, ny, f.w, f.h);
            nx = snapped.x; ny = snapped.y; snapGuides = snapped.guides;
        } else snapGuides = [];
        f.x = nx; f.y = ny;
    } else if (dragMode === 'resize') {
        const ratio = dragOrigRect.w / dragOrigRect.h;
        let nw = Math.max(64, Math.min(out_w - f.x, Math.round(dragOrigRect.w + dx)));
        let nh = Math.round(nw / ratio);
        if (nh > out_h - f.y) { nh = out_h - f.y; nw = Math.round(nh * ratio); }
        if (snapEnabled) {
            const snapped = computeSnapResize(primary, f.x, f.y, nw, nh, ratio);
            nw = snapped.w; nh = snapped.h; snapGuides = snapped.guides;
        } else snapGuides = [];
        f.w = nw % 2 === 0 ? nw : nw - 1;
        f.h = nh % 2 === 0 ? nh : nh - 1;
    }
    dessiner();
}

function canvasMouseUp() {
    dragMode = null; snapGuides = []; dessiner();
    selectedIdxs.forEach(idx => hotApplyWindow(idx));
}

function hotApplyWindow(idx) {
    if (editorVmid === null || !editorParams) return;
    const f = editorParams.flux_config[idx];
    if (!f) return;
    fetch(`/api/containers/${editorVmid}/plugin/window`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            idx,
            x: f.x, y: f.y,
            w: f.w % 2 === 0 ? f.w : f.w - 1,
            h: f.h % 2 === 0 ? f.h : f.h - 1,
            name:           computeDisplayName(f),
            show_label:     !!f.show_label,
            show_tally:     !!f.show_tally,
            tsl_index:      f.tsl_index ?? 0,
            meter_channels: f.meter_channels ?? 0,
            meter_position: f.meter_position || 'right',
            meter_inside:   !!f.meter_inside,
            meter_opacity:  f.meter_opacity ?? 70,
            meter_scale:    f.meter_scale || 'dbfs',
        })
    }).catch(() => {});
}

function hotApplyFull() {
    // Passe par le endpoint deploy pour que _multiview_hot_apply résolve in_w/in_h via _shm_dims.
    deployerEditor();
}

function hotApplyStyle() {
    if (editorVmid === null || !editorParams) return;
    fetch(`/api/containers/${editorVmid}/plugin/style`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            border_w:      editorParams.border_w,
            border_color:  editorParams.border_color,
            overlay_below: editorParams.overlay_below,
            label_size:    editorParams.label_size,
            frame_style:   editorParams.frame_style || 'none',
        })
    }).catch(() => {});
}

// ─── Snap ──────────────────────────────────────────────────────

function computeSnap(idx, x, y, w, h) {
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;
    const xTargets = [0, out_w, out_w / 2];     // bords + centre canvas
    const yTargets = [0, out_h, out_h / 2];
    editorParams.flux_config.forEach((o, i) => {
        if (i === idx) return;
        xTargets.push(o.x, o.x + o.w, o.x + o.w / 2);
        yTargets.push(o.y, o.y + o.h, o.y + o.h / 2);
    });
    // Pour X : on essaie left=x, right=x+w, center=x+w/2 contre chaque target
    let bestDx = SNAP_PX + 1, snapX = x;
    const guides = [];
    [{edge: 'l', val: x}, {edge: 'r', val: x + w}, {edge: 'c', val: x + w / 2}]
        .forEach(({edge, val}) => {
            xTargets.forEach(t => {
                const d = Math.abs(val - t);
                if (d <= SNAP_PX && d < bestDx) {
                    bestDx = d;
                    snapX = edge === 'l' ? t : edge === 'r' ? t - w : t - w / 2;
                }
            });
        });
    if (bestDx <= SNAP_PX) {
        // Recalcule la position effective de la guide (ligne verticale)
        const lineX = snapX + (snapX === x ? 0 : 0); // place sur l'edge snappée
        // On dessine simplement le target le plus proche
        let closest = xTargets[0], cd = Math.abs(xTargets[0] - snapX);
        xTargets.forEach(t => {
            [snapX, snapX + w, snapX + w/2].forEach(v => {
                const d = Math.abs(t - v);
                if (d < cd) { cd = d; closest = t; }
            });
        });
        guides.push({type: 'v', pos: closest});
    }
    let bestDy = SNAP_PX + 1, snapY = y;
    [{edge: 't', val: y}, {edge: 'b', val: y + h}, {edge: 'c', val: y + h / 2}]
        .forEach(({edge, val}) => {
            yTargets.forEach(t => {
                const d = Math.abs(val - t);
                if (d <= SNAP_PX && d < bestDy) {
                    bestDy = d;
                    snapY = edge === 't' ? t : edge === 'b' ? t - h : t - h / 2;
                }
            });
        });
    if (bestDy <= SNAP_PX) {
        let closest = yTargets[0], cd = Math.abs(yTargets[0] - snapY);
        yTargets.forEach(t => {
            [snapY, snapY + h, snapY + h/2].forEach(v => {
                const d = Math.abs(t - v);
                if (d < cd) { cd = d; closest = t; }
            });
        });
        guides.push({type: 'h', pos: closest});
    }
    return {x: Math.round(snapX), y: Math.round(snapY), guides};
}

function computeSnapResize(idx, x, y, w, h, ratio) {
    // Snap uniquement le coin bas-droit (resize est ancré haut-gauche)
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;
    const xTargets = [out_w, out_w / 2];
    const yTargets = [out_h, out_h / 2];
    editorParams.flux_config.forEach((o, i) => {
        if (i === idx) return;
        xTargets.push(o.x, o.x + o.w);
        yTargets.push(o.y, o.y + o.h);
    });
    const guides = [];
    let bestDx = SNAP_PX + 1, snapW = w;
    xTargets.forEach(t => {
        const d = Math.abs((x + w) - t);
        if (d <= SNAP_PX && d < bestDx) { bestDx = d; snapW = t - x; }
    });
    if (bestDx <= SNAP_PX) guides.push({type: 'v', pos: x + snapW});
    // h suit le ratio pour préserver l'aspect — mais on peut aussi snapper indépendamment
    let snapH = Math.round(snapW / ratio);
    let bestDy = SNAP_PX + 1;
    yTargets.forEach(t => {
        const d = Math.abs((y + snapH) - t);
        if (d <= SNAP_PX && d < bestDy) { bestDy = d; snapH = t - y; }
    });
    if (bestDy <= SNAP_PX) guides.push({type: 'h', pos: y + snapH});
    return {w: Math.max(64, snapW), h: Math.max(64, snapH), guides};
}

// ─── Déployer la composition ─────────────────────────────────

async function deployerEditor() {
    if (editorVmid === null) return;

    // Lit les champs globaux au cas où ils n'auraient pas déclenché onchange
    editorParams.max_inputs    = parseInt(document.getElementById('ed_max').value) || editorParams.max_inputs;
    editorParams.border_w      = parseInt(document.getElementById('ed_border_w').value) || 0;
    editorParams.border_color  = document.getElementById('ed_border_color').value;
    editorParams.overlay_below = document.getElementById('ed_overlay_below').checked;
    editorParams.label_size    = Math.max(6, parseInt(document.getElementById('ed_label_size').value) || 14);
    { const fs = document.getElementById('ed_frame_style'); if (fs) editorParams.frame_style = fs.value; }
    editorParams.genlock = document.getElementById('ed_genlock').checked;
    const tslPortEl = document.getElementById('ed_tsl_port');
    if (tslPortEl) editorParams.tsl_port = parseInt(tslPortEl.value) || 0;

    const flux_config = editorParams.flux_config.map(f => ({
        path: f.path,
        label_source: f.label_source || 'hostname',
        name: computeDisplayName(f),
        in_w: f.in_w,
        in_h: f.in_h,
        x: f.x, y: f.y,
        w: f.w % 2 === 0 ? f.w : f.w - 1,
        h: f.h % 2 === 0 ? f.h : f.h - 1,
        show_label: !!f.show_label,
        show_tally: !!f.show_tally,
        tsl_index: parseInt(f.tsl_index) || 0,
        // Peak meters audio
        meter_channels: parseInt(f.meter_channels) || 0,
        meter_position: f.meter_position || 'right',
        meter_inside:   !!f.meter_inside,
        meter_opacity:  Math.max(10, Math.min(100, parseInt(f.meter_opacity) || 70)),
        meter_scale:    f.meter_scale || 'dbfs',
    }));

    const params = {
        flux_config,
        shm_out:       editorParams.shm_out,
        out_width:     editorParams.out_width,
        out_height:    editorParams.out_height,
        border_w:      editorParams.border_w,
        border_color:  editorParams.border_color,
        overlay_below: editorParams.overlay_below,
        label_size:    editorParams.label_size,
        frame_style:   editorParams.frame_style || 'none',
        max_inputs:    editorParams.max_inputs,
        genlock:       editorParams.genlock,
        tsl_port:      editorParams.tsl_port ?? 4801
    };

    const r = await fetch('/api/containers/' + editorVmid + '/deploy', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({type: 'multiview', params, path: '/opt/script/main.py'})
    });
    if (r.ok) {
        const btns = document.querySelectorAll('#mw-editor .btn-orange');
        btns.forEach(b => { b.textContent = 'Déployé'; setTimeout(() => b.textContent = 'Déployer', 1500); });
        // Refresh la sidebar générique (badge version + mini-aperçu) après déploiement
        if (window.tpLoadInstances) setTimeout(window.tpLoadInstances, 800);
    } else {
        mwFlash('Erreur déploiement');
    }
}

// ─── Outils d'alignement ─────────────────────────────────────

function aligner(mode) {
    if (!editorParams || selectedIdxs.length === 0) return;
    const fc = editorParams.flux_config;
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;

    // Référence : primary si ≥2 sélectionnées, sinon le canvas
    let ref;
    if (selectedIdxs.length >= 2) {
        const p = fc[primaryIdx()];
        ref = {x: p.x, y: p.y, w: p.w, h: p.h};
    } else {
        ref = {x: 0, y: 0, w: out_w, h: out_h};
    }
    selectedIdxs.forEach(i => {
        const f = fc[i];
        if (selectedIdxs.length >= 2 && i === primaryIdx()) return;
        switch (mode) {
            case 'left':    f.x = ref.x; break;
            case 'right':   f.x = ref.x + ref.w - f.w; break;
            case 'hcenter': f.x = Math.round(ref.x + (ref.w - f.w) / 2); break;
            case 'top':     f.y = ref.y; break;
            case 'bottom':  f.y = ref.y + ref.h - f.h; break;
            case 'vcenter': f.y = Math.round(ref.y + (ref.h - f.h) / 2); break;
        }
        f.x = Math.max(0, Math.min(out_w - f.w, f.x));
        f.y = Math.max(0, Math.min(out_h - f.h, f.y));
    });
    dessiner();
}

function matchSize(mode) {
    if (!editorParams || selectedIdxs.length < 2) {
        mwFlash('Sélectionne au moins 2 fenêtres (la dernière sert de référence).');
        return;
    }
    const fc = editorParams.flux_config;
    const ref = fc[primaryIdx()];
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;
    selectedIdxs.forEach(i => {
        if (i === primaryIdx()) return;
        const f = fc[i];
        if (mode === 'w' || mode === 'both') {
            f.w = ref.w % 2 === 0 ? ref.w : ref.w - 1;
            if (f.x + f.w > out_w) f.x = out_w - f.w;
        }
        if (mode === 'h' || mode === 'both') {
            f.h = ref.h % 2 === 0 ? ref.h : ref.h - 1;
            if (f.y + f.h > out_h) f.y = out_h - f.h;
        }
    });
    dessiner();
}

function distribuer(axis) {
    if (!editorParams || selectedIdxs.length < 3) {
        mwFlash('Sélectionne au moins 3 fenêtres pour distribuer.');
        return;
    }
    const fc = editorParams.flux_config;
    const items = selectedIdxs.map(i => ({i, f: fc[i]}));
    if (axis === 'h') {
        items.sort((a, b) => a.f.x - b.f.x);
        const x0 = items[0].f.x;
        const xN = items[items.length - 1].f.x;
        const step = (xN - x0) / (items.length - 1);
        items.forEach((it, k) => { it.f.x = Math.round(x0 + step * k); });
    } else {
        items.sort((a, b) => a.f.y - b.f.y);
        const y0 = items[0].f.y;
        const yN = items[items.length - 1].f.y;
        const step = (yN - y0) / (items.length - 1);
        items.forEach((it, k) => { it.f.y = Math.round(y0 + step * k); });
    }
    dessiner();
}

// ─── Layouts (presets) ───────────────────────────────────────

let savedLayouts = [];

async function rafraichirListeLayouts() {
    try {
        const r = await fetch('/api/layouts');
        savedLayouts = await r.json();
    } catch(e) { savedLayouts = []; }
    const ul = document.getElementById('layout-saved-list');
    if (!ul) return;
    if (savedLayouts.length === 0) {
        ul.innerHTML = '<li class="meta">Aucun layout enregistré.</li>';
        return;
    }
    ul.innerHTML = savedLayouts.map(l => `
        <li>
            <div><b>${escapeHtml(l.name)}</b></div>
            <div class="meta">${l.created_at || ''} — ${(l.config.flux_config || []).length} entrées</div>
            <canvas data-layout-preview="${l.id}"></canvas>
            <div class="actions">
                <button class="btn btn-blue" onclick="appliquerLayout(${l.id})">Appliquer</button>
                <button class="btn btn-red"  onclick="supprimerLayout(${l.id}, '${escapeAttr(l.name)}')">Suppr</button>
            </div>
        </li>`).join('');
    savedLayouts.forEach(l => {
        const cv = ul.querySelector(`canvas[data-layout-preview="${l.id}"]`);
        drawMiniPreview(cv, l.config);
    });
}

function escapeHtml(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({
        '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
    }[c]));
}
function escapeAttr(s) {
    return String(s || '').replace(/'/g, "\\'");
}

async function enregistrerLayout() {
    const nameEl = document.getElementById('layout-save-name');
    const name = (nameEl.value || '').trim();
    if (!name) { mwFlash('Donne un nom au layout.'); return; }
    if (!editorParams) {
        mwFlash('Sélectionne d\'abord un multiview à éditer.');
        return;
    }
    // Sérialise la config courante (sans champs internes color/ratio)
    const config = {
        out_width:     editorParams.out_width,
        out_height:    editorParams.out_height,
        border_w:      editorParams.border_w,
        border_color:  editorParams.border_color,
        overlay_below: editorParams.overlay_below,
        label_size:    editorParams.label_size,
        frame_style:   editorParams.frame_style || 'none',
        max_inputs:    editorParams.max_inputs,
        // Path et in_w/in_h volontairement omis : un layout = réglages géométriques + style,
        // pas l'affectation de source. Les sources sont restaurées à l'apply depuis l'éditeur.
        flux_config:   (editorParams.flux_config || []).map(f => ({
            label_source: f.label_source || 'hostname',
            x: f.x, y: f.y, w: f.w, h: f.h,
            show_label: !!f.show_label,
            show_tally: !!f.show_tally
        }))
    };
    const r = await fetch('/api/layouts', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, config})
    });
    if (r.ok) {
        nameEl.value = '';
        rafraichirListeLayouts();
    } else {
        mwFlash('Erreur enregistrement layout');
    }
}

function appliquerLayout(lid) {
    if (!editorParams) {
        mwFlash('Sélectionne d\'abord un multiview à éditer.');
        return;
    }
    const l = savedLayouts.find(x => x.id === lid);
    if (!l) return;
    const cfg = l.config || {};
    // shm_out conservé tel quel (lié au container, pas au layout)
    editorParams.out_width     = cfg.out_width  || editorParams.out_width;
    editorParams.out_height    = cfg.out_height || editorParams.out_height;
    editorParams.border_w      = cfg.border_w || 0;
    editorParams.border_color  = cfg.border_color || '#ffffff';
    editorParams.overlay_below = !!cfg.overlay_below;
    editorParams.label_size    = cfg.label_size || editorParams.label_size || 14;
    editorParams.frame_style   = cfg.frame_style || editorParams.frame_style || 'none';
    editorParams.max_inputs    = cfg.max_inputs || editorParams.max_inputs;
    // Préserve les sources affectées dans l'éditeur (par index) — le layout n'apporte
    // que les réglages géométriques et de style.
    const existing = editorParams.flux_config || [];
    editorParams.flux_config = (cfg.flux_config || []).map((f, i) => {
        const prev = existing[i] || {};
        return Object.assign({
            color: COLORS[i % COLORS.length],
            path: prev.path || '',
            in_w: 0,
            in_h: 0,
            ratio: 16/9
        }, f);
    });
    selectedIdxs = [];
    const hostnameEl = document.getElementById('ed_hostname');
    const hostname = hostnameEl ? hostnameEl.textContent.trim() : '';
    renderEditor(hostname);
    deployerEditor();
}

async function supprimerLayout(lid, name) {
    if (!confirm(`Supprimer le layout "${name}" ?`)) return;
    const r = await fetch('/api/layouts/' + lid, {method: 'DELETE'});
    if (r.ok) rafraichirListeLayouts();
}

// Le montage est piloté par le shell générique Traitements (tpLoadInstances → tpMount →
// MXLPlugins.multiview.mount(el, vmid)). Pas d'init autonome ici.
