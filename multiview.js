// ─── i18n ─────────────────────────────────────────────────────
// Catalogue plugin.multiview.* (plugins/multiview/i18n/<lang>.json, fusionné par
// app/i18n.py ; préfixe `plugin.` exposé côté JS par js_catalog → window.t()).
const T = (k) => (window.t ? window.t(k) : k);

// Traduit le HTML statique du composer (injecté par le shell Traitements) :
// data-i18n (textContent), data-i18n-title, data-i18n-aria, data-i18n-ph.
function mwApplyI18n(root) {
    const r = root || document;
    r.querySelectorAll('[data-i18n]').forEach(n => { n.textContent = T(n.dataset.i18n); });
    r.querySelectorAll('[data-i18n-title]').forEach(n => { n.title = T(n.dataset.i18nTitle); });
    r.querySelectorAll('[data-i18n-aria]').forEach(n => { n.setAttribute('aria-label', T(n.dataset.i18nAria)); });
    r.querySelectorAll('[data-i18n-ph]').forEach(n => { n.placeholder = T(n.dataset.i18nPh); });
}

// ─── État global ──────────────────────────────────────────────

let allContainers = [];
let videoSources = [];   // sorties vidéo individuelles de la flotte (cf. loadVideoSources)
let editorVmid    = null;
let editorParams  = null;   // {flux_config, shm_out, out_width, out_height, border_w, border_color, overlay_below, max_inputs}
let selectedIdxs  = [];     // multi-select : le dernier est le primary (référence pour align/match-size)
let dragMode      = null;   // 'move' | 'resize'
let dragStart     = null;
let dragOrigRect  = null;
let dragGroupOrig = [];     // [{j, x, y}] positions d'origine de TOUTES les fenêtres sélectionnées
                            // (déplacement de groupe : même delta appliqué à chacune)
let snapEnabled   = true;
let snapGuides    = [];     // [{type:'v'|'h', pos:number}] dessinées pendant le drag

// Overlays (texte/horloge/image) : objets visuels séparés de flux_config (non câblés).
let selectedOverlay = -1;   // index dans editorParams.overlays, ou -1
let dragOverlay     = false; // true pendant un drag/resize d'overlay (réutilise dragMode/dragStart)
const _ovThumbCache = {};   // clé "slug|path" → Image (vignette média pour l'aperçu canvas)

const OVERLAY_FONTS = [
    ['dejavu-sans-bold',     'DejaVu Sans Bold'],
    ['dejavu-sans',          'DejaVu Sans'],
    ['dejavu-serif',         'DejaVu Serif'],
    ['dejavu-mono',          'DejaVu Mono'],
    ['liberation-sans',      'Liberation Sans'],
    ['liberation-sans-bold', 'Liberation Sans Bold'],
    ['liberation-mono',      'Liberation Mono'],
    ['inter',                'Inter'],
    ['roboto',               'Roboto'],
    ['firacode',             'Fira Code'],
];

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
    const ow = params.out_width  || 1920;
    const oh = params.out_height || 1080;
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

    (params.flux_config || []).filter(f => !f.hidden).forEach((f, i) => {
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

// ─── Noms de colonnes labels + connexions TSL (= niveaux de Tally) ───────────
let _tslLabelNames = ["Hostname", "MXL", "Label 2", "Label 3", "Label 4",
                      "Label 5", "Label 6", "Label 7", "Label 8", "Label 9"];
let _tslConns = [];   // [{id, name, …}] connexions = niveaux de Tally (choix par cellule)
let _labelRows = [];  // [{shm, name}] lignes du tableau /labels (overlay texte central)

async function _loadTslLabelNames() {
    try {
        const [rl, rc, rs, rsl] = await Promise.all([
            fetch('/api/tsl/label_names'),
            fetch('/api/tsl/connections'),
            fetch('/api/sources'),
            fetch('/api/source_labels'),
        ]);
        if (rl.ok) _tslLabelNames = await rl.json();
        if (rc.ok) _tslConns = await rc.json();
        // Lignes = sources réelles (tous kinds) + lignes manuelles/texte (source_labels).
        const rows = [], seen = new Set();
        if (rs.ok) for (const s of (await rs.json())) {
            const shm = s.shm || ''; if (!shm || seen.has(shm)) continue; seen.add(shm);
            rows.push({ shm, name: s.hostname || shm });
        }
        if (rsl.ok) for (const sl of (await rsl.json())) {
            const shm = sl.shm || ''; if (!shm || seen.has(shm)) continue; seen.add(shm);
            rows.push({ shm, name: shm.startsWith('__umd:') ? shm.slice(6) + ' (texte)' : shm });
        }
        _labelRows = rows;
    } catch(e) {}
    _tslPopulateSelects();
}

function _tslPopulateSelects() {
    // Colonne label = noms de colonnes ; niveau de Tally = connexions (par nom)
    const lc = document.getElementById('ed_label_col');
    if (lc) {
        const cur = lc.value;
        lc.innerHTML = _tslLabelNames.map((n, i) => `<option value="${i}">${i} — ${n}</option>`).join('');
        lc.value = cur;
    }
    const tl = document.getElementById('ed_tally_level');
    if (tl) {
        const cur = tl.value;
        // Niveaux de Tally = connexions actives, regroupées par bande tally_base → numéro 1-4.
        // La connexion qui sert le niveau (et son Rouge/Vert) est définie dans le service.
        const seen = new Set();
        let opts = '<option value="0">— Aucun —</option>';
        for (const c of (_tslConns || [])) {
            if (!c.enabled) continue;
            const n = Math.floor((c.tally_base || 0) / 3) + 1;
            if (seen.has(n)) continue;
            seen.add(n);
            opts += `<option value="${n}">Niveau ${n} — ${escapeHtmlMv(c.name)}</option>`;
        }
        tl.innerHTML = opts;
        tl.value = cur;
    }
    _tslPopulateOverlaySelects();
}

// Selects de l'éditeur d'overlay texte en mode central : ligne (toutes les lignes du tableau),
// colonne label, niveau de Tally. Mêmes données que les cellules PiP, cibles différentes.
function _tslPopulateOverlaySelects() {
    const lr = document.getElementById('ov_label_row');
    if (lr) {
        const cur = lr.value;
        lr.innerHTML = '<option value="">— Aucune —</option>' +
            _labelRows.map(r => `<option value="${escapeHtmlMv(r.shm)}">${escapeHtmlMv(r.name)}</option>`).join('');
        lr.value = cur;
    }
    const lc = document.getElementById('ov_label_col');
    if (lc) {
        const cur = lc.value;
        lc.innerHTML = _tslLabelNames.map((n, i) => `<option value="${i}">${i} — ${escapeHtmlMv(n)}</option>`).join('');
        lc.value = cur;
    }
    const tl = document.getElementById('ov_tally_level');
    if (tl) {
        const cur = tl.value;
        const seen = new Set();
        let opts = '<option value="0">— Aucun —</option>';
        for (const c of (_tslConns || [])) {
            if (!c.enabled) continue;
            const n = Math.floor((c.tally_base || 0) / 3) + 1;
            if (seen.has(n)) continue;
            seen.add(n);
            opts += `<option value="${n}">Niveau ${n} — ${escapeHtmlMv(c.name)}</option>`;
        }
        tl.innerHTML = opts;
        tl.value = cur;
    }
}

function escapeHtmlMv(s) {
    return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Renseigne les sélecteurs cellule : colonne label, niveau de Tally, couleurs.
function _tslSetSelects(labelCol, tallyLevel, tallyColors) {
    _tslPopulateSelects();
    const lc = document.getElementById('ed_label_col');
    const tl = document.getElementById('ed_tally_level');
    const tk = document.getElementById('ed_tally_colors');
    if (lc) lc.value = String(labelCol ?? 0);
    if (tl) tl.value = String(tallyLevel ?? 0);
    if (tk) tk.value = tallyColors || 'none';
}

async function chargerMw(vmid) {
    let c;
    try {
        await Promise.all([loadAllContainers(), loadVideoSources(), _loadTslLabelNames()]);
        const r = await fetch('/api/containers/' + vmid + '/config');
        if (!r.ok) throw new Error('HTTP ' + r.status);
        c = await r.json();
    } catch(e) {
        mwFlash(T('plugin.multiview.flash_load_failed'));
        return;
    }
    let dc = null;
    try { dc = c.deploy_config ? JSON.parse(c.deploy_config) : null; } catch(e) {}
    if (!dc || dc.type !== 'multiview') {
        mwFlash(T('plugin.multiview.flash_not_multiview'));
        return;
    }
    editorVmid   = vmid;
    editorParams = Object.assign({
        flux_config: [],
        shm_out: 'mxl_mix',
        border_w: 0,
        border_color: '#ffffff',
        overlay_below: false,
        label_size: 14,
        frame_style: 'none',
        show_no_signal: true,
        freeze_detect_s: 2,
        show_format: false,
        show_proxy: false,
        max_inputs: 4,
        genlock: true,
        tsl_mode: 'central',
        tsl_port: 4801,
        overlays: []
    }, dc.params || {});
    // Format de sortie : pas de littéral en dur → vient du réglage système (Formats vidéo) si la
    // config persistée ne le porte pas. L'explicite (dc.params, semé à la création) prime.
    // DÉFENSIF : le chargement des formats ne doit JAMAIS empêcher de sélectionner un multiview.
    // `scripts.js:loadVideoFormats` n'est PAS chargé sur la page Traitements (uniquement Containers/
    // Projects) → on a un chargeur AUTONOME pour ne pas dépendre de ce global (sinon dropdown vide
    // + repli 720p sur la page Traitements).
    try {
        if (typeof loadVideoFormats === 'function') await loadVideoFormats();
        else await _mvEnsureVideoFormats();
    } catch (e) { /* palette indisponible → on retombe sur le repli ci-dessous */ }
    const _sysf = systemDefaultFormat();
    editorParams.out_width  = editorParams.out_width  || _sysf.w;
    editorParams.out_height = editorParams.out_height || _sysf.h;
    editorParams.fps        = editorParams.fps        || _sysf.fps;
    editorParams.scan       = editorParams.scan       || _sysf.scan;
    // Couleurs locales pour le rendu (non sauvegardé)
    editorParams.flux_config = (editorParams.flux_config || []).map((f, i) => Object.assign({
        color: COLORS[i % COLORS.length],
        ratio: (f.in_w && f.in_h) ? f.in_w / f.in_h : 16/9
    }, f));
    editorParams.overlays = (editorParams.overlays || []);
    padBank();   // banque à indices stables : toujours max_inputs entrées
    selectedIdxs = [];
    selectedOverlay = -1;
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
    document.getElementById('ed_show_no_signal').checked = p.show_no_signal !== false;
    document.getElementById('ed_freeze_detect').value    = p.freeze_detect_s ?? 2;
    document.getElementById('ed_show_format').checked    = !!p.show_format;
    document.getElementById('ed_show_proxy').checked     = !!p.show_proxy;
    document.getElementById('ed_genlock').checked       = p.genlock !== false;
    { const _ft = document.getElementById('ed_fps_target'); if (_ft) _ft.value = String(parseInt(p.fps_target) || 0); }
    document.getElementById('ed_tsl_port').value        = p.tsl_port ?? 0;
    { const _tm = document.getElementById('ed_tsl_mode'); if (_tm) _tm.value = _resolveTslMode(p); }
    updateTslModeUI();
    document.getElementById('ed_snap').checked          = snapEnabled;
    try { populateOutFormatSelect(); } catch (e) { /* sélecteur de format non bloquant */ }
    try { populateOrientationSelect(); } catch (e) { /* sélecteur d'orientation non bloquant */ }
    // Réglages de sortie (colonne latérale) : visibles dès qu'une instance est chargée
    const settings = document.getElementById('mw-settings');
    if (settings) settings.hidden = false;

    resizeCanvas();
    dessiner();
    mwEnhanceSteppers(document.querySelector('.mw-compose'));   // boutons −/+ confortables (idempotent)
}

// ─── Steppers numériques : boutons −/+ confortables (remplacent les flèches natives) ─────────
// Les spinners natifs de <input type=number> sont minuscules → on enrobe chaque champ d'un
// stepper avec deux gros boutons (clic + appui maintenu pour répéter). Idempotent : on ignore un
// champ déjà enrobé. Conserve l'id et le onchange du champ (getElementById/handlers intacts).
function mwStep(input, dir) {
    const step = parseFloat(input.step) || 1;
    const min  = input.min !== '' ? parseFloat(input.min) : -Infinity;
    const max  = input.max !== '' ? parseFloat(input.max) :  Infinity;
    let v = parseFloat(input.value); if (isNaN(v)) v = 0;
    v = Math.min(max, Math.max(min, v + dir * step));
    v = parseFloat(v.toFixed(6));   // évite les imprécisions flottantes (step 0.5…)
    input.value = v;
    input.dispatchEvent(new Event('change', { bubbles: true }));
}
function _mwBindHold(btn, fn) {
    let to = null, iv = null;
    const stop = () => { clearTimeout(to); clearInterval(iv); to = iv = null; };
    btn.addEventListener('pointerdown', e => {
        e.preventDefault(); fn();
        to = setTimeout(() => { iv = setInterval(fn, 60); }, 350);
    });
    ['pointerup', 'pointerleave', 'pointercancel'].forEach(ev => btn.addEventListener(ev, stop));
}
function mwEnhanceSteppers(root) {
    (root || document).querySelectorAll('input[type="number"]').forEach(inp => {
        if (inp.closest('.num-stepper')) return;   // déjà enrobé
        inp.classList.add('mw-num');
        const wrap = document.createElement('div');
        wrap.className = 'num-stepper';
        inp.parentNode.insertBefore(wrap, inp);
        const mk = (cls, txt) => {
            const b = document.createElement('button');
            b.type = 'button'; b.className = 'num-btn ' + cls; b.tabIndex = -1;
            b.textContent = txt; b.setAttribute('aria-hidden', 'true');
            return b;
        };
        const dec = mk('num-dec', '−'), inc = mk('num-inc', '+');
        wrap.append(dec, inp, inc);
        _mwBindHold(dec, () => mwStep(inp, -1));
        _mwBindHold(inc, () => mwStep(inp, +1));
    });
}

// ─── Format de sortie (depuis le réglage système, jamais en dur) ─────────────

// Format par défaut SYSTÈME (palette Réglages → Formats vidéo). Repli ultime = celui du serveur
// (get_default_video_format : 1280×720) UNIQUEMENT si la palette est vide (DB vierge).
// Chargeur AUTONOME des formats vidéo (la page Traitements ne charge pas scripts.js). Peuple les
// mêmes globals (window._videoFormats / window._videoFormatDefault) que scripts.js:loadVideoFormats.
async function _mvEnsureVideoFormats() {
    if (Array.isArray(window._videoFormats) && window._videoFormats.length) return window._videoFormats;
    try {
        const r = await fetch('/api/settings');
        if (!r.ok) return [];
        const s = await r.json();
        window._videoFormatDefault = s.video_format_default || '';
        window._videoFormats = (s.video_formats || '').split('\n').map(l => l.trim()).filter(Boolean).map(l => {
            const p = l.split(';').map(x => x.trim());
            return { label: p[0] || '', w: parseInt(p[1]) || 0, h: parseInt(p[2]) || 0,
                     fps: parseFloat(p[3]) || 25, scan: (p[4] || 'p').toLowerCase() === 'i' ? 'i' : 'p',
                     chroma: ['420','422','444'].includes(p[5]) ? p[5] : '422',
                     bit_depth: [8,10,12].includes(parseInt(p[6])) ? parseInt(p[6]) : 10,
                     colorimetry: (p[7] || '709').toLowerCase() };
        }).filter(f => f.label && f.w && f.h);
    } catch (e) { window._videoFormats = []; }
    return window._videoFormats;
}

function systemDefaultFormat() {
    const list = window._videoFormats || [];
    const def  = window._videoFormatDefault || '';
    const f = list.find(x => x.label === def) || list[0];
    if (f) return { w: f.w, h: f.h, fps: f.fps, scan: f.scan || 'p' };
    return { w: 1920, h: 1080, fps: 50, scan: 'p' };   // repli broadcast (jamais 720p) si aucun format configuré
}

// Peuple le sélecteur de format de sortie depuis la palette système, sélectionne le format courant
// (ou ajoute une option « (actuel) » si la résolution courante n'est pas dans la palette).
function populateOutFormatSelect() {
    const sel = document.getElementById('ed_out_format');
    if (!sel) return;
    const list = window._videoFormats || [];
    const cur  = editorParams;
    let html = '', matched = false;
    list.forEach(f => {
        const isCur = f.w === cur.out_width && f.h === cur.out_height
            && Math.abs((f.fps || 0) - (cur.fps || 0)) < 0.01 && (f.scan || 'p') === (cur.scan || 'p');
        if (isCur) matched = true;
        html += `<option value="${escapeHtml(f.label)}"${isCur ? ' selected' : ''}>${escapeHtml(f.label)}</option>`;
    });
    if (!matched) {
        const lbl = `${cur.out_width}×${cur.out_height}${cur.scan || 'p'}${cur.fps}`;
        html = `<option value="__current__" selected>${escapeHtml(lbl)}</option>` + html;
    }
    sel.innerHTML = html;
}

function _mwIsPortrait(o) { return o === 'portrait_cw' || o === 'portrait_ccw'; }

// Reflète l'orientation courante dans le sélecteur (à l'ouverture de l'éditeur).
function populateOrientationSelect() {
    const sel = document.getElementById('ed_orientation');
    if (!sel) return;
    sel.value = _mwIsPortrait(editorParams.orientation) ? editorParams.orientation : 'landscape';
}

// Changement de format de sortie : structurel (résolution/cadence) → exige un redéploiement.
// out_width/out_height = canevas LOGIQUE : en portrait on swappe (le mur se compose en vertical).
function onFormatChange() {
    const sel = document.getElementById('ed_out_format');
    if (!sel) return;
    const f = (window._videoFormats || []).find(x => x.label === sel.value);
    if (!f) return;   // « (actuel) » : rien à changer
    const portrait = _mwIsPortrait(editorParams.orientation);
    editorParams.out_width  = portrait ? f.h : f.w;
    editorParams.out_height = portrait ? f.w : f.h;
    editorParams.fps        = f.fps;
    editorParams.scan       = f.scan || 'p';
    resizeCanvas();
    dessiner();
    mwFlash(T('plugin.multiview.format_changed_redeploy'));
}

// Changement d'orientation : structurel (recrée le flux). Le canevas LOGIQUE doit être « haut »
// (h>w) en portrait, « large » (w>h) en paysage. On corrige les dims SEULEMENT si elles ne
// correspondent pas à l'orientation cible → robuste même si l'état persisté était incohérent
// (ex. orientation perdue par un ancien déploiement éditeur). cw↔ccw : mêmes dims, pas de swap.
function onOrientationChange() {
    const sel = document.getElementById('ed_orientation');
    if (!sel) return;
    const next = sel.value;
    editorParams.orientation = next;
    const w = editorParams.out_width, h = editorParams.out_height;
    if (_mwIsPortrait(next) !== (h > w)) {
        editorParams.out_width = h; editorParams.out_height = w;
    }
    resizeCanvas();
    dessiner();
    mwFlash(T('plugin.multiview.format_changed_redeploy'));
}

// ─── Handlers globaux ────────────────────────────────────────

// Mode tally/UMD : "central" (push orchestrateur) | "direct" (serveur TSL local).
// Dérivation depuis l'ancien schéma tsl_port/tsl_remote si tsl_mode absent.
function _resolveTslMode(p) {
    if (p && p.tsl_mode) return p.tsl_mode;
    return ((parseInt(p && p.tsl_port) || 0) > 0 && !(p && p.tsl_remote)) ? 'direct' : 'central';
}
function updateTslModeUI() {
    const mode = document.getElementById('ed_tsl_mode')?.value || 'central';
    const portRow = document.getElementById('ed_tsl_port_row');
    const note    = document.getElementById('ed_tsl_central_note');
    if (portRow) portRow.style.display = (mode === 'direct') ? '' : 'none';
    if (note)    note.style.display    = (mode === 'direct') ? 'none' : '';
    // En central, l'index TSL d'un PiP est inutile : il est déduit de la source du PiP
    // (lookup inverse source_shm → tsl_index dans le tsl_mapping de la connexion).
    const idxRow = document.getElementById('ed_tsl_index_row');
    if (idxRow) idxRow.style.display = (mode === 'direct') ? '' : 'none';
}
function onTslModeChange() {
    const _tm = document.getElementById('ed_tsl_mode');
    if (_tm && editorParams) editorParams.tsl_mode = _tm.value;
    updateTslModeUI();
    if (typeof refreshOverlayPanel === 'function') refreshOverlayPanel();   // champs overlay selon le mode
}

function onGlobalChange() {
    editorParams.border_w      = parseInt(document.getElementById('ed_border_w').value) || 0;
    editorParams.border_color  = document.getElementById('ed_border_color').value;
    editorParams.overlay_below = document.getElementById('ed_overlay_below').checked;
    editorParams.label_size    = Math.max(6, parseInt(document.getElementById('ed_label_size').value) || 14);
    { const fs = document.getElementById('ed_frame_style'); if (fs) editorParams.frame_style = fs.value; }
    editorParams.show_no_signal  = document.getElementById('ed_show_no_signal').checked;
    editorParams.freeze_detect_s = Math.max(0, parseFloat(document.getElementById('ed_freeze_detect').value) || 0);
    editorParams.show_format     = document.getElementById('ed_show_format').checked;
    editorParams.show_proxy      = document.getElementById('ed_show_proxy').checked;
    { const _tm = document.getElementById('ed_tsl_mode'); if (_tm) editorParams.tsl_mode = _tm.value; }
    { const _tp = document.getElementById('ed_tsl_port'); if (_tp) editorParams.tsl_port = parseInt(_tp.value) || 0; }
    dessiner();
    hotApplyStyle();
}

// Max entrées : redimensionne la banque localement (appliqué au prochain déploiement).
function onMaxChange() {
    if (!editorParams) return;
    editorParams.max_inputs = Math.max(1, parseInt(document.getElementById('ed_max').value) || editorParams.max_inputs);
    padBank();
    dessiner();
}

function resizeCanvas() {
    const canvas = document.getElementById('ed_canvas');
    if (!canvas) return;
    const w = editorParams.out_width;
    const h = editorParams.out_height;
    canvas.width  = w;
    canvas.height = h;
    // Taille d'affichage explicite : tient dans la largeur de l'éditeur ET dans
    // ~65 % de la hauteur de fenêtre (sorties verticales/carrées), sans dépasser
    // la résolution native. Le wrap (fit-content) colle au canvas : la surface
    // sombre correspond exactement à la zone où l'on peut poser des fenêtres.
    const wrap   = canvas.parentElement;
    const availW = (wrap && wrap.parentElement && wrap.parentElement.clientWidth) || w;
    const availH = Math.max(240, Math.round(window.innerHeight * 0.65));
    const scale  = Math.min(1, availW / w, availH / h);
    canvas.style.width  = Math.round(w * scale) + 'px';
    canvas.style.height = 'auto';
    canvas.style.maxWidth = '100%';
}

// ─── Banque d'entrées / PiP ──────────────────────────────────
// flux_config = banque de max_inputs entrées à indices STABLES (slot de câblage =
// tally = Ember). Un PiP est une entrée non masquée (hidden) ; retirer un PiP de
// l'image NE COUPE PAS sa source. Miroir de hooks.pad_input_bank côté orchestrateur.

function newEntry(idx, hidden) {
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;
    return {
        path: '', hidden: !!hidden, name: '',
        label_source: 'hostname',
        in_w: 0, in_h: 0,
        x: 0, y: 0,
        w: Math.round(out_w / 2) & ~1,
        h: Math.round(out_h / 2) & ~1,
        ratio: 16/9,
        color: COLORS[idx % COLORS.length],
        show_label: true,
        show_tally: false,           // dérivé (rouge||vert) ; reste le gate de rendu
        label_proportional: false,   // taille du label proportionnelle à la fenêtre
        tsl_index: 0,
        label_col: 0,
        tally_level: 0,              // niveau de Tally (0 = aucun ; 1-4 = bande tally_base)
        tally_red: false,
        tally_green: false,
        // Peak meters
        meter_channels: 0,           // 0 = désactivé ; sinon 2/4/6/8
        meter_position: 'right',     // left | right
        meter_inside: false,         // false = à côté (réduit la vidéo), true = overlay
        meter_opacity: 70,           // 10..100 (utilisé si inside=true)
        meter_scale: 'dbfs',         // dbfs | ppm (EBU)
    };
}

function padBank() {
    const max = Math.max(1, parseInt(editorParams.max_inputs) || 1);
    const fc = editorParams.flux_config;
    while (fc.length < max) fc.push(newEntry(fc.length, true));
    // Au-delà de max : on garde (ne jamais détruire une entrée potentiellement câblée).
}

// Position/taille d'un NOUVEAU PiP (entrée jamais positionnée) : grille de
// demi-fenêtres tant qu'il reste des cases entièrement DANS la zone, puis PiP
// « cascade » posé PAR-DESSUS à 1/8 de la surface (≈ 35 % des dimensions),
// décalé à chaque ajout. Aucune image créée hors de la zone de sortie.
function placerNouveauPip(f) {
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;
    const win_w = Math.round(out_w / 2) & ~1;
    const win_h = Math.round(out_h / 2) & ~1;
    const cols  = Math.max(1, Math.floor(out_w / win_w));
    const rows  = Math.max(1, Math.floor(out_h / win_h));
    const nVis  = editorParams.flux_config.filter(o => o !== f && !o.hidden).length;
    if (nVis < cols * rows) {
        f.w = win_w; f.h = win_h;
        f.x = (nVis % cols) * win_w;
        f.y = Math.floor(nVis / cols) * win_h;
        return;
    }
    // Grille pleine → cascade : 1/8 de la SURFACE = dimensions / √8
    const s = 1 / Math.sqrt(8);
    f.w = Math.round(out_w * s) & ~1;
    f.h = Math.round(out_h * s) & ~1;
    const step   = Math.max(16, Math.round(out_h * 0.05));
    const maxX   = Math.max(0, out_w - f.w);
    const maxY   = Math.max(0, out_h - f.h);
    const perRun = Math.max(1, Math.floor(Math.min(maxX, maxY) / step));   // wrap au bord
    const k      = (nVis - cols * rows) % perRun;
    f.x = Math.min(maxX, (k + 1) * step) & ~1;
    f.y = Math.min(maxY, (k + 1) * step) & ~1;
}

function ajouterEntree() {
    padBank();
    // « Ajouter un PiP » = réafficher la première entrée masquée de la banque
    // (sa source câblée éventuelle est conservée et réapparaît).
    const idx = editorParams.flux_config.findIndex(f => f.hidden);
    if (idx < 0) {
        mwFlash(T('plugin.multiview.flash_bank_full').replace('{n}', editorParams.max_inputs));
        return;
    }
    const f = editorParams.flux_config[idx];
    f.hidden = false;
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;
    const win_w = Math.round(out_w / 2) & ~1;
    const win_h = Math.round(out_h / 2) & ~1;
    // Entrée jamais positionnée (défauts de la banque) → placement automatique
    if (!f.x && !f.y && f.w === win_w && f.h === win_h) {
        placerNouveauPip(f);
    }
    // Jamais hors zone (y compris une géométrie mémorisée d'un ancien PiP)
    f.w = Math.min(f.w, out_w); f.h = Math.min(f.h, out_h);
    f.x = Math.max(0, Math.min(out_w - f.w, f.x));
    f.y = Math.max(0, Math.min(out_h - f.h, f.y));
    selectedIdxs = [idx];
    dessiner();
    hotApplyFull();
}

function supprimerEntreeSelectionnee() {
    if (selectedIdxs.length === 0) return;
    // Retire le PiP de l'image SANS couper la source (entrée masquée, câble conservé).
    selectedIdxs.forEach(i => {
        const f = editorParams.flux_config[i];
        if (f) f.hidden = true;
    });
    selectedIdxs = [];
    dessiner();
    hotApplyFull();
}

function toggleEntryHidden(i, shown) {
    const f = editorParams.flux_config[i];
    if (!f) return;
    f.hidden = !shown;
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
    document.getElementById('ed_label_proportional').checked = !!f.label_proportional;
    document.getElementById('ed_tsl_index').value    = f.tsl_index || 0;
    const _colors = f.tally_red && f.tally_green ? 'both' : f.tally_red ? 'red' : f.tally_green ? 'green' : 'none';
    _tslSetSelects(f.label_col ?? 0, f.tally_level ?? 0, _colors);
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
        return `<option value="${escapeHtml('/dev/shm/' + s.shm)}">${escapeHtml(txt)}</option>`;
    });
    // PiP vide : option explicite « aucune source » en tête (path = '').
    opts.unshift('<option value="">' + escapeHtml(T('plugin.multiview.no_source_option')) + '</option>');
    // Inclure le path actuel même si introuvable dans la liste (container détruit p.ex.).
    if (f.path && !videoSources.some(s => '/dev/shm/' + s.shm === f.path)) {
        opts.splice(1, 0, `<option value="${escapeHtml(f.path)}">${escapeHtml(f.path)}</option>`);
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
    f.label_proportional = document.getElementById('ed_label_proportional').checked;
    f.tsl_index      = parseInt(document.getElementById('ed_tsl_index').value) || 0;
    f.label_col      = parseInt(document.getElementById('ed_label_col').value) || 0;
    f.tally_level    = parseInt(document.getElementById('ed_tally_level').value) || 0;
    const _colors    = document.getElementById('ed_tally_colors').value || 'none';
    f.tally_red      = (_colors === 'red'   || _colors === 'both');
    f.tally_green    = (_colors === 'green' || _colors === 'both');
    f.show_tally     = !!(f.tally_level && (f.tally_red || f.tally_green));   // gate de rendu
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
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;
    // Clamp à la zone de sortie : aucune image ne doit exister hors zone.
    let nw = Math.min(out_w, parseInt(document.getElementById('ed_w').value) || 64);
    let nh = Math.min(out_h, parseInt(document.getElementById('ed_h').value) || 64);
    f.w = nw % 2 === 0 ? nw : nw - 1;
    f.h = nh % 2 === 0 ? nh : nh - 1;
    f.x = Math.max(0, Math.min(out_w - f.w, parseInt(document.getElementById('ed_x').value) || 0));
    f.y = Math.max(0, Math.min(out_h - f.h, parseInt(document.getElementById('ed_y').value) || 0));
    dessiner();   // refreshEntryPanel() re-reflète les valeurs clampées dans les champs
    hotApplyWindow(primary);
}

// ─── Copier / coller les réglages d'une fenêtre ──────────────
// Copie le format + l'apparence de la fenêtre primaire, SANS la position (x/y)
// ni la source (path/in_w/in_h/ratio/color), puis colle dans toutes les fenêtres
// sélectionnées en une fois.
const COPY_FIELDS = ['w', 'h', 'label_source', 'show_label', 'show_tally', 'label_proportional', 'tsl_index',
    'label_col', 'tally_level', 'tally_red', 'tally_green',
    'meter_channels', 'meter_position', 'meter_inside', 'meter_opacity', 'meter_scale'];
let reglagesClipboard = null;

function mwFlash(msg) {
    const t = document.getElementById('mw-toast');
    if (!t) return;
    t.textContent = msg;
    t.classList.add('visible');
    clearTimeout(mwFlash._t);
    mwFlash._t = setTimeout(() => t.classList.remove('visible'), 2600);
}

function copierReglagesFenetre() {
    const primary = primaryIdx();
    if (!editorParams || primary < 0) { mwFlash(T('plugin.multiview.flash_select_copy')); return; }
    const f = editorParams.flux_config[primary];
    reglagesClipboard = {};
    COPY_FIELDS.forEach(k => { if (f[k] !== undefined) reglagesClipboard[k] = f[k]; });
    updateToolbar();
    mwFlash(T('plugin.multiview.flash_copied'));
}

function collerReglagesFenetre() {
    if (!editorParams || !reglagesClipboard) { mwFlash(T('plugin.multiview.flash_nothing_to_paste')); return; }
    if (selectedIdxs.length === 0) { mwFlash(T('plugin.multiview.flash_select_targets')); return; }
    selectedIdxs.forEach(i => {
        const f = editorParams.flux_config[i];
        if (f) Object.assign(f, reglagesClipboard);
    });
    dessiner();
    selectedIdxs.forEach(i => hotApplyWindow(i));
    mwFlash(T('plugin.multiview.flash_pasted').replace('{n}', selectedIdxs.length));
}

// ─── Dessin ──────────────────────────────────────────────────

// Tokens thème pour le canvas : lus une fois par dessiner() (pas à chaque frame de
// drag). Fallbacks = valeurs du thème default si le token manque (vieil orchestrateur).
let _tok = null;
function _readTokens() {
    const cs = getComputedStyle(document.documentElement);
    const v = (name, fb) => (cs.getPropertyValue(name) || '').trim() || fb;
    _tok = {
        canvasBg: v('--canvas-bg', '#0d1117'),
        grid:     v('--border-soft', '#21262d'),
        muted:    v('--text-muted', '#8b949e'),
        accent:   v('--accent', '#7aa2c8'),
        warning:  v('--status-warning-fg', '#c4a667'),
        overlay:  v('--overlay-accent', '#c49fd8'),
    };
    return _tok;
}

// Miroir de _frame_metrics (script.py) : marges réservées par l'habillage
// (frame_style) autour de l'image. t = 3 % du petit côté, borné 3..24 px.
function mwFrameMetrics(f) {
    const w = f.w, h = f.h;
    const labelSize = Math.max(6, editorParams.label_size || 14);
    const effRaw = f.label_proportional
        ? Math.max(6, Math.round(labelSize * (2 * h / editorParams.out_height)))
        : Math.max(6, Math.min(labelSize, Math.floor(h * 0.30)));
    const band = Math.min(Math.max(14, Math.round(effRaw * 2)), Math.max(8, Math.floor(h * 0.40)));
    const t = Math.max(3, Math.min(24, Math.round(Math.min(w, h) * 0.03)));
    const barOn = !!(f.show_label || f.show_tally);
    const style = editorParams.frame_style || 'none';
    let ml = 0, mt = 0, mr = 0, mb = 0;
    if (style === 'stylized') {              // Moniteur : bezel + menton nom/LED
        const bez = Math.max(4, Math.round(t * 2.2));
        ml = mr = mt = bez;
        mb = bez + (barOn ? band : 0);
    } else if (style === 'classic') {        // UMD : cadre fin + boîtier sous l'image
        const fr = Math.max(2, Math.floor(t / 2));
        ml = mr = mt = fr;
        mb = fr + (barOn ? band + Math.max(2, Math.floor(t / 2)) : 0);
    } else if (style === 'tally_border') {   // cadre FIN (3 px) + onglet nom AU-DESSUS
        const b = 3;
        ml = mr = mb = b;
        mt = b + (f.show_label ? band : 0);
    } else if (style === 'viewfinder') {     // équerres + chip nom
        ml = mr = mb = t;
        mt = f.show_label ? band + 4 : t;
    } else if (style === 'flat') {           // nom + soulignement
        mb = t + (f.show_label ? band + 2 : 0);
    }
    if (h - mt - mb < 16) {
        mb = Math.max(0, Math.min(mb, h - 16 - mt));
        if (h - mt - mb < 16) mt = Math.max(0, h - 16 - mb);
    }
    if (w - ml - mr < 16) {
        const side = Math.max(0, Math.floor((w - 16) / 2));
        ml = Math.min(ml, side); mr = Math.min(mr, side);
    }
    return { t, ml, mt, mr, mb, band };
}

function _roundRectPath(ctx, x, y, w, h, r) {
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
    else ctx.rect(x, y, w, h);
}

// Corps du bezel « Moniteur » : dessiné AVANT le rectangle vidéo factice
// (les autres habillages n'occupent que les marges → dessinés après).
function drawDressingBack(ctx, f, fm, style) {
    if (style !== 'stylized') return;
    const rad = Math.max(3, fm.t);
    const inset = Math.max(1, Math.floor(fm.t / 2));
    ctx.save();
    ctx.fillStyle = '#2e2e35';
    _roundRectPath(ctx, f.x, f.y, f.w, f.h, rad);
    ctx.fill();
    ctx.fillStyle = '#1b1b20';
    _roundRectPath(ctx, f.x + inset, f.y + inset, f.w - 2 * inset, f.h - 2 * inset,
                   Math.max(2, rad - inset));
    ctx.fill();
    ctx.restore();
}

// Habillage schématique (état tally « repos ») — miroir visuel de
// render_border / render_static / render_dynamic (script.py).
function drawDressing(ctx, f, fm, style, vx, vy, vw, vh, eff) {
    const x = f.x, y = f.y, w = f.w, h = f.h;
    const name = computeDisplayName(f);
    ctx.save();
    ctx.font = `bold ${eff}px monospace`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    if (style === 'stylized') {
        ctx.strokeStyle = '#62626a'; ctx.lineWidth = 1;
        ctx.strokeRect(vx - 0.5, vy - 0.5, vw + 1, vh + 1);
        const cy = y + h - fm.mb / 2;
        if (f.show_label) {
            ctx.fillStyle = '#a8a8b2'; ctx.textAlign = 'center';
            ctx.fillText(name, x + w / 2, cy);
        }
        if (f.show_tally) {
            const r = Math.max(3, Math.min(Math.max(3, Math.floor(fm.band / 3)),
                                           Math.round(eff * 0.45)));
            ctx.fillStyle = '#3a3a40';
            [x + fm.ml + r + 4, x + w - fm.mr - r - 4].forEach(cx => {
                ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
            });
        }

    } else if (style === 'classic') {
        const fr = Math.max(2, Math.floor(fm.t / 2));
        ctx.strokeStyle = '#46464e'; ctx.lineWidth = fr;
        ctx.strokeRect(vx - fr / 2, vy - fr / 2, vw + fr, vh + fr);
        if (f.show_label || f.show_tally) {
            const bw = Math.max(24, Math.min(Math.floor(w * 0.55), w - 60));
            const bx = x + (w - bw) / 2;
            const by = y + h - fm.band;
            if (f.show_label) {
                ctx.fillStyle = '#08080a'; ctx.fillRect(bx, by, bw, fm.band - 1);
                ctx.strokeStyle = '#82828a'; ctx.lineWidth = 1;
                ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, fm.band - 2);
                ctx.fillStyle = '#f0f0f5'; ctx.textAlign = 'center';
                ctx.fillText(name, bx + bw / 2, by + fm.band / 2);
            }
            if (f.show_tally) {
                const sz = Math.max(4, Math.round(eff * 1.4));
                const ty = by + (fm.band - sz) / 2;
                ctx.fillStyle = '#28282c'; ctx.strokeStyle = '#cdcdd4'; ctx.lineWidth = 1;
                ctx.fillRect(bx - 6 - sz, ty, sz, sz); ctx.strokeRect(bx - 6 - sz, ty, sz, sz);
                ctx.fillRect(bx + bw + 6, ty, sz, sz); ctx.strokeRect(bx + bw + 6, ty, sz, sz);
            }
        }

    } else if (style === 'tally_border') {
        const b = fm.ml;
        const band = Math.max(0, fm.mt - b);
        // Cadre fin autour de l'IMAGE seule ; onglet nom AU-DESSUS du cadre (dehors).
        ctx.strokeStyle = '#46464e'; ctx.lineWidth = b;
        ctx.strokeRect(vx - b / 2, vy - b / 2, vw + b, vh + b);
        if (f.show_label) {
            const tabW = Math.max(24, Math.min(vw + 2 * b, ctx.measureText(name).width + 16));
            ctx.fillStyle = '#323238';
            ctx.fillRect(vx - b, vy - b - band, tabW, band);
            ctx.fillStyle = '#f5f5fa';
            ctx.fillText(name, vx - b + 6, vy - b - band / 2);
        }

    } else if (style === 'viewfinder') {
        const bt = Math.max(2, Math.floor(fm.t / 2));
        const gap = bt;
        const arm = Math.max(8, Math.round(Math.min(vw, vh) * 0.14));
        ctx.fillStyle = '#e1e1e8';
        const x0 = vx - gap, y0 = vy - gap, x1 = vx + vw + gap, y1 = vy + vh + gap;
        [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]]
            .forEach(([cx, cy, sx, sy]) => {
                ctx.fillRect(Math.min(cx, cx + sx * arm), Math.min(cy, cy + sy * bt), arm, bt);
                ctx.fillRect(Math.min(cx, cx + sx * bt), Math.min(cy, cy + sy * arm), bt, arm);
            });
        if (f.show_label) {
            const dotR = f.show_tally ? Math.max(2, Math.round(eff * 0.3)) : 0;
            const chipH = fm.band;
            const chipW = Math.max(20, Math.min(w - 8,
                ctx.measureText(name).width + 16 + (dotR ? dotR * 2 + 4 : 0)));
            ctx.fillStyle = 'rgba(10,10,12,0.85)';
            _roundRectPath(ctx, vx, y + 1, chipW, chipH, chipH / 2);
            ctx.fill();
            let tx = vx + 8;
            if (dotR) {
                ctx.fillStyle = '#5a5a62';
                ctx.beginPath(); ctx.arc(tx + dotR, y + 1 + chipH / 2, dotR, 0, Math.PI * 2); ctx.fill();
                tx += dotR * 2 + 4;
            }
            ctx.fillStyle = '#f0f0f5';
            ctx.fillText(name, tx, y + 1 + chipH / 2);
        }

    } else if (style === 'flat') {
        ctx.fillStyle = '#5a5a62';
        ctx.fillRect(vx, y + h - fm.t, vw, fm.t);
        if (f.show_label) {
            ctx.fillStyle = '#ebebf0';
            ctx.fillText(name, vx + 2, y + h - fm.t - 2 - fm.band / 2);
        }
    }
    ctx.restore();
}

function dessiner() {
    renderEntryTable();
    refreshEntryPanel();
    refreshOverlayPanel();
    updateToolbar();
    _readTokens();
    drawCanvas();
    mwRefreshSidebarPreview();
}

// Rafraîchit le mini-aperçu (schéma) de l'instance ÉDITÉE dans la liste de gauche, à partir de la
// disposition LIVE de l'éditeur (editorParams) — sans round-trip serveur. La sidebar reflète ainsi
// les changements au fil de l'édition, en plus du rafraîchissement après déploiement (tpLoadInstances).
function mwRefreshSidebarPreview() {
    if (editorVmid == null || !editorParams || typeof drawMiniPreview !== 'function') return;
    const cv = document.querySelector('.tp-inst-preview[data-vmid="' + editorVmid + '"]');
    if (cv) { try { drawMiniPreview(cv, editorParams); } catch (_) {} }
}

// Peinture du canvas seule : appelée à chaque frame de drag — le DOM (table,
// panneaux, dropdown source) n'est resynchronisé qu'aux changements de sélection.
function drawCanvas() {
    const canvas = document.getElementById('ed_canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    const t = _tok || _readTokens();

    ctx.fillStyle = t.canvasBg;
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = t.grid;
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 64) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = 0; y < h; y += 64) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

    const borderW      = editorParams.border_w || 0;
    const borderColor  = editorParams.border_color || '#ffffff';
    const overlayBelow = !!editorParams.overlay_below;
    const labelSize    = Math.max(6, editorParams.label_size || 14);

    // Images de fond (overlay layer=background) : sous les fenêtres vidéo.
    drawOverlayLayer(ctx, 'background');

    const primary = primaryIdx();
    editorParams.flux_config.forEach((f, i) => {
        if (f.hidden) return;   // entrée de la banque non affichée
        const sel = isSelected(i);
        const isPrimary = i === primary;
        const barOn = f.show_label || f.show_tally;
        // Plafonds PAR FENÊTRE — formules miroir de _label_metrics (script.py) :
        // texte ≤ 30 % de la hauteur du PiP, bandeau ≤ 40 %. En mode proportionnel,
        // texte = labelSize quand la fenêtre fait 1/4 de l'image (h = sortie/2).
        const effRaw     = f.label_proportional
            ? Math.max(6, Math.round(labelSize * (2 * f.h / editorParams.out_height)))
            : Math.max(6, Math.min(labelSize, Math.floor(f.h * 0.30)));
        const BAR_H      = Math.min(Math.max(14, Math.round(effRaw * 2)), Math.max(8, Math.floor(f.h * 0.40)));
        const eff        = Math.max(6, Math.min(effRaw, BAR_H - 4));
        const TALLY_SIZE = Math.max(4, Math.min(Math.round(eff * 1.4), BAR_H - 2));
        const TALLY_PAD  = Math.max(2, Math.round(eff * 0.35));
        // Habillage (frame_style) : marges réservées AUTOUR de l'image — miroir de
        // _frame_metrics/_video_rect (script.py). Le bandeau legacy (overlay_below)
        // ne s'applique qu'au style 'none'.
        const style = editorParams.frame_style || 'none';
        const fm = style !== 'none' ? mwFrameMetrics(f) : null;
        let videoX, videoY, videoW, videoH;
        if (fm) {
            const availW = Math.max(2, f.w - fm.ml - fm.mr);
            const availH = Math.max(2, f.h - fm.mt - fm.mb);
            const sc = Math.min(availW / f.w, availH / f.h);
            videoW = Math.max(2, Math.round(f.w * sc));
            videoH = Math.max(2, Math.round(f.h * sc));
            videoX = f.x + fm.ml + Math.floor((availW - videoW) / 2);
            videoY = f.y + fm.mt + Math.floor((availH - videoH) / 2);
        } else {
            videoY = f.y;
            videoH = (overlayBelow && barOn) ? Math.max(2, f.h - BAR_H) : f.h;
            // Bandeau sous l'image : largeur réduite au même ratio (pillarbox centré,
            // pas d'étirement) — même géométrie que _video_rect (script.py).
            videoW = videoH < f.h ? Math.max(2, Math.round(f.w * videoH / f.h)) : f.w;
            videoX = f.x + Math.floor((f.w - videoW) / 2);
        }

        if (fm) drawDressingBack(ctx, f, fm, style);

        ctx.globalAlpha = sel ? 0.8 : 0.4;
        ctx.fillStyle   = f.color;
        ctx.fillRect(videoX, videoY, videoW, videoH);
        ctx.globalAlpha = 1;

        if (!fm && borderW > 0) {
            ctx.strokeStyle = borderColor;
            ctx.lineWidth   = borderW;
            ctx.strokeRect(f.x + borderW/2, f.y + borderW/2,
                           f.w - borderW, f.h - borderW);
        }

        ctx.strokeStyle = isPrimary ? '#ffffff' : (sel ? t.accent : f.color);
        ctx.lineWidth   = sel ? 2 : 1;
        ctx.setLineDash(isPrimary ? [6, 4] : (sel ? [3, 3] : []));
        ctx.strokeRect(f.x, f.y, f.w, f.h);
        ctx.setLineDash([]);

        if (fm) {
            // Habillage v0.12 : aperçu schématique fidèle (cadre/bezel/UMD/chip/trait).
            drawDressing(ctx, f, fm, style, videoX, videoY, videoW, videoH, eff);
        } else if (barOn) {
            const barTop = f.y + f.h - BAR_H;
            ctx.fillStyle = 'rgba(0,0,0,0.7)';
            ctx.fillRect(f.x, barTop, f.w, BAR_H);

            let textL = f.x, textR = f.x + f.w;

            if (f.show_tally) {
                // Pavés tally centrés sur le rectangle VIDÉO (videoX/videoW), pas la cellule
                // entière (qui inclut la bande VU audio) — miroir de render_dynamic 'none'.
                const ty = barTop + (BAR_H - TALLY_SIZE) / 2;
                ctx.fillStyle = 'rgba(128,128,128,0.7)';
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1;
                ctx.fillRect(videoX + TALLY_PAD, ty, TALLY_SIZE, TALLY_SIZE);
                ctx.strokeRect(videoX + TALLY_PAD, ty, TALLY_SIZE, TALLY_SIZE);
                ctx.fillRect(videoX + videoW - TALLY_PAD - TALLY_SIZE, ty, TALLY_SIZE, TALLY_SIZE);
                ctx.strokeRect(videoX + videoW - TALLY_PAD - TALLY_SIZE, ty, TALLY_SIZE, TALLY_SIZE);
                textL += TALLY_PAD + TALLY_SIZE + 4;
                textR -= TALLY_PAD + TALLY_SIZE + 4;
            }

            if (f.show_label) {
                ctx.fillStyle = '#ffffff';
                ctx.font = `bold ${eff}px monospace`;
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
                mx = f.x + (fm ? fm.ml : 0) + 2;
            } else {
                mx = f.x + f.w - (fm ? fm.mr : 0) - meterW - 2;
            }
            const my = videoY + 2;
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
        const cy = videoY + videoH / 2;
        ctx.fillStyle = isPrimary ? '#ffffff' : (sel ? t.accent : f.color);
        ctx.beginPath();
        ctx.arc(cx, cy, badgeR, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = t.canvasBg;
        ctx.font = `bold ${Math.round(badgeR * 1.1)}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(i + 1), cx, cy);
        ctx.textAlign = 'start';
        ctx.textBaseline = 'alphabetic';

        ctx.fillStyle = t.muted;
        ctx.font = '11px monospace';
        ctx.fillText(`${f.w}×${f.h}`, videoX + 4, videoY + 14);

        if (isPrimary) {
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(f.x + f.w - HANDLE_SIZE, f.y + f.h - HANDLE_SIZE,
                         HANDLE_SIZE, HANDLE_SIZE);
        }
    });

    // Overlays texte/horloge/logo (layer=foreground) : par-dessus les fenêtres.
    drawOverlayLayer(ctx, 'foreground');

    // Lignes guides de snap (pendant le drag)
    if (dragMode && snapGuides.length) {
        ctx.strokeStyle = t.warning;
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

    ctx.fillStyle = t.muted;
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
        const cls = (isP ? 'is-primary' : (sel ? 'is-selected' : '')) + (f.hidden ? ' is-hidden' : '');
        return `
        <tr class="mw-entry-row ${cls}" tabindex="0" onclick="selectEntry(${i}, event)" onkeydown="entryRowKey(event, ${i})">
            <td><span class="mw-entry-color" style="background:${f.color}"></span>${i + 1}</td>
            <td>${escapeHtml(sourceHostname(f)) || '—'}</td>
            <td class="mw-entry-path">${escapeHtml(f.path)}</td>
            <td>${f.hidden ? '—' : `${f.x}, ${f.y}`}</td>
            <td>${f.hidden ? '—' : `${f.w}×${f.h}`}</td>
            <td><input type="checkbox" class="ios-toggle" ${f.hidden ? '' : 'checked'}
                onclick="event.stopPropagation(); toggleEntryHidden(${i}, this.checked)"
                title="${escapeHtml(T('plugin.multiview.row_toggle_title'))}"></td>
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
    // Pointer Events : capture du geste (souris, stylet ou doigt) jusqu'au relâchement.
    if (e.pointerId !== undefined && e.target.setPointerCapture) {
        try { e.target.setPointerCapture(e.pointerId); } catch(_) {}
    }
    e.preventDefault();
    const pos = getCanvasPos(e);
    // 1. Overlays de premier plan (au-dessus de la vidéo)
    let hit = hitOverlay(pos, 'foreground');
    if (hit) return beginOverlayDrag(hit, pos);
    // 2. Fenêtres vidéo
    const primary = primaryIdx();
    for (let i = editorParams.flux_config.length - 1; i >= 0; i--) {
        const f = editorParams.flux_config[i];
        if (f.hidden) continue;   // entrées masquées : pas dans le canvas
        if (i === primary &&
            pos.x >= f.x + f.w - HANDLE_SIZE && pos.y >= f.y + f.h - HANDLE_SIZE) {
            selectedOverlay = -1;
            dragMode = 'resize'; dragOverlay = false; dragStart = pos; dragOrigRect = {...f};
            _setCanvasCursor('nwse-resize');
            return;
        }
        if (pos.x >= f.x && pos.x <= f.x + f.w &&
            pos.y >= f.y && pos.y <= f.y + f.h) {
            selectedOverlay = -1;
            // Si on commence le drag sur une fenêtre DÉJÀ sélectionnée (sans Maj), on garde la
            // sélection multiple pour la déplacer en groupe ; sinon clic = sélection simple.
            if (!(isSelected(i) && !e.shiftKey)) toggleSelection(i, e.shiftKey);
            dragMode = 'move'; dragOverlay = false; dragStart = pos; dragOrigRect = {...editorParams.flux_config[primaryIdx()]};
            dragGroupOrig = selectedIdxs.map(j => ({ j, x: editorParams.flux_config[j].x, y: editorParams.flux_config[j].y }));
            _setCanvasCursor('move');
            dessiner();
            return;
        }
    }
    // 3. Images de fond (sous la vidéo)
    hit = hitOverlay(pos, 'background');
    if (hit) return beginOverlayDrag(hit, pos);
    if (!e.shiftKey) selectedIdxs = [];
    selectedOverlay = -1;
    dragMode = null;
    dessiner();
}

function hitOverlay(pos, layer) {
    const ovs = editorParams.overlays || [];
    for (let i = ovs.length - 1; i >= 0; i--) {
        const o = ovs[i];
        const isBg = (o.kind === 'image' && o.layer === 'background');
        if (layer === 'background' ? !isBg : isBg) continue;
        if (i === selectedOverlay &&
            pos.x >= o.x + o.w - HANDLE_SIZE && pos.y >= o.y + o.h - HANDLE_SIZE) {
            return {i, mode: 'resize'};
        }
        if (pos.x >= o.x && pos.x <= o.x + o.w && pos.y >= o.y && pos.y <= o.y + o.h) {
            return {i, mode: 'move'};
        }
    }
    return null;
}

function beginOverlayDrag(hit, pos) {
    selectedOverlay = hit.i;
    selectedIdxs = [];
    dragOverlay = true;
    dragMode = hit.mode;
    dragStart = pos;
    dragOrigRect = {...editorParams.overlays[hit.i]};
    _setCanvasCursor(hit.mode === 'resize' ? 'nwse-resize' : 'move');
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

// Curseur attendu à une position donnée (hors drag) : coin bas-droite d'une fenêtre/overlay
// SÉLECTIONNÉ → redimensionnement ; corps → déplacement ; vide → défaut. Mire la logique de
// canvasMouseDown/hitOverlay pour que l'aspect du curseur corresponde TOUJOURS à l'action réelle.
function _cursorForPos(pos) {
    let hit = hitOverlay(pos, 'foreground');
    if (hit) return hit.mode === 'resize' ? 'nwse-resize' : 'move';
    const primary = primaryIdx();
    for (let i = editorParams.flux_config.length - 1; i >= 0; i--) {
        const f = editorParams.flux_config[i];
        if (f.hidden) continue;
        if (i === primary && pos.x >= f.x + f.w - HANDLE_SIZE && pos.y >= f.y + f.h - HANDLE_SIZE)
            return 'nwse-resize';
        if (pos.x >= f.x && pos.x <= f.x + f.w && pos.y >= f.y && pos.y <= f.y + f.h)
            return 'move';
    }
    hit = hitOverlay(pos, 'background');
    if (hit) return hit.mode === 'resize' ? 'nwse-resize' : 'move';
    return 'default';
}

function _setCanvasCursor(c) {
    const cv = document.getElementById('ed_canvas');
    if (cv) cv.style.cursor = c;
}

function canvasMouseMove(e) {
    // Hors drag : retour visuel au survol (déplacement vs redimensionnement) pour lever l'ambiguïté.
    if (!dragMode) { _setCanvasCursor(_cursorForPos(getCanvasPos(e))); return; }
    if (dragOverlay) return overlayMouseMove(e);
    const primary = primaryIdx();
    if (primary < 0) return;
    const pos = getCanvasPos(e);
    const dx = pos.x - dragStart.x;
    const dy = pos.y - dragStart.y;
    const f = editorParams.flux_config[primary];
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;

    if (dragMode === 'move') {
        // Delta visé d'après la fenêtre primaire (avec snap), puis appliqué À TOUT LE GROUPE.
        let nx = Math.max(0, Math.min(out_w - f.w, Math.round(dragOrigRect.x + dx)));
        let ny = Math.max(0, Math.min(out_h - f.h, Math.round(dragOrigRect.y + dy)));
        if (snapEnabled) {
            const snapped = computeSnap(primary, nx, ny, f.w, f.h);
            nx = snapped.x; ny = snapped.y; snapGuides = snapped.guides;
        } else snapGuides = [];
        let ddx = nx - dragOrigRect.x, ddy = ny - dragOrigRect.y;
        // Clamp le delta pour qu'AUCUNE fenêtre du groupe ne sorte du cadre.
        const grp = dragGroupOrig.length ? dragGroupOrig : [{ j: primary, x: dragOrigRect.x, y: dragOrigRect.y }];
        let loX = -Infinity, hiX = Infinity, loY = -Infinity, hiY = Infinity;
        grp.forEach(g => {
            const gf = editorParams.flux_config[g.j];
            loX = Math.max(loX, -g.x); hiX = Math.min(hiX, out_w - gf.w - g.x);
            loY = Math.max(loY, -g.y); hiY = Math.min(hiY, out_h - gf.h - g.y);
        });
        ddx = Math.max(loX, Math.min(hiX, ddx));
        ddy = Math.max(loY, Math.min(hiY, ddy));
        grp.forEach(g => {
            const gf = editorParams.flux_config[g.j];
            gf.x = Math.round(g.x + ddx);
            gf.y = Math.round(g.y + ddy);
        });
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
    drawCanvas();
    syncGeomFields();
}

function overlayMouseMove(e) {
    if (!dragMode || selectedOverlay < 0) return;
    const pos = getCanvasPos(e);
    const dx = pos.x - dragStart.x, dy = pos.y - dragStart.y;
    const o = editorParams.overlays[selectedOverlay];
    const out_w = editorParams.out_width, out_h = editorParams.out_height;
    if (dragMode === 'move') {
        o.x = Math.max(0, Math.min(out_w - o.w, Math.round(dragOrigRect.x + dx)));
        o.y = Math.max(0, Math.min(out_h - o.h, Math.round(dragOrigRect.y + dy)));
    } else if (dragMode === 'resize') {
        let nw = Math.max(16, Math.min(out_w - o.x, Math.round(dragOrigRect.w + dx)));
        let nh = Math.max(16, Math.min(out_h - o.y, Math.round(dragOrigRect.h + dy)));
        o.w = nw % 2 === 0 ? nw : nw - 1;
        o.h = nh % 2 === 0 ? nh : nh - 1;
    }
    drawCanvas();
    syncGeomFields();
}

function canvasMouseUp() {
    _setCanvasCursor('default');   // le prochain survol réévaluera (move/resize/défaut)
    if (dragOverlay) {
        dragOverlay = false; dragMode = null; dessiner();
        hotApplyFull();   // résout l'image base64 + hot-apply /overlays (pas de coupure)
        return;
    }
    dragMode = null; snapGuides = []; dessiner();
    selectedIdxs.forEach(idx => hotApplyWindow(idx));
}

// Pendant un drag : ne resynchronise que les champs géométrie (pas de rebuild DOM).
function syncGeomFields() {
    if (dragOverlay && selectedOverlay >= 0) {
        const o = editorParams.overlays[selectedOverlay];
        if (!o) return;
        _ovSetVal('ov_x', o.x); _ovSetVal('ov_y', o.y);
        _ovSetVal('ov_w', o.w); _ovSetVal('ov_h', o.h);
        return;
    }
    const p = primaryIdx();
    if (p < 0) return;
    const f = editorParams.flux_config[p];
    [['ed_x', f.x], ['ed_y', f.y], ['ed_w', f.w], ['ed_h', f.h]].forEach(([id, v]) => {
        const e = document.getElementById(id);
        if (e) e.value = v;
    });
}

// Boutons d'outils : actifs selon le nombre de fenêtres sélectionnées (data-min).
function updateToolbar() {
    const n = selectedIdxs.length;
    document.querySelectorAll('#mw-toolbar .tool-btn[data-min]').forEach(b => {
        b.disabled = n < parseInt(b.dataset.min);
    });
    const paste = document.getElementById('ed_paste_btn');
    if (paste) paste.disabled = !reglagesClipboard || n === 0;
}

// Sélection d'une entrée au clavier depuis le tableau (Entrée / Espace).
function entryRowKey(ev, i) {
    if (ev.key !== 'Enter' && ev.key !== ' ') return;
    ev.preventDefault();
    selectEntry(i, ev);
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
            hidden:         !!f.hidden,
            name:           computeDisplayName(f),
            show_label:     !!f.show_label,
            show_tally:     !!f.show_tally,
            label_proportional: !!f.label_proportional,
            tsl_index:      f.tsl_index ?? 0,
            label_col:      f.label_col ?? 0,
            tally_level:    f.tally_level ?? 0,
            tally_red:      !!f.tally_red,
            tally_green:    !!f.tally_green,
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
            show_no_signal:  editorParams.show_no_signal !== false,
            freeze_detect_s: editorParams.freeze_detect_s ?? 2,
            show_format:     !!editorParams.show_format,
            show_proxy:      !!editorParams.show_proxy,
        })
    }).catch(() => {});
}

// ─── Overlays : texte / horloge / image ──────────────────────
// Objets visuels posés sur le canvas, séparés des fenêtres vidéo (non câblés).
// Add/edit/déplacement → hotApplyFull() (deploy) pour résoudre l'image en base64 côté
// serveur puis appliquer à chaud via :8082/overlays — aucune coupure de la sortie.

function newOverlay(kind) {
    const ow = editorParams.out_width, oh = editorParams.out_height;
    let w = Math.round(ow * 0.25) & ~1, h = Math.round(oh * 0.12) & ~1;
    if (kind === 'image') { w = Math.round(ow * 0.2) & ~1; h = Math.round(oh * 0.2) & ~1; }
    const o = {
        id: 'ov' + Date.now().toString(36) + Math.floor(Math.random() * 1000),
        kind, layer: 'foreground',
        x: Math.round((ow - w) / 2) & ~1, y: Math.round((oh - h) / 2) & ~1, w, h,
        font: 'dejavu-sans-bold', font_size: 0, align: 'center',
        color: '#ffffff', bg_color: '', bg_opacity: 100,
        tally_index: 0, color_on: '#ffd400', bg_color_on: '#cc0000',
    };
    if (kind === 'text')  Object.assign(o, { text: 'TEXTE', text_source: 'local', tsl_index: 0,
        label_row: '', label_col: 0, tally_level: 0, tally_red: false, tally_green: false });
    if (kind === 'clock') Object.assign(o, {
        clock_source: 'ptp', show_hh: true, show_mm: true, show_ss: true, show_ff: false,
        offset_ms: 0, chrono_start: '00:00:00', chrono_running: false,
        tc_source: 0,   // index d'entrée vidéo dont on lit le timecode ANC
        cd_warn: true, cd_warn_orange: 10, cd_warn_red: 5,
        cd_color_orange: '#ff9000', cd_color_red: '#ff3030',
        bg_color: '#000000', bg_opacity: 60 });
    if (kind === 'image') Object.assign(o, { image_b64: '', image_name: '', fit: 'contain', opacity: 100 });
    return o;
}

function ajouterOverlay(kind) {
    if (!editorParams) return;
    editorParams.overlays = editorParams.overlays || [];
    editorParams.overlays.push(newOverlay(kind));
    selectedOverlay = editorParams.overlays.length - 1;
    selectedIdxs = [];
    dessiner();
    hotApplyFull();
}

function supprimerOverlay() {
    if (selectedOverlay < 0) return;
    editorParams.overlays.splice(selectedOverlay, 1);
    selectedOverlay = -1;
    dessiner();
    hotApplyFull();
}

function serializeOverlays() {
    const ev = v => { v = parseInt(v) || 0; return v % 2 === 0 ? v : v - 1; };
    const clamp = (v, d) => Math.max(0, Math.min(100, parseInt(v ?? d)));
    return (editorParams.overlays || []).map(o => {
        const base = { id: o.id, kind: o.kind, layer: o.layer || 'foreground',
            x: parseInt(o.x) || 0, y: parseInt(o.y) || 0, w: ev(o.w), h: ev(o.h) };
        if (o.kind === 'text' || o.kind === 'clock') Object.assign(base, {
            font: o.font || 'dejavu-sans-bold', font_size: parseInt(o.font_size) || 0,
            align: o.align || 'center', color: o.color || '#ffffff',
            bg_color: o.bg_color || '', bg_opacity: clamp(o.bg_opacity, 100),
            tally_index: parseInt(o.tally_index) || 0,
            color_on: o.color_on || '#ffffff', bg_color_on: o.bg_color_on || '' });
        if (o.kind === 'text') Object.assign(base, {
            text: o.text || '', text_source: o.text_source || 'local',
            tsl_index: parseInt(o.tsl_index) || 0,
            label_row: o.label_row || '', label_col: parseInt(o.label_col) || 0,
            tally_level: parseInt(o.tally_level) || 0,
            tally_red: !!o.tally_red, tally_green: !!o.tally_green });
        if (o.kind === 'clock') Object.assign(base, {
            clock_source: o.clock_source || 'ptp',
            show_hh: o.show_hh !== false, show_mm: o.show_mm !== false,
            show_ss: o.show_ss !== false, show_ff: !!o.show_ff,
            offset_ms: parseInt(o.offset_ms) || 0,
            chrono_start: o.chrono_start || '00:00:00', chrono_running: !!o.chrono_running,
            tc_source: parseInt(o.tc_source) || 0,
            cd_warn: o.cd_warn !== false,
            cd_warn_orange: parseInt(o.cd_warn_orange ?? 10),
            cd_warn_red: parseInt(o.cd_warn_red ?? 5),
            cd_color_orange: o.cd_color_orange || '#ff9000',
            cd_color_red: o.cd_color_red || '#ff3030' });
        if (o.kind === 'image') Object.assign(base, {
            image_b64: o.image_b64 || '', image_name: o.image_name || '',
            fit: o.fit || 'contain', opacity: clamp(o.opacity, 100) });
        return base;
    });
}

// ── Rendu des overlays sur le canvas de l'éditeur ──
function _ovImg(o) {
    const b64 = o.image_b64 || '';
    if (!b64) return null;
    const sig = b64.length + ':' + b64.slice(0, 24);
    const c = _ovThumbCache[o.id];
    if (c && c.sig === sig) return c.img;
    const img = new Image();
    img.onload = () => dessiner();
    img.onerror = () => {};
    img.src = 'data:image/png;base64,' + b64;
    _ovThumbCache[o.id] = { sig, img };
    return img;
}

function hexA(hex, alpha) {
    const m = /^#?([0-9a-f]{6})$/i.exec((hex || '').trim());
    if (!m) return `rgba(0,0,0,${alpha})`;
    const n = parseInt(m[1], 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

function clockSample(o) {
    // L'éditeur n'a pas le timecode ANC live → placeholder en tirets pour la source ANC.
    const dash = o.clock_source === 'anc';
    const p = [];
    if (o.show_hh !== false) p.push(dash ? '--' : '12');
    if (o.show_mm !== false) p.push(dash ? '--' : '34');
    if (o.show_ss !== false) p.push(dash ? '--' : '56');
    if (o.show_ff) p.push(dash ? '--' : '00');
    return p.join(':') || (dash ? '--:--:--' : '12:34:56');
}

function drawImageFit(ctx, img, o) {
    const iw = img.naturalWidth, ih = img.naturalHeight;
    const fit = o.fit || 'contain';
    if (fit === 'stretch') { ctx.drawImage(img, o.x, o.y, o.w, o.h); return; }
    const scale = fit === 'cover' ? Math.max(o.w / iw, o.h / ih) : Math.min(o.w / iw, o.h / ih);
    const nw = Math.max(1, iw * scale), nh = Math.max(1, ih * scale);
    ctx.save(); ctx.beginPath(); ctx.rect(o.x, o.y, o.w, o.h); ctx.clip();
    ctx.drawImage(img, o.x + (o.w - nw) / 2, o.y + (o.h - nh) / 2, nw, nh);
    ctx.restore();
}

function drawOverlayLayer(ctx, layer) {
    const t = _tok || _readTokens();
    (editorParams.overlays || []).forEach((o, i) => {
        const isBg = (o.kind === 'image' && o.layer === 'background');
        if (layer === 'background' ? !isBg : isBg) return;
        const sel = (i === selectedOverlay);
        ctx.save();
        if (o.kind === 'image') {
            const img = _ovImg(o);
            if (img && img.complete && img.naturalWidth) {
                ctx.globalAlpha = Math.max(0.2, (o.opacity ?? 100) / 100);
                drawImageFit(ctx, img, o);
                ctx.globalAlpha = 1;
            } else {
                ctx.globalAlpha = 0.22; ctx.fillStyle = t.overlay;
                ctx.fillRect(o.x, o.y, o.w, o.h); ctx.globalAlpha = 1;
                ctx.fillStyle = t.overlay; ctx.font = '12px monospace';
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText(o.image_name || '(importer une image)', o.x + o.w / 2, o.y + o.h / 2);
                ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
            }
        } else {
            if (o.bg_color) {
                ctx.fillStyle = hexA(o.bg_color, (o.bg_opacity ?? 100) / 100);
                ctx.fillRect(o.x, o.y, o.w, o.h);
            }
            const txt = o.kind === 'clock' ? clockSample(o)
                : (o.text_source === 'tsl'
                    ? (_resolveTslMode(editorParams) !== 'direct'
                        ? `(${(_labelRows.find(r => r.shm === o.label_row) || {}).name || 'ligne ?'})`
                        : `(TSL #${o.tsl_index || 0})`)
                    : (o.text || ''));
            const fs = (o.font_size > 0 ? o.font_size : Math.max(8, Math.round(o.h * 0.7)));
            ctx.fillStyle = o.color || '#ffffff';
            ctx.font = `bold ${fs}px sans-serif`;
            ctx.textBaseline = 'middle';
            ctx.textAlign = o.align === 'left' ? 'left' : o.align === 'right' ? 'right' : 'center';
            const tx = o.align === 'left' ? o.x + 4 : o.align === 'right' ? o.x + o.w - 4 : o.x + o.w / 2;
            ctx.save();
            ctx.beginPath(); ctx.rect(o.x, o.y, o.w, o.h); ctx.clip();
            // Multiligne : aperçu cohérent avec le rendu baké (bloc centré verticalement).
            const lines = String(txt).split('\n');
            const lh = fs * 1.2;
            const startY = o.y + o.h / 2 - (lines.length - 1) * lh / 2;
            lines.forEach((ln, li) => ctx.fillText(ln, tx, startY + li * lh));
            ctx.restore();
            ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
        }
        ctx.strokeStyle = sel ? '#ffffff' : t.overlay;
        ctx.lineWidth = sel ? 2 : 1;
        ctx.setLineDash(sel ? [6, 4] : [4, 3]);
        ctx.strokeRect(o.x, o.y, o.w, o.h);
        ctx.setLineDash([]);
        ctx.fillStyle = t.overlay; ctx.font = '10px monospace';
        ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
        ctx.fillText(o.kind + (isBg ? ' (fond)' : ''), o.x + 3, o.y + 11);
        if (sel) {
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(o.x + o.w - HANDLE_SIZE, o.y + o.h - HANDLE_SIZE, HANDLE_SIZE, HANDLE_SIZE);
        }
        ctx.restore();
    });
}

// ── Panneau de propriétés overlay ──
function _ovSetVal(id, v) { const e = document.getElementById(id); if (e) e.value = v; }
function _ovSetChk(id, v) { const e = document.getElementById(id); if (e) e.checked = !!v; }
function _ovFillFonts() {
    const sel = document.getElementById('ov_font');
    if (!sel || sel.dataset.filled) return;
    sel.innerHTML = OVERLAY_FONTS.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
    sel.dataset.filled = '1';
}

function refreshOverlayPanel() {
    const panel = document.getElementById('ed_overlay_panel');
    if (!panel) return;
    if (selectedOverlay < 0 || !editorParams.overlays || !editorParams.overlays[selectedOverlay]) {
        panel.hidden = true; return;
    }
    const o = editorParams.overlays[selectedOverlay];
    panel.hidden = false;
    const ttl = document.getElementById('ov_title');
    if (ttl) ttl.textContent = ({text: T('plugin.multiview.ov_kind_text'), clock: T('plugin.multiview.ov_kind_clock'), image: T('plugin.multiview.ov_kind_image')})[o.kind] || T('plugin.multiview.overlay');
    panel.querySelectorAll('.ov-grp').forEach(g => {
        const kinds = (g.dataset.kind || '').split(',').filter(Boolean);
        g.hidden = kinds.length > 0 && !kinds.includes(o.kind);
    });
    _ovSetVal('ov_x', o.x); _ovSetVal('ov_y', o.y); _ovSetVal('ov_w', o.w); _ovSetVal('ov_h', o.h);
    _ovFillFonts();
    _ovSetVal('ov_font', o.font || 'dejavu-sans-bold');
    _ovSetVal('ov_font_size', o.font_size || 0);
    _ovSetVal('ov_align', o.align || 'center');
    _ovSetVal('ov_color', o.color || '#ffffff');
    _ovSetVal('ov_bg_color', o.bg_color || '#000000');
    _ovSetChk('ov_bg_on', !!o.bg_color);
    _ovSetVal('ov_bg_opacity', o.bg_opacity ?? 100);
    _ovSetVal('ov_tally_index', o.tally_index || 0);
    _ovSetVal('ov_color_on', o.color_on || '#ffd400');
    _ovSetVal('ov_bg_color_on', o.bg_color_on || '#cc0000');
    _ovSetVal('ov_text', o.text || '');
    _ovSetVal('ov_text_source', o.text_source || 'local');
    _ovSetVal('ov_tsl_index', o.tsl_index || 0);
    _tslPopulateOverlaySelects();
    _ovSetVal('ov_label_row', o.label_row || '');
    _ovSetVal('ov_label_col', o.label_col || 0);
    _ovSetVal('ov_tally_level', o.tally_level || 0);
    _ovSetVal('ov_tally_colors', o.tally_red && o.tally_green ? 'both'
                               : o.tally_red ? 'red' : o.tally_green ? 'green' : 'none');
    _ovSetVal('ov_clock_source', o.clock_source || 'ptp');
    _ovSetChk('ov_show_hh', o.show_hh !== false);
    _ovSetChk('ov_show_mm', o.show_mm !== false);
    _ovSetChk('ov_show_ss', o.show_ss !== false);
    _ovSetChk('ov_show_ff', !!o.show_ff);
    _ovSetVal('ov_offset_ms', o.offset_ms || 0);
    _ovSetVal('ov_chrono_start', o.chrono_start || '00:00:00');
    _ovSetChk('ov_cd_warn', o.cd_warn !== false);
    _ovSetVal('ov_cd_warn_orange', o.cd_warn_orange ?? 10);
    _ovSetVal('ov_cd_warn_red', o.cd_warn_red ?? 5);
    _ovSetVal('ov_cd_color_orange', o.cd_color_orange || '#ff9000');
    _ovSetVal('ov_cd_color_red', o.cd_color_red || '#ff3030');
    _ovFillTcSources(o.tc_source);
    _ovSetVal('ov_layer', o.layer || 'foreground');
    _ovSetVal('ov_fit', o.fit || 'contain');
    _ovSetVal('ov_opacity', o.opacity ?? 100);
    const mp = document.getElementById('ov_media_label');
    if (mp) mp.textContent = o.image_name || (o.image_b64 ? T('plugin.multiview.image_imported') : T('plugin.multiview.no_media'));
    const sub = (id, on) => { const e = document.getElementById(id); if (e) e.hidden = !on; };
    // Texte sourcé TSL : champs différents selon le mode tally/UMD du multiview.
    //  - Direct  : index TSL local (ov_tsl_index) + Tally On par index (ov_tally_index).
    //  - Central : ligne + colonne du tableau (texte) + niveau + couleurs (allumage).
    const tslSrc  = o.kind === 'text' && o.text_source === 'tsl';
    const central = _resolveTslMode(editorParams) !== 'direct';
    sub('ov_text_tsl_grp',     tslSrc && !central);
    sub('ov_label_row_grp',    tslSrc &&  central);
    sub('ov_label_col_grp',    tslSrc &&  central);
    sub('ov_tally_level_grp',  tslSrc &&  central);
    sub('ov_tally_colors_grp', tslSrc &&  central);
    sub('ov_tally_index_grp',  !central);   // index manuel d'allumage : Direct uniquement
    sub('ov_chrono_grp',   o.kind === 'clock' && (o.clock_source === 'chrono' || o.clock_source === 'countdown'));
    sub('ov_cdwarn_grp',   o.kind === 'clock' && o.clock_source === 'countdown');
    sub('ov_ptp_grp',      o.kind === 'clock' && o.clock_source === 'ptp');
    sub('ov_anc_grp',      o.kind === 'clock' && o.clock_source === 'anc');
}

// Peuple le sélecteur d'entrée vidéo dont l'horloge ANC lit le timecode embarqué.
function _ovFillTcSources(selected) {
    const sel = document.getElementById('ov_tc_source');
    if (!sel) return;
    const opts = (editorParams.flux_config || []).map((f, i) => {
        const nm = computeDisplayName(f) || sourceHostname(f) || T('plugin.multiview.entry_n').replace('{n}', i + 1);
        return `<option value="${i}">#${i + 1} ${escapeHtml(nm)}</option>`;
    });
    sel.innerHTML = opts.join('') || ('<option value="0">' + escapeHtml(T('plugin.multiview.no_entry_option')) + '</option>');
    sel.value = String(selected ?? 0);
}

function onOverlayChange() {
    if (selectedOverlay < 0) return;
    const o = editorParams.overlays[selectedOverlay];
    const g = id => document.getElementById(id);
    if (o.kind === 'text' || o.kind === 'clock') {
        o.font = g('ov_font').value;
        o.font_size = parseInt(g('ov_font_size').value) || 0;
        o.align = g('ov_align').value;
        o.color = g('ov_color').value;
        o.bg_color = g('ov_bg_on').checked ? g('ov_bg_color').value : '';
        o.bg_opacity = Math.max(0, Math.min(100, parseInt(g('ov_bg_opacity').value) || 100));
        o.tally_index = parseInt(g('ov_tally_index').value) || 0;
        o.color_on = g('ov_color_on').value;
        o.bg_color_on = g('ov_bg_color_on').value;
    }
    if (o.kind === 'text') {
        o.text = g('ov_text').value;
        o.text_source = g('ov_text_source').value;
        o.tsl_index = parseInt(g('ov_tsl_index').value) || 0;
        // Central : ligne + colonne du tableau (texte) + niveau + couleurs (allumage).
        o.label_row = g('ov_label_row').value || '';
        o.label_col = parseInt(g('ov_label_col').value) || 0;
        o.tally_level = parseInt(g('ov_tally_level').value) || 0;
        const tc = g('ov_tally_colors').value;
        o.tally_red   = (tc === 'red'  || tc === 'both');
        o.tally_green = (tc === 'green' || tc === 'both');
        refreshOverlayPanel();   // re-bascule la visibilité des champs (source TSL ↔ local)
    }
    if (o.kind === 'clock') {
        o.clock_source = g('ov_clock_source').value;
        o.show_hh = g('ov_show_hh').checked;
        o.show_mm = g('ov_show_mm').checked;
        o.show_ss = g('ov_show_ss').checked;
        o.show_ff = g('ov_show_ff').checked;
        o.offset_ms = parseInt(g('ov_offset_ms').value) || 0;
        o.chrono_start = g('ov_chrono_start').value || '00:00:00';
        o.tc_source = parseInt(g('ov_tc_source').value) || 0;
        o.cd_warn = g('ov_cd_warn').checked;
        o.cd_warn_orange = parseInt(g('ov_cd_warn_orange').value);
        if (isNaN(o.cd_warn_orange)) o.cd_warn_orange = 10;
        o.cd_warn_red = parseInt(g('ov_cd_warn_red').value);
        if (isNaN(o.cd_warn_red)) o.cd_warn_red = 5;
        o.cd_color_orange = g('ov_cd_color_orange').value;
        o.cd_color_red = g('ov_cd_color_red').value;
    }
    if (o.kind === 'image') {
        o.layer = g('ov_layer').value;
        o.fit = g('ov_fit').value;
        o.opacity = Math.max(0, Math.min(100, parseInt(g('ov_opacity').value) || 100));
    }
    dessiner();
    hotApplyFull();
}

// Aperçu local pendant la saisie (pas de déploiement à chaque frappe ; le hot-apply
// se fait au blur via onchange → onOverlayChange).
function onOverlayTextInput() {
    if (selectedOverlay < 0) return;
    const o = editorParams.overlays[selectedOverlay];
    if (o.kind !== 'text') return;
    const e = document.getElementById('ov_text');
    o.text = e ? e.value : o.text;
    dessiner();
}

function onOverlayGeomChange() {
    if (selectedOverlay < 0) return;
    const o = editorParams.overlays[selectedOverlay];
    const g = id => document.getElementById(id);
    o.x = parseInt(g('ov_x').value) || 0;
    o.y = parseInt(g('ov_y').value) || 0;
    let nw = parseInt(g('ov_w').value) || 16;
    let nh = parseInt(g('ov_h').value) || 16;
    o.w = nw % 2 === 0 ? nw : nw - 1;
    o.h = nh % 2 === 0 ? nh : nh - 1;
    dessiner();
    hotApplyFull();
}

function chronoAction(action) {
    if (selectedOverlay < 0 || editorVmid === null) return;
    const o = editorParams.overlays[selectedOverlay];
    if (!o || o.kind !== 'clock') return;
    if (action === 'start') o.chrono_running = true;
    if (action === 'stop')  o.chrono_running = false;
    if (action === 'reset') o.chrono_running = false;  // raz = retour départ + arrêt
    fetch(`/api/containers/${editorVmid}/plugin/chrono`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ id: o.id, action })
    }).then(() => {
        const lbl = ({start: T('plugin.multiview.flash_chrono_start'),
                      stop: T('plugin.multiview.flash_chrono_stop'),
                      reset: T('plugin.multiview.flash_chrono_reset')})[action] || action;
        mwFlash(lbl);
    }).catch(() => mwFlash(T('plugin.multiview.flash_chrono_unreachable')));
}

// ── Import d'image depuis l'ordinateur (overlay image) ──
// Le fichier est lu et réduit côté navigateur (côté le plus long ≤ résolution de sortie),
// puis stocké en base64 PNG dans l'overlay (autonome : pas de stockage serveur).
function _downscaleToB64(file, maxSide, cb) {
    const reader = new FileReader();
    reader.onload = () => {
        const img = new Image();
        img.onload = () => {
            const w = img.naturalWidth, h = img.naturalHeight;
            const scale = Math.min(1, maxSide / Math.max(w, h));
            const cw = Math.max(1, Math.round(w * scale));
            const ch = Math.max(1, Math.round(h * scale));
            const cv = document.createElement('canvas');
            cv.width = cw; cv.height = ch;
            cv.getContext('2d').drawImage(img, 0, 0, cw, ch);
            cb((cv.toDataURL('image/png').split(',')[1]) || '', file.name);
        };
        img.onerror = () => cb('', file.name);
        img.src = reader.result;
    };
    reader.onerror = () => cb('', file.name);
    reader.readAsDataURL(file);
}

function importOverlayImage(input) {
    if (selectedOverlay < 0) return;
    const file = input.files && input.files[0];
    if (!file) return;
    const o = editorParams.overlays[selectedOverlay];
    const maxSide = Math.max(editorParams.out_width, editorParams.out_height, 1280);
    _downscaleToB64(file, maxSide, (b64, name) => {
        o.image_b64 = b64;
        o.image_name = name;
        input.value = '';   // autorise la réimportation du même fichier
        dessiner();
        hotApplyFull();
    });
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
    // Sérialise les déploiements : un appel pendant un POST en cours est rejoué à la
    // fin (jamais deux deploys concurrents, jamais un dernier changement perdu).
    if (deployerEditor._busy) { deployerEditor._pending = true; return; }
    deployerEditor._busy = true;

    // Lit les champs globaux au cas où ils n'auraient pas déclenché onchange
    editorParams.max_inputs    = parseInt(document.getElementById('ed_max').value) || editorParams.max_inputs;
    editorParams.border_w      = parseInt(document.getElementById('ed_border_w').value) || 0;
    editorParams.border_color  = document.getElementById('ed_border_color').value;
    editorParams.overlay_below = document.getElementById('ed_overlay_below').checked;
    editorParams.label_size    = Math.max(6, parseInt(document.getElementById('ed_label_size').value) || 14);
    { const fs = document.getElementById('ed_frame_style'); if (fs) editorParams.frame_style = fs.value; }
    editorParams.show_no_signal  = document.getElementById('ed_show_no_signal').checked;
    editorParams.freeze_detect_s = Math.max(0, parseFloat(document.getElementById('ed_freeze_detect').value) || 0);
    editorParams.show_format     = document.getElementById('ed_show_format').checked;
    editorParams.show_proxy      = document.getElementById('ed_show_proxy').checked;
    editorParams.genlock = document.getElementById('ed_genlock').checked;
    { const _ft = document.getElementById('ed_fps_target'); if (_ft) editorParams.fps_target = parseInt(_ft.value) || 0; }
    const tslPortEl = document.getElementById('ed_tsl_port');
    if (tslPortEl) editorParams.tsl_port = parseInt(tslPortEl.value) || 0;
    { const _tm = document.getElementById('ed_tsl_mode'); if (_tm) editorParams.tsl_mode = _tm.value; }
    padBank();   // max_inputs a pu changer → complète la banque

    const flux_config = editorParams.flux_config.map(f => ({
        path: f.path,
        hidden: !!f.hidden,
        label_source: f.label_source || 'hostname',
        name: computeDisplayName(f),
        in_w: f.in_w,
        in_h: f.in_h,
        x: f.x, y: f.y,
        w: f.w % 2 === 0 ? f.w : f.w - 1,
        h: f.h % 2 === 0 ? f.h : f.h - 1,
        show_label: !!f.show_label,
        show_tally: !!f.show_tally,
        label_proportional: !!f.label_proportional,
        tsl_index: parseInt(f.tsl_index) || 0,
        label_col: parseInt(f.label_col) || 0,
        tally_level: parseInt(f.tally_level) || 0,
        tally_red: !!f.tally_red,
        tally_green: !!f.tally_green,
        // Peak meters audio
        meter_channels: parseInt(f.meter_channels) || 0,
        meter_position: f.meter_position || 'right',
        meter_inside:   !!f.meter_inside,
        meter_opacity:  Math.max(10, Math.min(100, parseInt(f.meter_opacity) || 70)),
        meter_scale:    f.meter_scale || 'dbfs',
    }));

    const params = {
        flux_config,
        overlays:      serializeOverlays(),
        shm_out:       editorParams.shm_out,
        out_width:     editorParams.out_width,
        out_height:    editorParams.out_height,
        orientation:   editorParams.orientation || 'landscape',
        fps:           editorParams.fps,
        scan:          editorParams.scan || 'p',
        border_w:      editorParams.border_w,
        border_color:  editorParams.border_color,
        overlay_below: editorParams.overlay_below,
        label_size:    editorParams.label_size,
        frame_style:   editorParams.frame_style || 'none',
        show_no_signal:  editorParams.show_no_signal !== false,
        freeze_detect_s: editorParams.freeze_detect_s ?? 2,
        show_format:     !!editorParams.show_format,
        show_proxy:      !!editorParams.show_proxy,
        max_inputs:    editorParams.max_inputs,
        genlock:       editorParams.genlock,
        fps_target:    editorParams.fps_target || 0,
        tsl_mode:      editorParams.tsl_mode || 'central',
        tsl_port:      editorParams.tsl_port ?? 4801
    };

    const btn = document.getElementById('ed_deploy_btn');
    if (btn) { btn.disabled = true; btn.textContent = T('plugin.multiview.deploying'); }
    let r = null;
    try {
        r = await fetch('/api/containers/' + editorVmid + '/deploy', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({type: 'multiview', params, path: '/opt/script/main.py'})
        });
    } catch(e) {}
    if (btn) { btn.disabled = false; btn.textContent = T('plugin.multiview.deploy'); }
    if (r && r.ok) {
        if (btn) {
            btn.textContent = T('plugin.multiview.deployed');
            clearTimeout(deployerEditor._t);
            deployerEditor._t = setTimeout(() => { btn.textContent = T('plugin.multiview.deploy'); }, 1500);
        }
        // Refresh la sidebar générique (badge version + mini-aperçu) après déploiement
        if (window.tpLoadInstances) setTimeout(window.tpLoadInstances, 800);
    } else {
        mwFlash(T('plugin.multiview.flash_deploy_failed'));
    }
    deployerEditor._busy = false;
    if (deployerEditor._pending) { deployerEditor._pending = false; deployerEditor(); }
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
    selectedIdxs.forEach(i => hotApplyWindow(i));
}

function matchSize(mode) {
    if (!editorParams || selectedIdxs.length < 2) {
        mwFlash(T('plugin.multiview.flash_select2'));
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
    selectedIdxs.forEach(i => hotApplyWindow(i));
}

function distribuer(axis) {
    if (!editorParams || selectedIdxs.length < 3) {
        mwFlash(T('plugin.multiview.flash_select3'));
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
    selectedIdxs.forEach(i => hotApplyWindow(i));
}

// Remplir : les fenêtres sélectionnées se TUILENT le long d'un axe (largeur en parts
// égales, ou hauteur), dans leur ordre courant. L'AUTRE dimension suit le RATIO de
// chaque fenêtre (pas de déformation) ; la fenêtre est centrée sur cet autre axe et,
// si le ratio la ferait déborder, elle est réduite pour rester dans l'image.
// Ex. 4 PiP (un à gauche, un à droite, deux au centre) → 4 colonnes égales, chacune
// à la bonne hauteur 16:9 et centrée verticalement.
function remplir(axis) {
    if (!editorParams || selectedIdxs.length < 1) return;
    const fc = editorParams.flux_config;
    const out_w = editorParams.out_width, out_h = editorParams.out_height;
    const items = selectedIdxs.map(i => fc[i]);
    const n = items.length;
    const evDim = v => Math.max(2, Math.round(v) & ~1);   // dimension paire ≥ 2
    const evPos = v => Math.max(0, Math.round(v) & ~1);   // position paire ≥ 0
    const ratioOf = f => (f.ratio && f.ratio > 0) ? f.ratio
                       : (f.in_w && f.in_h) ? f.in_w / f.in_h
                       : (f.h ? f.w / f.h : 16 / 9);
    if (axis === 'h') {
        items.sort((a, b) => a.x - b.x);
        const colW = Math.max(2, Math.floor(out_w / n) & ~1);
        items.forEach((f, k) => {
            const r = ratioOf(f);
            f.x = k * colW;
            let w = (k === n - 1) ? (out_w - k * colW) : colW;
            let h = w / r;
            if (h > out_h) { h = out_h; w = h * r; }      // ne pas déborder en hauteur
            f.w = evDim(w); f.h = evDim(h);
            if (f.x + f.w > out_w) f.x = out_w - f.w;
            f.y = evPos((out_h - f.h) / 2);               // centré verticalement
        });
    } else {
        items.sort((a, b) => a.y - b.y);
        const rowH = Math.max(2, Math.floor(out_h / n) & ~1);
        items.forEach((f, k) => {
            const r = ratioOf(f);
            f.y = k * rowH;
            let h = (k === n - 1) ? (out_h - k * rowH) : rowH;
            let w = h * r;
            if (w > out_w) { w = out_w; h = w / r; }       // ne pas déborder en largeur
            f.w = evDim(w); f.h = evDim(h);
            if (f.y + f.h > out_h) f.y = out_h - f.h;
            f.x = evPos((out_w - f.w) / 2);               // centré horizontalement
        });
    }
    dessiner();
    selectedIdxs.forEach(i => hotApplyWindow(i));
}

// ─── Modèles de grille (templates) ───────────────────────────
// Géométries prédéfinies appliquées aux fenêtres de la banque DANS L'ORDRE des
// indices (sources/labels/meters/tsl conservés, comme appliquerLayout) ; le
// surplus de banque est masqué (source câblée préservée). Cellules en coordonnées
// de grille [c0, r0, c1, r1] sur une base cols×rows.

function _gridCells(n) {
    const cells = [];
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) cells.push([c, r, c + 1, r + 1]);
    return {cols: n, rows: n, cells};
}

const MW_TEMPLATES = {
    '1x1': _gridCells(1),
    '2x2': _gridCells(2),
    '3x3': _gridCells(3),
    '4x4': _gridCells(4),
    '1+5': {cols: 3, rows: 3, cells: [
        [0,0,2,2], [2,0,3,1], [2,1,3,2], [0,2,1,3], [1,2,2,3], [2,2,3,3]]},
    '1+7': {cols: 4, rows: 4, cells: [
        [0,0,3,3], [3,0,4,1], [3,1,4,2], [3,2,4,3], [0,3,1,4], [1,3,2,4], [2,3,3,4], [3,3,4,4]]},
    '2+8': {cols: 4, rows: 4, cells: [
        [0,0,2,2], [2,0,4,2],
        [0,2,1,3], [1,2,2,3], [2,2,3,3], [3,2,4,3],
        [0,3,1,4], [1,3,2,4], [2,3,3,4], [3,3,4,4]]},
};

function appliquerTemplate(key) {
    const tpl = MW_TEMPLATES[key];
    if (!tpl || !editorParams) return;
    padBank();
    const ow = editorParams.out_width, oh = editorParams.out_height;
    // Bords de grille partagés, arrondis pairs : pas de trous ni de chevauchements,
    // dimensions paires garanties (alignement chroma, même contrainte que deployerEditor).
    const ex = c => (c >= tpl.cols) ? (ow & ~1) : (Math.round(ow * c / tpl.cols) & ~1);
    const ey = r => (r >= tpl.rows) ? (oh & ~1) : (Math.round(oh * r / tpl.rows) & ~1);
    const fc = editorParams.flux_config || [];
    const n = Math.min(tpl.cells.length, fc.length);
    for (let i = 0; i < fc.length; i++) {
        if (i < n) {
            const [c0, r0, c1, r1] = tpl.cells[i];
            fc[i].x = ex(c0); fc[i].y = ey(r0);
            fc[i].w = ex(c1) - ex(c0); fc[i].h = ey(r1) - ey(r0);
            fc[i].hidden = false;
        } else {
            fc[i].hidden = true;   // hors image, source câblée conservée
        }
    }
    if (tpl.cells.length > fc.length) {
        mwFlash(T('plugin.multiview.flash_template_short').replace('{name}', key).replace('{n}', n));
    }
    selectedIdxs = [];
    selectedOverlay = -1;
    dessiner();
    hotApplyFull();   // /reconfigure atomique à chaud (géométrie + hidden de toute la banque)
}

// ─── Layouts (presets) ───────────────────────────────────────

let savedLayouts = [];

async function rafraichirListeLayouts() {
    const ul = document.getElementById('layout-saved-list');
    try {
        const r = await fetch('/api/layouts');
        if (!r.ok) throw new Error('HTTP ' + r.status);
        savedLayouts = await r.json();
    } catch(e) {
        savedLayouts = [];
        if (ul) ul.innerHTML = '<li class="meta">' + escapeHtml(T('plugin.multiview.layouts_unavailable')) + '</li>';
        return;
    }
    if (!ul) return;
    if (savedLayouts.length === 0) {
        ul.innerHTML = '<li class="meta">' + escapeHtml(T('plugin.multiview.layouts_empty')) + '</li>';
        return;
    }
    ul.innerHTML = savedLayouts.map(l => `
        <li>
            <div><b>${escapeHtml(l.name)}</b></div>
            <div class="meta">${escapeHtml(l.created_at || '')} · ${(l.config.flux_config || []).length} ${escapeHtml(T('plugin.multiview.entries'))}</div>
            <canvas data-layout-preview="${l.id}"></canvas>
            <div class="actions">
                <button class="btn btn-blue" onclick="appliquerLayout(${l.id})">${escapeHtml(T('plugin.multiview.apply'))}</button>
                <button class="btn" onclick="exporterLayout(${l.id})" title="${escapeHtml(T('plugin.multiview.export_title'))}">${escapeHtml(T('plugin.multiview.export'))}</button>
                <button class="btn btn-red" onclick="supprimerLayout(${l.id})">${escapeHtml(T('plugin.multiview.delete'))}</button>
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

async function enregistrerLayout() {
    const nameEl = document.getElementById('layout-save-name');
    const name = (nameEl.value || '').trim();
    if (!name) { mwFlash(T('plugin.multiview.flash_layout_name')); return; }
    if (!editorParams) {
        mwFlash(T('plugin.multiview.flash_select_mv_first'));
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
        // Seuls les PiP AFFICHÉS font partie du layout (les entrées masquées de la banque, non).
        flux_config:   (editorParams.flux_config || []).filter(f => !f.hidden).map(f => ({
            label_source: f.label_source || 'hostname',
            x: f.x, y: f.y, w: f.w, h: f.h,
            show_label: !!f.show_label,
            show_tally: !!f.show_tally,
            label_proportional: !!f.label_proportional,
            label_col: f.label_col ?? 0,
            tally_level: f.tally_level ?? 0,
            tally_red: !!f.tally_red,
            tally_green: !!f.tally_green
        }))
    };
    let r = null;
    try {
        r = await fetch('/api/layouts', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, config})
        });
    } catch(e) {}
    if (r && r.ok) {
        nameEl.value = '';
        mwFlash(T('plugin.multiview.flash_layout_saved').replace('{name}', () => name));
        rafraichirListeLayouts();
    } else {
        mwFlash(T('plugin.multiview.flash_layout_save_failed'));
    }
}

function appliquerLayout(lid) {
    if (!editorParams) {
        mwFlash(T('plugin.multiview.flash_select_mv_first'));
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
    const applied = (cfg.flux_config || []).map((f, i) => {
        const prev = existing[i] || {};
        return Object.assign({
            color: COLORS[i % COLORS.length],
            path: prev.path || '',
            in_w: 0,
            in_h: 0,
            ratio: 16/9
        }, f, {hidden: false});
    });
    // Entrées existantes au-delà du layout : conservées MASQUÉES (sources préservées).
    for (let i = applied.length; i < existing.length; i++) {
        applied.push(Object.assign({}, existing[i], {hidden: true}));
    }
    editorParams.flux_config = applied;
    padBank();
    selectedIdxs = [];
    const hostnameEl = document.getElementById('ed_hostname');
    const hostname = hostnameEl ? hostnameEl.textContent.trim() : '';
    renderEditor(hostname);
    deployerEditor();
}

async function supprimerLayout(lid) {
    const l = savedLayouts.find(x => x.id === lid);
    if (!confirm(T('plugin.multiview.confirm_delete_layout').replace('{name}', () => (l ? l.name : lid)))) return;
    let r = null;
    try { r = await fetch('/api/layouts/' + lid, {method: 'DELETE'}); } catch(e) {}
    if (r && r.ok) rafraichirListeLayouts();
    else mwFlash(T('plugin.multiview.flash_layout_delete_failed'));
}

// ─── Import / export de layouts (fichier .json {name, config}) ──

function importerLayout() {
    const inp = document.getElementById('layout-import-input');
    if (inp) inp.click();
}

function onImportLayoutFile(input) {
    const file = input.files && input.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onerror = () => { input.value = ''; mwFlash(T('plugin.multiview.flash_file_read_failed')); };
    reader.onload = async () => {
        input.value = '';   // autorise la réimportation du même fichier
        let data = null;
        try { data = JSON.parse(reader.result); } catch(e) {}
        const config = data && data.config;
        const name = ((data && data.name) || file.name.replace(/\.json$/i, '')).trim();
        if (!config || !Array.isArray(config.flux_config)) {
            mwFlash(T('plugin.multiview.flash_invalid_layout_file'));
            return;
        }
        let r = null;
        try {
            r = await fetch('/api/layouts', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name, config})
            });
        } catch(e) {}
        if (r && r.ok) {
            mwFlash(T('plugin.multiview.flash_layout_imported').replace('{name}', () => name));
            rafraichirListeLayouts();
        } else {
            mwFlash(T('plugin.multiview.flash_layout_import_failed'));
        }
    };
    reader.readAsText(file);
}

function exporterLayout(lid) {
    const l = savedLayouts.find(x => x.id === lid);
    if (!l) return;
    const blob = new Blob([JSON.stringify({name: l.name, config: l.config}, null, 2)],
                          {type: 'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (l.name || 'layout').replace(/[^\w.à-üÀ-Ü-]+/g, '_') + '.json';
    a.click();
    URL.revokeObjectURL(a.href);
}

// Le montage est piloté par le shell générique Traitements (tpLoadInstances → tpMount →
// MXLPlugins.multiview.mount(el, vmid)). Pas d'init autonome ici.
