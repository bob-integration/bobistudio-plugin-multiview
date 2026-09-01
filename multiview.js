// ─── i18n ─────────────────────────────────────────────────────
// Catalogue plugin.multiview.* (plugins/multiview/i18n/<lang>.json, fusionné par
// app/i18n.py ; préfixe `plugin.` exposé côté JS par js_catalog → window.t()).
// `window.t` rend la CLÉ BRUTE quand elle manque du catalogue. Écrire ce retour tel quel dans le
// DOM remplacerait le français du gabarit par « plugin.multiview.fps_target » à l'écran — vécu le
// 2026-08-21 (catalogues écrasés par une écriture concurrente). On teste donc le retour.
const T = (k, repli) => {
    const v = window.t ? window.t(k) : k;
    return (v && v !== k) ? v : (repli !== undefined ? repli : k);
};

// Traduit le HTML statique du composer (injecté par le shell Traitements) :
// data-i18n (textContent), data-i18n-title, data-i18n-aria, data-i18n-ph.
function mwApplyI18n(root) {
    const r = root || document;
    // Le gabarit porte le français EN DUR : on ne l'écrase que si la clé est RÉSOLUE (2e argument
    // = ce qui est déjà à l'écran), sinon une clé manquante afficherait son propre nom.
    r.querySelectorAll('[data-i18n]').forEach(n => { n.textContent = T(n.dataset.i18n, n.textContent); });
    r.querySelectorAll('[data-i18n-title]').forEach(n => { n.title = T(n.dataset.i18nTitle, n.title); });
    r.querySelectorAll('[data-i18n-aria]').forEach(n => { n.setAttribute('aria-label', T(n.dataset.i18nAria, n.getAttribute('aria-label'))); });
    r.querySelectorAll('[data-i18n-ph]').forEach(n => { n.placeholder = T(n.dataset.i18nPh, n.placeholder); });
}

// ─── État global ──────────────────────────────────────────────

let allContainers = [];
let videoSources = [];   // sorties vidéo individuelles de la flotte (cf. loadVideoSources)
let audioSources = [];   // sorties audio individuelles de la flotte (cf. loadAudioSources)
let editorVmid    = null;
let editorParams  = null;   // {flux_config, shm_out, out_width, out_height, max_inputs, default_template…}
let selectedIdxs  = [];     // multi-select : le dernier est le primary (référence pour align/match-size)
let dragMode      = null;   // 'move' | 'resize'
let dragStart     = null;
let dragOrigRect  = null;
let dragGroupOrig = [];     // [{j, x, y}] positions d'origine de TOUTES les fenêtres sélectionnées
                            // (déplacement de groupe : même delta appliqué à chacune)
let dragGeomOrig  = null;   // {idx: {x,y,w,h}} géométrie au DÉBUT du geste — sert au relâchement à
                            // ne hot-appliquer que les fenêtres réellement modifiées (cf.
                            // canvasMouseUp). null = aucun geste en cours sur une fenêtre.
let snapEnabled   = true;
let snapGuides    = [];     // [{type:'v'|'h', pos:number}] dessinées pendant le drag

// Overlays (texte/horloge/image) : objets visuels séparés de flux_config (non câblés).
let selectedOverlay = -1;   // index dans editorParams.overlays, ou -1
let dragOverlay     = false; // true pendant un drag/resize d'overlay (réutilise dragMode/dragStart)
const _ovThumbCache = {};   // clé "slug|path" → Image (vignette média pour l'aperçu canvas)

// Blocs VU-mètres de MUR (meter_blocks) : posés en coordonnées ABSOLUES du canvas (comme les
// overlays), CÂBLÉS (source audio propre, page Câbles) mais indépendants de toute fenêtre —
// contrairement aux overlays. Édités en PIXELS canvas (comme les overlays) ; converti en
// fractions 0..1 du mur SEULEMENT à la sérialisation (deployerEditor), cf. spec meter_blocks.
let selectedBlock = -1;     // index dans editorParams.meter_blocks, ou -1
let dragBlock      = false; // true pendant un drag/resize de bloc (réutilise dragMode/dragStart)

// Blocs d'HISTORIQUE de MUR (frise vidéo / frise audio) : mêmes conventions que les blocs
// VU-mètres (pixels canvas à l'édition → fractions 0..1 du mur à la sérialisation, source
// câblée en propre). Deux listes distinctes (essences différentes côté câblage) manipulées
// par UN SEUL jeu de fonctions, paramétrées par `kind` ('video' | 'audio').
let selectedHist     = -1;        // index dans la liste du kind courant, ou -1
let selectedHistKind = 'video';   // 'video' | 'audio' — liste visée par selectedHist
let dragHist         = false;
const HIST_LIST = {video: 'video_history_blocks', audio: 'audio_history_blocks'};

// ─── SÉLECTION INTER-FAMILLES (0.111.0) ──────────────────────────────────────
// Historique : chaque famille d'objets avait SA variable de sélection — `selectedIdxs` (fenêtres,
// multiple), `selectedOverlay`, `selectedBlock`, `selectedHist`+`selectedHistKind` (une seule à la
// fois, et sélectionner l'une effaçait les autres). Conséquence : impossible d'aligner un texte
// SUR une fenêtre, ni de distribuer trois horloges — alors que c'est le geste naturel d'un mur.
//
// `mwSel` devient la SOURCE DE VÉRITÉ : une liste ordonnée de références {k, i}, la RÉFÉRENCE
// (primary) en DERNIER — même convention que la multi-sélection de fenêtres, et que l'aide
// affichée sous le canvas. Les quatre variables historiques sont recalculées à chaque changement
// par `_selSync()` : elles ne désignent plus que le PRIMAIRE. Tout le code existant (panneaux,
// poignée de redimensionnement, drag, dessin, hot-apply) continue donc de fonctionner sans être
// réécrit — il lit simplement « l'objet courant » là où il le lisait déjà.
//   k : 'win' (fenêtre vidéo) | 'ov' (overlay) | 'blk' (bloc VU) | 'vh' / 'ah' (frises)
// Ce sont les MÊMES clés que `_snapRects` : un aimant et une sélection parlent enfin la même langue.
let mwSel = [];

const _SEL_LIST = {
    win: () => (editorParams && editorParams.flux_config) || [],
    ov:  () => (editorParams && editorParams.overlays) || [],
    blk: () => (editorParams && editorParams.meter_blocks) || [],
    vh:  () => (editorParams && editorParams.video_history_blocks) || [],
    ah:  () => (editorParams && editorParams.audio_history_blocks) || [],
};
// Objet visé par une référence, ou null si elle est périmée (liste raccourcie entre-temps).
function _refObj(r) { return r ? (_SEL_LIST[r.k] ? _SEL_LIST[r.k]()[r.i] : null) || null : null; }
function _refKey(r) { return r.k + ':' + r.i; }
function _selPrimary()  { return mwSel.length ? mwSel[mwSel.length - 1] : null; }
function _selHas(k, i)  { return mwSel.some(r => r.k === k && r.i === i); }
function _selCount()    { return mwSel.length; }
// Références encore valides (un objet supprimé sort de la sélection tout seul).
function _selRefs()     { return mwSel.filter(r => _refObj(r)); }

// Les quatre variables historiques = VUE du primaire (et, pour les fenêtres, de tous les membres :
// leur multi-sélection existait déjà et le reste du code s'en sert).
function _selSync() {
    mwSel = _selRefs();
    selectedIdxs = mwSel.filter(r => r.k === 'win').map(r => r.i);
    const p = _selPrimary();
    selectedOverlay = (p && p.k === 'ov')  ? p.i : -1;
    selectedBlock   = (p && p.k === 'blk') ? p.i : -1;
    if (p && (p.k === 'vh' || p.k === 'ah')) {
        selectedHist = p.i;
        selectedHistKind = (p.k === 'vh') ? 'video' : 'audio';
    } else {
        selectedHist = -1;
    }
}
// Clic simple = sélection unique ; Maj+clic = bascule dans la sélection, l'objet cliqué devenant
// la nouvelle référence (il passe en dernier) — exactement la règle déjà appliquée aux fenêtres.
function _selSet(k, i, additive) {
    if (!additive) { mwSel = [{k, i}]; _selSync(); return; }
    const at = mwSel.findIndex(r => r.k === k && r.i === i);
    if (at >= 0) mwSel.splice(at, 1);   // Maj+clic sur un objet déjà pris = le RETIRER
    else mwSel.push({k, i});
    _selSync();
}
// Idem, mais un objet DÉJÀ sélectionné qu'on saisit sans Maj ne réduit pas la sélection : on le
// promeut en référence et on garde le groupe (sinon impossible de déplacer un groupe à la souris).
function _selGrab(k, i, additive) {
    if (!additive && _selHas(k, i)) {
        mwSel = mwSel.filter(r => !(r.k === k && r.i === i));
        mwSel.push({k, i});
        _selSync();
        return;
    }
    _selSet(k, i, additive);
}
function _selClear() { mwSel = []; _selSync(); }
// Une famille a perdu un élément d'indice `i` (suppression) : les refs au-delà se décalent.
function _selDrop(k, i) {
    mwSel = mwSel.filter(r => !(r.k === k && r.i === i))
                 .map(r => (r.k === k && r.i > i) ? {k, i: r.i - 1} : r);
    _selSync();
}
// ─── Déplacement de GROUPE, toutes familles ──────────────────────────────────
// `dragGroupOrig` mémorise l'origine de CHAQUE membre au début du geste : [{r, x, y}]. Le delta
// est calculé sur l'objet réellement tiré (aimant compris), borné pour qu'aucun membre ne sorte
// du cadre, puis appliqué à tous depuis leur origine — jamais en cumulant, sinon le groupe dérive.
function _beginGroupDrag() {
    dragGroupOrig = _selRefs().map(r => {
        const o = _refObj(r);
        return {r, x: o.x, y: o.y};
    });
}
function _applyGroupMove(nx, ny) {
    const ow = editorParams.out_width, oh = editorParams.out_height;
    let ddx = nx - dragOrigRect.x, ddy = ny - dragOrigRect.y;
    let loX = -Infinity, hiX = Infinity, loY = -Infinity, hiY = Infinity;
    dragGroupOrig.forEach(g => {
        const o = _refObj(g.r);
        if (!o) return;
        loX = Math.max(loX, -g.x); hiX = Math.min(hiX, ow - o.w - g.x);
        loY = Math.max(loY, -g.y); hiY = Math.min(hiY, oh - o.h - g.y);
    });
    // Groupe plus large (ou plus haut) que le cadre : aucune position ne satisfait tout le monde
    // → on fige l'axe concerné plutôt que d'appliquer une borne inversée.
    if (loX > hiX) loX = hiX = 0;
    if (loY > hiY) loY = hiY = 0;
    ddx = Math.max(loX, Math.min(hiX, ddx));
    ddy = Math.max(loY, Math.min(hiY, ddy));
    dragGroupOrig.forEach(g => {
        const o = _refObj(g.r);
        if (!o) return;
        o.x = Math.round(g.x + ddx);
        o.y = Math.round(g.y + ddy);
    });
}
// Le geste a-t-il RÉELLEMENT modifié la géométrie ? Un simple clic de SÉLECTION emprunte le même
// chemin qu'un déplacement (mousedown → mouseup) : sans cette question, il déclencherait un
// déploiement — et, sur un mur shardé, une re-planification du tissu donc une coupure de sortie,
// pour un clic qui n'a rien changé. La garde existait pour les fenêtres (`dragGeomOrig`) ; elle
// manquait aux overlays, blocs et frises, dont chaque clic postait un deploy complet.
function _gestureChanged() {
    if (dragGroupOrig.some(g => {
        const o = _refObj(g.r);
        return o && (o.x !== g.x || o.y !== g.y);
    })) return true;
    const o = _refObj(_selPrimary());
    return !!(o && dragOrigRect && (o.w !== dragOrigRect.w || o.h !== dragOrigRect.h));
}

// Cibles d'aimant à EXCLURE pendant un geste : tous les membres du groupe (un objet ne doit pas
// s'aimanter sur ses propres compagnons, qui se déplacent avec lui).
function _dragSkip(k, i) {
    return dragGroupOrig.length ? dragGroupOrig.map(g => ({kind: g.r.k, i: g.r.i})) : [{kind: k, i}];
}

// Fenêtre PRIMAIRE au sens strict : -1 si la référence courante n'est pas une fenêtre. Sert à
// n'afficher qu'UN cadre blanc et QU'UNE poignée de redimensionnement sur tout le mur.
function _winPrimary() {
    const p = _selPrimary();
    return (p && p.k === 'win') ? p.i : -1;
}

// ─── ÉDITION À PLUSIEURS : verrou consultatif + garde de révision ────────────
// Le composer poste TOUT l'état du mur à chaque geste : sans garde-fou, deux personnes s'écrasent
// l'une l'autre sans jamais savoir pourquoi leur travail disparaît (« l'image de A disparaît dès
// que B modifie quelque chose », 2026-08-11). Deux protections, volontairement distinctes :
//   • le VERROU (app/edit_lock.py) dit QUI a la main, et laisse reprendre la main sciemment ;
//   • la RÉVISION (`base_rev` du déploiement) refuse un envoi bâti sur un état périmé — elle
//     couvre aussi ce qui ne passe pas par ici (page Câbles, macros, restauration de projet).
// Le verrou n'interdit rien côté serveur : c'est un outil de conversation, pas de sécurité.
let mwRev        = null;    // révision de deploy_config sur laquelle notre édition est bâtie
let mwVerrou     = {a_moi: true, user_name: '', depuis_s: 0, libre: true};
let _mwBattement = null;    // minuterie du battement de cœur

function mwLectureSeule() { return !mwVerrou.a_moi; }

// Un geste d'écriture est-il permis ? Sert de garde AUX POINTS D'ÉCRITURE (et pas seulement
// dans l'UI) : un raccourci clavier, une macro ou un appel resté en vol ne doivent pas passer
// derrière le bandeau.
function mwPeutEcrire() {
    if (mwLectureSeule()) {
        mwFlash(T('plugin.multiview.lock_readonly_flash').replace('{nom}', mwVerrou.user_name || '?'));
        return false;
    }
    return true;
}

function mwMajVerrouUI() {
    const box = document.getElementById('mw_verrou');
    const txt = document.getElementById('mw_verrou_txt');
    const ed  = document.getElementById('mw-editor');
    if (ed) ed.classList.toggle('lecture-seule', mwLectureSeule());
    if (!box || !txt) return;
    box.hidden = !mwLectureSeule();
    if (mwLectureSeule()) {
        const min = Math.max(1, Math.round((mwVerrou.depuis_s || 0) / 60));
        txt.textContent = T('plugin.multiview.lock_held_by')
            .replace('{nom}', mwVerrou.user_name || '?')
            .replace('{min}', String(min));
    }
}

async function mwVerrouPrendre(force) {
    if (editorVmid === null) return;
    try {
        const r = await fetch(`/api/containers/${editorVmid}/edit-lock`, {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({force: !!force})
        });
        const j = await r.json();
        mwVerrou = j;
        if (j.rev !== undefined && mwRev === null) mwRev = j.rev;
    } catch (e) {
        // Réseau perdu : on ne bascule PAS en lecture seule (ce serait punir l'utilisateur pour
        // une panne côté serveur) ; le prochain battement rétablira la vérité.
    }
    mwMajVerrouUI();
}

function mwPrendreLaMain() { mwVerrouPrendre(true); }

// Conflit d'écriture : quelqu'un d'autre a modifié ce mur depuis qu'on l'a ouvert. Notre envoi
// n'est pas parti. On bascule l'éditeur en lecture seule le temps de la décision — continuer à
// éditer une copie périmée ne produirait que des refus en série — et on propose de recharger.
function mwConflit(par) {
    const box = document.getElementById('mw_verrou');
    const txt = document.getElementById('mw_verrou_txt');
    const btn = document.getElementById('mw_verrou_btn');
    if (box && txt && btn) {
        box.hidden = false;
        txt.textContent = par
            ? T('plugin.multiview.lock_conflict_by').replace('{nom}', par)
            : T('plugin.multiview.lock_conflict');
        btn.textContent = T('plugin.multiview.lock_reload');
        btn.onclick = () => { const v = editorVmid; btn.onclick = mwPrendreLaMain; chargerMw(v); };
    }
    const ed = document.getElementById('mw-editor');
    if (ed) ed.classList.add('lecture-seule');
    mwFlash(T('plugin.multiview.lock_conflict'));
}

function mwVerrouRendre() {
    if (_mwBattement) { clearInterval(_mwBattement); _mwBattement = null; }
    const vmid = editorVmid;
    if (vmid === null) return;
    // `keepalive` : la requête survit à la fermeture de l'onglet ou au changement de page.
    try {
        fetch(`/api/containers/${vmid}/edit-lock`, {method: 'DELETE', keepalive: true}).catch(() => {});
    } catch (e) {}
}

function mwVerrouDemarrer(battement_s) {
    if (_mwBattement) clearInterval(_mwBattement);
    const p = Math.max(5, parseInt(battement_s) || 20) * 1000;
    // Le battement RENOUVELLE notre verrou et, quand on ne l'a pas, RÉINTERROGE : un mur libéré
    // par l'autre (fermeture d'onglet, expiration) redevient éditable sans recharger la page.
    _mwBattement = setInterval(() => {
        if (editorVmid === null) return;
        mwVerrouPrendre(false);
    }, p);
}

function histList(kind) {
    const k = HIST_LIST[kind || selectedHistKind];
    editorParams[k] = editorParams[k] || [];
    return editorParams[k];
}

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

// Entier d'un champ, où ZÉRO est une valeur LÉGITIME. `parseInt(v) || defaut` est le piège :
// 0 est falsy, donc saisir 0 rendait le défaut — « l'opacité du fond ne peut pas valoir 0, elle
// se met à 100 » (signalé le 2026-08-11). Le défaut ne doit servir que si le champ est vide ou
// illisible. Les bornes restent à l'appelant (min/max propres à chaque réglage).
function numOu(v, defaut) {
    const n = parseInt(v, 10);
    return Number.isNaN(n) ? defaut : n;
}

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

    // ── FENÊTRES (PiP) : géométrie en PIXELS DU MUR ────────────────────────────
    (params.flux_config || []).filter(f => !f.hidden).forEach((f, i) => {
        const x = f.x * sx, y = f.y * sy;
        const w = f.w * sx, h = f.h * sy;
        ctx.globalAlpha = 0.67;
        ctx.fillStyle   = COLORS[i % COLORS.length];
        ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = 1;
    });

    // ── LES AUTRES ÉLÉMENTS DU MUR ─────────────────────────────────────────────
    // ⚠ DEUX SYSTÈMES D'UNITÉS, à ne surtout pas mélanger :
    //   • overlays[]              → PIXELS du mur (comme les fenêtres) ;
    //   • meter_blocks[], video_history_blocks[], audio_history_blocks[]
    //                             → FRACTIONS 0..1 du mur ENTIER.
    // Chaque famille a son aplat de couleur + un liseré : on doit les distinguer des PiP
    // au premier coup d'œil (ce sont des vignettes de quelques centaines de pixels, pas un
    // rendu fidèle). `hidden` est respecté sur CHAQUE liste.
    const drawBlock = (x, y, w, h, color) => {
        if (!(w > 0) || !(h > 0)) return;
        ctx.globalAlpha = 0.45;
        ctx.fillStyle = color;
        ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, Math.max(1, w - 1), Math.max(1, h - 1));
    };
    // fractions 0..1 → pixels du canevas d'aperçu
    const frac = (b, color) => drawBlock(b.x * cw, b.y * ch, b.w * cw, b.h * ch, color);

    (params.overlays || []).filter(o => !o.hidden).forEach(o => {
        // overlays = PIXELS du mur → même échelle que les fenêtres (sx/sy)
        drawBlock((o.x || 0) * sx, (o.y || 0) * sy, (o.w || 0) * sx, (o.h || 0) * sy,
                  OVERLAY_COLORS[o.kind] || OVERLAY_COLORS.text);
    });
    (params.meter_blocks || []).filter(b => !b.hidden).forEach(b => frac(b, EL_COLORS.meters));
    (params.video_history_blocks || []).filter(b => !b.hidden).forEach(b => frac(b, EL_COLORS.vhist));
    (params.audio_history_blocks || []).filter(b => !b.hidden).forEach(b => frac(b, EL_COLORS.ahist));
}

// Résumé du CONTENU d'un layout : le compteur historique ne comptait que les fenêtres
// (`flux_config.length` + « entrées ») → un layout fait d'habillages et de frises s'affichait
// « 0 entrées ». On énumère chaque famille PRÉSENTE (masqués exclus, comme l'aperçu).
function layoutSummary(cfg) {
    cfg = cfg || {};
    const live = a => (a || []).filter(e => e && !e.hidden).length;
    const parts = [];
    const push = (n, key) => { if (n > 0) parts.push(n + ' ' + T(key)); };
    push(live(cfg.flux_config),          'plugin.multiview.count_windows');
    push(live(cfg.overlays),             'plugin.multiview.count_overlays');
    push(live(cfg.meter_blocks),         'plugin.multiview.count_meters');
    push(live(cfg.video_history_blocks), 'plugin.multiview.count_vhist');
    push(live(cfg.audio_history_blocks), 'plugin.multiview.count_ahist');
    return parts.length ? parts.join(' · ') : T('plugin.multiview.count_empty');
}

// Couleurs des éléments NON-PiP de l'aperçu (distinctes de la palette COLORS des fenêtres).
const OVERLAY_COLORS = {
    text:  '#f0b429',   // texte      — ambre
    clock: '#f2711c',   // horloge    — orange
    image: '#d64bc8',   // image      — magenta
};
const EL_COLORS = {
    meters: '#2fbf71',  // blocs VU-mètres      — vert
    vhist:  '#3d8bfd',  // frise historique vidéo — bleu
    ahist:  '#00b8c4',  // frise historique audio — cyan
};

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

// Sorties AUDIO individuelles de la flotte, pour le sélecteur « Source audio » du panneau
// d'entrée (cf. refreshEntryPanel) — même dérivation DB-only que loadVideoSources.
async function loadAudioSources() {
    try {
        const srcs = await (await fetch('/api/sources?kind=audio')).json();
        audioSources = (srcs || []).map(s => ({
            vmid: s.vmid, hostname: s.hostname || ('mxl' + s.vmid),
            shm: s.shm, label: s.label || '' }));
    } catch(e) { audioSources = []; }
}

// Sentinel `audio_path` = « explicitement aucune source audio » (VU-mètres coupés), à
// distinguer de '' = auto (dérivé de la source vidéo). Miroir de AUDIO_PATH_NONE,
// plugins/multiview/script.py (_audio_name_for) — NE JAMAIS confondre avec un vrai nom de shm.
const AUDIO_PATH_NONE = '__none__';

// ─── Noms de colonnes labels + connexions TSL (= niveaux de Tally) ───────────
let _tslLabelNames = ["Hostname", "MXL", "Label 2", "Label 3", "Label 4",
                      "Label 5", "Label 6", "Label 7", "Label 8", "Label 9"];
let _tslConns = [];   // [{id, name, …}] connexions TSL (diagnostic ; PLUS les niveaux)
// Niveaux de Tally du site — des ENTITÉS NOMMÉES (`tally_levels`), plus des bandes déduites
// d'une base TSL. Une tuile en choisit un ; le service dit qui le sert.
let _tslNiveaux = [];
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
        try {
            const rn = await fetch('/api/tally/levels');
            if (rn.ok) _tslNiveaux = await rn.json();
        } catch (e) { _tslNiveaux = []; }
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
        tl.innerHTML = _tallyLevelOptions();
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
        tl.innerHTML = _tallyLevelOptions();
        tl.value = cur;
    }
}

// ★ LES NIVEAUX SONT NOMMÉS, PLUS DÉDUITS. Ce menu listait `tally_base / 3 + 1` : le pas de 3
// du mot de contrôle TSL 5.0, recopié dans l'éditeur du mur, avec pour effet un plafond de
// quatre niveaux et des numéros qui bougeaient dès qu'une connexion changeait de base. Un niveau
// est maintenant une entité de `tally_levels` : il a un identifiant stable et un nom écrit par
// l'exploitant, et le protocole qui le sert n'entre plus dans son identité.
function _tallyLevelOptions() {
    // ★ LA VALEUR EST L'UUID, LE LIBELLÉ PORTE LE NUMÉRO. Le numéro n'est qu'un rang
    // d'affichage : réordonner les niveaux le réécrit, et une tuile qui l'aurait mémorisé
    // pointerait ensuite un autre niveau — sans rien afficher d'anormal.
    let opts = '<option value="">— Aucun —</option>';
    for (const n of (_tslNiveaux || [])) {
        opts += `<option value="${n.uuid}">${n.num} — ${escapeHtmlMv(n.nom || '')}</option>`;
    }
    return opts;
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
        await Promise.all([loadAllContainers(), loadVideoSources(), loadAudioSources(),
                           _loadTslLabelNames(), loadPipTemplates(),
                           // Catalogue de polices (+ @font-face de la bibliothèque) : chargé dès
                           // l'ouverture pour que l'APERÇU canvas rende la vraie police.
                           window.BobiFonts ? window.BobiFonts.load() : Promise.resolve()]);
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
    // On quitte le mur précédent : on lui rend la main avant de changer d'éditeur.
    if (editorVmid !== null && editorVmid !== vmid) mwVerrouRendre();
    editorVmid   = vmid;
    // Révision de la configuration CHARGÉE : c'est l'état sur lequel toute notre édition sera
    // bâtie, et ce que le serveur comparera à chaque déploiement (garde anti-écrasement).
    mwRev = (c.config_rev === undefined || c.config_rev === null) ? null : parseInt(c.config_rev);
    mwVerrou = {a_moi: true, user_name: '', depuis_s: 0, libre: true};
    mwVerrouPrendre(false).then(() => mwVerrouDemarrer(mwVerrou.battement_s));
    mwFabricStart();   // régions de calcul + annonce des réorganisations (murs shardés)
    editorParams = Object.assign({
        flux_config: [],
        shm_out: 'mxl_mix',
        show_no_signal: true,
        freeze_detect_s: 2,
        show_proxy: false,
        max_inputs: 4,
        genlock: true,
        tsl_mode: 'central',
        tsl_port: 4801,
        overlays: [],
        meter_blocks: [],
        video_history_blocks: [],
        audio_history_blocks: [],
        default_template: null,
        default_template_ref: ''
    }, dc.params || {});
    populateDefaultTemplateSelect();
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
    // out_width/out_height = résolution de SORTIE (le script émet à ces dims). Repli sur width/height
    // AVANT le format système : un multiview créé via la palette pose width/height mais pas
    // out_width ; sans ce repli l'éditeur écrasait out_width avec le défaut système (ex. 640×360),
    // désynchronisant la sortie réelle du width/height voulu (sortie en 640 alors que la page Câbles
    // affichait le width). Le format système ne sert que si NI out_width NI width ne sont posés.
    editorParams.out_width  = editorParams.out_width  || editorParams.width  || _sysf.w;
    editorParams.out_height = editorParams.out_height || editorParams.height || _sysf.h;
    editorParams.fps        = editorParams.fps        || _sysf.fps;
    editorParams.scan       = editorParams.scan       || _sysf.scan;
    // Couleur : purement locale (rendu de l'éditeur), jamais persistée.
    // `ratio` = format VOULU de la fenêtre. La SOURCE fait foi quand ses dimensions sont
    // connues (résolues à chaque déploiement depuis le flow_def), sinon le ratio PERSISTÉ
    // (fenêtre non câblée : format du modèle de PiP, ou 16:9 par défaut). Ne jamais le
    // dériver de w/h : une fenêtre déformée verrouillerait sa déformation.
    editorParams.flux_config = (editorParams.flux_config || []).map((f, i) => Object.assign({}, f, {
        color: COLORS[i % COLORS.length],
        ratio: (f.in_w && f.in_h) ? f.in_w / f.in_h
             : ((f.ratio && f.ratio > 0) ? f.ratio : 16/9)
    }));
    editorParams.overlays = (editorParams.overlays || []);
    // Blocs VU-mètres de MUR : stockés côté serveur en FRACTIONS 0..1 du mur (meter_blocks) →
    // convertis en PIXELS canvas pour l'édition (comme les overlays), reconvertis en fractions
    // à la sérialisation (deployerEditor). out_width/out_height sont résolus juste au-dessus.
    editorParams.meter_blocks = (editorParams.meter_blocks || []).map((b, i) => ({
        id: b.id || ('mb' + i + '_' + Date.now().toString(36)),
        x: Math.round((b.x || 0) * editorParams.out_width),
        y: Math.round((b.y || 0) * editorParams.out_height),
        w: Math.max(24, Math.round((b.w || 0.15) * editorParams.out_width)),
        h: Math.max(24, Math.round((b.h || 0.3) * editorParams.out_height)),
        channels: b.channels ?? 2,
        ch_start: b.ch_start ?? 1,
        scale: b.scale || 'dbfs',
        opacity: b.opacity ?? 100,
        align: b.align || 'left',
        width_mode: b.width_mode || 'auto',
        audio_path: b.audio_path || '',
        label: b.label || '',
        hidden: !!b.hidden,
    }));
    // Blocs d'historique (frises) : fractions 0..1 du mur en base → PIXELS canvas pour l'édition
    // (même conversion que les blocs VU-mètres ci-dessus ; re-sérialisés en fractions au deploy).
    ['video', 'audio'].forEach(kind => {
        const key = HIST_LIST[kind];
        editorParams[key] = (editorParams[key] || []).map((b, i) => Object.assign(newHistBlock(kind), b, {
            id: b.id || ('hb' + kind + i + '_' + Date.now().toString(36)),
            x: Math.round((b.x || 0) * editorParams.out_width),
            y: Math.round((b.y || 0) * editorParams.out_height),
            w: Math.max(48, Math.round((b.w || 0.3) * editorParams.out_width)),
            h: Math.max(24, Math.round((b.h || 0.12) * editorParams.out_height)),
        }));
    });
    padBank();   // banque à indices stables : toujours max_inputs entrées
    _selClear();
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
    document.getElementById('ed_show_no_signal').checked = p.show_no_signal !== false;
    document.getElementById('ed_freeze_detect').value    = p.freeze_detect_s ?? 2;
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
    editorParams.show_no_signal  = document.getElementById('ed_show_no_signal').checked;
    editorParams.freeze_detect_s = Math.max(0, parseFloat(document.getElementById('ed_freeze_detect').value) || 0);
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

// Métadonnées ANC affichables sur l'image d'une cellule (miroir de ANC_FIELDS, script.py).
const ANC_FLAGS = ['anc_types', 'anc_tc', 'anc_cc', 'anc_afd', 'anc_st352', 'anc_scte', 'anc_crc'];

function newEntry(idx, hidden) {
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;
    return {
        path: '', hidden: !!hidden, name: '',
        // '' = auto (dérivé de `path`, cf. _audio_name_for côté moteur) ; AUDIO_PATH_NONE =
        // explicitement aucune (VU-mètres coupés) ; sinon nom de shm audio explicite.
        audio_path: '',
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
        tally_level: '',             // niveau de Tally (vide = aucun ; sinon un UUID `tally_levels`)
        tally_red: false,
        tally_green: false,
        // Peak meters
        meter_channels: 0,           // 0 = désactivé ; sinon 2/4/6/8
        meter_position: 'right',     // left | right
        meter_inside: false,         // false = à côté (réduit la vidéo), true = overlay
        meter_opacity: 100,          // 10..100 (utilisé si inside=true)
        meter_scale: 'dbfs',         // dbfs | ppm (EBU)
        // Métadonnées ANC incrustées sur l'image de la cellule — TOUT À FALSE par défaut :
        // rien ne s'affiche tant que l'utilisateur n'a rien coché (port ANC à câbler par ailleurs).
        anc_types: false,            // inventaire des métadonnées portées (ATC, CC/708, AFD…)
        anc_tc: false,               // timecode embarqué (ATC RP 188)
        anc_cc: false,               // sous-titres (texte CEA-608 si décodable, sinon présence)
        anc_afd: false,              // format d'image actif (ST 2016-3)
        anc_st352: false,            // format DÉCLARÉ par le signal (ST 352)
        anc_scte: false,             // déclencheur SCTE-104
        anc_crc: false,              // paquets au checksum invalide (métadonnée corrompue)
        anc_position: 'bottom',      // bottom | top
        anc_opacity: 60,             // 0..100 (fond du bandeau)
        // Modèle de PiP (bibliothèque Réglages → PiP) : null = hériter (défaut du mur, sinon
        // « Classique » généré depuis les flags ci-dessus) ; sinon {name, components} RÉSOLU
        // (embarqué dans le deploy_config).
        template: null,
        template_ref: '',            // '' = hériter du mur | id bibliothèque
    };
}

// ─── Modèles de PiP (bibliothèque /api/pip_templates) ────────
let mwPipTemplates = [];
async function loadPipTemplates() {
    try {
        const r = await fetch('/api/pip_templates');
        mwPipTemplates = r.ok ? await r.json() : [];
    } catch (e) { mwPipTemplates = []; }
    populateDefaultTemplateSelect();
    refreshEntryPanel();
}

function mwResolveTemplate(ref) {
    const tp = mwPipTemplates.find(x => String(x.id) === String(ref));
    if (!tp) return null;
    const out = { name: tp.name,
                  components: JSON.parse(JSON.stringify((tp.config || {}).components || [])) };
    // Format LIBRE du modèle (éditeur PiP) : ratio L/H de la cellule cible. Absent = 16:9
    // implicite (modèles historiques). Embarqué avec le modèle → suit dans deploy_config
    // (le moteur l'ignore : les composants restent relatifs au rect de la tuile).
    const a = parseFloat((tp.config || {}).aspect);
    if (a > 0.1 && a < 10) out.aspect = a;
    return out;
}

// Aspect du MODÈLE EFFECTIF d'une fenêtre (explicite sinon défaut du mur), ou 0 si le modèle
// n'en déclare pas (16:9 implicite / habillage Classique) — le tuilage (remplir) et le snap
// d'affectation s'en servent pour donner à la tuile le ratio natif de son habillage.
function mwTemplateAspect(f) {
    const t = mwEffectiveTemplate(f);
    const a = t ? parseFloat(t.aspect) : 0;
    return (a > 0.1 && a < 10) ? a : 0;
}

// HÉRITAGE : modèle EFFECTIF d'une fenêtre = son modèle explicite, sinon le modèle PAR DÉFAUT
// du mur. null = modèle « Classique » GÉNÉRÉ côté moteur depuis les flags par-fenêtre
// (_classic_comps, script.py) — l'aperçu bandeau/tally/VU de drawCanvas en est le miroir.
function mwEffectiveTemplate(f) {
    if (f.template && f.template.components && f.template.components.length) return f.template;
    const d = editorParams && editorParams.default_template;
    return (d && d.components && d.components.length) ? d : null;
}

function _pipTemplateOptions(selectedRef, withInherit) {
    const opts = [];
    if (withInherit) {
        opts.push('<option value="">' + escapeHtml(T('plugin.multiview.pip_template_inherit')) + '</option>');
    } else {
        opts.push('<option value="">' + escapeHtml(T('plugin.multiview.pip_template_classic')) + '</option>');
    }
    mwPipTemplates.forEach(tp => {
        opts.push(`<option value="${escapeHtml(String(tp.id))}">${escapeHtml(tp.name)}</option>`);
    });
    // Réf affectée mais absente de la bibliothèque (modèle supprimé) : option conservée.
    if (selectedRef
            && !mwPipTemplates.some(tp => String(tp.id) === String(selectedRef))) {
        opts.push(`<option value="${escapeHtml(String(selectedRef))}">${escapeHtml(String(selectedRef))}</option>`);
    }
    return opts.join('');
}

function populateDefaultTemplateSelect() {
    const sel = document.getElementById('ed_default_template');
    if (!sel || !editorParams) return;
    sel.innerHTML = _pipTemplateOptions(editorParams.default_template_ref || '', false);
    sel.value = editorParams.default_template_ref || '';
}

function onDefaultTemplateChange() {
    if (!editorParams) return;
    const sel = document.getElementById('ed_default_template');
    const ref = sel.value || '';
    editorParams.default_template_ref = ref;
    editorParams.default_template = ref ? mwResolveTemplate(ref) : null;
    dessiner();
    refreshEntryPanel();
    hotApplyStyle();
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
    _selSet('win', idx);
    dessiner();
    hotApplyFull();
}

// ─── Blocs VU-mètres de MUR (posés directement sur le layout du mur, indépendants de toute
// fenêtre — cf. deploy_config.params.meter_blocks, plugins/multiview/script.py render_meters).
// Édités en PIXELS canvas (comme les overlays) ; convertis en fractions 0..1 du mur à la
// sérialisation (deployerEditor). ───
function newMeterBlock() {
    const ow = editorParams.out_width, oh = editorParams.out_height;
    const w = Math.max(24, Math.round(ow * 0.12) & ~1);
    const h = Math.max(24, Math.round(oh * 0.30) & ~1);
    return {
        id: 'mb' + Date.now().toString(36) + Math.floor(Math.random() * 1000),
        x: Math.round((ow - w) / 2) & ~1, y: Math.round((oh - h) / 2) & ~1, w, h,
        channels: 2, ch_start: 1, scale: 'dbfs', opacity: 100, align: 'left',
        width_mode: 'auto', audio_path: '', label: '', hidden: false,
    };
}

function ajouterBlocVU() {
    if (!editorParams) return;
    editorParams.meter_blocks = editorParams.meter_blocks || [];
    editorParams.meter_blocks.push(newMeterBlock());
    _selSet('blk', editorParams.meter_blocks.length - 1);
    dessiner();
    hotApplyFull();
}

function supprimerBlocVU() {
    if (selectedBlock < 0) return;
    const _i = selectedBlock;
    editorParams.meter_blocks.splice(_i, 1);
    _selDrop('blk', _i);   // décale les références au-delà, retire la sienne
    dessiner();
    hotApplyFull();
}

// ─── Blocs d'HISTORIQUE de MUR (frise vidéo / frise audio) ───
// Frise vidéo : une vignette par seconde + ruban gel/noir/perte de signal (la vignette de
// l'INSTANT de l'événement est épinglée). Frise audio : enveloppe des crêtes + saturation
// (rouge, persistante) + silence. Cf. plugins/multiview/script.py, render_history_tiles.
function newHistBlock(kind) {
    const ow = editorParams.out_width, oh = editorParams.out_height;
    const w = Math.max(48, Math.round(ow * 0.32) & ~1);
    const h = Math.max(24, Math.round(oh * (kind === 'audio' ? 0.10 : 0.13)) & ~1);
    const b = {
        id: 'hb' + Date.now().toString(36) + Math.floor(Math.random() * 1000),
        kind,
        x: Math.round((ow - w) / 2) & ~1, y: Math.round((oh - h) / 2) & ~1, w, h,
        duration: 30, opacity: 100, label: '', hidden: false,
    };
    if (kind === 'audio') { b.audio_path = ''; b.channels = 2; b.ch_start = 1; }
    else { b.path = ''; b.events = true; }
    return b;
}

function ajouterBlocHist(kind) {
    if (!editorParams) return;
    histList(kind).push(newHistBlock(kind));
    selectedHistKind = kind;
    _selSet(kind === 'audio' ? 'ah' : 'vh', histList(kind).length - 1);
    dessiner();
    hotApplyFull();
}

function supprimerBlocHist() {
    if (selectedHist < 0) return;
    const _i = selectedHist, _k = (selectedHistKind === 'audio') ? 'ah' : 'vh';
    histList(selectedHistKind).splice(_i, 1);
    _selDrop(_k, _i);
    dessiner();
    hotApplyFull();
}

function hitHist(pos) {
    for (const kind of ['audio', 'video']) {
        const bs = histList(kind);
        for (let i = bs.length - 1; i >= 0; i--) {
            const b = bs[i];
            if (b.hidden) continue;
            if (kind === selectedHistKind && i === selectedHist &&
                pos.x >= b.x + b.w - HANDLE_SIZE && pos.y >= b.y + b.h - HANDLE_SIZE) {
                return {i, kind, mode: 'resize'};
            }
            if (pos.x >= b.x && pos.x <= b.x + b.w && pos.y >= b.y && pos.y <= b.y + b.h) {
                return {i, kind, mode: 'move'};
            }
        }
    }
    return null;
}

function beginHistDrag(hit, pos, additive) {
    selectedHistKind = hit.kind;   // la liste visée doit être posée AVANT la sélection
    _selGrab(hit.kind === 'audio' ? 'ah' : 'vh', hit.i, additive);
    dragHist = true;
    dragMode = hit.mode;
    dragStart = pos;
    dragOrigRect = {...histList(hit.kind)[hit.i]};
    _beginGroupDrag();
    _setCanvasCursor(hit.mode === 'resize' ? 'nwse-resize' : 'move');
    dessiner();
}

function histMouseMove(e) {
    if (!dragMode || selectedHist < 0) return;
    const pos = getCanvasPos(e);
    const dx = pos.x - dragStart.x, dy = pos.y - dragStart.y;
    const b = histList(selectedHistKind)[selectedHist];
    if (!b) return;
    const ow = editorParams.out_width, oh = editorParams.out_height;
    // AIMANT : les frises l'ignoraient purement et simplement — aucun appel à computeSnap/
    // computeSnapResize ici, alors que les overlays et les blocs VU passent par les deux. Elles
    // étaient donc les seuls objets du mur impossibles à aligner au geste. `_snapRects` les
    // connaissait déjà (familles 'vh'/'ah'), y compris comme CIBLES : le reste du mur s'alignait
    // sur elles, mais elles ne s'alignaient sur rien.
    const skip = { kind: selectedHistKind === 'audio' ? 'ah' : 'vh', i: selectedHist };
    if (dragMode === 'move') {
        let nx = Math.max(0, Math.min(ow - b.w, Math.round(dragOrigRect.x + dx)));
        let ny = Math.max(0, Math.min(oh - b.h, Math.round(dragOrigRect.y + dy)));
        if (snapEnabled) {
            const sn = computeSnap(_dragSkip(skip.kind, selectedHist), nx, ny, b.w, b.h);
            nx = sn.x; ny = sn.y; snapGuides = sn.guides;
        } else snapGuides = [];
        _applyGroupMove(nx, ny);
    } else if (dragMode === 'resize') {
        let nw = Math.max(48, Math.min(ow - b.x, Math.round(dragOrigRect.w + dx)));
        let nh = Math.max(24, Math.min(oh - b.y, Math.round(dragOrigRect.h + dy)));
        if (snapEnabled) {
            // Frise à format LIBRE (aucun ratio imposé, comme un overlay) → ratio = 0 : les deux
            // axes s'aimantent indépendamment (cf. computeSnapResize).
            const sn = computeSnapResize(skip, b.x, b.y, nw, nh, 0);
            nw = Math.max(48, sn.w); nh = Math.max(24, sn.h); snapGuides = sn.guides;
        } else snapGuides = [];
        b.w = nw % 2 === 0 ? nw : nw - 1;
        b.h = nh % 2 === 0 ? nh : nh - 1;
    }
    drawCanvas();
    syncGeomFields();
}

// Aperçu SCHÉMATIQUE sur le canvas de l'éditeur (pas de flux live ici) : cases de vignettes +
// ruban pour la frise vidéo, enveloppe mock + colonne rouge pour la frise audio.
function drawHistLayer(ctx) {
    ['video', 'audio'].forEach(kind => {
        histList(kind).forEach((b, i) => {
            if (b.hidden) return;
            const sel = _selHas(kind === 'audio' ? 'ah' : 'vh', i);
            ctx.save();
            ctx.fillStyle = 'rgba(14,16,20,0.85)';
            ctx.fillRect(b.x, b.y, b.w, b.h);
            if (kind === 'video') {
                const n = Math.max(1, Math.min(120, parseInt(b.duration) || 30));
                const rb = b.events === false ? 0 : Math.max(3, Math.min(10, Math.round(b.h / 8)));
                const sh = Math.max(3, b.h - rb - 2);
                const cw = b.w / n;
                for (let k = 0; k < n; k++) {
                    ctx.fillStyle = (k % 2) ? 'rgba(52,56,66,0.9)' : 'rgba(40,44,52,0.9)';
                    ctx.fillRect(b.x + k * cw, b.y, Math.max(1, cw - 1), sh);
                }
                if (rb) {
                    ctx.fillStyle = 'rgba(44,48,56,0.95)';
                    ctx.fillRect(b.x, b.y + sh + 1, b.w, rb);
                    ctx.fillStyle = 'rgba(240,184,44,0.95)';   // gel (exemple)
                    ctx.fillRect(b.x + b.w * 0.55, b.y + sh + 1, Math.max(2, b.w * 0.08), rb);
                    ctx.fillStyle = 'rgba(236,72,60,0.95)';    // perte de signal (exemple)
                    ctx.fillRect(b.x + b.w * 0.8, b.y + sh + 1, Math.max(2, b.w * 0.05), rb);
                }
            } else {
                const cy = b.y + b.h / 2;
                ctx.strokeStyle = 'rgba(96,210,140,0.9)';
                ctx.beginPath();
                for (let x = 0; x < b.w; x += 2) {
                    const a = (Math.abs(Math.sin(x / 7)) * 0.4 + Math.abs(Math.sin(x / 23)) * 0.5) * (b.h / 2 - 2);
                    ctx.moveTo(b.x + x, cy - a);
                    ctx.lineTo(b.x + x, cy + a);
                }
                ctx.stroke();
                ctx.fillStyle = 'rgba(236,72,60,0.95)';        // saturation (exemple)
                ctx.fillRect(b.x + b.w * 0.62, b.y, 2, b.h);
            }
            ctx.strokeStyle = sel ? '#ffffff' : '#c084fc';
            ctx.lineWidth = sel ? 2 : 1;
            ctx.setLineDash(sel ? [6, 4] : [4, 3]);
            ctx.strokeRect(b.x, b.y, b.w, b.h);
            ctx.setLineDash([]);
            ctx.fillStyle = '#c084fc'; ctx.font = '10px monospace';
            ctx.fillText((kind === 'video' ? 'HIST V ' : 'HIST A ') + (b.label || (i + 1)) +
                         ' · ' + (b.duration || 30) + 's', b.x + 3, b.y + 11);
            if (sel) {
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(b.x + b.w - HANDLE_SIZE, b.y + b.h - HANDLE_SIZE, HANDLE_SIZE, HANDLE_SIZE);
            }
            ctx.restore();
        });
    });
}

// ── Panneau de propriétés d'un bloc d'historique ──
function _hbSetVal(id, v) { const e = document.getElementById(id); if (e) e.value = v; }

function refreshHistPanel() {
    const panel = document.getElementById('ed_hb_panel');
    if (!panel) return;
    const b = selectedHist >= 0 ? histList(selectedHistKind)[selectedHist] : null;
    if (!b) { panel.hidden = true; return; }
    panel.hidden = false;
    const isVideo = selectedHistKind === 'video';
    const ttl = document.getElementById('hb_title');
    if (ttl) ttl.textContent = T(isVideo ? 'plugin.multiview.hb_title_video' : 'plugin.multiview.hb_title_audio');
    const vr = document.getElementById('hb_video_row');
    const ar = document.getElementById('hb_audio_row');
    if (vr) vr.hidden = !isVideo;
    if (ar) ar.hidden = isVideo;
    _hbSetVal('hb_x', b.x); _hbSetVal('hb_y', b.y);
    _hbSetVal('hb_w', b.w); _hbSetVal('hb_h', b.h);
    _hbSetVal('hb_duration', String(b.duration ?? 30));
    _hbSetVal('hb_opacity', b.opacity ?? 100);
    _hbSetVal('hb_label', b.label || '');
    const hbHid = document.getElementById('hb_hidden');
    if (hbHid) hbHid.checked = !!b.hidden;
    const ev = document.getElementById('hb_events');
    if (ev) ev.checked = b.events !== false;
    if (!isVideo) {
        _hbSetVal('hb_channels', String(b.channels ?? 2));
        _hbSetVal('hb_ch_start', b.ch_start ?? 1);
    }
    // Source : flux vidéo ou audio de la flotte (la page Câbles câble le même port).
    const sel = document.getElementById('hb_source');
    if (sel) {
        const list = isVideo ? videoSources : audioSources;
        const cur = (isVideo ? b.path : b.audio_path) || '';
        const opts = list.map(s => {
            const base = s.label ? `${s.hostname} → ${s.label} (${s.shm})` : `${s.hostname} → ${s.shm}`;
            // Libellé d'exploitation du niveau courant devant le nom technique (window.SourceLabels,
            // sélecteur global de la barre de navigation) — l'opérateur choisit sa source sous le nom
            // qu'il lui donne, pas sous « 2110-io-dl360_3 ».
            const txt = window.SourceLabels ? window.SourceLabels.display(s.shm, base) : base;
            return `<option value="${escapeHtml('/dev/shm/' + s.shm)}">${escapeHtml(txt)}</option>`;
        });
        opts.unshift('<option value="">' + escapeHtml(T('plugin.multiview.mb_audio_none_option')) + '</option>');
        if (cur && !list.some(s => '/dev/shm/' + s.shm === cur)) {
            opts.splice(1, 0, `<option value="${escapeHtml(cur)}">${escapeHtml(cur)}</option>`);
        }
        sel.innerHTML = opts.join('');
        sel.value = cur;
    }
}

function onHistChange() {
    if (selectedHist < 0) return;
    const b = histList(selectedHistKind)[selectedHist];
    if (!b) return;
    const isVideo = selectedHistKind === 'video';
    b.duration = parseInt(document.getElementById('hb_duration').value) || 30;
    b.opacity = Math.max(10, Math.min(100, numOu(document.getElementById('hb_opacity').value, 85)));
    b.label = document.getElementById('hb_label').value || '';
    // Masquer SANS supprimer : le moteur saute les blocs `hidden` (script.py, _hist_units) — la
    // config et le câblage sont conservés. Utile pour comparer le coût du mur avec/sans la frise.
    b.hidden = !!document.getElementById('hb_hidden').checked;
    const src = document.getElementById('hb_source').value || '';
    if (isVideo) {
        b.path = src;
        b.events = !!document.getElementById('hb_events').checked;
    } else {
        b.audio_path = src;
        b.channels = parseInt(document.getElementById('hb_channels').value) || 2;
        b.ch_start = Math.max(1, Math.min(16, parseInt(document.getElementById('hb_ch_start').value) || 1));
    }
    dessiner();
    hotApplyFull();
}

function onHistGeomChange() {
    if (selectedHist < 0) return;
    const b = histList(selectedHistKind)[selectedHist];
    if (!b) return;
    const ow = editorParams.out_width, oh = editorParams.out_height;
    b.x = Math.max(0, Math.min(ow - 2, parseInt(document.getElementById('hb_x').value) || 0));
    b.y = Math.max(0, Math.min(oh - 2, parseInt(document.getElementById('hb_y').value) || 0));
    b.w = Math.max(48, Math.min(ow - b.x, parseInt(document.getElementById('hb_w').value) || 48));
    b.h = Math.max(24, Math.min(oh - b.y, parseInt(document.getElementById('hb_h').value) || 24));
    dessiner();
    hotApplyFull();
}

// Sérialisation : pixels canvas → fractions 0..1 du MUR (contrat serveur, cf. script.py).
function serializeHistBlocks(kind) {
    const ow = Math.max(1, editorParams.out_width), oh = Math.max(1, editorParams.out_height);
    const frac = (v, span) => Math.max(0, Math.min(1, +(v / span).toFixed(4)));
    const DURS = [10, 30, 60, 120];
    return histList(kind).map(b => {
        const d = parseInt(b.duration) || 30;
        const o = {
            x: frac(b.x, ow), y: frac(b.y, oh), w: frac(b.w, ow), h: frac(b.h, oh),
            duration: DURS.includes(d) ? d : 30,
            opacity: Math.max(10, Math.min(100, numOu(b.opacity, 85))),
            label: b.label || '',
            hidden: !!b.hidden,
        };
        if (kind === 'audio') {
            o.audio_path = b.audio_path || '';
            o.channels = parseInt(b.channels) || 2;
            o.ch_start = Math.max(1, Math.min(16, parseInt(b.ch_start) || 1));
        } else {
            o.path = b.path || '';
            o.events = b.events !== false;
        }
        return o;
    });
}

function supprimerEntreeSelectionnee() {
    if (selectedIdxs.length === 0) return;
    // Retire le PiP de l'image SANS couper la source (entrée masquée, câble conservé).
    selectedIdxs.forEach(i => {
        const f = editorParams.flux_config[i];
        if (f) f.hidden = true;
    });
    // Les fenêtres masquées sortent de la sélection ; les objets d'autres familles y restent.
    mwSel = mwSel.filter(r => r.k !== 'win');
    _selSync();
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
    // Un seul panneau ouvert à la fois : celui de la RÉFÉRENCE. Avec une sélection mixte, des
    // fenêtres peuvent être sélectionnées alors que la référence est un texte — le panneau de
    // fenêtre resterait ouvert sur un objet qui n'est plus celui qu'on édite.
    const primary = _winPrimary();
    if (primary < 0) {
        panel.hidden = true;
        return;
    }
    const f = editorParams.flux_config[primary];
    panel.hidden = false;
    // Modèle de PiP : select peuplé depuis la bibliothèque ; un modèle actif REMPLACE
    // l'habillage legacy → on masque les réglages ignorés (nom/tally visuels, VU, ANC),
    // en gardant source + routing TSL (le tally alimente les composants du modèle).
    const tplSel = document.getElementById('ed_pip_template');
    if (tplSel) {
        tplSel.innerHTML = _pipTemplateOptions(f.template_ref || '', true);
        tplSel.value = f.template_ref || '';
    }
    const hasTpl = !!mwEffectiveTemplate(f);
    ['ed_show_label_row', 'ed_label_proportional_row', 'ed_row_meters', 'ed_row_anc'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = hasTpl ? 'none' : '';
    });
    document.getElementById('ed_label_source').value = f.label_source || 'hostname';
    document.getElementById('ed_show_label').checked = !!f.show_label;
    document.getElementById('ed_label_proportional').checked = !!f.label_proportional;
    document.getElementById('ed_tsl_index').value    = f.tsl_index || 0;
    const _colors = f.tally_red && f.tally_green ? 'both' : f.tally_red ? 'red' : f.tally_green ? 'green' : 'none';
    _tslSetSelects(f.label_col ?? 0, f.tally_level ?? '', _colors);
    document.getElementById('ed_x').value = f.x;
    document.getElementById('ed_y').value = f.y;
    document.getElementById('ed_w').value = f.w;
    document.getElementById('ed_h').value = f.h;
    // Peak meters
    document.getElementById('ed_meter_channels').value = String(f.meter_channels ?? 0);
    document.getElementById('ed_meter_position').value = f.meter_position || 'right';
    document.getElementById('ed_meter_inside').value   = (f.meter_inside ? '1' : '0');
    document.getElementById('ed_meter_opacity').value  = f.meter_opacity ?? 100;
    document.getElementById('ed_meter_scale').value    = f.meter_scale || 'dbfs';
    // Métadonnées ANC (par fenêtre)
    ANC_FLAGS.forEach(k => { document.getElementById('ed_' + k).checked = !!f[k]; });
    document.getElementById('ed_anc_position').value = f.anc_position || 'bottom';
    document.getElementById('ed_anc_opacity').value  = f.anc_opacity ?? 60;

    // Re-peuple la dropdown source pour cette entrée
    const pathSel = document.getElementById('ed_path');
    // Une <option> par sortie vidéo individuelle (libellé hostname → label (shm)).
    const opts = videoSources.map(s => {
        const base = s.label ? `${s.hostname} → ${s.label} (${s.shm})`
                             : `${s.hostname} → ${s.shm}`;
        const txt = window.SourceLabels ? window.SourceLabels.display(s.shm, base) : base;
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

    // Re-peuple la dropdown SOURCE AUDIO (VU-mètres) : 3 états — auto (dérivé de la vidéo,
    // '' ) / aucune (AUDIO_PATH_NONE, VU-mètres coupés) / flux explicite (page Câbles ou
    // choisi ici). cf. _audio_name_for, plugins/multiview/script.py.
    const audioSel = document.getElementById('ed_audio_path');
    if (audioSel) {
        const audioOpts = audioSources.map(s => {
            const base = s.label ? `${s.hostname} → ${s.label} (${s.shm})`
                                 : `${s.hostname} → ${s.shm}`;
            const txt = window.SourceLabels ? window.SourceLabels.display(s.shm, base) : base;
            return `<option value="${escapeHtml('/dev/shm/' + s.shm)}">${escapeHtml(txt)}</option>`;
        });
        audioOpts.unshift('<option value="' + escapeHtml(AUDIO_PATH_NONE) + '">'
            + escapeHtml(T('plugin.multiview.audio_none_option')) + '</option>');
        audioOpts.unshift('<option value="">' + escapeHtml(T('plugin.multiview.audio_auto_option')) + '</option>');
        const curAudio = f.audio_path || '';
        if (curAudio && curAudio !== AUDIO_PATH_NONE
                && !audioSources.some(s => '/dev/shm/' + s.shm === curAudio)) {
            audioOpts.splice(2, 0, `<option value="${escapeHtml(curAudio)}">${escapeHtml(curAudio)}</option>`);
        }
        audioSel.innerHTML = audioOpts.join('');
        audioSel.value = curAudio;
    }
}

// Bouton « Recharger » (sous ed_pip_template) : re-fetch la bibliothèque de modèles en
// contournant tout cache (mwPipTemplates n'est chargé QU'UNE FOIS au chargement du composer,
// cf. loadPipTemplates() plus haut — éditer un modèle dans Réglages → PiP pendant que ce mur
// reste ouvert ne le rafraîchit jamais tout seul), puis RE-RÉSOUT le modèle effectif de la
// fenêtre courante (direct, ou hérité du mur) et le repousse à chaud au conteneur en cours
// d'exécution. Aucun redéploiement requis : le moteur applique le modèle reçu tel quel via
// les endpoints /plugin/window et /plugin/style (script.py ne relit jamais la bibliothèque
// lui-même — il n'y a pas d'accès réseau depuis le container vers l'orchestrateur pour ça).
// Rafraîchit le MUR ENTIER : le modèle par défaut (héritage) ET chaque fenêtre qui porte un
// modèle propre. Ne dépend d'AUCUNE sélection — appelable depuis « Sortie & habillage » (bouton
// sous ed_default_template) comme depuis le panneau d'une fenêtre (bouton sous ed_pip_template).
async function reloadPipTemplate() {
    if (!editorParams) return;
    try {
        const r = await fetch('/api/pip_templates', { cache: 'no-store' });
        if (r.ok) mwPipTemplates = await r.json();
    } catch (e) { /* échec réseau : on retente la résolution avec la liste déjà en mémoire */ }
    populateDefaultTemplateSelect();

    // 1. Modèle PAR DÉFAUT du mur → /plugin/style : rafraîchit d'un coup toutes les fenêtres
    //    qui en héritent (template_ref vide).
    if (editorParams.default_template_ref) {
        editorParams.default_template = mwResolveTemplate(editorParams.default_template_ref) || editorParams.default_template;
        hotApplyStyle();
    }
    // 2. Fenêtres portant un modèle PROPRE → /plugin/window, une par une (elles n'héritent pas,
    //    le push de style ci-dessus ne les touche pas).
    (editorParams.flux_config || []).forEach((f, i) => {
        if (!f.template_ref) return;
        f.template = mwResolveTemplate(f.template_ref) || f.template;
        hotApplyWindow(i);
    });

    refreshEntryPanel();
    dessiner();
    mwFlash(T('plugin.multiview.flash_pip_template_reloaded'));
}

function onEntryChange() {
    const primary = primaryIdx();
    if (primary < 0) return;
    const f = editorParams.flux_config[primary];
    const tplSel = document.getElementById('ed_pip_template');
    if (tplSel) {
        const ref = tplSel.value || '';
        if (ref !== (f.template_ref || '')) {
            // '' = hériter du mur (défaut, sinon « Classique » généré) ; sinon id bibliothèque.
            f.template_ref = ref;
            f.template = ref ? (mwResolveTemplate(ref) || f.template) : null;
            // Modèle à format LIBRE : la tuile prend le ratio natif de l'habillage (hauteur
            // recalculée depuis la largeur, paire, clampée) — sinon la vidéo 16:9 interne
            // serait déformée/letterboxée. Uniquement à l'AFFECTATION explicite : on ne touche
            // jamais la géométrie des autres tuiles.
            const _ta = mwTemplateAspect(f);
            if (_ta && f.w > 0) {
                const out_h = editorParams.out_height;
                let nh = Math.max(2, Math.round(f.w / _ta) & ~1);
                if (f.y + nh > out_h) {
                    nh = Math.max(2, (out_h - f.y) & ~1);
                    f.w = Math.max(2, Math.round(nh * _ta) & ~1);   // déborde → réduit à ratio constant
                }
                f.h = nh;
                f.ratio = _ta;   // le resize à la poignée conserve ce ratio
            }
            refreshEntryPanel();   // re-masque/affiche les réglages du repli Classique
        }
    }
    f.label_source = document.getElementById('ed_label_source').value || 'hostname';
    // SOURCE VIDÉO : le hot-apply /plugin/window ne transporte pas `path` et le proxy ne le
    // persiste pas (liste blanche `_MV_WINDOW_PERSIST`, app/routes/plugin_registry.py) — changer
    // la source depuis ce panneau ne touchait donc NI le conteneur NI la base : la nouvelle source
    // s'affichait dans l'éditeur puis disparaissait au rechargement. On passe par le déploiement
    // (hot-apply /reconfigure côté serveur), seul chemin qui résout les dimensions de la nouvelle
    // source, les proxies de pyramide et le tissu des murs shardés — et qui ne ferme QUE le Reader
    // de la fenêtre concernée (les autres tuiles ne se figent pas).
    // La source AUDIO, elle, RESTE sur /plugin/window : c'est lui qui purge les états audio ouverts
    // de la fenêtre (`_do_window`), là où `/reconfigure` ne purge que ceux des BLOCS VU de mur —
    // l'aiguiller vers le déploiement laisserait les VU sur l'ANCIENNE source. Elle est désormais
    // persistée côté proxy (même liste blanche).
    const _pathAvant = f.path || '';
    f.path         = document.getElementById('ed_path').value;
    { const _as = document.getElementById('ed_audio_path'); if (_as) f.audio_path = _as.value; }
    const _srcChange = (f.path || '') !== _pathAvant;
    f.show_label   = document.getElementById('ed_show_label').checked;
    f.label_proportional = document.getElementById('ed_label_proportional').checked;
    f.tsl_index      = parseInt(document.getElementById('ed_tsl_index').value) || 0;
    f.label_col      = parseInt(document.getElementById('ed_label_col').value) || 0;
    // ⚠ PAS DE `parseInt` : un niveau est un UUID depuis le dénouement, et `parseInt` d'un
    // UUID rend NaN — donc `|| 0`, donc AUCUN niveau. Le menu proposait bien les bons
    // identifiants, mais toute modification d'une tuile effaçait son tally, en silence.
    f.tally_level    = document.getElementById('ed_tally_level').value || '';
    const _colors    = document.getElementById('ed_tally_colors').value || 'none';
    f.tally_red      = (_colors === 'red'   || _colors === 'both');
    f.tally_green    = (_colors === 'green' || _colors === 'both');
    f.show_tally     = !!(f.tally_level && (f.tally_red || f.tally_green));   // gate de rendu
    f.meter_channels = parseInt(document.getElementById('ed_meter_channels').value) || 0;
    f.meter_position = document.getElementById('ed_meter_position').value || 'right';
    f.meter_inside   = document.getElementById('ed_meter_inside').value === '1';
    f.meter_opacity  = Math.max(10, Math.min(100, numOu(document.getElementById('ed_meter_opacity').value, 70)));
    f.meter_scale    = document.getElementById('ed_meter_scale').value || 'dbfs';
    ANC_FLAGS.forEach(k => { f[k] = document.getElementById('ed_' + k).checked; });
    f.anc_position   = document.getElementById('ed_anc_position').value || 'bottom';
    f.anc_opacity    = Math.max(0, Math.min(100, numOu(document.getElementById('ed_anc_opacity').value, 60)));
    dessiner();
    // Changement de source → déploiement (le seul chemin qui l'applique ET la persiste, cf.
    // ci-dessus). Tout le reste tient dans le hot-apply de fenêtre, sans coupure ni redeploy.
    if (_srcChange) hotApplyFull();
    else hotApplyWindow(primary);
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
    'meter_channels', 'meter_position', 'meter_inside', 'meter_opacity', 'meter_scale',
    ...ANC_FLAGS, 'anc_position', 'anc_opacity', 'template', 'template_ref'];
let reglagesClipboard = null;

// ─── Tissu de composition : régions de calcul et attente ─────────────────────
// Un mur lourd n'est pas composé par un seul conteneur : il est découpé en RÉGIONS, une par
// conteneur. Déplacer une fenêtre À L'INTÉRIEUR de sa région se propage à chaud — immédiat.
// Lui faire franchir une frontière change la découpe : il faut construire un conteneur, et la
// sortie ne bascule qu'une fois qu'il produit (~5-10 s). Sans rien montrer, le même geste paraît
// tantôt instantané, tantôt en panne — et rien à l'écran ne permet de deviner pourquoi. D'où
// deux ajouts : les frontières sont DESSINÉES (on voit qu'on va en franchir une), et l'attente
// est ANNONCÉE quand elle a lieu.
let mwFabric = { sharded: false, regions: [], etat: null };
let _mwFabricTimer = null;
let mwShowRegions = true;   // interrupteur (à côté du snap) — n'apparaît que sur un mur shardé

function mwToggleRegions(el) {
    mwShowRegions = !!el.checked;
    drawCanvas();
}

async function mwFabricRefresh() {
    if (editorVmid === null) return;
    let d;
    try {
        d = await (await fetch(`/api/containers/${editorVmid}/fabric`, { cache: 'no-store' })).json();
    } catch (e) { return; }   // réseau indisponible : on garde le dernier état connu, sans clignoter
    const avant = JSON.stringify(mwFabric);
    mwFabric = d || { sharded: false, regions: [], etat: null };
    const note = document.getElementById('mw_fabric_note');
    if (note) {
        note.hidden = (mwFabric.etat !== 'reorganisation');
        if (!note.hidden) note.textContent = T('plugin.multiview.fabric_reorganizing');
    }
    // L'interrupteur et la légende n'ont de sens que sur un mur SHARDÉ : sur un mur composé par
    // un seul conteneur il n'y a pas de région à montrer, et tout y est instantané.
    const montrable = !!(mwFabric.sharded && mwFabric.regions.length);
    const wrap = document.getElementById('ed_regions_wrap');
    if (wrap) wrap.hidden = !montrable;
    const hint = document.getElementById('mw_fabric_hint');
    if (hint) hint.hidden = !(montrable && mwShowRegions);
    if (JSON.stringify(mwFabric) !== avant) drawCanvas();
}

function mwFabricStart() {
    clearInterval(_mwFabricTimer);
    mwFabricRefresh();
    _mwFabricTimer = setInterval(mwFabricRefresh, 2500);
}

// Après une édition, la décision « mutation à chaud ou reconstruction ? » se prend en moins
// d'une seconde côté orchestrateur. Au rythme de veille (2,5 s), l'annonce arrivait avec
// jusqu'à 2,5 s de retard sur une attente qui n'en dure que quelques-unes — donc trop tard pour
// servir. On interroge serré le temps que ça se décide, et tant que ça dure.
let _mwFabricBurstLeft = 0;
function mwFabricBurst() {
    _mwFabricBurstLeft = 20;                    // 20 × 600 ms ≈ 12 s
    clearInterval(_mwFabricTimer);
    _mwFabricTimer = setInterval(async () => {
        await mwFabricRefresh();
        if (mwFabric.etat === 'reorganisation') _mwFabricBurstLeft = 20;   // ça dure → on suit
        if (--_mwFabricBurstLeft <= 0) mwFabricStart();
    }, 600);
}

// Frontières des régions : pointillés sobres, PAR-DESSUS les fenêtres (sous les fenêtres d'un mur
// dense, on ne les verrait pas). Plus marquées pendant une réorganisation, pour relier ce qu'on
// voit bouger au message affiché sous le canvas.
function drawFabricRegions(ctx, t) {
    if (!mwShowRegions || !mwFabric.sharded || !mwFabric.regions.length) return;
    ctx.save();
    ctx.setLineDash([14, 12]);
    ctx.lineWidth = 3;
    ctx.strokeStyle = t.muted;
    ctx.globalAlpha = (mwFabric.etat === 'reorganisation') ? 0.8 : 0.32;
    mwFabric.regions.forEach(r => {
        if (r.w > 2 && r.h > 2) ctx.strokeRect(r.x + 1.5, r.y + 1.5, r.w - 3, r.h - 3);
    });
    ctx.restore();
}

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

function _roundRectPath(ctx, x, y, w, h, r) {
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
    else ctx.rect(x, y, w, h);
}

function dessiner() {
    renderEntryTable();
    refreshEntryPanel();
    refreshOverlayPanel();
    refreshBlockPanel();
    refreshHistPanel();
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
// Aperçu schématique d'une fenêtre à MODÈLE de PiP : le composant vidéo en fond (couleur de la
// fenêtre), les autres composants en boîtes translucides étiquetées. L'édition fine du modèle se
// fait dans Réglages → PiP ; ici on montre l'encombrement dans le mur.
const _TPL_PREVIEW_COLORS = { umd: 'rgba(74,74,82,0.85)', tally: 'rgba(220,40,40,0.8)',
    meters: 'rgba(60,200,60,0.55)', anc: 'rgba(143,106,210,0.6)', clock: 'rgba(210,160,40,0.7)',
    text: 'rgba(200,200,208,0.5)', format: 'rgba(90,138,210,0.65)' };
function drawTemplatePreview(ctx, f, tpl, sel, isPrimary, t) {
    const comps = tpl.components;
    const px = c => ({ x: f.x + (c.x || 0) * f.w, y: f.y + (c.y || 0) * f.h,
                       w: Math.max(2, (c.w || 0) * f.w), h: Math.max(2, (c.h || 0) * f.h) });
    const vid = comps.find(c => c.type === 'video');
    if (vid) {
        const r = px(vid);
        ctx.globalAlpha = sel ? 0.8 : 0.4;
        ctx.fillStyle = f.color;
        ctx.fillRect(r.x, r.y, r.w, r.h);
        ctx.globalAlpha = 1;
    }
    comps.forEach(c => {
        if (c.type === 'video') return;
        const r = px(c);
        ctx.fillStyle = _TPL_PREVIEW_COLORS[c.type] || 'rgba(128,128,128,0.5)';
        ctx.fillRect(r.x, r.y, r.w, r.h);
    });
    // Nom + badge modèle dans la fenêtre
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 12px monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText(computeDisplayName(f) + ' · ' + (tpl.name || 'PiP'), f.x + 4, f.y + 4);
    ctx.textBaseline = 'alphabetic';
    // Cadre de sélection (même convention que les fenêtres legacy)
    ctx.strokeStyle = isPrimary ? '#ffffff' : (sel ? t.accent : f.color);
    ctx.lineWidth = sel ? 2 : 1;
    ctx.setLineDash(isPrimary ? [6, 4] : (sel ? [3, 3] : []));
    ctx.strokeRect(f.x, f.y, f.w, f.h);
    ctx.setLineDash([]);
}

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


    // Images de fond (overlay layer=background) : sous les fenêtres vidéo.
    drawOverlayLayer(ctx, 'background');

    // Cadre blanc + poignée : sur LA référence du mur (globale), pas une par famille.
    const primary = _winPrimary();
    editorParams.flux_config.forEach((f, i) => {
        if (f.hidden) return;   // entrée de la banque non affichée
        const sel = isSelected(i);
        const isPrimary = i === primary;
        // Modèle de PiP EFFECTIF (explicite ou hérité du mur) : aperçu schématique par composants.
        const efft = mwEffectiveTemplate(f);
        if (efft) {
            drawTemplatePreview(ctx, f, efft, sel, isPrimary, t);
            return;
        }
        const barOn = f.show_label || f.show_tally;
        // Aperçu du modèle « Classique » GÉNÉRÉ (miroir de _classic_comps, script.py) :
        // vidéo pleine cellule + bandeau nom translucide SUR le bas de l'image + pavés
        // tally + bande VU opt-in.
        const BAR_H      = Math.min(28, Math.max(14, Math.floor(f.h * 0.18)));
        const eff        = Math.max(6, Math.min(14, BAR_H - 4));
        const TALLY_SIZE = Math.max(6, Math.round(BAR_H * 0.7));
        const TALLY_PAD  = Math.max(2, Math.round(BAR_H * 0.25));
        const videoX = f.x, videoY = f.y, videoW = f.w, videoH = f.h;

        ctx.globalAlpha = sel ? 0.8 : 0.4;
        ctx.fillStyle   = f.color;
        ctx.fillRect(videoX, videoY, videoW, videoH);
        ctx.globalAlpha = 1;

        ctx.strokeStyle = isPrimary ? '#ffffff' : (sel ? t.accent : f.color);
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
            const mx = (f.meter_position === 'left') ? f.x + 2
                                                      : f.x + f.w - meterW - 2;
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

    // Blocs VU-mètres de mur : par-dessus les fenêtres (toujours manipulables), sous le texte.
    drawBlocksLayer(ctx, t);
    // Frises d'historique de mur : même couche que les blocs VU-mètres.
    drawHistLayer(ctx);

    // Overlays texte/horloge/logo (layer=foreground) : par-dessus les fenêtres.
    drawOverlayLayer(ctx, 'foreground');

    // Frontières des régions de calcul (murs shardés) — au-dessus des fenêtres pour rester
    // lisibles sur un mur dense, mais sous les guides de snap, qui sont l'aide ACTIVE du geste.
    drawFabricRegions(ctx, t);

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
    if (mwLectureSeule()) return;   // regarder oui, déplacer non
    // Pointer Events : capture du geste (souris, stylet ou doigt) jusqu'au relâchement.
    if (e.pointerId !== undefined && e.target.setPointerCapture) {
        try { e.target.setPointerCapture(e.pointerId); } catch(_) {}
    }
    e.preventDefault();
    dragGeomOrig = null;   // réarmé ci-dessous par les seules branches « fenêtre vidéo »
    const pos = getCanvasPos(e);
    // 1. Overlays de premier plan (au-dessus de la vidéo)
    let hit = hitOverlay(pos, 'foreground');
    if (hit) return beginOverlayDrag(hit, pos, e.shiftKey);
    // 2. Blocs de mur (VU-mètres, frises d'historique) — au-dessus des fenêtres, toujours manipulables
    const hhit = hitHist(pos);
    if (hhit) return beginHistDrag(hhit, pos, e.shiftKey);
    const bhit = hitBlock(pos);
    if (bhit) return beginBlockDrag(bhit, pos, e.shiftKey);
    // 3. Fenêtres vidéo
    const primary = _winPrimary();   // poignée de redimensionnement : sur LA référence, et elle seule
    for (let i = editorParams.flux_config.length - 1; i >= 0; i--) {
        const f = editorParams.flux_config[i];
        if (f.hidden) continue;   // entrées masquées : pas dans le canvas
        if (i === primary &&
            pos.x >= f.x + f.w - HANDLE_SIZE && pos.y >= f.y + f.h - HANDLE_SIZE) {
            dragMode = 'resize'; dragOverlay = false; dragBlock = false; dragStart = pos; dragOrigRect = {...f};
            dragGeomOrig = _geomSnapshot();
            _setCanvasCursor('nwse-resize');
            return;
        }
        if (pos.x >= f.x && pos.x <= f.x + f.w &&
            pos.y >= f.y && pos.y <= f.y + f.h) {
            // Saisir une fenêtre DÉJÀ sélectionnée (sans Maj) garde le groupe et la promeut en
            // référence ; Maj bascule son appartenance ; sinon clic = sélection simple.
            _selGrab('win', i, e.shiftKey);
            dragMode = 'move'; dragOverlay = false; dragBlock = false; dragStart = pos; dragOrigRect = {...f};
            _beginGroupDrag();
            dragGeomOrig = _geomSnapshot();
            _setCanvasCursor('move');
            dessiner();
            return;
        }
    }
    // 4. Images de fond (sous la vidéo)
    hit = hitOverlay(pos, 'background');
    if (hit) return beginOverlayDrag(hit, pos, e.shiftKey);
    if (!e.shiftKey) _selClear();   // clic dans le vide = tout désélectionner (Maj = ne rien changer)
    dragMode = null;
    dessiner();
}

function hitBlock(pos) {
    const bs = editorParams.meter_blocks || [];
    for (let i = bs.length - 1; i >= 0; i--) {
        const b = bs[i];
        if (b.hidden) continue;
        if (i === selectedBlock &&
            pos.x >= b.x + b.w - HANDLE_SIZE && pos.y >= b.y + b.h - HANDLE_SIZE) {
            return {i, mode: 'resize'};
        }
        if (pos.x >= b.x && pos.x <= b.x + b.w && pos.y >= b.y && pos.y <= b.y + b.h) {
            return {i, mode: 'move'};
        }
    }
    return null;
}

function beginBlockDrag(hit, pos, additive) {
    _selGrab('blk', hit.i, additive);
    dragBlock = true;
    dragMode = hit.mode;
    dragStart = pos;
    dragOrigRect = {...editorParams.meter_blocks[hit.i]};
    _beginGroupDrag();
    _setCanvasCursor(hit.mode === 'resize' ? 'nwse-resize' : 'move');
    dessiner();
}

function hitOverlay(pos, layer) {
    // MÊME ORDRE que le dessin (cf. drawOverlayLayer : les overlays dynamiques passent au-dessus),
    // parcouru à l'envers — on saisit ce qu'on voit. Sans ça, cliquer sur une horloge posée sur une
    // image aurait attrapé l'image, qui est pourtant derrière à l'écran.
    const ovs = (editorParams.overlays || []).map((o, i) => ({o, i}));
    ovs.sort((a, b) => (_ovDynamique(a.o) ? 1 : 0) - (_ovDynamique(b.o) ? 1 : 0));
    for (let k = ovs.length - 1; k >= 0; k--) {
        const {o, i} = ovs[k];
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

function beginOverlayDrag(hit, pos, additive) {
    _selGrab('ov', hit.i, additive);
    dragOverlay = true;
    dragMode = hit.mode;
    dragStart = pos;
    dragOrigRect = {...editorParams.overlays[hit.i]};
    _beginGroupDrag();
    _setCanvasCursor(hit.mode === 'resize' ? 'nwse-resize' : 'move');
    dessiner();
}

// Sélection d'une FENÊTRE : passe par l'API inter-familles (cf. `mwSel`). Maj+clic ajoute/retire
// sans toucher aux objets d'autres familles déjà sélectionnés — c'est là tout l'intérêt.
function toggleSelection(i, additive) {
    _selGrab('win', i, additive);
}

// Curseur attendu à une position donnée (hors drag) : coin bas-droite d'une fenêtre/overlay
// SÉLECTIONNÉ → redimensionnement ; corps → déplacement ; vide → défaut. Mire la logique de
// canvasMouseDown/hitOverlay pour que l'aspect du curseur corresponde TOUJOURS à l'action réelle.
function _cursorForPos(pos) {
    let hit = hitOverlay(pos, 'foreground');
    if (hit) return hit.mode === 'resize' ? 'nwse-resize' : 'move';
    const hhit = hitHist(pos);
    if (hhit) return hhit.mode === 'resize' ? 'nwse-resize' : 'move';
    const bhit = hitBlock(pos);
    if (bhit) return bhit.mode === 'resize' ? 'nwse-resize' : 'move';
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
    if (dragBlock) return blockMouseMove(e);
    if (dragHist) return histMouseMove(e);
    const primary = primaryIdx();
    if (primary < 0) return;
    const pos = getCanvasPos(e);
    const dx = pos.x - dragStart.x;
    const dy = pos.y - dragStart.y;
    const f = editorParams.flux_config[primary];
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;

    if (dragMode === 'move') {
        // Delta visé d'après l'objet TIRÉ (avec aimant), puis appliqué À TOUT LE GROUPE — lequel
        // peut désormais mêler fenêtres, textes, horloges, VU et frises.
        let nx = Math.max(0, Math.min(out_w - f.w, Math.round(dragOrigRect.x + dx)));
        let ny = Math.max(0, Math.min(out_h - f.h, Math.round(dragOrigRect.y + dy)));
        if (snapEnabled) {
            const snapped = computeSnap(_dragSkip('win', primary), nx, ny, f.w, f.h);
            nx = snapped.x; ny = snapped.y; snapGuides = snapped.guides;
        } else snapGuides = [];
        _applyGroupMove(nx, ny);
    } else if (dragMode === 'resize') {
        const ratio = dragOrigRect.w / dragOrigRect.h;
        let nw = Math.max(64, Math.min(out_w - f.x, Math.round(dragOrigRect.w + dx)));
        let nh = Math.round(nw / ratio);
        if (nh > out_h - f.y) { nh = out_h - f.y; nw = Math.round(nh * ratio); }
        if (snapEnabled) {
            const snapped = computeSnapResize({ kind: 'win', i: primary }, f.x, f.y, nw, nh, ratio);
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
    const skip = { kind: 'ov', i: selectedOverlay };
    if (dragMode === 'move') {
        let nx = Math.max(0, Math.min(out_w - o.w, Math.round(dragOrigRect.x + dx)));
        let ny = Math.max(0, Math.min(out_h - o.h, Math.round(dragOrigRect.y + dy)));
        if (snapEnabled) {
            const sn = computeSnap(_dragSkip('ov', selectedOverlay), nx, ny, o.w, o.h);
            nx = sn.x; ny = sn.y; snapGuides = sn.guides;
        } else snapGuides = [];
        _applyGroupMove(nx, ny);
    } else if (dragMode === 'resize') {
        // Image en « contain » : la boîte EST l'image, son format est donc contraint — plus de
        // marges vides entre le cadre et l'image. Les autres overlays (et les images en « cover »
        // ou « étirer ») restent libres : leur boîte a un sens propre.
        const rImg = _ovBoiteContrainte(o);
        let nw = Math.max(16, Math.min(out_w - o.x, Math.round(dragOrigRect.w + dx)));
        let nh = rImg > 0 ? Math.max(16, Math.round(nw / rImg))
                          : Math.max(16, Math.min(out_h - o.y, Math.round(dragOrigRect.h + dy)));
        if (rImg > 0 && o.y + nh > out_h) {          // débordement → on réduit à format constant
            nh = Math.max(16, out_h - o.y);
            nw = Math.max(16, Math.round(nh * rImg));
        }
        if (snapEnabled) {
            const sn = computeSnapResize(_dragSkip('ov', selectedOverlay), o.x, o.y, nw, nh, rImg, 16);
            nw = sn.w; nh = sn.h; snapGuides = sn.guides;
        } else snapGuides = [];
        o.w = nw % 2 === 0 ? nw : nw - 1;
        o.h = nh % 2 === 0 ? nh : nh - 1;
    }
    drawCanvas();
    syncGeomFields();
}

function blockMouseMove(e) {
    if (!dragMode || selectedBlock < 0) return;
    const pos = getCanvasPos(e);
    const dx = pos.x - dragStart.x, dy = pos.y - dragStart.y;
    const b = editorParams.meter_blocks[selectedBlock];
    const out_w = editorParams.out_width, out_h = editorParams.out_height;
    const skip = { kind: 'blk', i: selectedBlock };
    if (dragMode === 'move') {
        let nx = Math.max(0, Math.min(out_w - b.w, Math.round(dragOrigRect.x + dx)));
        let ny = Math.max(0, Math.min(out_h - b.h, Math.round(dragOrigRect.y + dy)));
        if (snapEnabled) {
            const sn = computeSnap(_dragSkip('blk', selectedBlock), nx, ny, b.w, b.h);
            nx = sn.x; ny = sn.y; snapGuides = sn.guides;
        } else snapGuides = [];
        _applyGroupMove(nx, ny);
    } else if (dragMode === 'resize') {
        let nw = Math.max(24, Math.min(out_w - b.x, Math.round(dragOrigRect.w + dx)));
        let nh = Math.max(24, Math.min(out_h - b.y, Math.round(dragOrigRect.h + dy)));
        if (snapEnabled) {
            const sn = computeSnapResize(skip, b.x, b.y, nw, nh, 0);
            nw = Math.max(24, sn.w); nh = Math.max(24, sn.h); snapGuides = sn.guides;
        } else snapGuides = [];
        b.w = nw % 2 === 0 ? nw : nw - 1;
        b.h = nh % 2 === 0 ? nh : nh - 1;
    }
    drawCanvas();
    syncGeomFields();
}

function canvasMouseUp() {
    _setCanvasCursor('default');   // le prochain survol réévaluera (move/resize/défaut)
    // Repères d'aimant remis à zéro dans TOUS les cas : seule la branche « fenêtre » le faisait,
    // si bien que le premier rendu du geste SUIVANT (dessiner() du mousedown, dragMode déjà posé)
    // retraçait les repères du geste précédent le temps d'une image.
    snapGuides = [];
    const bouge = _gestureChanged();
    if (dragOverlay || dragBlock || dragHist) {
        dragOverlay = dragBlock = dragHist = false; dragMode = null; dragGeomOrig = null;
        dessiner();
        // Overlays / blocs / frises : un seul canal de persistance (deploy → hot-apply
        // /overlays + /reconfigure, sans coupure). Il couvre aussi les FENÊTRES éventuellement
        // déplacées dans le même groupe — inutile d'y ajouter des POST /window par fenêtre.
        if (bouge) hotApplyFull();
        return;
    }
    dragMode = null; dessiner();
    // Groupe MIXTE tiré par une fenêtre : les objets d'autres familles ont bougé eux aussi, et
    // eux ne savent pas se hot-appliquer fenêtre par fenêtre → un seul déploiement couvre tout.
    if (bouge && mwSel.some(r => r.k !== 'win')) {
        dragGeomOrig = null;
        hotApplyFull();
        return;
    }
    // Ne hot-appliquer que les fenêtres dont la géométrie a RÉELLEMENT bougé pendant le geste.
    // Un clic de SÉLECTION passe aussi par ici (mousedown sur une fenêtre → mouseup sans
    // déplacement) : il ne doit rien poster. Sur un mur SHARDÉ, chaque POST /plugin/window
    // déclenche une re-planification du tissu du nœud — donc, pour une sélection, une coupure de
    // la sortie sans qu'aucun paramètre n'ait changé.
    const orig = dragGeomOrig;
    dragGeomOrig = null;
    if (!orig) return;
    selectedIdxs.forEach(idx => {
        const f = editorParams.flux_config[idx], o = orig[idx];
        if (!f) return;
        if (o && f.x === o.x && f.y === o.y && f.w === o.w && f.h === o.h) return;
        hotApplyWindow(idx);
    });
}

// Géométrie des fenêtres sélectionnées à l'instant T, indexée par idx (cf. dragGeomOrig).
function _geomSnapshot() {
    const snap = {};
    selectedIdxs.forEach(j => {
        const f = editorParams && editorParams.flux_config[j];
        if (f) snap[j] = { x: f.x, y: f.y, w: f.w, h: f.h };
    });
    return snap;
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
    if (dragBlock && selectedBlock >= 0) {
        const b = editorParams.meter_blocks[selectedBlock];
        if (!b) return;
        _mbSetVal('mb_x', b.x); _mbSetVal('mb_y', b.y);
        _mbSetVal('mb_w', b.w); _mbSetVal('mb_h', b.h);
        return;
    }
    if (dragHist && selectedHist >= 0) {
        const b = histList(selectedHistKind)[selectedHist];
        if (!b) return;
        _hbSetVal('hb_x', b.x); _hbSetVal('hb_y', b.y);
        _hbSetVal('hb_w', b.w); _hbSetVal('hb_h', b.h);
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
    // La barre compte l'UNION de la sélection, toutes familles confondues. Avant 0.111.0 elle ne
    // lisait que `selectedIdxs` (les fenêtres) : cliquer sur un texte, un VU ou une frise grisait
    // TOUS les outils — signalé cinq fois par un testeur le 2026-08-11, une fois par famille.
    const n = _selCount();
    document.querySelectorAll('#mw-toolbar .tool-btn[data-min]').forEach(b => {
        b.disabled = n < parseInt(b.dataset.min);
    });
    // Copier/coller des RÉGLAGES reste propre aux fenêtres : les champs copiés (bandeau, tally,
    // VU, ANC, modèle de PiP) n'existent pas sur un texte ou une frise.
    const nWin = selectedIdxs.length;
    const copy = document.getElementById('ed_copy_btn');
    if (copy) copy.disabled = nWin === 0;
    const paste = document.getElementById('ed_paste_btn');
    if (paste) paste.disabled = !reglagesClipboard || nWin === 0;
    const why = (n > 0 && nWin === 0) ? T('plugin.multiview.tools_windows_only') : '';
    const row = document.getElementById('mw_row_settings');
    if (row) row.title = why;
}

// Sélection d'une entrée au clavier depuis le tableau (Entrée / Espace).
function entryRowKey(ev, i) {
    if (ev.key !== 'Enter' && ev.key !== ' ') return;
    ev.preventDefault();
    selectEntry(i, ev);
}

function hotApplyWindow(idx) {
    if (editorVmid === null || !editorParams) return;
    if (mwLectureSeule()) return;   // lecture seule : aucun envoi, même déclenché autrement que par l'UI
    mwFabricBurst();
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
            // Source des VU-mètres (cf. sélecteur Source audio) : '' = auto (dérivé de path),
            // AUDIO_PATH_NONE = explicitement aucune, sinon flux explicite. Serveur : purge les
            // états audio ouverts sur un changement (même effet que le câblage page Câbles).
            audio_path:     f.audio_path ?? '',
            show_label:     !!f.show_label,
            show_tally:     !!f.show_tally,
            label_proportional: !!f.label_proportional,
            tsl_index:      f.tsl_index ?? 0,
            label_col:      f.label_col ?? 0,
            tally_level:    f.tally_level ?? '',
            tally_red:      !!f.tally_red,
            tally_green:    !!f.tally_green,
            meter_channels: f.meter_channels ?? 0,
            meter_position: f.meter_position || 'right',
            meter_inside:   !!f.meter_inside,
            meter_opacity:  f.meter_opacity ?? 100,
            meter_scale:    f.meter_scale || 'dbfs',
            // Métadonnées ANC par fenêtre (absentes du hot-apply avant 0.29.0 — les cases
            // cochées dans le composer n'atteignaient jamais le container).
            ...Object.fromEntries(ANC_FLAGS.map(k => [k, f[k] ? 1 : 0])),
            anc_position:   f.anc_position || 'bottom',
            anc_opacity:    f.anc_opacity ?? 60,
            // Modèle de PiP (null = hériter : défaut du mur, sinon « Classique » généré).
            // `template_ref` accompagne le modèle RÉSOLU : sans lui, persister le modèle sans sa
            // référence laisserait le sélecteur de l'éditeur sur l'ancienne entrée de
            // bibliothèque au rechargement — le mur rendrait B en affichant « A ».
            template:       f.template || null,
            template_ref:   f.template_ref || '',
        })
    }).catch(() => {});
}

function hotApplyFull() {
    // Passe par le endpoint deploy pour que _multiview_hot_apply résolve in_w/in_h via _shm_dims.
    deployerEditor();
}

function hotApplyStyle() {
    if (editorVmid === null || !editorParams) return;
    if (mwLectureSeule()) return;
    fetch(`/api/containers/${editorVmid}/plugin/style`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            show_no_signal:  editorParams.show_no_signal !== false,
            freeze_detect_s: editorParams.freeze_detect_s ?? 2,
            show_proxy:      !!editorParams.show_proxy,
            // Modèle de PiP PAR DÉFAUT du mur (héritage) — appliqué à chaud. Le ref accompagne
            // le modèle résolu pour que la persistance côté proxy (deploy_config) garde le lien
            // bibliothèque (sinon le select retombe sur '' au rechargement de l'éditeur).
            default_template: editorParams.default_template || null,
            default_template_ref: editorParams.default_template_ref || '',
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
    // Contenu par défaut d'un nouvel habillage texte : c'est une VALEUR (elle part dans la vidéo),
    // mais une valeur de CRÉATION — comme « Nouveau dossier » d'un explorateur de fichiers. Elle
    // suit donc la langue de celui qui crée, et ne bougera plus ensuite.
    if (kind === 'text')  Object.assign(o, { text: T('plugin.multiview.new_text_default', 'TEXTE'),
        text_source: 'local', tsl_index: 0,
        label_row: '', label_col: 0, tally_level: '', tally_red: false, tally_green: false });
    if (kind === 'clock') Object.assign(o, {
        // tz vide = fuseau du MUR (réglage système). Une horloge neuve suit donc le mur, et
        // n'affiche une autre ville que si l'utilisateur le demande explicitement.
        clock_source: 'ptp', tz: '', show_hh: true, show_mm: true, show_ss: true, show_ff: false,
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
    _selSet('ov', editorParams.overlays.length - 1);
    dessiner();
    hotApplyFull();
}

function supprimerOverlay() {
    if (selectedOverlay < 0) return;
    const _i = selectedOverlay;
    editorParams.overlays.splice(_i, 1);
    _selDrop('ov', _i);
    dessiner();
    hotApplyFull();
}

// Blocs VU-mètres de mur : édités en PIXELS canvas (comme les overlays), SÉRIALISÉS en fractions
// 0..1 du mur (contrat serveur, meter_blocks — cf. plugins/multiview/script.py render_meters).
function serializeMeterBlocks() {
    const ow = Math.max(1, editorParams.out_width), oh = Math.max(1, editorParams.out_height);
    const frac = (v, span) => Math.max(0, Math.min(1, +(v / span).toFixed(4)));
    return (editorParams.meter_blocks || []).map(b => ({
        x: frac(b.x, ow), y: frac(b.y, oh), w: frac(b.w, ow), h: frac(b.h, oh),
        channels: parseInt(b.channels) || 2,
        ch_start: Math.max(1, Math.min(16, parseInt(b.ch_start) || 1)),
        scale: b.scale || 'dbfs',
        opacity: Math.max(10, Math.min(100, numOu(b.opacity, 70))),
        align: b.align || 'left',
        width_mode: b.width_mode || 'auto',
        audio_path: b.audio_path || '',
        label: b.label || '',
        hidden: !!b.hidden,
    }));
}

function serializeOverlays() {
    const ev = v => { v = parseInt(v) || 0; return v % 2 === 0 ? v : v - 1; };
    const clamp = (v, d) => Math.max(0, Math.min(100, numOu(v, d)));
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
            tally_level: o.tally_level || '',
            tally_red: !!o.tally_red, tally_green: !!o.tally_green });
        if (o.kind === 'clock') Object.assign(base, {
            clock_source: o.clock_source || 'ptp',
            tz: o.tz || '',
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

// Format NATIF d'une image importée (0 si pas encore décodée). Sert à faire coller la boîte à
// l'image : en mode « contain », toute marge est du vide, et c'est ce vide qui faisait dire au
// testeur que « l'encadré est plus grand que l'image et empêche de la redimensionner » — la
// poignée se trouvait au coin de la boîte, loin du coin visible de l'image (2026-08-11).
function _ovImageRatio(o) {
    if (!o || o.kind !== 'image') return 0;
    const img = _ovImg(o);
    if (!img || !img.complete || !img.naturalWidth || !img.naturalHeight) return 0;
    return img.naturalWidth / img.naturalHeight;
}
// La boîte doit-elle suivre le format de l'image ? Uniquement en « contain » : « cover » recadre
// volontairement, « étirer » déforme volontairement — dans les deux cas la boîte a un sens propre.
function _ovBoiteContrainte(o) {
    return (o && o.kind === 'image' && (o.fit || 'contain') === 'contain') ? _ovImageRatio(o) : 0;
}
// Ajuste la boîte au format `r` en gardant la LARGEUR (et en rentrant dans le cadre).
function _ovColleAuFormat(o, r) {
    if (!(r > 0)) return;
    const ow = editorParams.out_width, oh = editorParams.out_height;
    let w = Math.min(o.w, ow - o.x), h = Math.round(w / r);
    if (o.y + h > oh) { h = oh - o.y; w = Math.round(h * r); }
    o.w = Math.max(16, w % 2 === 0 ? w : w - 1);
    o.h = Math.max(16, h % 2 === 0 ? h : h - 1);
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

// Un overlay est-il DYNAMIQUE au sens du moteur ? Miroir exact de `_dyn_overlays()`
// (script.py) : horloges et textes à variables. Le moteur les rend en tuiles PER-FRAME, blendées
// APRÈS la couche d'habillage bakée qui porte les images et les textes fixes. Sur l'écran, une
// horloge est donc TOUJOURS au-dessus d'une image, quel que soit l'ordre de la liste — alors que
// l'éditeur, lui, dessinait dans l'ordre du tableau et pouvait montrer l'image par-dessus
// l'horloge (signalé le 2026-08-11). L'aperçu suit désormais l'ordre du moteur.
// ⚠ Ce n'est pas un choix d'ergonomie mais une CONSÉQUENCE du découpage bake / per-frame, sur
// lequel repose la tenue de cadence : la couche bakée ne se redessine pas à chaque trame.
function _ovDynamique(o) {
    return o.kind === 'clock' || (o.kind === 'text' && String(o.text || '').includes('%'));
}

function drawOverlayLayer(ctx, layer) {
    const t = _tok || _readTokens();
    // Index d'origine conservé (sélection, poignée) : on ne réordonne que le DESSIN.
    const ordonnes = (editorParams.overlays || []).map((o, i) => ({o, i}));
    ordonnes.sort((a, b) => (_ovDynamique(a.o) ? 1 : 0) - (_ovDynamique(b.o) ? 1 : 0));
    ordonnes.forEach(({o, i}) => {
        const isBg = (o.kind === 'image' && o.layer === 'background');
        if (layer === 'background' ? !isBg : isBg) return;
        const sel = _selHas('ov', i);
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
            // Police RÉELLE de l'overlay (bibliothèque incluse : @font-face posé par BobiFonts) —
            // sinon l'aperçu mentirait sur le rendu du conteneur.
            const fam = window.BobiFonts ? window.BobiFonts.cssFamily(o.font) : 'sans-serif';
            ctx.font = `bold ${fs}px ${fam}`;
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

// ── Rendu des blocs VU-mètres de mur sur le canvas de l'éditeur ──
// ★ GÉOMÉTRIE FIDÈLE AU MOTEUR (0.111.0). L'aperçu étalait `n` barres sur toute la largeur du
// bloc, ignorant la colonne de graduations, la largeur RÉELLE des barres, le mode de largeur et
// l'alignement : « la largeur des peak meters et l'alignement ne sont pas représentatifs de ce
// qui est diffusé » (2026-08-11). Les constantes et le calcul ci-dessous sont le miroir exact de
// `_meter_grad`, `_meter_layout`, `_meter_fit_dims` et `_draw_meter` (script.py) — mêmes valeurs,
// mêmes seuils, même ordre. Seuls les NIVEAUX restent fictifs : l'éditeur n'a pas de flux audio.
// ⚠ Deux copies de la même géométrie : toute correction faite dans script.py doit être portée ici.
const MW_METER = {
    BAR_W: 5, GAP: 1,          // METER_BAR_W / METER_GAP
    TICK_DBFS: 22, TICK_PPM: 26, TICK_MARKS: 6,   // METER_TICK_W_*
    LABEL_H: 12,               // bande des numéros de canal, sous les barres
};
// Niveau de graduations EFFECTIF et largeur de colonne — miroir de `_meter_grad`. En mode auto,
// la décision se prend sur la place réellement allouée (seuil 40 % de la largeur du bloc).
function _mwMeterGrad(b, scale, rw) {
    const mode = String(b.graduations || 'auto').toLowerCase();
    const twFull = (scale === 'ppm') ? MW_METER.TICK_PPM : MW_METER.TICK_DBFS;
    if (mode === 'full')  return {grad: 'full',  tick: twFull};
    if (mode === 'marks') return {grad: 'marks', tick: MW_METER.TICK_MARKS};
    if (mode === 'none')  return {grad: 'none',  tick: 0};
    if (rw > 0 && twFull <= rw * 0.40)              return {grad: 'full',  tick: twFull};
    if (rw > 0 && MW_METER.TICK_MARKS <= rw * 0.40) return {grad: 'marks', tick: MW_METER.TICK_MARKS};
    return {grad: 'none', tick: 0};
}
function _mwMeterLayout(n, tick, bar, gap) { return tick + n * bar + (n - 1) * gap; }
// Mode `fit` : SEULES les barres s'élargissent — colonne et espacement restent fixes, sinon
// l'échelle dB et ses repères se déforment. Repli sur la largeur intrinsèque si c'est trop étroit.
function _mwMeterFitDims(n, rw, tick) {
    const gap = MW_METER.GAP;
    const bar0 = Math.floor((rw - tick - (n - 1) * gap) / n);
    const bar = (bar0 < 2) ? MW_METER.BAR_W : bar0;
    return {tick, bar, gap, mw: _mwMeterLayout(n, tick, bar, gap)};
}
// Géométrie complète d'un bloc VU : ce que le moteur dessinera, aux mêmes pixels près.
function mwMeterGeom(b) {
    const n = Math.max(1, Math.min(16, parseInt(b.channels) || 2));
    const rw = b.w, rh = b.h;
    const scale = b.scale || 'dbfs';
    const g = _mwMeterGrad(b, scale, rw);
    const side = String(b.grad_side || 'left').toLowerCase();
    const tick = (side === 'inside' || g.grad === 'none') ? 0 : g.tick;   // « inside » : colonne superposée
    const mh = Math.max(20, rh - 1);
    let dims, mx;
    if ((b.width_mode || 'auto') === 'fit') {
        dims = _mwMeterFitDims(n, rw, tick);
        mx = b.x;
    } else {
        dims = {tick, bar: MW_METER.BAR_W, gap: MW_METER.GAP,
                mw: _mwMeterLayout(n, tick, MW_METER.BAR_W, MW_METER.GAP)};
        const al = b.align || 'left';
        mx = b.x + (al === 'center' ? Math.floor((rw - dims.mw) / 2)
                  : al === 'right'  ? (rw - dims.mw) : 0);
    }
    // Barres après la colonne quand elle est à gauche ; dès le bord quand elle est à droite
    // (ou absente — tick vaut alors 0 et les deux expressions coïncident).
    const barX0 = (side === 'right') ? mx : mx + dims.tick;
    return {n, mx, my: b.y, mw: dims.mw, mh, tick: dims.tick, bar: dims.bar, gap: dims.gap,
            side, grad: g.grad, barX0, barsH: Math.max(20, mh - MW_METER.LABEL_H)};
}

function drawBlocksLayer(ctx, t) {
    (editorParams.meter_blocks || []).forEach((b, i) => {
        if (b.hidden) return;
        const sel = _selHas('blk', i);
        const G = mwMeterGeom(b);
        ctx.save();
        // Emprise du BLOC (la zone que l'utilisateur pose) — teintée, discrète.
        ctx.fillStyle = 'rgba(20,160,180,0.10)';
        ctx.fillRect(b.x, b.y, b.w, b.h);
        // Emprise du MÈTRE lui-même : là où le moteur peindra vraiment. C'est tout l'objet du
        // correctif — le mètre n'occupe pas forcément tout le bloc, et sa position dans le bloc
        // dépend de l'alignement.
        ctx.fillStyle = 'rgba(0,0,0,0.55)';
        ctx.fillRect(G.mx, G.my, G.mw, G.mh);
        // Colonne de graduations (traits seuls ou traits + chiffres, selon la place disponible).
        if (G.tick > 0) {
            const gx = (G.side === 'right') ? (G.mx + G.mw - G.tick) : G.mx;
            ctx.fillStyle = 'rgba(255,255,255,0.10)';
            ctx.fillRect(gx, G.my, G.tick, G.mh);
            ctx.strokeStyle = 'rgba(220,220,220,0.55)';
            ctx.lineWidth = 1;
            for (let k = 0; k <= 4; k++) {
                const y = Math.round(G.my + G.barsH * k / 4) + 0.5;
                const x1 = (G.side === 'right') ? gx + 4 : gx + G.tick;
                const x0 = (G.side === 'right') ? gx : gx + G.tick - 4;
                ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
            }
        }
        // Barres : largeur et espacement RÉELS. Les niveaux, eux, sont fictifs (pas d'audio ici).
        const bottom = G.my + G.barsH;
        for (let ch = 0; ch < G.n; ch++) {
            const bx = G.barX0 + ch * (G.bar + G.gap);
            const lvl = 0.35 + 0.3 * Math.abs(Math.sin(ch + 1));
            const bh = Math.max(2, G.barsH * lvl);
            ctx.fillStyle = 'rgba(90,210,140,0.85)';
            ctx.fillRect(bx, bottom - bh, G.bar, bh);
        }
        // Bande des numéros de canal (12 px réservés par le moteur sous les barres).
        ctx.fillStyle = 'rgba(220,220,220,0.30)';
        ctx.fillRect(G.mx, bottom, G.mw, Math.min(MW_METER.LABEL_H, G.my + G.mh - bottom));
        ctx.strokeStyle = sel ? '#ffffff' : '#20c4d8';
        ctx.lineWidth = sel ? 2 : 1;
        ctx.setLineDash(sel ? [6, 4] : [4, 3]);
        ctx.strokeRect(b.x, b.y, b.w, b.h);
        ctx.setLineDash([]);
        ctx.fillStyle = '#20c4d8'; ctx.font = '10px monospace';
        ctx.fillText('VU ' + (b.label || (i + 1)), b.x + 3, b.y + 11);
        if (sel) {
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(b.x + b.w - HANDLE_SIZE, b.y + b.h - HANDLE_SIZE, HANDLE_SIZE, HANDLE_SIZE);
        }
        ctx.restore();
    });
}

// ── Panneau de propriétés d'un bloc VU-mètres de mur ──
function _mbSetVal(id, v) { const e = document.getElementById(id); if (e) e.value = v; }

function refreshBlockPanel() {
    const panel = document.getElementById('ed_mb_panel');
    if (!panel) return;
    if (selectedBlock < 0 || !editorParams.meter_blocks || !editorParams.meter_blocks[selectedBlock]) {
        panel.hidden = true; return;
    }
    const b = editorParams.meter_blocks[selectedBlock];
    panel.hidden = false;
    _mbSetVal('mb_x', b.x); _mbSetVal('mb_y', b.y);
    _mbSetVal('mb_w', b.w); _mbSetVal('mb_h', b.h);
    _mbSetVal('mb_channels', String(b.channels ?? 2));
    _mbSetVal('mb_ch_start', b.ch_start ?? 1);
    _mbSetVal('mb_scale', b.scale || 'dbfs');
    _mbSetVal('mb_opacity', b.opacity ?? 100);
    _mbSetVal('mb_align', b.align || 'left');
    _mbSetVal('mb_width_mode', b.width_mode || 'auto');
    _mbSetVal('mb_label', b.label || '');
    const mbHid = document.getElementById('mb_hidden');
    if (mbHid) mbHid.checked = !!b.hidden;
    // Source audio : flux explicite de la flotte (page Câbles peut aussi câbler ce port) —
    // pas de dérivation « auto » possible ici (un bloc de mur n'a pas de source vidéo).
    const sel = document.getElementById('mb_audio_path');
    if (sel) {
        const opts = audioSources.map(s => {
            const base = s.label ? `${s.hostname} → ${s.label} (${s.shm})` : `${s.hostname} → ${s.shm}`;
            const txt = window.SourceLabels ? window.SourceLabels.display(s.shm, base) : base;
            return `<option value="${escapeHtml('/dev/shm/' + s.shm)}">${escapeHtml(txt)}</option>`;
        });
        opts.unshift('<option value="">' + escapeHtml(T('plugin.multiview.mb_audio_none_option')) + '</option>');
        const cur = b.audio_path || '';
        if (cur && !audioSources.some(s => '/dev/shm/' + s.shm === cur)) {
            opts.splice(1, 0, `<option value="${escapeHtml(cur)}">${escapeHtml(cur)}</option>`);
        }
        sel.innerHTML = opts.join('');
        sel.value = cur;
    }
}

function onBlockChange() {
    if (selectedBlock < 0) return;
    const b = editorParams.meter_blocks[selectedBlock];
    if (!b) return;
    b.channels = parseInt(document.getElementById('mb_channels').value) || 2;
    b.ch_start = Math.max(1, Math.min(16, parseInt(document.getElementById('mb_ch_start').value) || 1));
    b.scale = document.getElementById('mb_scale').value;
    b.opacity = Math.max(10, Math.min(100, numOu(document.getElementById('mb_opacity').value, 70)));
    b.align = document.getElementById('mb_align').value;
    b.width_mode = document.getElementById('mb_width_mode').value;
    b.label = document.getElementById('mb_label').value || '';
    // Masquer SANS supprimer : le moteur saute les blocs `hidden` (script.py, render_meters).
    b.hidden = !!document.getElementById('mb_hidden').checked;
    b.audio_path = document.getElementById('mb_audio_path').value || '';
    dessiner();
    hotApplyFull();
}

function onBlockGeomChange() {
    if (selectedBlock < 0) return;
    const b = editorParams.meter_blocks[selectedBlock];
    if (!b) return;
    const ow = editorParams.out_width, oh = editorParams.out_height;
    b.x = Math.max(0, Math.min(ow - 2, parseInt(document.getElementById('mb_x').value) || 0));
    b.y = Math.max(0, Math.min(oh - 2, parseInt(document.getElementById('mb_y').value) || 0));
    b.w = Math.max(24, Math.min(ow - b.x, parseInt(document.getElementById('mb_w').value) || 24));
    b.h = Math.max(24, Math.min(oh - b.y, parseInt(document.getElementById('mb_h').value) || 24));
    dessiner();
    hotApplyFull();
}

// ── Panneau de propriétés overlay ──
// ─── Sélecteur de FUSEAU HORAIRE d'une horloge ─────────────────────────────────────────────
// La liste vient de /timezones (proxy plugin) = la tzdata RÉELLEMENT présente dans l'image du
// conteneur, jamais une liste codée en dur : proposer un fuseau absent ferait afficher l'heure du
// mur sans le moindre signal. Chargée UNE fois puis mémorisée (486 entrées, ~15 ko).
let _tzCache = null, _tzPending = null;
function _tzLoad() {
    if (_tzCache) return Promise.resolve(_tzCache);
    if (_tzPending) return _tzPending;
    _tzPending = fetch(`/api/containers/${editorVmid}/plugin/timezones`)
        .then(r => r.ok ? r.json() : {items: []})
        .then(j => { _tzCache = j.items || []; return _tzCache; })
        .catch(() => []);
    return _tzPending;
}
// Peuple le <select> et sélectionne `val`. Les options sont groupées par région (une liste plate
// de ~500 entrées est inutilisable) ; l'entrée vide « fuseau du mur » reste en tête.
// Fuseaux « UTC±N » PRÉSENTABLES. La liste servie par le conteneur est la tzdata brute
// (`zoneinfo.available_timezones()`), et elle porte trois pièges signalés le 2026-08-11 :
//   1. ★ LE SIGNE EST INVERSÉ. `Etc/GMT+2` vaut UTC−2 — c'est la convention POSIX, pas un bug de
//      tzdata : le signe y est celui de l'opération à appliquer au temps local pour revenir à UTC.
//      Affichée telle quelle, l'option « GMT+2 » posait donc une horloge à UTC−2. Vérifié sur ce
//      contrôleur : Etc/GMT+2 → décalage réel UTC−2, Etc/GMT-2 → UTC+2.
//   2. Des doublons : `Etc/GMT`, `Etc/GMT+0` et `Etc/GMT-0` désignent tous UTC, plus les alias
//      sans région (GMT, GMT0, Zulu, Universal, Greenwich, UCT…).
//   3. Des valeurs qui semblent absurdes lues comme des offsets : `Etc/GMT-13` et `Etc/GMT-14`
//      sont réels — ce sont UTC+13 et UTC+14 (Tonga, Kiribati) — mais illisibles ainsi.
// On construit donc un groupe « UTC » SYNTHÉTIQUE, étiqueté dans le sens que lit un humain
// (UTC+2 = deux heures d'avance), dont la VALEUR reste le nom de zone correct. Les zones
// géographiques (Europe/…, America/…) passent inchangées : elles portent l'heure d'été, ce qu'un
// offset fixe ne fait pas.
const _TZ_ALIAS_PLATS = new Set(['GMT', 'GMT0', 'GMT+0', 'GMT-0', 'UCT', 'UTC', 'Universal',
                                 'Greenwich', 'Zulu']);
function _tzOffsetOptions(items) {
    // Zones réellement servies (donc réellement présentes dans l'image du conteneur).
    const dispo = new Set(items.map(it => it.value));
    const out = [];
    if (dispo.has('UTC')) out.push({value: 'UTC', label: 'UTC (±0)'});
    for (let n = 14; n >= -12; n--) {
        if (n === 0) continue;                       // couvert par « UTC (±0) » ci-dessus
        const zone = 'Etc/GMT' + (n > 0 ? '-' : '+') + Math.abs(n);   // ★ signe INVERSÉ (POSIX)
        if (dispo.has(zone)) out.push({value: zone, label: 'UTC' + (n > 0 ? '+' : '−') + Math.abs(n)});
    }
    return out;
}

// Peuple le <select> et sélectionne `val`. Les options sont groupées par région (une liste plate
// de ~500 entrées est inutilisable) ; l'entrée vide « fuseau du mur » reste en tête.
function _tzFill(id, val) {
    const el = document.getElementById(id);
    if (!el) return;
    _tzLoad().then(items => {
        if (el.dataset.filled !== '1') {
            el.innerHTML = '';
            const groups = new Map();
            const mkGroup = nom => {
                if (!groups.has(nom)) {
                    const g = document.createElement('optgroup');
                    g.label = nom; groups.set(nom, g); el.appendChild(g);
                }
                return groups.get(nom);
            };
            const vide = items.find(it => !it.value);
            if (vide) el.appendChild(new Option(vide.label, ''));   // « hérite du mur », en tête
            // Décalages fixes, en premier : c'est ce que cherche un opérateur pressé.
            _tzOffsetOptions(items).forEach(o => mkGroup('UTC').appendChild(new Option(o.label, o.value)));
            items.forEach(it => {
                if (!it.value) return;
                // Alias plats et famille Etc/ : remplacés par le groupe UTC ci-dessus.
                if (it.value.startsWith('Etc/') || _TZ_ALIAS_PLATS.has(it.value)) return;
                const reg = it.value.includes('/') ? it.value.split('/')[0] : 'Autres';
                mkGroup(reg).appendChild(new Option(it.label, it.value));
            });
            el.dataset.filled = '1';
        }
        // Valeur enregistrée absente des options (alias écarté, zone retirée de la tzdata) : on
        // l'ajoute telle quelle plutôt que de laisser le select retomber en silence sur « hérite
        // du mur » — l'horloge afficherait alors autre chose que ce que la base contient.
        if (val && !Array.from(el.options).some(o => o.value === val)) {
            el.appendChild(new Option(val, val));
        }
        el.value = val || '';
    });
}
// ─── Insertion de VARIABLES dans un champ texte ────────────────────────────────────────────
// Catalogue servi par /api/text-variables : défini UNE SEULE FOIS côté orchestrateur et partagé
// avec l'éditeur de modèles de PiP. Ici on ne propose que le groupe « Système » : les variables
// de source décrivent la source d'une FENÊTRE, et un overlay de mur n'en a pas.
let _mvVarCat = null, _mvVarLoading = null;
function _mvLoadVars() {
    if (_mvVarCat) return Promise.resolve(_mvVarCat);
    if (_mvVarLoading) return _mvVarLoading;
    _mvVarLoading = fetch('/api/text-variables')
        .then(r => (r.ok ? r.json() : { system: [] }))
        .then(j => { _mvVarCat = j; return _mvVarCat; })
        .catch(() => ({ system: [] }));
    return _mvVarLoading;
}
// Écrit %nom% à la position du curseur dans #ov_text, puis déclenche la persistance comme une
// frappe utilisateur (onOverlayTextInput + onOverlayChange), sinon la saisie serait perdue.
function _mvFillVarSelect() {
    const sel = document.getElementById('ov_vars');
    if (!sel || sel.dataset.filled === '1') return;
    _mvLoadVars().then(cat => {
        sel.innerHTML = '';
        sel.appendChild(new Option(T('plugin.multiview.insert_var_pick') || 'Insérer une variable…', ''));
        // Un menu se parcourt à la souris : une liste plate de trente entrées se lit mal. On
        // groupe donc par question posée — ce conteneur, la machine qui le porte, ses liens, le
        // contrôleur. Les variables de SOURCE restent absentes : elles décrivent la source d'une
        // FENÊTRE, et un overlay de mur n'en a pas.
        const grp = (titre, liste) => {
            if (!liste || !liste.length) return;
            const og = document.createElement('optgroup');
            og.label = titre;
            liste.forEach(v => og.appendChild(new Option(v.label + '  (%' + v.name + '%)', v.name)));
            sel.appendChild(og);
        };
        grp(T('plugin.multiview.vars_grp_sys') || 'Ce conteneur', cat.system);
        grp(T('plugin.multiview.vars_grp_node') || 'Nœud', cat.noeud);
        grp(T('plugin.multiview.vars_grp_rdma') || 'RDMA', cat.rdma);
        grp(T('plugin.multiview.vars_grp_orch') || 'Orchestrateur', cat.orchestrateur);
        sel.dataset.filled = '1';
        sel.onchange = () => {
            const v = sel.value; sel.value = '';
            const inp = document.getElementById('ov_text');
            if (!v || !inp) return;
            const tok = '%' + v + '%';
            const p0 = inp.selectionStart, p1 = inp.selectionEnd;
            inp.value = (p0 === null) ? (inp.value + tok)
                                      : inp.value.slice(0, p0) + tok + inp.value.slice(p1);
            if (p0 !== null) { const np = p0 + tok.length; inp.setSelectionRange(np, np); }
            inp.focus();
            if (typeof onOverlayTextInput === 'function') onOverlayTextInput();
            onOverlayChange();
        };
    });
}

function _ovSetVal(id, v) { const e = document.getElementById(id); if (e) e.value = v; }
function _ovSetChk(id, v) { const e = document.getElementById(id); if (e) e.checked = !!v; }
// Pose la police d'un overlay dans le select. Si la clé n'est pas (encore / plus) au catalogue
// — police de la bibliothèque supprimée, ou API indisponible — on AJOUTE une option explicite
// plutôt que de laisser le select vide, qui écraserait silencieusement le choix de l'utilisateur.
function _ovSetFont(id, key) {
    const sel = document.getElementById(id);
    if (!sel) return;
    if (key && !Array.from(sel.options).some(o => o.value === key)) {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = key + ' — ' + T('plugin.multiview.font_missing');
        sel.appendChild(opt);
    }
    sel.value = key;
}

// Sélecteur de police des overlays TEXTE et HORLOGE : alimenté par le catalogue
// /api/fonts (window.BobiFonts) = polices de l'image runtime + bibliothèque téléversée
// (Réglages → Polices, clés `lib:<sha16>`). Repli sur la liste figée OVERLAY_FONTS si l'API
// est inaccessible (le contrôle reste utilisable, jamais un select vide).
function _ovFillFonts() {
    const sel = document.getElementById('ov_font');
    if (!sel || sel.dataset.filled) return;       // `filled` = le CATALOGUE est posé (pas le repli)
    // Repli AFFICHÉ tout de suite (le contrôle reste utilisable si l'API tarde ou échoue) — mais on
    // ne marque PAS le select comme rempli : sinon, si BobiFonts n'est pas encore défini au premier
    // appel (ordre de chargement) ou si /api/fonts échoue, on resterait à VIE sur la liste figée des
    // 10 polices d'image, sans jamais voir les polices de la BIBLIOTHÈQUE — et le garde-fou du haut
    // interdirait toute nouvelle tentative. On retente donc à chaque ouverture du panneau.
    if (!sel.options.length) {
        sel.innerHTML = OVERLAY_FONTS.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
    }
    if (!window.BobiFonts) return;
    window.BobiFonts.load().then(({ fonts }) => {
        if (!fonts.length) return;
        const o = (editorParams.overlays || [])[selectedOverlay];
        window.BobiFonts.fillSelect(sel, (o && o.font) || 'dejavu-sans-bold');
        sel.dataset.filled = '1';                 // catalogue RÉELLEMENT posé → plus rien à faire
        dessiner();   // l'aperçu doit refléter la police réelle dès que les @font-face sont posées
    }).catch(() => { /* catalogue indisponible : on garde le repli et on retentera */ });
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
    _ovSetFont('ov_font', o.font || 'dejavu-sans-bold');
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
    _ovSetVal('ov_tally_level', o.tally_level || '');
    _ovSetVal('ov_tally_colors', o.tally_red && o.tally_green ? 'both'
                               : o.tally_red ? 'red' : o.tally_green ? 'green' : 'none');
    _ovSetVal('ov_clock_source', o.clock_source || 'ptp');
    _ovSetChk('ov_show_hh', o.show_hh !== false);
    _ovSetChk('ov_show_mm', o.show_mm !== false);
    _ovSetChk('ov_show_ss', o.show_ss !== false);
    _ovSetChk('ov_show_ff', !!o.show_ff);
    _ovSetVal('ov_offset_ms', o.offset_ms || 0);
    _tzFill('ov_tz', o.tz || '');
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
    // Le fuseau n'a de sens que pour l'HEURE DU JOUR : un chrono, un décompte ou un timecode
    // embarqué sont des durées/valeurs absolues, les décaler d'un fuseau n'aurait aucun sens.
    sub('ov_tz_grp',       o.kind === 'clock' && o.clock_source === 'ptp');
    // Variables : seulement sur un texte LOCAL (un texte piloté par TSL vient du protocole).
    sub('ov_vars_grp',     o.kind === 'text' && (o.text_source || 'local') === 'local');
    if (o.kind === 'text') _mvFillVarSelect();
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
        o.bg_opacity = Math.max(0, Math.min(100, numOu(g('ov_bg_opacity').value, 100)));
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
        // Même piège que les tuiles : un niveau est un UUID, `parseInt` rend NaN → 0.
        o.tally_level = g('ov_tally_level').value || '';
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
        o.tz = g('ov_tz').value || '';
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
        o.opacity = Math.max(0, Math.min(100, numOu(g('ov_opacity').value, 100)));
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
            const ratioNatif = (w > 0 && h > 0) ? w / h : 0;
            const scale = Math.min(1, maxSide / Math.max(w, h));
            const cw = Math.max(1, Math.round(w * scale));
            const ch = Math.max(1, Math.round(h * scale));
            const cv = document.createElement('canvas');
            cv.width = cw; cv.height = ch;
            cv.getContext('2d').drawImage(img, 0, 0, cw, ch);
            cb((cv.toDataURL('image/png').split(',')[1]) || '', file.name, ratioNatif);
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
    _downscaleToB64(file, maxSide, (b64, name, ratioNatif) => {
        o.image_b64 = b64;
        o.image_name = name;
        // La boîte prend le FORMAT du fichier importé : sans ça elle gardait ses 20 % × 20 % du
        // mur, l'image s'y affichait avec des marges (mode « contain ») et la poignée de
        // redimensionnement se retrouvait loin du coin visible de l'image.
        if ((o.fit || 'contain') === 'contain') _ovColleAuFormat(o, ratioNatif);
        input.value = '';   // autorise la réimportation du même fichier
        dessiner();
        hotApplyFull();
    });
}

// ─── Snap ──────────────────────────────────────────────────────

// Rectangles de TOUS les objets posés sur le mur, pour le snap. Le composeur ne considérait que
// `flux_config` : une horloge, un champ texte, une image ou un bloc VU n'était donc NI une cible de
// snap, NI snappé lui-même (overlayMouseMove/blockMouseMove n'appelaient pas le snap du tout). On
// alignait à l'œil tout ce qui n'était pas une fenêtre vidéo.
// `skip` = {kind, i} de l'objet en cours de déplacement, à exclure de ses propres cibles.
function _snapRects(skip) {
    const out = [];
    // `skip` = une exclusion {kind,i} ou une LISTE (déplacement de groupe : chaque membre bouge,
    // aucun ne doit servir de cible aux autres).
    const skips = !skip ? [] : (Array.isArray(skip) ? skip : [skip]);
    const add = (kind, list) => (list || []).forEach((o, i) => {
        if (!o || o.hidden) return;
        if (skips.some(sk => sk && sk.kind === kind && sk.i === i)) return;
        if (!(o.w > 0) || !(o.h > 0)) return;
        out.push(o);
    });
    add('win', editorParams.flux_config);
    add('ov',  editorParams.overlays);
    add('blk', editorParams.meter_blocks);
    add('vh',  editorParams.video_history_blocks);
    add('ah',  editorParams.audio_history_blocks);
    return out;
}

function computeSnap(skip, x, y, w, h) {
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;
    const xTargets = [0, out_w, out_w / 2];     // bords + centre canvas
    const yTargets = [0, out_h, out_h / 2];
    _snapRects(skip).forEach(o => {
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

// `minPx` = plancher de dimension : 64 px pour une fenêtre vidéo (défaut historique), plus bas
// pour un objet posé — une pastille, un petit logo sont légitimes.
function computeSnapResize(skip, x, y, w, h, ratio, minPx) {
    // Snap uniquement le coin bas-droit (resize est ancré haut-gauche)
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;
    const xTargets = [out_w, out_w / 2];
    const yTargets = [out_h, out_h / 2];
    _snapRects(skip).forEach(o => {
        xTargets.push(o.x, o.x + o.w);
        yTargets.push(o.y, o.y + o.h);
    });
    const guides = [];
    // ratio > 0 : objet à FORMAT CONTRAINT (fenêtre vidéo). Le coin bas-droit suit la souris,
    // donc les DEUX bords bougent — mais un rectangle à ratio fixe ne peut satisfaire qu'UNE
    // cible à la fois. On retient la plus PROCHE DU COIN (bord droit vs cibles verticales,
    // bord bas vs cibles horizontales, distance en pixels canvas), puis on recalcule l'autre
    // dimension au ratio : les deux axes bougent, un seul est aimanté.
    // ⚠ Avant 0.111.0, les deux axes étaient aimantés INDÉPENDAMMENT : la hauteur déduite du
    // ratio était écrasée par `snapH = t - y` sans toucher à la largeur → l'aimant DÉFORMAIT
    // la fenêtre. Constaté en production (mur multiview-Vincent : PiP 6..9 à 1,79/1,77/1,82
    // au lieu de 1,778), et propagé ensuite par « Remplir » qui conserve fidèlement le ratio
    // courant d'une fenêtre déjà déformée.
    if (ratio > 0) {
        const MN = minPx || 64;
        let best = null; // {d, w, h, axis}
        xTargets.forEach(t => {
            const d = Math.abs((x + w) - t);
            if (d > SNAP_PX) return;
            const nw = t - x, nh = Math.round(nw / ratio);
            // Une cible dont la dimension DÉDUITE sortirait du cadre (ou passerait sous le
            // plancher) n'est pas atteignable à ratio constant : on ne la propose pas.
            if (nw < MN || nh < MN || y + nh > out_h) return;
            if (!best || d < best.d) best = {d, w: nw, h: nh, axis: 'v'};
        });
        yTargets.forEach(t => {
            const d = Math.abs((y + h) - t);
            if (d > SNAP_PX) return;
            const nh = t - y, nw = Math.round(nh * ratio);
            if (nw < MN || nh < MN || x + nw > out_w) return;
            if (!best || d < best.d) best = {d, w: nw, h: nh, axis: 'h'};
        });
        if (!best) return {w: Math.max(MN, w), h: Math.max(MN, h), guides};
        guides.push(best.axis === 'v' ? {type: 'v', pos: x + best.w}
                                      : {type: 'h', pos: y + best.h});
        // L'autre bord peut tomber PILE sur une cible (grille régulière) : on trace aussi son
        // repère, pour ne pas laisser croire que seul un côté est aligné.
        if (best.axis === 'v') {
            if (yTargets.some(t => Math.abs((y + best.h) - t) <= 1)) guides.push({type: 'h', pos: y + best.h});
        } else if (xTargets.some(t => Math.abs((x + best.w) - t) <= 1)) {
            guides.push({type: 'v', pos: x + best.w});
        }
        return {w: best.w, h: best.h, guides};
    }
    // ratio ≤ 0 : objet à format LIBRE (overlay, bloc VU) — chaque axe s'aimante seul, il n'y a
    // aucune contrainte à respecter entre les deux. Sans ce cas, `snapW / 0` valait l'infini et
    // le redimensionnement d'un overlay explosait.
    let bestDx = SNAP_PX + 1, snapW = w;
    xTargets.forEach(t => {
        const d = Math.abs((x + w) - t);
        if (d <= SNAP_PX && d < bestDx) { bestDx = d; snapW = t - x; }
    });
    if (bestDx <= SNAP_PX) guides.push({type: 'v', pos: x + snapW});
    let snapH = h, bestDy = SNAP_PX + 1;
    yTargets.forEach(t => {
        const d = Math.abs((y + snapH) - t);
        if (d <= SNAP_PX && d < bestDy) { bestDy = d; snapH = t - y; }
    });
    if (bestDy <= SNAP_PX) guides.push({type: 'h', pos: y + snapH});
    // Plancher 16 px : un overlay peut légitimement être petit (pastille tally, chiffre d'horloge).
    return {w: Math.max(16, snapW), h: Math.max(16, snapH), guides};
}

// ─── Déployer la composition ─────────────────────────────────

async function deployerEditor() {
    mwFabricBurst();
    if (editorVmid === null) return;
    if (!mwPeutEcrire()) return;   // le bandeau dit déjà qui a la main ; on ne poste rien
    // Sérialise les déploiements : un appel pendant un POST en cours est rejoué à la
    // fin (jamais deux deploys concurrents, jamais un dernier changement perdu).
    if (deployerEditor._busy) { deployerEditor._pending = true; return; }
    deployerEditor._busy = true;

    // Lit les champs globaux au cas où ils n'auraient pas déclenché onchange
    editorParams.max_inputs    = parseInt(document.getElementById('ed_max').value) || editorParams.max_inputs;
    editorParams.show_no_signal  = document.getElementById('ed_show_no_signal').checked;
    editorParams.freeze_detect_s = Math.max(0, parseFloat(document.getElementById('ed_freeze_detect').value) || 0);
    editorParams.show_proxy      = document.getElementById('ed_show_proxy').checked;
    editorParams.genlock = document.getElementById('ed_genlock').checked;
    { const _ft = document.getElementById('ed_fps_target'); if (_ft) editorParams.fps_target = parseInt(_ft.value) || 0; }
    const tslPortEl = document.getElementById('ed_tsl_port');
    if (tslPortEl) editorParams.tsl_port = parseInt(tslPortEl.value) || 0;
    { const _tm = document.getElementById('ed_tsl_mode'); if (_tm) editorParams.tsl_mode = _tm.value; }
    padBank();   // max_inputs a pu changer → complète la banque

    const flux_config = editorParams.flux_config.map(f => ({
        path: f.path,
        // Ports SUIVEURS posés par la page Câbles (audio des VU / flux ANC) : sérialisés
        // VERBATIM — même bug de liste blanche que les flags ANC d'avant 0.29.0, un
        // redéploiement depuis l'éditeur les perdait (VU en ABSENCE, ANC vide, si la
        // dérivation _N est impossible). `undefined` disparaît du JSON → une entrée qui
        // n'a jamais eu la clé reste sans la clé (le hook serveur peut ré-hydrater) ;
        // une valeur posée, Y COMPRIS "" (décâblage volontaire), est préservée telle quelle.
        audio_path: f.audio_path,
        anc_path: f.anc_path,
        hidden: !!f.hidden,
        label_source: f.label_source || 'hostname',
        name: computeDisplayName(f),
        in_w: f.in_w,
        in_h: f.in_h,
        x: f.x, y: f.y,
        w: f.w % 2 === 0 ? f.w : f.w - 1,
        h: f.h % 2 === 0 ? f.h : f.h - 1,
        // Format VOULU de la fenêtre (aspect du modèle de PiP, sinon de la source, sinon 16:9).
        // Ce n'est PAS w/h : c'est la référence que « Remplir » et le resize à la poignée doivent
        // respecter. Il n'était pas sérialisé → au rechargement `ratioOf` retombait en dernier
        // recours sur la géométrie COURANTE d'une fenêtre déjà déformée, et « Remplir » figeait
        // la déformation au lieu de la corriger. Non pixel-déterminant (cf. la liste noire de
        // signature de cellule, app/compositor_fabric.py) : ne re-matérialise aucun shard.
        ratio: (f.ratio && f.ratio > 0) ? f.ratio : undefined,
        show_label: !!f.show_label,
        show_tally: !!f.show_tally,
        label_proportional: !!f.label_proportional,
        tsl_index: parseInt(f.tsl_index) || 0,
        label_col: parseInt(f.label_col) || 0,
        tally_level: f.tally_level || '',
        tally_red: !!f.tally_red,
        tally_green: !!f.tally_green,
        // Peak meters audio
        meter_channels: parseInt(f.meter_channels) || 0,
        meter_position: f.meter_position || 'right',
        meter_inside:   !!f.meter_inside,
        meter_opacity:  Math.max(10, Math.min(100, numOu(f.meter_opacity, 70))),
        meter_scale:    f.meter_scale || 'dbfs',
        // Métadonnées ANC par fenêtre (absentes de la sérialisation avant 0.29.0 →
        // les flags étaient PERDUS au déploiement).
        ...Object.fromEntries(ANC_FLAGS.map(k => [k, !!f[k]])),
        anc_position:   f.anc_position || 'bottom',
        anc_opacity:    Math.max(0, Math.min(100, numOu(f.anc_opacity, 60))),
        // Modèle de PiP (résolu, embarqué dans le deploy_config → snapshoté avec les projets)
        template:       f.template || null,
        template_ref:   f.template_ref || '',
    }));

    const params = {
        flux_config,
        overlays:      serializeOverlays(),
        meter_blocks:  serializeMeterBlocks(),
        video_history_blocks: serializeHistBlocks('video'),
        audio_history_blocks: serializeHistBlocks('audio'),
        shm_out:       editorParams.shm_out,
        out_width:     editorParams.out_width,
        out_height:    editorParams.out_height,
        orientation:   editorParams.orientation || 'landscape',
        fps:           editorParams.fps,
        scan:          editorParams.scan || 'p',
        show_no_signal:  editorParams.show_no_signal !== false,
        freeze_detect_s: editorParams.freeze_detect_s ?? 2,
        show_proxy:      !!editorParams.show_proxy,
        max_inputs:    editorParams.max_inputs,
        genlock:       editorParams.genlock,
        fps_target:    editorParams.fps_target || 0,
        tsl_mode:      editorParams.tsl_mode || 'central',
        tsl_port:      editorParams.tsl_port ?? 4801,
        // Modèle de PiP par défaut du mur (héritage)
        default_template:     editorParams.default_template || null,
        default_template_ref: editorParams.default_template_ref || ''
    };

    const btn = document.getElementById('ed_deploy_btn');
    if (btn) { btn.disabled = true; btn.textContent = T('plugin.multiview.deploying'); }
    let r = null;
    try {
        r = await fetch('/api/containers/' + editorVmid + '/deploy', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            // `base_rev` : l'état sur lequel notre travail est bâti. Le serveur refuse (409) si
            // QUELQU'UN D'AUTRE a écrit depuis — nos propres écritures ne nous bloquent jamais
            // (le déploiement est asynchrone : on ne peut pas connaître la révision qu'on vient
            // de produire). Absent = pas de garde, comportement historique.
            body: JSON.stringify({type: 'multiview', params, path: '/opt/script/main.py',
                                  ...(mwRev === null ? {} : {base_rev: mwRev})})
        });
    } catch(e) {}
    if (r && r.status === 409) {
        let j = {}; try { j = await r.json(); } catch (e) {}
        if (j.error === 'config_perimee') {
            // Le travail de l'autre est en base, le nôtre ne part pas : on le DIT, et on propose
            // la seule issue honnête — recharger. Écraser en silence est exactement le défaut
            // qu'on corrige ; réessayer en boucle le serait tout autant.
            if (btn) { btn.disabled = false; btn.textContent = T('plugin.multiview.deploy'); }
            deployerEditor._busy = false;
            deployerEditor._pending = false;
            mwConflit(j.par || '');
            return;
        }
    }
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

// Applique un alignement à UN rectangle {x,y,w,h} par rapport à `ref`, en le maintenant dans le
// canvas. Partagé par les fenêtres et par les objets à sélection unique (overlays, blocs VU,
// frises) — ces derniers n'étaient couverts par AUCUN outil d'alignement : les six boutons ne
// regardaient que `selectedIdxs`, c'est-à-dire les fenêtres vidéo. Sélectionner une horloge ou un
// champ texte et cliquer « centrer » ne faisait donc rien du tout, sans le moindre message.
function _alignRect(o, mode, ref, out_w, out_h) {
    switch (mode) {
        case 'left':    o.x = ref.x; break;
        case 'right':   o.x = ref.x + ref.w - o.w; break;
        case 'hcenter': o.x = Math.round(ref.x + (ref.w - o.w) / 2); break;
        case 'top':     o.y = ref.y; break;
        case 'bottom':  o.y = ref.y + ref.h - o.h; break;
        case 'vcenter': o.y = Math.round(ref.y + (ref.h - o.h) / 2); break;
    }
    o.x = Math.max(0, Math.min(out_w - o.w, o.x));
    o.y = Math.max(0, Math.min(out_h - o.h, o.y));
}


// Membres de la sélection, RÉFÉRENCE EN DERNIER, avec l'objet géométrique de chacun.
function _selMembres() {
    return _selRefs().map(r => ({r, o: _refObj(r)})).filter(m => m.o);
}
// Persistance après un outil de disposition : une seule règle. Dès qu'un objet d'une autre
// famille est touché, le déploiement (hot-apply /reconfigure + /overlays) couvre TOUT, fenêtres
// comprises ; s'il n'y a que des fenêtres, on garde les POST /window ciblés — moins coûteux, et
// surtout sans re-planification de tissu sur un mur shardé.
function _persistMembres(membres) {
    if (membres.some(m => m.r.k !== 'win')) { hotApplyFull(); return; }
    membres.forEach(m => hotApplyWindow(m.r.i));
}

function aligner(mode) {
    if (!editorParams) return;
    const membres = _selMembres();
    if (!membres.length) return;
    const ow = editorParams.out_width, oh = editorParams.out_height;
    // Référence : l'objet PRIMAIRE (dernier sélectionné) dès qu'il y a au moins deux membres,
    // sinon le cadre. Règle inchangée — elle vaut désormais pour un mélange de familles.
    const prim = membres[membres.length - 1].o;
    const ref = (membres.length >= 2) ? {x: prim.x, y: prim.y, w: prim.w, h: prim.h}
                                      : {x: 0, y: 0, w: ow, h: oh};
    membres.forEach((m, k) => {
        if (membres.length >= 2 && k === membres.length - 1) return;   // la référence ne bouge pas
        _alignRect(m.o, mode, ref, ow, oh);
    });
    dessiner();
    syncGeomFields();
    _persistMembres(membres);
}
function matchSize(mode) {
    const membres = editorParams ? _selMembres() : [];
    if (membres.length < 2) {
        mwFlash(T('plugin.multiview.flash_select2'));
        return;
    }
    const out_w = editorParams.out_width;
    const out_h = editorParams.out_height;
    const ref = membres[membres.length - 1].o;   // référence = dernier sélectionné, toutes familles
    membres.forEach((m, k) => {
        if (k === membres.length - 1) return;
        const f = m.o;
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
    syncGeomFields();
    _persistMembres(membres);
}

function distribuer(axis) {
    const membres = editorParams ? _selMembres() : [];
    if (membres.length < 3) {
        mwFlash(T('plugin.multiview.flash_select3'));
        return;
    }
    const items = membres.slice();   // trié ci-dessous : l'ordre de sélection n'est pas l'ordre à l'écran
    if (axis === 'h') {
        items.sort((a, b) => a.o.x - b.o.x);
        const x0 = items[0].o.x;
        const xN = items[items.length - 1].o.x;
        const step = (xN - x0) / (items.length - 1);
        items.forEach((it, k) => { it.o.x = Math.round(x0 + step * k); });
    } else {
        items.sort((a, b) => a.o.y - b.o.y);
        const y0 = items[0].o.y;
        const yN = items[items.length - 1].o.y;
        const step = (yN - y0) / (items.length - 1);
        items.forEach((it, k) => { it.o.y = Math.round(y0 + step * k); });
    }
    dessiner();
    syncGeomFields();
    _persistMembres(membres);
}

// Remplir : les objets sélectionnés (toutes familles) se TUILENT le long d'un axe — largeur en
// parts égales, ou hauteur — dans leur ordre À L'ÉCRAN. L'AUTRE dimension suit le RATIO de chaque
// objet (pas de déformation) ; l'objet est centré sur cet autre axe et, si le ratio le ferait
// déborder, il est réduit pour rester dans l'image.
// Ex. 4 PiP (un à gauche, un à droite, deux au centre) → 4 colonnes égales, chacune à la bonne
// hauteur 16:9 et centrée verticalement.
function remplir(axis) {
    if (!editorParams) return;
    const out_w = editorParams.out_width, out_h = editorParams.out_height;
    const membres = _selMembres();
    if (!membres.length) return;
    // UN SEUL objet à format LIBRE (texte, horloge, image, bloc VU, frise) : « remplir » ne peut
    // pas vouloir dire tuiler — il est seul — et il n'a aucun ratio à préserver. Il prend donc
    // toute la largeur (ou toute la hauteur) du cadre, l'autre dimension inchangée : le geste du
    // bandeau. Une FENÊTRE seule, elle, garde son format (c'est de la vidéo) et suit le tuilage
    // ci-dessous, qui la pose plein cadre à ratio constant.
    if (membres.length === 1 && membres[0].r.k !== 'win') {
        const o = membres[0].o;
        if (axis === 'h') { o.x = 0; o.w = out_w & ~1; }
        else              { o.y = 0; o.h = out_h & ~1; }
        dessiner();
        syncGeomFields();
        _persistMembres(membres);
        return;
    }
    const n = membres.length;
    const evDim = v => Math.max(2, Math.round(v) & ~1);   // dimension paire ≥ 2
    const evPos = v => Math.max(0, Math.round(v) & ~1);   // position paire ≥ 0
    // Ratio d'un membre : pour une FENÊTRE, l'aspect du modèle de PiP (format libre) prioritaire,
    // sinon le ratio voulu, sinon la source, sinon la géométrie. Pour les autres familles, leur
    // forme courante — c'est la seule notion de proportion qu'elles aient.
    const ratioOf = m => {
        const o = m.o;
        if (m.r.k === 'win') {
            return mwTemplateAspect(o)
                || ((o.ratio && o.ratio > 0) ? o.ratio
                : (o.in_w && o.in_h) ? o.in_w / o.in_h
                : (o.h ? o.w / o.h : 16 / 9));
        }
        return (o.h > 0) ? o.w / o.h : 16 / 9;
    };
    const items = membres.slice();
    if (axis === 'h') {
        items.sort((a, b) => a.o.x - b.o.x);
        const colW = Math.max(2, Math.floor(out_w / n) & ~1);
        items.forEach((m, k) => {
            const f = m.o, r = ratioOf(m);
            f.x = k * colW;
            let w = (k === n - 1) ? (out_w - k * colW) : colW;
            let h = w / r;
            if (h > out_h) { h = out_h; w = h * r; }      // ne pas déborder en hauteur
            f.w = evDim(w); f.h = evDim(h);
            if (f.x + f.w > out_w) f.x = out_w - f.w;
            f.y = evPos((out_h - f.h) / 2);               // centré verticalement
        });
    } else {
        items.sort((a, b) => a.o.y - b.o.y);
        const rowH = Math.max(2, Math.floor(out_h / n) & ~1);
        items.forEach((m, k) => {
            const f = m.o, r = ratioOf(m);
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
    syncGeomFields();
    _persistMembres(membres);
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
    _selClear();
    dessiner();
    hotApplyFull();   // /reconfigure atomique à chaud (géométrie + hidden de toute la banque)
}

// ─── Layouts (presets) ───────────────────────────────────────

let savedLayouts = [];
// Layout en cours d'édition en place (bouton « Modifier ») : null = mode « enregistrer nouveau ».
let editingLayoutId = null;

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
            <div class="meta">${escapeHtml(l.created_at || '')} · ${escapeHtml(layoutSummary(l.config))}</div>
            <canvas data-layout-preview="${l.id}"></canvas>
            <div class="actions">
                <button class="btn btn-blue" onclick="appliquerLayout(${l.id})">${escapeHtml(T('plugin.multiview.apply'))}</button>
                <button class="btn" onclick="modifierLayout(${l.id})" title="${escapeHtml(T('plugin.multiview.edit_layout_title'))}">${escapeHtml(T('plugin.multiview.edit'))}</button>
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

function _serialiserLayoutCourant() {
    // Sérialise la config courante (sans champs internes color/ratio)
    // Un layout décrit TOUT le mur (décision utilisateur 2026-07-14) — SAUF `shm_out`, qui est
    // l'identité de SORTIE du conteneur (ses consommateurs aval sont câblés dessus : la changer par
    // un rappel de layout casserait le câblage). Tout le reste suit : format, habillage, overlays,
    // blocs, frises, réglages d'affichage, TSL.
    const config = {
        out_width:     editorParams.out_width,
        out_height:    editorParams.out_height,
        orientation:   editorParams.orientation || 'landscape',
        fps:           editorParams.fps,
        scan:          editorParams.scan || 'p',
        genlock:       editorParams.genlock,
        fps_target:    editorParams.fps_target || 0,
        show_no_signal:  editorParams.show_no_signal !== false,
        freeze_detect_s: editorParams.freeze_detect_s ?? 2,
        show_proxy:      !!editorParams.show_proxy,
        tsl_mode:      editorParams.tsl_mode || 'central',
        tsl_port:      editorParams.tsl_port ?? 4801,
        max_inputs:    editorParams.max_inputs,
        // Overlays du mur (texte, horloge, image) : ils font partie du layout au même titre que le
        // reste. Sans ça ils SURVIVAIENT à tous les rappels (même oubli que les frises). Aucune
        // source à préserver : un overlay est autonome (l'image est embarquée).
        overlays:      serializeOverlays(),
        // Modèle de PiP par défaut du mur (héritage) : fait partie de l'habillage du layout.
        default_template:     editorParams.default_template || null,
        default_template_ref: editorParams.default_template_ref || '',
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
            tally_level: f.tally_level ?? '',
            tally_red: !!f.tally_red,
            tally_green: !!f.tally_green,
            // Modèle de PiP : fait partie de l'habillage mémorisé par le layout.
            template: f.template || null,
            template_ref: f.template_ref || ''
        })),
        // Blocs VU-mètres de mur : géométrie + style, SOURCE AUDIO OMISE (même convention que
        // `path` pour les fenêtres, ci-dessus) — restaurée depuis l'éditeur courant à l'apply.
        meter_blocks: (editorParams.meter_blocks || []).filter(b => !b.hidden).map(b => ({
            x: b.x, y: b.y, w: b.w, h: b.h,
            channels: b.channels ?? 2, ch_start: b.ch_start ?? 1,
            scale: b.scale || 'dbfs', opacity: b.opacity ?? 100,
            align: b.align || 'left', width_mode: b.width_mode || 'auto',
            label: b.label || ''
        })),
        // Frises d'HISTORIQUE (vidéo/audio) : même convention que les blocs VU-mètres —
        // géométrie + style mémorisés, SOURCE OMISE (elle appartient au conteneur, pas au
        // layout ; restaurée par index à l'apply). Sans ça, un layout ne les décrivait pas et
        // le rappel les laissait en place : elles SURVIVAIENT à tous les changements de layout.
        video_history_blocks: (editorParams.video_history_blocks || []).filter(b => !b.hidden).map(b => ({
            x: b.x, y: b.y, w: b.w, h: b.h,
            duration: b.duration ?? 30, opacity: b.opacity ?? 100,
            events: b.events !== false, label: b.label || ''
        })),
        audio_history_blocks: (editorParams.audio_history_blocks || []).filter(b => !b.hidden).map(b => ({
            x: b.x, y: b.y, w: b.w, h: b.h,
            duration: b.duration ?? 30, opacity: b.opacity ?? 100,
            channels: b.channels ?? 2, ch_start: b.ch_start ?? 1,
            label: b.label || ''
        }))
    };
    return config;
}

// Enregistre l'éditeur courant comme NOUVEAU layout. `nameOverride` (bouton « Enregistrer comme
// nouveau » du bandeau d'édition) court-circuite le champ #layout-save-name.
async function enregistrerLayout(nameOverride) {
    let nameEl = null, name;
    if (nameOverride != null) {
        name = (nameOverride || '').trim();
    } else {
        nameEl = document.getElementById('layout-save-name');
        name = (nameEl.value || '').trim();
    }
    if (!name) { mwFlash(T('plugin.multiview.flash_layout_name')); return; }
    if (!editorParams) {
        mwFlash(T('plugin.multiview.flash_select_mv_first'));
        return;
    }
    const config = _serialiserLayoutCourant();
    let r = null;
    try {
        r = await fetch('/api/layouts', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, config})
        });
    } catch(e) {}
    if (r && r.ok) {
        if (nameEl) nameEl.value = '';
        quitterEditionLayout();
        mwFlash(T('plugin.multiview.flash_layout_saved').replace('{name}', () => name));
        rafraichirListeLayouts();
    } else {
        mwFlash(T('plugin.multiview.flash_layout_save_failed'));
    }
}

// Écrase le layout en cours d'édition (PUT) — bouton « Enregistrer les modifications ».
async function enregistrerModifsLayout() {
    if (editingLayoutId == null) { enregistrerLayout(); return; }
    const nameEl = document.getElementById('layout-edit-name');
    const name = ((nameEl && nameEl.value) || '').trim();
    if (!name) { mwFlash(T('plugin.multiview.flash_layout_name')); return; }
    if (!editorParams) {
        mwFlash(T('plugin.multiview.flash_select_mv_first'));
        return;
    }
    const config = _serialiserLayoutCourant();
    let r = null;
    try {
        r = await fetch('/api/layouts/' + editingLayoutId, {
            method: 'PUT', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, config})
        });
    } catch(e) {}
    if (r && r.ok) {
        quitterEditionLayout();
        mwFlash(T('plugin.multiview.flash_layout_saved').replace('{name}', () => name));
        rafraichirListeLayouts();
    } else {
        mwFlash(T('plugin.multiview.flash_layout_save_failed'));
    }
}

function _chargerLayoutDansEditeur(l) {
    const cfg = l.config || {};
    // shm_out conservé tel quel (lié au container, pas au layout)
    editorParams.out_width     = cfg.out_width  || editorParams.out_width;
    editorParams.out_height    = cfg.out_height || editorParams.out_height;
    editorParams.max_inputs    = cfg.max_inputs || editorParams.max_inputs;
    // Le layout décrit TOUT le mur (sauf shm_out, cf. sauvegarde). Les clés absentes d'un layout
    // ANCIEN (enregistré avant 0.40.2) laissent la valeur courante — pas de régression.
    ['orientation', 'fps', 'scan', 'genlock', 'fps_target',
     'show_no_signal', 'freeze_detect_s', 'show_proxy', 'tsl_mode', 'tsl_port'].forEach(k => {
        if (cfg[k] !== undefined) editorParams[k] = cfg[k];
    });
    // Overlays : REMPLACÉS par ceux du layout (liste absente ⇒ liste vide, sinon ils survivraient
    // à tous les rappels). Autonomes : rien à préserver depuis l'éditeur courant.
    if (cfg.overlays !== undefined) editorParams.overlays = cfg.overlays.map(o => Object.assign({}, o));
    editorParams.default_template     = cfg.default_template || null;
    editorParams.default_template_ref = cfg.default_template_ref || '';
    populateDefaultTemplateSelect();
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
    // Blocs VU-mètres de mur : géométrie/style du layout, source audio PRÉSERVÉE par index
    // depuis l'éditeur courant (même logique que `path` ci-dessus pour les fenêtres).
    const existingBlocks = editorParams.meter_blocks || [];
    editorParams.meter_blocks = (cfg.meter_blocks || []).map((b, i) => {
        const prev = existingBlocks[i] || {};
        return Object.assign(newMeterBlock(), b, {
            id: prev.id || ('mb' + i + '_' + Date.now().toString(36)),
            audio_path: prev.audio_path || '',
            hidden: false,
        });
    });
    // Frises d'HISTORIQUE : REMPLACÉES par celles du layout (une liste absente du layout ⇒ liste
    // VIDE, sinon elles survivraient à tous les rappels — c'était le bug). Source préservée par
    // index depuis l'éditeur courant, comme pour les fenêtres et les blocs VU-mètres.
    [['video', 'video_history_blocks', 'path'],
     ['audio', 'audio_history_blocks', 'audio_path']].forEach(([kind, key, srcField]) => {
        const prevList = editorParams[key] || [];
        editorParams[key] = (cfg[key] || []).map((b, i) => {
            const prev = prevList[i] || {};
            return Object.assign(newHistBlock(kind), b, {
                id: prev.id || ('hb' + kind + i + '_' + Date.now().toString(36)),
                [srcField]: prev[srcField] || '',
                hidden: false,
            });
        });
    });
    padBank();
    _selClear();
    const hostnameEl = document.getElementById('ed_hostname');
    const hostname = hostnameEl ? hostnameEl.textContent.trim() : '';
    renderEditor(hostname);
}

function appliquerLayout(lid) {
    if (!editorParams) {
        mwFlash(T('plugin.multiview.flash_select_mv_first'));
        return;
    }
    const l = savedLayouts.find(x => x.id === lid);
    if (!l) return;
    _chargerLayoutDansEditeur(l);
    deployerEditor();
}

// ─── Édition en place d'un layout enregistré (bouton « Modifier ») ──
function modifierLayout(lid) {
    if (!editorParams) {
        mwFlash(T('plugin.multiview.flash_select_mv_first'));
        return;
    }
    const l = savedLayouts.find(x => x.id === lid);
    if (!l) return;
    // Charge la géométrie/l'habillage du layout dans l'éditeur SANS déployer : on édite un
    // preset, on ne perturbe pas le mur live. Les sources courantes (par index) restent en place
    // pour la prévisualisation ; elles ne font de toute façon pas partie du layout.
    _chargerLayoutDansEditeur(l);
    editingLayoutId = lid;
    const nameEl = document.getElementById('layout-edit-name');
    if (nameEl) nameEl.value = l.name || '';
    const saveBar = document.getElementById('layout-save-bar');
    const editBar = document.getElementById('layout-edit-bar');
    if (saveBar) saveBar.hidden = true;
    if (editBar) editBar.hidden = false;
    mwFlash(T('plugin.multiview.flash_editing_layout').replace('{name}', () => l.name || ''));
}

function quitterEditionLayout() {
    editingLayoutId = null;
    const saveBar = document.getElementById('layout-save-bar');
    const editBar = document.getElementById('layout-edit-bar');
    if (saveBar) saveBar.hidden = false;
    if (editBar) editBar.hidden = true;
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
