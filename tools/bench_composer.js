// Banc LOGIQUE du composer (0.111.0) : sélection inter-familles + boîte d'une image importée.
// `node tools/bench_composer.js`
// — aucune dépendance, aucun navigateur, aucun conteneur touché.
//
// Pourquoi ce fichier : le composer n'a aucun test, et la sélection vient de passer de quatre
// variables indépendantes (une par famille d'objets) à une liste unique `mwSel` dont ces variables
// sont des vues. C'est le genre de refonte qui casse en silence, à la souris, des semaines plus
// tard. Le banc charge le VRAI multiview.js dans un DOM minimal et rejoue les gestes.
const fs = require('fs'), vm = require('vm'), path = require('path');

const noop = () => {};
// Éléments PERSISTANTS par id, avec un classList qui se souvient : sans mémoire, on ne peut pas
// vérifier qu'un état (« lecture seule ») a bien été posé sur l'éditeur.
const _els = new Map();
const elStub = (id) => {
    if (id && _els.has(id)) return _els.get(id);
    const cls = new Set();
    const el = { value: '', checked: false, textContent: '', innerHTML: '', hidden: false,
                 style: {}, dataset: {}, title: '', onclick: null,
                 classList: { add: c => cls.add(c), remove: c => cls.delete(c),
                              toggle: (c, v) => (v === undefined ? (cls.has(c) ? cls.delete(c) : cls.add(c))
                                                                 : (v ? cls.add(c) : cls.delete(c))),
                              contains: c => cls.has(c) },
                 querySelectorAll: () => [], querySelector: () => null,
                 append: noop, appendChild: noop, insertBefore: noop, remove: noop,
                 addEventListener: noop, setAttribute: noop, getAttribute: () => null };
    if (id) _els.set(id, el);
    return el;
};
const sandbox = {
    console,
    document: { getElementById: id => elStub(id), querySelectorAll: () => [], querySelector: () => null,
                createElement: () => elStub(), addEventListener: noop, documentElement: {lang: 'fr'} },
    window: { addEventListener: noop, removeEventListener: noop, innerHeight: 900, t: null },
    navigator: { platform: 'Linux' },
    getComputedStyle: () => ({ getPropertyValue: () => '' }),
    requestAnimationFrame: noop,
    fetch: async (url, opt) => {
        sandbox.__appels.push({url: String(url), methode: (opt && opt.method) || 'GET',
                               corps: opt && opt.body ? JSON.parse(opt.body) : null});
        const rep = sandbox.__reponses.shift();
        return rep || { ok: true, status: 200, json: async () => ({}), text: async () => '' };
    },
    setTimeout, clearTimeout, setInterval, clearInterval, Image: function () {},
};
sandbox.__appels = [];      // journal des requêtes émises par le composer
sandbox.__reponses = [];    // réponses scénarisées (FIFO)
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname, '..', 'multiview.js'), 'utf8'), sandbox);

// Neutralise ce qui touche au rendu / au réseau (on teste la LOGIQUE de disposition).
['dessiner', 'drawCanvas', 'syncGeomFields', 'mwFlash', 'mwFabricBurst',
 'refreshEntryPanel', 'refreshOverlayPanel', 'refreshBlockPanel', 'refreshHistPanel',
 'renderEntryTable', 'updateToolbar', 'mwRefreshSidebarPreview', '_readTokens'].forEach(f => {
    vm.runInContext(`${f} = function(){};`, sandbox);
});
const posts = [];
Object.assign(sandbox, {__posts: posts});
vm.runInContext('__vraiHotApplyWindow = hotApplyWindow;', sandbox);   // gardé pour les cas 15-16
vm.runInContext('hotApplyWindow = function(i){ __posts.push(i); };', sandbox);

const run = code => vm.runInContext(code, sandbox);
// `vm.runInContext` n'autorise pas l'await racine : on emballe dans une fonction async et on
// attend la promesse depuis le banc (les chemins d'écriture du composer sont asynchrones).
const runA = code => vm.runInContext(`(async () => { ${code} })()`, sandbox);

// ── Mur d'essai : 2 fenêtres 16:9, 1 texte, 1 bloc VU, 1 frise vidéo ─────────
function mur() {
    run(`editorParams = {
        out_width: 1920, out_height: 1080, max_inputs: 2,
        flux_config: [
            {path:'/dev/shm/a', x: 100, y: 100, w: 640, h: 360, ratio: 16/9, hidden:false},
            {path:'/dev/shm/b', x: 900, y: 500, w: 480, h: 270, ratio: 16/9, hidden:false}],
        overlays:            [{id:'ov1', kind:'text', x: 40, y: 800, w: 480, h: 120}],
        meter_blocks:        [{id:'mb1', x: 1500, y: 60, w: 200, h: 300}],
        video_history_blocks:[{id:'hb1', x: 200, y: 950, w: 600, h: 100}],
        audio_history_blocks: [],
    }; mwSel = []; _selSync(); __posts.length = 0;`);
}
const geo = expr => run(`(o => ({x:o.x, y:o.y, w:o.w, h:o.h}))(${expr})`);
const F = i => geo(`editorParams.flux_config[${i}]`);
const OV = () => geo('editorParams.overlays[0]');
const MB = () => geo('editorParams.meter_blocks[0]');
const HB = () => geo('editorParams.video_history_blocks[0]');

let ko = 0;
const ok = (nom, cond, detail) => {
    console.log((cond ? '  ok   ' : '  ÉCHEC') + ' ' + nom + (detail ? '   ' + detail : ''));
    if (!cond) ko++;
};

(async () => {
console.log('\n1. Sélection mixte');
mur();
run(`_selSet('win', 0); _selSet('ov', 0, true); _selSet('blk', 0, true);`);
ok('trois familles sélectionnées ensemble', run('_selCount()') === 3);
ok('la référence est le dernier cliqué', run('JSON.stringify(_selPrimary())') === '{"k":"blk","i":0}');
ok('selectedIdxs ne voit que les fenêtres', run('JSON.stringify(selectedIdxs)') === '[0]');
ok('selectedBlock = primaire', run('selectedBlock') === 0);
ok('selectedOverlay effacé (le primaire est ailleurs)', run('selectedOverlay') === -1);
run(`_selSet('ov', 0, true);`);
ok('Maj+clic sur un objet déjà pris le retire', run('_selCount()') === 2 && !run(`_selHas('ov',0)`));

console.log('\n2. Aligner un texte SUR une fenêtre (le geste demandé)');
mur();
run(`_selSet('ov', 0); _selSet('win', 0, true);`);   // référence = fenêtre 0 (x=100)
run(`aligner('left')`);
ok('le texte prend le bord gauche de la fenêtre', OV().x === 100, `x=${OV().x}`);
ok('la référence n a pas bougé', F(0).x === 100 && F(0).y === 100);

console.log('\n3. Centrer un objet seul = sur le cadre');
mur();
run(`_selSet('blk', 0); aligner('hcenter');`);
ok('bloc VU centré horizontalement', MB().x === (1920 - 200) / 2, `x=${MB().x}`);

console.log('\n4. Distribuer trois objets de familles différentes');
mur();
run(`_selSet('win', 0); _selSet('ov', 0, true); _selSet('blk', 0, true); distribuer('h');`);
{
    const xs = [F(0).x, OV().x, MB().x].sort((a, b) => a - b);
    ok('pas égalisées', xs[1] - xs[0] === xs[2] - xs[1], `x = ${xs.join(', ')}`);
}

console.log('\n5. Même taille que la référence, familles mêlées');
mur();
run(`_selSet('ov', 0); _selSet('win', 1, true); matchSize('both');`);   // réf = fenêtre 1 (480×270)
ok('le texte prend la taille de la fenêtre', OV().w === 480 && OV().h === 270, `${OV().w}×${OV().h}`);

console.log('\n6. Déplacement de GROUPE mixte (fenêtre tirée, texte suivi)');
mur();
run(`_selSet('ov', 0); _selSet('win', 0, true);
     dragOrigRect = {...editorParams.flux_config[0]}; dragMode='move'; _beginGroupDrag();
     _applyGroupMove(300, 100);`);   // fenêtre 0 : 100,100 → 300,100  (delta +200, 0)
ok('la fenêtre tirée va où on la met', F(0).x === 300 && F(0).y === 100);
ok('le texte suit du même delta', OV().x === 240 && OV().y === 800, `${OV().x},${OV().y}`);

console.log('\n7. Le groupe ne sort JAMAIS du cadre');
mur();
run(`_selSet('blk', 0); _selSet('win', 0, true);
     dragOrigRect = {...editorParams.flux_config[0]}; dragMode='move'; _beginGroupDrag();
     _applyGroupMove(5000, 5000);`);   // très au-delà du bord
ok('bloc VU dans le cadre', MB().x + MB().w <= 1920 && MB().y + MB().h <= 1080, `${MB().x},${MB().y}`);
ok('fenêtre dans le cadre', F(0).x + F(0).w <= 1920 && F(0).y + F(0).h <= 1080, `${F(0).x},${F(0).y}`);

console.log('\n8. Remplir : tuilage mixte, chaque objet garde son format');
mur();
run(`_selSet('win', 0); _selSet('ov', 0, true); remplir('h');`);
// Tuilage dans l'ordre À L'ÉCRAN : le texte (x=40) précède la fenêtre (x=100), il prend donc la
// première colonne. C'est bien l'ordre visuel qui commande, pas l'ordre de sélection.
ok('deux colonnes égales, dans l ordre visuel', OV().x === 0 && F(0).x === 960,
   `texte x=${OV().x}, fenêtre x=${F(0).x}`);
ok('la fenêtre garde 16:9', Math.abs(F(0).w / F(0).h - 16 / 9) < 0.01, `${F(0).w}×${F(0).h}`);
ok('le texte garde son format 4:1', Math.abs(OV().w / OV().h - 4) < 0.02, `${OV().w}×${OV().h}`);

console.log('\n9. Remplir un objet libre SEUL = bandeau pleine largeur');
mur();
run(`_selSet('vh', 0); remplir('h');`);
ok('frise pleine largeur, hauteur inchangée', HB().x === 0 && HB().w === 1920 && HB().h === 100,
   `${HB().x},${HB().w}×${HB().h}`);

console.log('\n10. Persistance : la bonne route selon les familles touchées');
mur();
run(`_selSet('win', 0); _selSet('win', 1, true); aligner('top');`);
ok('fenêtres seules → POST /window ciblés', posts.length === 2, `posts=${posts.length}`);
posts.length = 0;
run(`_selSet('win', 0); _selSet('ov', 0, true); aligner('top');`);
ok('sélection mixte → un seul déploiement, aucun POST /window', posts.length === 0);

console.log('\n11. Suppression : les références se décalent');
mur();
run(`editorParams.overlays.push({id:'ov2', kind:'clock', x:0,y:0,w:100,h:100});
     _selSet('ov', 1); _selDrop('ov', 0);`);
ok('la ref survivante pointe le bon objet', run(`JSON.stringify(_selPrimary())`) === '{"k":"ov","i":0}');

console.log('\n12. Objet disparu = référence purgée, pas de plantage');
mur();
run(`_selSet('win', 0); _selSet('ov', 0, true); editorParams.overlays.length = 0; _selSync();`);
ok('la sélection se nettoie seule', run('_selCount()') === 1 && run('selectedOverlay') === -1);

console.log('\n13. Boîte d une image : elle prend le FORMAT du fichier');
mur();
run(`editorParams.overlays[0] = {id:'im1', kind:'image', fit:'contain', x:100, y:100, w:384, h:216};
     _ovColleAuFormat(editorParams.overlays[0], 4);`);   // logo large 4:1
ok('largeur gardée, hauteur déduite', OV().w === 384 && OV().h === 96, `${OV().w}×${OV().h}`);
mur();
run(`editorParams.overlays[0] = {id:'im1', kind:'image', fit:'contain', x:100, y:1000, w:384, h:60};
     _ovColleAuFormat(editorParams.overlays[0], 0.5);`);  // portrait, près du bord bas
ok('rentré dans le cadre à format constant', OV().y + OV().h <= 1080 && Math.abs(OV().w / OV().h - 0.5) < 0.05,
   `${OV().w}×${OV().h} à y=${OV().y}`);

console.log('\n14. Redimensionnement contraint d une image (aimant compris)');
mur();
{
    const r = run(`JSON.stringify(computeSnapResize([], 100, 100, 800, 200, 4, 16))`);
    const sn = JSON.parse(r);
    ok('le format 4:1 est tenu', Math.abs(sn.w / sn.h - 4) < 0.02, `${sn.w}×${sn.h}`);
}
{
    const r = JSON.parse(run(`JSON.stringify(computeSnapResize([], 0, 0, 955, 60, 4, 16))`));
    ok('plancher 16 px respecté, pas 64', r.h >= 16, `h=${r.h}`);
}

// ── Édition à plusieurs ─────────────────────────────────────────────────────
const appels = () => sandbox.__appels;
const razAppels = () => { sandbox.__appels.length = 0; };

console.log('\n15. Verrou : lecture seule quand un autre a la main');
mur();
// On rebranche le VRAI hotApplyWindow : c'est LUI qui porte la garde, le pister ne suffit pas.
run(`hotApplyWindow = __vraiHotApplyWindow; dragMode = null; dragOverlay = dragBlock = dragHist = false;
     editorVmid = 972; mwRev = 7;
     mwVerrou = {a_moi: false, libre: false, user_name: 'Vincent', depuis_s: 180};`);
ok('mwLectureSeule() vrai', run('mwLectureSeule()') === true);
razAppels();
run(`hotApplyWindow(0);`);
ok('aucun POST /plugin/window', appels().length === 0);
run(`hotApplyStyle();`);
ok('aucun POST /plugin/style', appels().length === 0);
await runA('await deployerEditor();');
ok('aucun déploiement', appels().filter(a => a.url.includes('/deploy')).length === 0);
run(`canvasMouseDown({clientX: 10, clientY: 10, preventDefault(){}, shiftKey: false, target: {}});`);
ok('le canvas ne démarre aucun geste', run('dragMode') === null);

console.log('\n16. Verrou : la main rendue, tout repart');
run(`mwVerrou = {a_moi: true, libre: false, user_name: 'moi', depuis_s: 0};`);
razAppels();
run(`hotApplyWindow(0);`);
ok('le hot-apply repart', appels().filter(a => a.url.includes('/plugin/window')).length === 1);

console.log('\n17. Garde de révision : la base est jointe au déploiement');
mur();
run(`editorVmid = 972; mwRev = 12; mwVerrou = {a_moi: true, libre: false, user_name: '', depuis_s: 0};`);
razAppels();
await runA('await deployerEditor();');
{
    const dep = appels().find(a => a.url.includes('/deploy'));
    ok('base_rev jointe', dep && dep.corps && dep.corps.base_rev === 12, dep ? JSON.stringify(dep.corps.base_rev) : 'aucun');
}
run(`mwRev = null;`); razAppels();
await runA('await deployerEditor();');
{
    const dep = appels().find(a => a.url.includes('/deploy'));
    ok('révision inconnue → aucune garde (rétrocompatible)', dep && dep.corps && dep.corps.base_rev === undefined);
}

console.log('\n18. Conflit 409 : rien n écrase, l éditeur passe en lecture seule');
mur();
run(`editorVmid = 972; mwRev = 3; mwVerrou = {a_moi: true, libre: false, user_name: '', depuis_s: 0};`);
sandbox.__reponses.push({ ok: false, status: 409,
                          json: async () => ({error: 'config_perimee', rev: 9, par: 'Anne'}) });
razAppels();
await runA('await deployerEditor();');
ok('un seul envoi, pas de réessai en boucle', appels().filter(a => a.url.includes('/deploy')).length === 1);
ok('éditeur verrouillé après conflit', run(`document.getElementById('mw-editor').classList.contains('lecture-seule')`) !== false);
ok('le déploiement n est pas resté « occupé »', run('deployerEditor._busy') === false);
ok('aucun déploiement en attente rejoué', run('deployerEditor._pending') === false);

console.log(ko ? `\n${ko} ÉCHEC(S)\n` : '\nTous les cas passent\n');
process.exit(ko ? 1 : 0);
})();
