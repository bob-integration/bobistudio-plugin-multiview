// Panneau Monitoring « Détail Multiviews » (plugin multiview) — vue en GRAPHE.
// Enregistre window.MXLMonitorPanels["multiview"] = { mount(el), unmount() }.
// Rend, pour chaque multiview shardé, un graphe 3 colonnes (sources | shards parallèles |
// assembleur→sortie) avec arêtes SVG en courbe (façon page Câbles). Les shards sont empilés
// verticalement dans UNE colonne → on voit qu'ils tournent en parallèle, pas en série.
// Lecture seule (/api/fabric/overview + /api/settings) ; autonome (helpers locaux).
(function () {
    function esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
    }
    function fmtLat(ms) { return (ms == null || !isFinite(ms)) ? '—' : Math.round(ms) + ' ms'; }
    function latCls(ms, budget) {
        if (ms == null) return '';
        if (ms <= budget) return 'lat-ok';
        if (ms <= budget * 1.5) return 'lat-warn';
        return 'lat-err';
    }
    function fmtFmt(p) { return p ? `${esc(p.chroma || '?')} · ${p.bit_depth || '?'} bit` : ''; }

    function pill(ms, budget) {
        return `<span class="lat-pill ${latCls(ms, budget)}">${fmtLat(ms)}</span>`;
    }

    // ── Construit le HTML d'un graphe (un assembleur) ───────────────────────────
    function graphHtml(a, gi, budget) {
        const shards = a.shards || [];
        // Sources amont dédupliquées (union des sources de tous les shards) + index.
        const srcIndex = new Map();
        shards.forEach(s => (s.sources || []).forEach(x => { if (!srcIndex.has(x)) srcIndex.set(x, srcIndex.size); }));
        const sources = Array.from(srcIndex.keys());

        const srcNodes = sources.length
            ? sources.map((s, si) => `<div class="mvg-node mvg-src" id="g${gi}-s${si}">
                    <div class="mvg-node-title">Source</div>
                    <code>${esc(s)}</code></div>`).join('')
            : `<div class="mvg-node mvg-src"><div class="meta">aucune source</div></div>`;

        const shardNodes = shards.map((s, hi) => {
            const srcs = (s.sources || []).map(x => srcIndex.get(x)).filter(v => v != null);
            const shared = s.kind === 'shared';
            return `<div class="mvg-node ${shared ? 'mvg-shared' : 'mvg-shard'}" id="g${gi}-h${hi}"
                         data-srcs="${srcs.join(',')}">
                <div class="mvg-node-title">${shared ? 'Partagé' : 'Shard ' + (hi + 1)}
                    <span class="mvg-n">${s.cells || 0} cell.</span></div>
                <code>${esc(s.shm || '')}</code>
                <div class="mvg-node-meta">${s.out_w || '?'}×${s.out_h || '?'} · ${fmtFmt(s.format)}</div>
                <div class="mvg-node-meta">propre ${pill(s.own_latency_ms, budget)}</div>
            </div>`;
        }).join('');

        const asmNode = `<div class="mvg-node mvg-asm" id="g${gi}-a">
            <div class="mvg-node-title">Assembleur <span class="card-vmid">#${a.vmid}</span></div>
            <code>${esc(a.shm || '')}</code>
            <div class="mvg-node-meta">${fmtFmt(a.format)}</div>
            <div class="mvg-node-meta">propre ${pill(a.own_latency_ms, budget)}</div>
        </div>`;

        const cum = a.cumulative_latency_ms;
        return `<div class="mvg-graph" data-gi="${gi}">
            <div class="mvg-head">
                <strong>${esc(a.hostname || ('#' + a.vmid))}</strong>
                <span class="meta">${shards.length} shard${shards.length > 1 ? 's' : ''} en parallèle</span>
                <span style="flex:1"></span>
                <span class="meta">latence cumulée (chemin critique)</span>
                ${pill(cum, budget)}
                <span class="meta">/ budget ${budget} ms</span>
            </div>
            <svg class="mvg-edges"></svg>
            <div class="mvg-cols">
                <div class="mvg-col"><div class="mvg-col-label">Sources</div>${srcNodes}</div>
                <div class="mvg-col"><div class="mvg-col-label">Shards (parallèles)</div>${shardNodes}</div>
                <div class="mvg-col"><div class="mvg-col-label">Assembleur → sortie</div>${asmNode}</div>
            </div>
        </div>`;
    }

    // ── Trace les arêtes SVG d'un graphe (après layout) ─────────────────────────
    function drawEdges(graphEl) {
        const svg = graphEl.querySelector('svg.mvg-edges');
        const cols = graphEl.querySelector('.mvg-cols');
        if (!svg || !cols) return;
        const box = cols.getBoundingClientRect();
        svg.innerHTML = '';
        const NS = 'http://www.w3.org/2000/svg';
        const gi = graphEl.dataset.gi;

        function pt(el, side) {
            const r = el.getBoundingClientRect();
            return { x: (side === 'out' ? r.right : r.left) - box.left, y: r.top + r.height / 2 - box.top };
        }
        function edge(fromEl, toEl, active) {
            const A = pt(fromEl, 'out'), B = pt(toEl, 'in');
            const cx = Math.max(28, (B.x - A.x) * 0.45);
            const p = document.createElementNS(NS, 'path');
            p.setAttribute('d', `M ${A.x} ${A.y} C ${A.x + cx} ${A.y}, ${B.x - cx} ${B.y}, ${B.x} ${B.y}`);
            p.setAttribute('class', 'mvg-edge' + (active ? ' mvg-edge-active' : ''));
            svg.appendChild(p);
        }
        const asm = graphEl.querySelector(`#g${gi}-a`);
        graphEl.querySelectorAll('.mvg-col .mvg-node[id^="g' + gi + '-h"]').forEach(sh => {
            // sources → ce shard
            (sh.dataset.srcs || '').split(',').filter(x => x !== '').forEach(si => {
                const src = graphEl.querySelector(`#g${gi}-s${si}`);
                if (src) edge(src, sh, true);
            });
            // shard → assembleur
            if (asm) edge(sh, asm, true);
        });
        // Position l'SVG sur la zone des colonnes (le SVG remplit le .mvg-graph, on cale le
        // viewBox sur le décalage des colonnes pour que (0,0) = coin haut-gauche de .mvg-cols).
        const gbox = graphEl.getBoundingClientRect();
        svg.style.left = (box.left - gbox.left) + 'px';
        svg.style.top = (box.top - gbox.top) + 'px';
        svg.style.width = box.width + 'px';
        svg.style.height = box.height + 'px';
        svg.setAttribute('viewBox', `0 0 ${box.width} ${box.height}`);
    }

    function renderProxies(proxies, budget) {
        if (!proxies || !proxies.length) return '';
        return `<div class="mvg-proxies">
            <span class="meta">Proxies pyramide (infra partagée) :</span>
            ${proxies.map(p => `<span>${esc(p.hostname || ('#' + p.vmid))} ${pill(p.own_latency_ms, budget)}</span>`).join('')}
        </div>`;
    }

    function render(body, j, budget) {
        const asm = (j && j.assemblers) || [];
        const proxies = (j && j.proxies) || [];
        if (!asm.length) {
            body.innerHTML = `<div class="meta">Aucun multiview shardé actuellement. Quand un multiview
                sature, l'orchestrateur le découpe en shards parallèles et le graphe apparaît ici.</div>`
                + renderProxies(proxies, budget);
            return;
        }
        body.innerHTML = asm.map((a, gi) => graphHtml(a, gi, budget)).join('') + renderProxies(proxies, budget);
        // Trace les arêtes une fois la mise en page faite.
        requestAnimationFrame(() => body.querySelectorAll('.mvg-graph').forEach(drawEdges));
    }

    let _timer = null, _budget = 20, _onResize = null, _el = null;

    function redraw() { if (_el) _el.querySelectorAll('.mvg-graph').forEach(drawEdges); }

    async function refresh(el) {
        const body = el.querySelector('#mv-fabric-body');
        if (!body) return;
        try {
            const j = await (await fetch('/api/fabric/overview')).json();
            render(body, j, _budget);
        } catch (e) {
            if (body.querySelector('.meta')) body.querySelector('.meta').textContent = '✕ ' + e.message;
        }
    }

    window.MXLMonitorPanels = window.MXLMonitorPanels || {};
    window.MXLMonitorPanels["multiview"] = {
        async mount(el) {
            _el = el;
            try {
                const s = await (await fetch('/api/settings')).json();
                const b = parseInt(s.fabric_budget_ms);
                if (isFinite(b) && b > 0) _budget = b;
            } catch (e) {}
            refresh(el);
            clearInterval(_timer);
            _timer = setInterval(() => refresh(el), 5000);
            _onResize = () => { clearTimeout(_onResize._t); _onResize._t = setTimeout(redraw, 120); };
            window.addEventListener('resize', _onResize);
        },
        unmount() {
            clearInterval(_timer); _timer = null;
            if (_onResize) window.removeEventListener('resize', _onResize);
            _onResize = null; _el = null;
        }
    };
})();
