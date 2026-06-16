// Panneau Monitoring « Détail Multiviews » (plugin multiview).
// Enregistre window.MXLMonitorPanels["multiview"] = { mount(el), unmount() }.
// La page Monitoring appelle mount() à l'activation de l'onglet ; le panneau gère son
// propre rafraîchissement (interval) et le coupe dans unmount(). Lecture seule
// (/api/fabric/overview + /api/settings pour le budget) ; autonome (helpers locaux).
(function () {
    function esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
    }
    function fmtLat(ms) {
        if (ms == null || !isFinite(ms)) return '—';
        return Math.round(ms) + ' ms';
    }
    function latCls(ms, budget) {
        if (ms == null) return '';
        if (ms <= budget) return 'lat-ok';
        if (ms <= budget * 1.5) return 'lat-warn';
        return 'lat-err';
    }
    function fmtFmt(p) { return p ? `${esc(p.chroma || '?')} · ${p.bit_depth || '?'} bit` : ''; }
    const arrow = `<span class="fab-arrow" aria-hidden="true">→</span>`;

    function renderProxies(proxies, budget) {
        if (!proxies || !proxies.length) return '';
        return `<div class="fab-proxies">
            <span class="fab-proxies-label">Proxies pyramide (infra partagée) :</span>
            ${proxies.map(p => `<span class="fab-proxy">${esc(p.hostname || ('#' + p.vmid))}
                <span class="lat-pill ${latCls(p.own_latency_ms, budget)}">${fmtLat(p.own_latency_ms)}</span></span>`).join('')}
        </div>`;
    }

    function render(body, j, budget) {
        const asm = (j && j.assemblers) || [];
        const proxies = (j && j.proxies) || [];
        if (!asm.length) {
            body.innerHTML = `<div class="meta">Aucun multiview shardé actuellement. Quand un multiview
                sature, l'orchestrateur le découpe en shards parallèles et la chaîne complète apparaît ici.</div>`
                + renderProxies(proxies, budget);
            return;
        }
        let html = '';
        asm.forEach(a => {
            const shards = a.shards || [];
            const srcSet = new Set();
            shards.forEach(s => (s.sources || []).forEach(x => srcSet.add(x)));
            const sources = Array.from(srcSet);
            const cum = a.cumulative_latency_ms;

            const srcCard = `<div class="fab-step fab-step-src">
                <div class="fab-step-head">Sources <span class="fab-step-n">${sources.length}</span></div>
                <div class="fab-step-body">${
                    sources.length ? sources.map(s => `<code>${esc(s)}</code>`).join('') : '<span class="meta">—</span>'
                }</div></div>`;

            const shardCards = shards.map((s, i) => {
                const shared = s.kind === 'shared';
                return `<div class="fab-step fab-step-shard${shared ? ' fab-step-shared' : ''}">
                    <div class="fab-step-head">${shared ? 'Partagé' : 'Shard ' + (i + 1)}
                        <span class="fab-step-n">${s.cells || 0} cell.</span></div>
                    <div class="fab-step-body">
                        <code>${esc(s.shm || '')}</code>
                        <div class="fab-meta">${s.out_w || '?'}×${s.out_h || '?'} · ${fmtFmt(s.format)}</div>
                        <div class="fab-meta">propre <span class="lat-pill ${latCls(s.own_latency_ms, budget)}">${fmtLat(s.own_latency_ms)}</span></div>
                    </div></div>`;
            }).join(arrow);

            const asmCard = `<div class="fab-step fab-step-asm">
                <div class="fab-step-head">Assembleur <span class="card-vmid">#${a.vmid}</span></div>
                <div class="fab-step-body">
                    <code>${esc(a.shm || '')}</code>
                    <div class="fab-meta">${fmtFmt(a.format)}</div>
                    <div class="fab-meta">propre <span class="lat-pill ${latCls(a.own_latency_ms, budget)}">${fmtLat(a.own_latency_ms)}</span></div>
                </div></div>`;

            html += `<div class="fab-assembler">
                <div class="fab-assembler-head">
                    <strong>${esc(a.hostname || ('#' + a.vmid))}</strong>
                    <span class="meta">${shards.length} shard${shards.length > 1 ? 's' : ''} parallèle${shards.length > 1 ? 's' : ''}</span>
                    <span style="flex:1"></span>
                    <span class="meta">latence cumulée (chemin critique)</span>
                    <span class="lat-pill ${latCls(cum, budget)}">${fmtLat(cum)}</span>
                    <span class="meta">/ budget ${budget} ms</span>
                </div>
                <div class="fab-chain">${srcCard}${arrow}${shardCards}${arrow}${asmCard}</div>
            </div>`;
        });
        html += renderProxies(proxies, budget);
        body.innerHTML = html;
    }

    let _timer = null;
    let _budget = 20;

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
            // Budget de trame courant (réglage fabric_budget_ms) pour colorer les latences.
            try {
                const s = await (await fetch('/api/settings')).json();
                const b = parseInt(s.fabric_budget_ms);
                if (isFinite(b) && b > 0) _budget = b;
            } catch (e) {}
            refresh(el);
            clearInterval(_timer);
            _timer = setInterval(() => refresh(el), 5000);
        },
        unmount() { clearInterval(_timer); _timer = null; }
    };
})();
