/* Dashboard and onboarding behaviour.
 *
 * One file drives all three sector dashboards. Nothing here knows which sector
 * it is rendering: the templates declare their KPI keys, their charts and their
 * focus queue through data attributes, and the metrics endpoint supplies the
 * values. Adding a fourth sector is a new template, not new JavaScript.
 *
 * Every string that reaches the DOM goes through a text node. Uploaded exports
 * contain free text (ticket subjects, dispute reasons) that ends up in reason
 * strings and playbook payloads, so innerHTML is reserved for markup this file
 * writes itself.
 */

(function () {
    'use strict';

    const API = {
        tenants: '/api/v1/tenants/',
        analyze: '/api/v1/upload/analyze',
        demoData: '/api/v1/upload/demo-data',
        metrics: '/api/v1/analytics/metrics',
        customer: '/api/v1/analytics/customer',
        status: '/api/v1/analytics/status',
    };

    const MAX_TABLE_ROWS = 250;
    const MAX_QUEUE_ROWS = 50;
    const TIER_ORDER = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];
    const TIER_COLORS = {
        CRITICAL: '#b42318',
        HIGH: '#f79009',
        MEDIUM: '#175cd3',
        LOW: '#067647',
    };
    const SERIES_FALLBACK = ['#98a2b3', '#f2b824', '#7a5af8', '#dd2590', '#4e5ba6'];

    document.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('[data-export]').forEach(button => button.addEventListener('click', async () => {
            const status = document.getElementById('export-status');
            button.disabled = true; status.textContent = 'Preparing export…';
            const format = button.dataset.export;
            const query = new URLSearchParams({tenant_id: currentTenantId(), format,
                tier: document.getElementById('tier-filter').value,
                search: document.getElementById('search-filter').value.trim()});
            try {
                const response = await fetch('/api/v1/exports?' + query);
                if (response.status === 401) { window.location.assign('/login'); return; }
                if (!response.ok) { const data = await response.json(); throw new Error(data.detail || 'Export failed'); }
                const url = URL.createObjectURL(await response.blob());
                const link = document.createElement('a'); link.href = url;
                link.download = format === 'zip' ? 'original-uploads.zip' : 'churn-analysis.' + format; document.body.append(link); link.click(); link.remove();
                setTimeout(() => URL.revokeObjectURL(url), 30000);
                status.textContent = 'Export downloaded.';
            } catch (err) { status.textContent = err.message; }
            finally { button.disabled = false; }
        }));
    });

    // -- small DOM helpers --------------------------------------------------

    function el(tag, attrs, ...children) {
        const node = document.createElement(tag);
        for (const [key, value] of Object.entries(attrs || {})) {
            if (value === null || value === undefined || value === false) continue;
            if (key === 'class') node.className = value;
            else if (key === 'text') node.textContent = value;
            else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
            else node.setAttribute(key, value === true ? '' : String(value));
        }
        for (const child of children.flat(Infinity)) {
            if (child === null || child === undefined || child === false) continue;
            node.append(child.nodeType ? child : document.createTextNode(String(child)));
        }
        return node;
    }

    function byId(id) {
        return document.getElementById(id);
    }

    function cssVar(name, fallback) {
        const value = getComputedStyle(document.body).getPropertyValue(name).trim();
        return value || fallback;
    }

    function clear(node) {
        while (node && node.firstChild) node.removeChild(node.firstChild);
        return node;
    }

    function show(node, visible) {
        if (node) node.hidden = !visible;
    }

    function percent(fraction, digits) {
        return (fraction * 100).toFixed(digits === undefined ? 1 : digits) + '%';
    }

    /* A numeric reading out of a pre-formatted highlight value ("$1,327.71",
       "33.3%", "3"). Returns 0 when the value is not a measurement. */
    function readNumber(formatted) {
        const match = String(formatted).replace(/[^0-9.\-]/g, '');
        const value = parseFloat(match);
        return Number.isFinite(value) ? value : 0;
    }

    function currentTenantId() {
        return document.body.getAttribute('data-tenant-id') || localStorage.getItem('tenant_id') || '';
    }

    function customerUrl(entityId) {
        return `/customer?tenant_id=${encodeURIComponent(currentTenantId())}` +
            `&entity_id=${encodeURIComponent(entityId)}`;
    }

    async function requestJson(url, options) {
        const response = await fetch(url, options);
        if (response.status === 401) { window.location.assign('/login'); throw new Error('Your session expired. Please log in again.'); }
        let body = null;
        try {
            body = await response.json();
        } catch (err) {
            body = null;
        }
        if (!response.ok) {
            const detail = body && (body.detail || body.message);
            throw new Error(typeof detail === 'string' ? detail : `Request failed with status ${response.status}`);
        }
        return body;
    }

    // -- banners ------------------------------------------------------------

    function banner(kind, text) {
        return el('div', { class: `banner banner--${kind}`, role: kind === 'warning' ? 'alert' : 'status' },
            el('span', { text: kind === 'warning' ? '\u26A0' : '\u2139' }),
            el('span', { text }));
    }

    function renderBanners(messages) {
        const slot = byId('banner-slot');
        if (!slot) return;
        clear(slot);
        for (const [kind, text] of messages) slot.append(banner(kind, text));
    }

    // -- onboarding ---------------------------------------------------------

    function initOnboarding() {
        const form = byId('onboard-form');
        if (!form) return false;

        const errorBox = byId('onboard-error');
        const submit = byId('onboard-submit');

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            const name = byId('company-name').value.trim();
            // namedItem resolves a radio group to its checked value and a select
            // to its own, so the picker's markup can change without this handler.
            const sector = form.elements.namedItem('sector').value;
            if (!name) {
                errorBox.textContent = 'A company name is required.';
                show(errorBox, true);
                return;
            }

            submit.disabled = true;
            show(errorBox, false);
            try {
                const tenant = await requestJson(API.tenants, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, sector }),
                });
                localStorage.setItem('tenant_id', tenant.tenant_id);
                localStorage.setItem('tenant_name', tenant.name);
                localStorage.setItem('tenant_sector', tenant.sector);
                window.location.href = '/dashboard?tenant_id=' + encodeURIComponent(tenant.tenant_id);
            } catch (err) {
                errorBox.textContent = err.message;
                show(errorBox, true);
                submit.disabled = false;
            }
        });
        return true;
    }

    // -- charts -------------------------------------------------------------

    const liveCharts = new Map();

    function destroyChart(key) {
        const existing = liveCharts.get(key);
        if (existing) {
            existing.destroy();
            liveCharts.delete(key);
        }
    }

    function seriesColor(spec, dataset, index) {
        if (TIER_COLORS[dataset.label]) return TIER_COLORS[dataset.label];
        if (dataset.label === 'At risk') return cssVar('--critical', '#b42318');
        if (dataset.label === 'Healthy') return cssVar('--accent', '#4f46e5');
        return SERIES_FALLBACK[index % SERIES_FALLBACK.length];
    }

    function tooltipFor(spec) {
        if (spec.kind === 'scatter') {
            return {
                callbacks: {
                    label: (ctx) => {
                        const id = ctx.raw && ctx.raw.label ? `${ctx.raw.label}: ` : '';
                        return `${id}${spec.x_label || 'x'} ${ctx.parsed.x}, ${spec.y_label || 'y'} ${ctx.parsed.y}`;
                    },
                },
            };
        }
        return {};
    }

    function chartConfig(spec) {
        const isDoughnut = spec.kind === 'doughnut';
        const datasets = spec.datasets.map((dataset, index) => {
            const color = seriesColor(spec, dataset, index);
            if (spec.kind === 'scatter') {
                return {
                    label: dataset.label,
                    data: dataset.points.map((p) => ({ x: p.x, y: p.y, label: p.label })),
                    backgroundColor: color + 'cc',
                    borderColor: color,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                };
            }
            if (isDoughnut) {
                // A doughnut's slices are its labels, so colour per slice —
                // that is what makes the risk-tier ring read as red-to-green.
                const sliceColors = spec.labels.map((label, i) => TIER_COLORS[label] || SERIES_FALLBACK[i % SERIES_FALLBACK.length]);
                return {
                    label: dataset.label,
                    data: dataset.values,
                    backgroundColor: sliceColors,
                    borderWidth: 2,
                    borderColor: cssVar('--surface', '#ffffff'),
                };
            }
            return {
                label: dataset.label,
                data: dataset.values,
                backgroundColor: color,
                borderColor: color,
                borderWidth: spec.kind === 'line' ? 2 : 0,
                borderRadius: spec.kind === 'line' ? 0 : 4,
                maxBarThickness: 46,
            };
        });

        const scales = isDoughnut ? {} : {
            x: {
                stacked: !!spec.stacked,
                title: spec.x_label ? { display: true, text: spec.x_label, color: cssVar('--muted', '#7b8291'), font: { size: 11 } } : undefined,
                grid: { display: false },
                ticks: { color: cssVar('--muted', '#7b8291'), font: { size: 11 } },
            },
            y: {
                stacked: !!spec.stacked,
                beginAtZero: true,
                title: spec.y_label ? { display: true, text: spec.y_label, color: cssVar('--muted', '#7b8291'), font: { size: 11 } } : undefined,
                grid: { color: cssVar('--border', '#eef0f3') },
                ticks: { color: cssVar('--muted', '#7b8291'), font: { size: 11 } },
            },
        };

        return {
            type: spec.kind === 'scatter' ? 'scatter' : spec.kind,
            data: { labels: isDoughnut || spec.kind === 'scatter' ? [] : spec.labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: isDoughnut ? 'nearest' : 'index', intersect: false },
                plugins: {
                    legend: {
                        display: isDoughnut || datasets.length > 1,
                        position: isDoughnut ? 'right' : 'top',
                        align: 'start',
                        labels: { boxWidth: 10, boxHeight: 10, color: cssVar('--ink-soft', '#4b5160'), font: { size: 11 }, usePointStyle: true },
                    },
                    tooltip: tooltipFor(spec),
                },
                scales,
            },
        };
    }

    let currentChartSpecs = null;
    window.addEventListener("themechange", () => { if (currentChartSpecs) renderCharts(currentChartSpecs); });

    function renderCharts(charts) {
        currentChartSpecs = charts;
        document.querySelectorAll('[data-chart]').forEach((canvas) => {
            const key = canvas.getAttribute('data-chart');
            const panel = canvas.closest('.panel');
            const spec = charts[key];
            destroyChart(key);
            if (!spec) {
                // A chart whose inputs the upload did not contain. Hiding the
                // panel beats drawing an empty axis with a title that promises
                // evidence.
                if (panel) panel.hidden = true;
                return;
            }
            if (panel) panel.hidden = false;

            const titleNode = document.querySelector(`[data-chart-title="${key}"]`);
            const subtitleNode = document.querySelector(`[data-chart-subtitle="${key}"]`);
            if (titleNode && spec.title) titleNode.textContent = spec.title;
            if (subtitleNode) subtitleNode.textContent = spec.subtitle || '';

            liveCharts.set(key, new Chart(canvas.getContext('2d'), chartConfig(spec)));
        });
    }

    // -- KPI cards ----------------------------------------------------------

    function renderKpis(kpis) {
        const grid = byId('kpi-grid');
        if (!grid) return;
        clear(grid);
        const keys = (grid.getAttribute('data-kpi-keys') || '')
            .split(',')
            .map((key) => key.trim())
            .filter(Boolean);

        for (const key of keys) {
            const kpi = kpis[key];
            if (!kpi) continue;
            grid.append(el('article', { class: 'kpi-card', 'data-tone': kpi.tone || 'neutral' },
                el('p', { class: 'kpi-card__label', text: kpi.label }),
                el('p', { class: 'kpi-card__value', text: kpi.value }),
                kpi.detail ? el('p', { class: 'kpi-card__detail', text: kpi.detail }) : null));
        }
    }

    // -- schema discovery ---------------------------------------------------

    function renderSchema(mapping, offline) {
        const panel = byId('schema-panel');
        if (!panel || !mapping) return;
        panel.hidden = false;
        byId('schema-key').textContent = mapping.primary_entity_key;
        byId('schema-table-count').textContent = String(mapping.tables.length);
        const confirmed = mapping.tables.some(t => (t.columns || []).some(c => c.confidence === 1 &&
            /confirmed/i.test(c.reasoning || '')));
        byId('schema-source').textContent = confirmed ? 'Includes workspace-confirmed column mappings'
            : offline ? 'Resolved by the system model (rule-based)' : 'Resolved by the AI model';

        const body = clear(byId('schema-rows'));
        for (const table of mapping.tables) {
            body.append(el('tr', {},
                el('td', { class: 'mono', text: table.file_name }),
                el('td', {}, el('span', { class: 'role-tag', text: table.role })),
                el('td', { class: 'mono', text: table.primary_entity_key }),
                el('td', { class: 'mono', text: table.timestamp_column || '\u2014' }),
                el('td', { class: 'noise-list' },
                    table.noise_columns && table.noise_columns.length
                        ? el('span', { class: 'mono', text: table.noise_columns.join(', ') })
                        : el('span', { text: 'none' }))));
        }
        enableNavLink('schema-panel', true);
    }

    // -- predictions table --------------------------------------------------

    /* Columns are the ordered union of every highlight label present, because a
       highlight is dropped for an individual customer when the upload lacked the
       feature it reads. Deriving the header from the first row alone would
       silently truncate the table for everyone behind it. */
    function highlightLabels(rows) {
        const labels = [];
        for (const row of rows) {
            for (const highlight of row.highlights || []) {
                if (!labels.includes(highlight.label)) labels.push(highlight.label);
            }
        }
        return labels;
    }

    function highlightMap(row) {
        const map = {};
        for (const highlight of row.highlights || []) map[highlight.label] = highlight.value;
        return map;
    }

    let tableState = { rows: [], labels: [], tier: 'ALL', query: '' };

    function probabilityCell(value) {
        return el('td', {},
            el('div', { class: 'prob-cell' },
                el('div', { class: 'prob-bar' },
                    el('div', {
                        class: 'prob-bar__fill',
                        style: `width:${Math.round(Math.min(Math.max(value, 0), 1) * 100)}%;` +
                            `background:${TIER_COLORS[tierOf(value)] || cssVar('--accent', '#4f46e5')}`,
                    })),
                el('span', { class: 'prob-value', text: percent(value) })));
    }

    function tierOf(value) {
        if (value >= 0.8) return 'CRITICAL';
        if (value >= 0.55) return 'HIGH';
        if (value >= 0.3) return 'MEDIUM';
        return 'LOW';
    }

    function renderTable() {
        const head = byId('predictions-head');
        const body = byId('predictions-body');
        if (!head || !body) return;

        clear(head);
        head.append(el('tr', {},
            el('th', { text: 'Customer' }),
            el('th', { text: 'Risk' }),
            el('th', { text: 'Probability' }),
            ...tableState.labels.map((label) => el('th', { text: label })),
            el('th', {})));

        const query = tableState.query.trim().toLowerCase();
        const matching = tableState.rows.filter((row) => {
            // Default is every tier. An earlier version hard-filtered LOW out,
            // which hid most of a healthy customer base and made the tool look
            // broken rather than selective.
            if (tableState.tier !== 'ALL' && row.risk_tier !== tableState.tier) return false;
            return !query || row.entity_id.toLowerCase().includes(query);
        });

        clear(body);
        const shown = matching.slice(0, MAX_TABLE_ROWS);
        for (const row of shown) {
            const values = highlightMap(row);
            body.append(el('tr', {},
                el('td', { class: 'mono', text: row.entity_id }),
                el('td', {}, el('span', { class: `pill pill--${row.risk_tier}`, text: row.risk_tier })),
                probabilityCell(row.churn_probability),
                ...tableState.labels.map((label) =>
                    el('td', { text: values[label] === undefined ? '\u2014' : values[label] })),
                el('td', {},
                    el('div', { style: 'display:flex;gap:.75rem;align-items:center;white-space:nowrap' },
                        el('button', {
                            type: 'button',
                            class: 'link-btn',
                            text: 'View playbook',
                            onclick: () => openDrawer(row),
                        }),
                        el('a', { class: 'link-btn', href: customerUrl(row.entity_id), text: 'Detail' })))));
        }

        if (!shown.length) {
            body.append(el('tr', {},
                el('td', { colspan: String(5 + tableState.labels.length) },
                    el('div', { class: 'empty-state' },
                        el('p', { class: 'empty-state__title', text: 'No customers match this filter' }),
                        el('p', { class: 'empty-state__hint', text: 'Widen the tier filter or clear the search box.' })))));
        }

        const note = byId('row-count-note');
        if (note) {
            const parts = [`${matching.length} of ${tableState.rows.length} customers`];
            if (matching.length > shown.length) parts.push(`showing the ${shown.length} highest risk`);
            note.textContent = parts.join(' \u00B7 ');
        }
    }

    function initTableControls() {
        const tierFilter = byId('tier-filter');
        const search = byId('search-filter');
        if (tierFilter) {
            tierFilter.addEventListener('change', () => {
                tableState.tier = tierFilter.value;
                renderTable();
            });
        }
        if (search) {
            search.addEventListener('input', () => {
                tableState.query = search.value;
                renderTable();
            });
        }
    }

    // -- focus queue --------------------------------------------------------

    function renderFocusQueue(rows) {
        const panel = byId('focus-queue');
        if (!panel) return;
        const body = byId('queue-body');
        const count = byId('queue-count');
        const label = panel.getAttribute('data-queue-label');
        const group = panel.getAttribute('data-queue-group');
        const mode = panel.getAttribute('data-queue-match') || 'equals';
        const wanted = panel.getAttribute('data-queue-value') || '';
        const emptyMessage = panel.getAttribute('data-queue-empty') || 'Nothing to action right now.';

        const matches = (value) => {
            if (value === undefined) return false;
            if (mode === 'nonzero') return readNumber(value) > 0;
            return String(value).toLowerCase() === wanted.toLowerCase();
        };

        const selected = rows.filter((row) => matches(highlightMap(row)[label]));
        panel.hidden = false;
        if (count) count.textContent = `${selected.length} customer${selected.length === 1 ? '' : 's'}`;
        enableNavLink('focus-queue', selected.length > 0);

        clear(body);
        if (!selected.length) {
            body.append(el('tr', {},
                el('td', { colspan: '5' },
                    el('div', { class: 'empty-state' },
                        el('p', { class: 'empty-state__hint', text: emptyMessage })))));
            return;
        }

        for (const row of selected.slice(0, MAX_QUEUE_ROWS)) {
            const values = highlightMap(row);
            body.append(el('tr', {},
                el('td', { text: values[group] === undefined ? '\u2014' : values[group] }),
                el('td', { class: 'mono', text: row.entity_id }),
                el('td', {}, el('span', { class: `pill pill--${row.risk_tier}`, text: row.risk_tier })),
                el('td', { class: 'prob-value', text: percent(row.churn_probability) }),
                el('td', {},
                    el('div', { style: 'display:flex;gap:.75rem;align-items:center;white-space:nowrap' },
                        el('button', {
                            type: 'button',
                            class: 'link-btn',
                            text: 'Open',
                            onclick: () => openDrawer(row),
                        }),
                        el('a', { class: 'link-btn', href: customerUrl(row.entity_id), text: 'Detail' })))));
        }
    }

    // -- drawer -------------------------------------------------------------

    let currentRow = null;

    function openDrawer(row) {
        currentRow = row;
        const drawer = byId('drawer');
        const body = byId('drawer-body');
        byId('drawer-title').textContent = row.entity_id;
        byId('drawer-subtitle').textContent =
            `${percent(row.churn_probability)} predicted churn \u00B7 ${row.risk_tier} risk`;

        clear(body);
        body.append(el('div', { class: 'meta-row' },
            el('span', { class: `pill pill--${row.risk_tier}`, text: row.risk_tier }),
            el('span', { class: 'pill pill--plain', text: `${percent(row.churn_probability, 1)} churn probability` })));

        if (row.reason) {
            body.append(el('p', { class: 'section__title', text: 'Why this customer' }),
                el('p', { class: 'payload-box', style: 'margin-bottom:1.1rem', text: row.reason }));
        }

        if (row.highlights && row.highlights.length) {
            body.append(el('p', { class: 'section__title', text: 'Measured evidence' }));
            const evidence = el('div', { class: 'evidence', style: 'margin-bottom:1.1rem' });
            for (const highlight of row.highlights) {
                evidence.append(el('div', { class: 'evidence__row' },
                    el('span', { class: 'evidence__label', text: highlight.label }),
                    el('span', { class: 'evidence__value', text: highlight.value })));
            }
            body.append(evidence);
        }

        const playbook = row.playbook || {};
        body.append(el('p', { class: 'section__title', text: 'Recommended intervention' }),
            el('div', { class: 'evidence', style: 'margin-bottom:.8rem' },
                el('div', { class: 'evidence__row' },
                    el('span', { class: 'evidence__label', text: 'Action' }),
                    el('span', { class: 'evidence__value', text: playbook.action_type || '\u2014' })),
                el('div', { class: 'evidence__row' },
                    el('span', { class: 'evidence__label', text: 'Channel' }),
                    el('span', { class: 'evidence__value', text: playbook.channel || '\u2014' }))),
            el('div', { class: 'payload-box', text: playbook.action_payload || 'No payload was returned.' }),
            el('div', { style: 'margin-top:1.1rem' },
                el('a', {
                    class: 'link-btn',
                    href: customerUrl(row.entity_id),
                    text: 'Open full customer detail \u2192',
                })));

        byId('deploy-note').textContent = '';
        drawer.classList.add('is-open');
        drawer.setAttribute('aria-hidden', 'false');
        byId('drawer-scrim').classList.add('is-open');
    }

    function closeDrawer() {
        currentRow = null;
        const drawer = byId('drawer');
        drawer.classList.remove('is-open');
        drawer.setAttribute('aria-hidden', 'true');
        byId('drawer-scrim').classList.remove('is-open');
    }

    function initDrawer() {
        byId('close-drawer').addEventListener('click', closeDrawer);
        byId('drawer-scrim').addEventListener('click', closeDrawer);
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') closeDrawer();
        });

        /* There is no delivery channel behind this button, so it does the one
           honest thing available: put the message on the clipboard and say that
           nothing was sent. */
        byId('deploy-btn').addEventListener('click', async () => {
            const note = byId('deploy-note');
            const payload = currentRow && currentRow.playbook ? currentRow.playbook.action_payload : '';
            if (!payload) {
                note.textContent = 'Nothing to copy for this customer.';
                return;
            }
            try {
                await navigator.clipboard.writeText(payload);
                note.textContent = 'Message copied. No delivery channel is wired up in this build.';
            } catch (err) {
                note.textContent = 'Clipboard unavailable here. No delivery channel is wired up in this build.';
            }
        });
    }

    // -- sidebar ------------------------------------------------------------

    function enableNavLink(targetId, enabled) {
        document.querySelectorAll(`[data-scroll-to="${targetId}"]`).forEach((link) => {
            link.disabled = !enabled;
        });
    }

    function initSidebar() {
        document.querySelectorAll('[data-scroll-to]').forEach((link) => {
            link.addEventListener('click', () => {
                const target = byId(link.getAttribute('data-scroll-to'));
                if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
        });
        // Only the upload panel exists before a run, so everything else starts
        // disabled rather than scrolling to a hidden section.
        ['schema-panel', 'kpi-section', 'predictions-panel', 'focus-queue'].forEach((id) => enableNavLink(id, false));
        document.querySelectorAll('[data-scroll-to^="chart-"]').forEach((link) => { link.disabled = true; });
    }

    function syncSidebar(summary) {
        enableNavLink('kpi-section', true);
        enableNavLink('predictions-panel', true);
        document.querySelectorAll('[data-scroll-to^="chart-"]').forEach((link) => {
            const key = link.getAttribute('data-scroll-to').slice('chart-'.length);
            link.disabled = !summary.charts[key];
        });
    }

    // -- upload -------------------------------------------------------------

    let selectedFiles = [];
    let isBusy = false;
    let aiEngineAvailable = false;
    let aiModelName = '';

    /* One place decides what in the upload panel is clickable, because it
       depends on three things now: files present, nothing in flight, and for the
       AI button only, a provider key the server could actually resolve. */
    function syncUploadButtons() {
        const hasFiles = selectedFiles.length > 0;
        const blocked = isBusy || !hasFiles;
        const aiBtn = byId('analyze-ai-btn');
        const systemBtn = byId('analyze-system-btn');
        const clearBtn = byId('clear-btn');
        const demoBtn = byId('demo-btn');
        if (aiBtn) aiBtn.disabled = blocked || !aiEngineAvailable;
        if (systemBtn) systemBtn.disabled = blocked;
        if (clearBtn) clearBtn.disabled = blocked;
        if (demoBtn) demoBtn.disabled = isBusy;
    }

    function renderChips() {
        const row = byId('file-chips');
        clear(row);
        for (const [index, file] of selectedFiles.entries()) {
            row.append(el('span', { class: 'chip' },
                el('span', { text: file.name }),
                el('button', {
                    type: 'button',
                    class: 'chip__remove',
                    'aria-label': `Remove ${file.name}`,
                    text: '\u00D7',
                    onclick: (event) => {
                        event.stopPropagation();
                        selectedFiles.splice(index, 1);
                        renderChips();
                    },
                })));
        }
        byId('upload-count').textContent =
            `${selectedFiles.length} file${selectedFiles.length === 1 ? '' : 's'}`;
        syncUploadButtons();
    }

    function addFiles(list) {
        const incoming = Array.from(list || []);
        for (const file of incoming) {
            // Re-selecting the same export must not upload it twice under two
            // table names.
            if (!selectedFiles.some((existing) => existing.name === file.name)) selectedFiles.push(file);
        }
        renderChips();
    }

    function initDropzone(tenantId) {
        const zone = byId('dropzone');
        const input = byId('file-input');
        if (!zone || !input) return;

        const openPicker = () => input.click();
        zone.addEventListener('click', openPicker);
        zone.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                openPicker();
            }
        });
        input.addEventListener('change', () => {
            addFiles(input.files);
            input.value = '';
        });

        ['dragenter', 'dragover'].forEach((name) =>
            zone.addEventListener(name, (event) => {
                event.preventDefault();
                zone.classList.add('is-dragover');
            }));
        ['dragleave', 'drop'].forEach((name) =>
            zone.addEventListener(name, (event) => {
                event.preventDefault();
                zone.classList.remove('is-dragover');
            }));
        zone.addEventListener('drop', (event) => addFiles(event.dataTransfer && event.dataTransfer.files));

        byId('clear-btn').addEventListener('click', () => {
            selectedFiles = [];
            renderChips();
        });

        byId('analyze-ai-btn').addEventListener('click', () => runAnalysis(tenantId, 'ai'));
        byId('analyze-system-btn').addEventListener('click', () => runAnalysis(tenantId, 'system'));
        byId('demo-btn').addEventListener('click', () => loadDemoData(tenantId));
        renderChips();
    }

    function setBusy(busy, message) {
        isBusy = busy;
        show(byId('upload-progress'), busy);
        if (message) byId('upload-progress-text').textContent = message;
        syncUploadButtons();
    }

    async function runAnalysis(tenantId, engine) {
        if (!selectedFiles.length) return;
        setBusy(true, engine === 'ai'
            ? 'Calling the AI model\u2026 a free tier answers slowly, so this can take minutes'
            : 'Scoring locally and synthesizing features\u2026');
        renderBanners([]);

        const form = new FormData();
        form.append('tenant_id', tenantId);
        form.append('engine', engine);
        form.append('review_mapping', byId('review-mapping').checked ? 'true' : 'false');
        for (const file of selectedFiles) form.append('files', file);

        try {
            const result = await requestJson(API.analyze, { method: 'POST', body: form });
            if (result.requires_human_review) {
                setBusy(false);
                openSchemaReview(tenantId, result);
                return;
            }
            await finishAnalysis(tenantId, result);
        } catch (err) {
            setBusy(false);
            renderBanners([['warning', `Analysis failed: ${err.message}`]]);
        }
    }

    async function finishAnalysis(tenantId, result) {
            setBusy(true, 'Aggregating sector metrics\u2026');
            const summary = await loadMetrics(tenantId);
            const messages = [];
            // offline_mode now describes this run rather than the deployment, so
            // it is the one trustworthy statement about who scored it.
            messages.push(['info', result.offline_mode
                ? 'Scored by the system model: deterministic, computed locally from your exports, no AI provider called.'
                : `Scored by the AI model${aiModelName ? ` (${aiModelName})` : ''}.`]);
            for (const warning of result.warnings || []) messages.push(['warning', warning]);
            if (summary && result.entities_analyzed < result.entities_uploaded) {
                messages.push(['warning',
                    `${result.entities_uploaded - result.entities_analyzed} of ${result.entities_uploaded} entities were not scored.`]);
            }
            renderBanners(messages);
            setBusy(false);
    }

    function openSchemaReview(tenantId, review) {
        const dialog = byId('schema-review-dialog');
        const form = byId('schema-review-form');
        const container = byId('schema-review-columns');
        const error = byId('schema-review-error');
        const confirm = byId('schema-review-confirm');
        const cancel = byId('schema-review-cancel');
        const controls = [];
        const tables = [];
        let submitting = false;
        container.replaceChildren();
        error.textContent = review.schema_mapping.review_reasons.join('; ');
        const labels = { UNKNOWN: 'Unknown', CUSTOMER_ID: 'Customer ID', TIMESTAMP: 'Timestamp', TRANSACTION_AMOUNT: 'Transaction amount',
            STATUS: 'Status', EVENT_TYPE: 'Event type', TEXT: 'Text / feedback', ATTRIBUTE: 'Attribute (keep original name)',
            NOISE_IGNORE: 'Ignore / Drop Column', CUSTOM: 'Add Custom Name', SUBSCRIPTION_PLAN: 'Subscription Plan', USAGE_ACTIVITY: 'Usage Activity' };
        const element = (tag, text) => { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; return el; };
        const validateMappings = () => {
            const names = new Map();
            for (const control of controls) {
                control.row.classList.remove('mapping-conflict');
                control.select.removeAttribute('aria-invalid');
                const role = control.select.value;
                if (!role || role === 'NOISE_IGNORE') continue;
                const target = role === 'CUSTOM' ? control.custom.value.trim()
                    : (review.canonical_names || {})[role] || control.source_column;
                if (!target) continue;
                const key = JSON.stringify([control.file_name, target]);
                if (!names.has(key)) names.set(key, {target, entries: []});
                names.get(key).entries.push(control);
            }
            const conflicts = [];
            for (const {target, entries} of names.values()) {
                if (entries.length < 2) continue;
                for (const entry of entries) {
                    entry.row.classList.add('mapping-conflict');
                    entry.select.setAttribute('aria-invalid', 'true');
                }
                conflicts.push(`${entries[0].file_name}: ${entries.map(c => c.source_column).join(', ')} all map to "${target}". Keep one in this role; choose Attribute or distinct custom names for the others.`);
            }
            error.textContent = conflicts.join(' ');
            confirm.disabled = submitting || conflicts.length > 0;
            return conflicts.length === 0;
        };
        for (const table of review.schema_mapping.tables) {
            const section = element('section'); section.className = 'schema-review__table';
            section.appendChild(element('h3', table.file_name));
            const tableLabel = element('label', 'Table type');
            const tableSelect = element('select');
            for (const [role, name] of Object.entries({DIMENSION:'Customer attributes', TIME_SERIES_EVENT:'Activity events', TRANSACTIONAL:'Transactions', UNSTRUCTURED_TEXT:'Feedback / support text'})) {
                const option = element('option', name); option.value = role; tableSelect.appendChild(option);
            }
            tableSelect.value = table.role; tableLabel.appendChild(tableSelect); section.appendChild(tableLabel);
            tables.push({file_name: table.file_name, select: tableSelect});
            const scroll = element('div'); scroll.className = 'table-scroll';
            const grid = element('table');
            const head = element('thead'); const header = element('tr');
            for (const name of ['Uploaded column', 'Sample values', 'Confidence', 'Use as']) header.appendChild(element('th', name));
            head.appendChild(header); grid.appendChild(head);
            const body = element('tbody');
            for (const column of table.columns) {
                const row = element('tr');
                if (column.confidence < .8 || column.status === 'REQUIRES_HUMAN_REVIEW') row.className = 'needs-review';
                const source = element('td', column.source_column);
                source.appendChild(element('small', `Suggested: ${labels[column.canonical_role] || column.canonical_role}`));
                source.appendChild(element('small', column.reasoning || 'Please select a role.'));
                row.appendChild(source);
                const sample = element('td');
                for (const value of column.sample_values) sample.appendChild(element('div', value));
                if (!column.sample_values.length) sample.textContent = 'No non-empty sample values';
                row.appendChild(sample);
                row.appendChild(element('td', `${Math.round(column.confidence * 100)}%`));
                const cell = element('td'); const select = element('select');
                select.setAttribute('aria-label', `Role for ${column.source_column} in ${table.file_name}`);
                select.required = true;
                const placeholder = element('option', 'Choose a role'); placeholder.value = ''; select.appendChild(placeholder);
                for (const role of review.canonical_roles) {
                    const option = element('option', labels[role] || role); option.value = role; select.appendChild(option);
                }
                select.value = column.canonical_role === 'UNKNOWN' ? '' : column.canonical_role;
                const custom = element('input'); custom.type = 'text'; custom.maxLength = 100;
                custom.placeholder = 'e.g. Priority SLA Tier'; custom.value = column.custom_label || '';
                custom.setAttribute('aria-label', `Custom name for ${column.source_column} in ${table.file_name}`);
                const syncCustom = () => { custom.hidden = select.value !== 'CUSTOM'; custom.required = !custom.hidden; };
                select.addEventListener('change', () => { syncCustom(); validateMappings(); });
                custom.addEventListener('input', validateMappings); syncCustom();
                cell.append(select, custom); row.appendChild(cell); body.appendChild(row);
                controls.push({file_name: table.file_name, source_column: column.source_column, select, custom, row});
            }
            grid.appendChild(body); scroll.appendChild(grid); section.appendChild(scroll); container.appendChild(section);
        }
        cancel.onclick = () => { if (!submitting) dialog.close(); };
        dialog.oncancel = (event) => { if (submitting) event.preventDefault(); };
        form.onsubmit = async (event) => {
            event.preventDefault();
            if (submitting || !validateMappings() || !form.reportValidity()) return;
            const mappings = controls.map(c => ({file_name: c.file_name, source_column: c.source_column,
                canonical_role: c.select.value, custom_label: c.select.value === 'CUSTOM' ? c.custom.value.trim() : null}));
            submitting = true; confirm.disabled = true; cancel.disabled = true;
            confirm.textContent = 'Analyzing…'; error.textContent = '';
            setBusy(true, 'Applying confirmed mappings and analyzing customers…');
            try {
                const result = await requestJson('/api/v1/upload/confirm-mapping', {
                    method: 'POST', headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({tenant_id: tenantId, upload_session_id: review.upload_session_id, mappings,
                        table_roles: Object.fromEntries(tables.map(t => [t.file_name, t.select.value]))})
                });
                dialog.close();
                await finishAnalysis(tenantId, result);
            } catch (err) {
                error.textContent = err.message; setBusy(false);
            } finally {
                submitting = false; confirm.disabled = false; cancel.disabled = false;
                confirm.textContent = 'Confirm & Run Analysis';
            }
        };
        validateMappings();
        dialog.showModal();
    }

    // -- sample data --------------------------------------------------------

    function decodeDemoFile(entry) {
        const binary = atob(entry.content_base64);
        const bytes = new Uint8Array(binary.length);
        for (let index = 0; index < binary.length; index += 1) {
            bytes[index] = binary.charCodeAt(index);
        }
        return new File([bytes], entry.name, { type: 'text/csv' });
    }

    /* Hands the bundled exports to the same addFiles() a file picker feeds, so
       the chips, the counters and the analyze buttons all behave as they do
       after a manual pick. Nothing is analyzed here: the user still chooses an
       engine, which is the point of the two buttons. */
    async function loadDemoData(tenantId) {
        setBusy(true, 'Fetching the sample exports\u2026');
        renderBanners([]);
        try {
            const payload = await requestJson(
                `${API.demoData}?tenant_id=${encodeURIComponent(tenantId)}`);
            addFiles(payload.files.map(decodeDemoFile));
            renderBanners([['info',
                `Loaded ${payload.files.length} sample ${payload.sector_label} exports. ` +
                'Choose an engine and analyze.']]);
        } catch (err) {
            renderBanners([['warning', `Could not load the sample data: ${err.message}`]]);
        }
        setBusy(false);
    }

    function initEnginePicker(status) {
        const note = byId('engine-note');
        if (!note) return;

        if (!status) {
            aiEngineAvailable = false;
            note.textContent = 'Could not read the scoring engines from the server.';
            syncUploadButtons();
            return;
        }

        aiEngineAvailable = Boolean(status.ai_available);
        aiModelName = status.model || '';
        note.textContent = aiEngineAvailable
            ? `AI model: ${aiModelName}, ${status.batch_size} customers per call. ` +
              'A full 100-customer base takes minutes on a free tier, and a hosted ' +
              'function may time out before it finishes \u2014 the system model answers in seconds.'
            : 'AI model unavailable: no provider key is configured. Add GEMINI_API_KEY or GROQ_API_KEY ' +
              '(or HF_TOKEN, OPENROUTER_API_KEY, DASHSCOPE_API_KEY) to api_key.env to enable it. ' +
              'The system model scores locally, deterministically, in seconds.';
        syncUploadButtons();
    }

    // -- metrics ------------------------------------------------------------

    function renderSummary(summary) {
        byId('tenant-name').textContent = summary.tenant_name;
        show(byId('results-section'), true);
        show(byId('empty-state'), false);

        renderKpis(summary.kpis);
        renderSchema(summary.schema_mapping, summary.offline_mode);
        renderCharts(summary.charts);

        tableState = {
            rows: summary.rows,
            labels: highlightLabels(summary.rows),
            tier: byId('tier-filter') ? byId('tier-filter').value : 'ALL',
            query: byId('search-filter') ? byId('search-filter').value : '',
        };
        renderTable();
        renderFocusQueue(summary.rows);
        syncSidebar(summary);
    }

    async function loadMetrics(tenantId) {
        try {
            const summary = await requestJson(`${API.metrics}?tenant_id=${encodeURIComponent(tenantId)}`);
            renderSummary(summary);
            return summary;
        } catch (err) {
            // A 404 here just means "no upload yet", which the empty state says.
            show(byId('results-section'), false);
            show(byId('empty-state'), true);
            if (!/No analysis has been run/.test(err.message)) {
                renderBanners([['warning', `Could not load metrics: ${err.message}`]]);
            }
            return null;
        }
    }

    async function loadStatus() {
        const badge = byId('mode-badge');
        const note = byId('mode-note');
        try {
            const status = await requestJson(API.status);
            if (badge) {
                // Availability, not mode: each run now picks its own engine, so
                // the honest sidebar statement is what can be chosen at all.
                badge.textContent = status.ai_available
                    ? `AI available \u00B7 ${status.model}`
                    : 'System model only';
                badge.className = `pill ${status.ai_available ? 'pill--LOW' : 'pill--MEDIUM'}`;
            }
            if (note) note.textContent = status.batch_size_note;
            return status;
        } catch (err) {
            if (badge) badge.textContent = 'Mode unknown';
            return null;
        }
    }

    // -- customer detail ------------------------------------------------------

    /* The detail page shares the dashboard's helpers but none of its data flow:
       one customer, one fetch, every number positioned against the whole base. */

    function formatMeasure(value) {
        if (Number.isInteger(value)) return value.toLocaleString();
        const abs = Math.abs(value);
        const digits = abs >= 100 ? 1 : abs >= 1 ? 2 : 4;
        return value.toLocaleString(undefined, { maximumFractionDigits: digits });
    }

    function toneForTier(tier) {
        if (tier === 'CRITICAL') return 'critical';
        if (tier === 'HIGH') return 'warning';
        if (tier === 'LOW') return 'good';
        return 'neutral';
    }

    function detailCard(label, value, detail, tone) {
        return el('article', { class: 'kpi-card', 'data-tone': tone || 'neutral' },
            el('p', { class: 'kpi-card__label', text: label }),
            el('p', { class: 'kpi-card__value', text: value }),
            detail ? el('p', { class: 'kpi-card__detail', text: detail }) : null);
    }

    function percentileCell(feature) {
        if (feature.percentile === null || feature.percentile === undefined) {
            return el('td', { text: '\u2014' });
        }
        return el('td', {},
            el('div', { class: 'prob-cell' },
                el('div', { class: 'prob-bar' },
                    el('div', {
                        class: 'prob-bar__fill',
                        style: `width:${Math.round(feature.percentile * 100)}%;` +
                            `background:${cssVar('--accent', '#4f46e5')}`,
                    })),
                el('span', { class: 'prob-value', text: percent(feature.percentile, 0) })));
    }

    function renderDetail(detail) {
        const tone = toneForTier(detail.risk_tier);
        const score = clear(byId('detail-score'));
        score.append(
            detailCard('Churn probability', percent(detail.churn_probability),
                `Ranked ${detail.risk_rank} of ${detail.population_size} in this customer base`, tone),
            detailCard('Risk tier', detail.risk_tier,
                detail.reason || 'No explanation was recorded for this score', tone),
            detailCard('Peers compared', String(detail.population_size),
                'Customers scored in the latest analysis', 'neutral'),
            detailCard('Analyzed',
                detail.created_at ? new Date(detail.created_at).toLocaleString() : '\u2014',
                'From the latest upload for this tenant', 'neutral'));

        const head = clear(byId('detail-evidence-head'));
        head.append(el('tr', {},
            el('th', { text: 'Signal' }),
            el('th', { text: 'This customer' }),
            el('th', { text: 'Tenant median' }),
            el('th', { text: 'Percentile in base' })));

        const evidence = clear(byId('detail-evidence'));
        for (const feature of detail.features) {
            evidence.append(el('tr', {},
                el('td', { text: feature.key.replace(/_/g, ' ') }),
                el('td', { class: 'mono', text: formatMeasure(feature.value) }),
                el('td', {
                    class: 'mono',
                    text: feature.tenant_median === null || feature.tenant_median === undefined
                        ? '\u2014' : formatMeasure(feature.tenant_median),
                }),
                percentileCell(feature)));
        }
        if (!detail.features.length) {
            evidence.append(el('tr', {},
                el('td', { colspan: '4' },
                    el('div', { class: 'empty-state' },
                        el('p', { class: 'empty-state__hint',
                            text: 'No numeric signals were stored for this customer.' })))));
        }
        const count = byId('evidence-count');
        if (count) {
            count.textContent = `${detail.features.length} signal${detail.features.length === 1 ? '' : 's'}`;
        }

        const playbook = clear(byId('detail-playbook'));
        if (detail.reason) {
            playbook.append(el('p', { class: 'section__title', text: 'Why this score' }),
                el('p', { class: 'payload-box', style: 'margin-bottom:1.1rem', text: detail.reason }));
        }
        const action = detail.playbook || {};
        playbook.append(el('p', { class: 'section__title', text: 'Recommended intervention' }),
            el('div', { class: 'evidence', style: 'margin-bottom:.8rem' },
                el('div', { class: 'evidence__row' },
                    el('span', { class: 'evidence__label', text: 'Action' }),
                    el('span', { class: 'evidence__value', text: action.action_type || '\u2014' })),
                el('div', { class: 'evidence__row' },
                    el('span', { class: 'evidence__label', text: 'Channel' }),
                    el('span', { class: 'evidence__value', text: action.channel || '\u2014' }))),
            el('div', { class: 'payload-box', text: action.action_payload || 'No payload was returned.' }));
    }

    function initCustomerDetail() {
        const body = document.body;
        const tenantId = body.getAttribute('data-tenant-id');
        const entityId = body.getAttribute('data-entity-id');

        (async () => {
            await loadStatus();
            const messages = [];
            try {
                const detail = await requestJson(
                    `${API.customer}?tenant_id=${encodeURIComponent(tenantId)}` +
                    `&entity_id=${encodeURIComponent(entityId)}`);
                renderDetail(detail);
                // The run remembers its own engine, so this says who scored
                // *this* customer rather than what the deployment defaults to.
                messages.push(['info', detail.offline_mode
                    ? 'This score comes from the system model: deterministic, computed from the ' +
                      'uploaded features, and not model output.'
                    : 'This score comes from the AI model.']);
            } catch (err) {
                show(byId('detail-score-section'), false);
                show(byId('detail-evidence-panel'), false);
                show(byId('detail-playbook-panel'), false);
                show(byId('detail-empty'), true);
                // The 404 text already says which of the two causes it is.
                const hint = byId('detail-empty-hint');
                if (hint) hint.textContent = err.message;
            }
            renderBanners(messages);
        })();
    }

    // -- boot ---------------------------------------------------------------

    function initDashboard() {
        const body = document.body;
        const tenantId = body.getAttribute('data-tenant-id') || localStorage.getItem('tenant_id');
        if (!tenantId) {
            window.location.replace('/');
            return;
        }
        if (!body.getAttribute('data-tenant-id')) {
            // The server did not recognise this tenant, so it rendered the
            // bootstrap page; hand the id back to the routing endpoint.
            window.location.replace('/dashboard?tenant_id=' + encodeURIComponent(tenantId));
            return;
        }

        initSidebar();
        initTableControls();
        initDrawer();
        initDropzone(tenantId);

        (async () => {
            const status = await loadStatus();
            initEnginePicker(status);
            const summary = await loadMetrics(tenantId);
            const messages = [];
            // The stored run remembers which engine produced it, so this describes
            // the numbers actually on screen rather than what a new run would
            // default to.
            if (summary && summary.offline_mode) {
                messages.push(['info',
                    'These predictions came from the system model: deterministic, computed from ' +
                    'the uploaded features, and not model output. Re-analyze with the AI model to compare.']);
            }
            if (summary) {
                for (const warning of summary.warnings || []) messages.push(['warning', warning]);
            }
            renderBanners(messages);
        })();
    }

    document.addEventListener('DOMContentLoaded', () => {
        if (typeof Chart !== 'undefined') {
            Chart.defaults.font.family = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
            Chart.defaults.font.size = 11;
            Chart.defaults.color = '#4b5160';
        }
        if (initOnboarding()) return;
        if (document.body.hasAttribute('data-entity-id')) {
            initCustomerDetail();
            return;
        }
        if (byId('dropzone')) initDashboard();
    });
})();
