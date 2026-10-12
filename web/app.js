'use strict';

/* ---------- helpers ---------- */

const $ = (sel, root = document) => root.querySelector(sel);

function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (['value', 'disabled', 'hidden'].includes(k)) el[k] = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

const ICONS = {
  play: '<polygon points="5 3 19 12 5 21 5 3"/>',
  copy: '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
};
function icon(name) {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('class', 'icon');
  s.setAttribute('viewBox', '0 0 24 24');
  s.setAttribute('aria-hidden', 'true');
  s.innerHTML = ICONS[name] || '';
  return s;
}

const store = {
  get(key, fallback) { try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ } },
};

function toast(message) {
  const t = h('div', { class: 'toast', text: message });
  $('#toasts').append(t);
  setTimeout(() => t.remove(), 2000);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = h('textarea', { 'aria-hidden': 'true' });
    ta.value = text;
    document.body.append(ta);
    ta.select();
    try { document.execCommand('copy'); } catch { /* ignored */ }
    ta.remove();
  }
  toast('Copied to clipboard');
}

const copyBtn = (label, getText) => h('button', { type: 'button', class: 'btn btn-sm', 'aria-label': label, onclick: () => copyText(getText()) }, icon('copy'), 'Copy');

function jsonNodes(text) {
  const frag = document.createDocumentFragment();
  const re = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) frag.append(text.slice(last, m.index));
    const cls = m[1] ? (m[2] ? 'tok-key' : 'tok-str') : m[3] ? 'tok-lit' : 'tok-num';
    frag.append(h('span', { class: cls, text: m[1] || m[0] }));
    if (m[2]) frag.append(m[2]);
    last = re.lastIndex;
  }
  frag.append(text.slice(last));
  return frag;
}

/* inline **bold** and _italic_ (the subset our Markdown and Adaptive Card text uses) */
function inline(text) {
  const out = [];
  const re = /\*\*(.+?)\*\*|(?:^|(?<=\s|\())_(.+?)_(?=$|[\s).,])/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    out.push(m[1] != null ? h('strong', { text: m[1] }) : h('em', { text: m[2] }));
    last = re.lastIndex;
  }
  out.push(text.slice(last));
  return out;
}

/* ---------- card renderers ---------- */

function renderAdaptiveCard(card) {
  const build = el => {
    switch (el.type) {
      case 'TextBlock': {
        const cls = ['ac-tb'];
        if (el.isSubtle) cls.push('subtle');
        if (el.weight === 'Bolder') cls.push('bold');
        if (el.size === 'Small') cls.push('small');
        else if (el.size === 'Medium') cls.push('medium');
        else if (el.size === 'Large') cls.push('large');
        else if (el.size === 'ExtraLarge') cls.push('xl');
        if (el.color) cls.push('c-' + el.color.toLowerCase());
        return h('div', { class: cls.join(' ') }, inline(el.text));
      }
      case 'Container':
        return h('div', { class: 'ac-container' + (el.separator ? ' sep' : '') }, (el.items || []).map(build));
      case 'ColumnSet':
        return h('div', { class: 'ac-columnset' }, (el.columns || []).map(build));
      case 'Column':
        return h('div', { class: 'ac-col ' + (el.width === 'auto' ? 'auto' : 'stretch') }, (el.items || []).map(build));
      case 'Table':
        return h('div', { class: 'ac-table-wrap' }, h('table', { class: 'ac-table' }, h('tbody', null,
          (el.rows || []).map(r => h('tr', null, (r.cells || []).map(c => h('td', null, (c.items || []).map(build))))))));
      default:
        return h('div', { class: 'ac-tb subtle', text: `[unsupported element: ${el.type}]` });
    }
  };
  return h('div', { class: 'ac', role: 'group', 'aria-label': 'Adaptive Card preview' }, (card.body || []).map(build));
}

function renderMarkdown(md) {
  const box = h('div', { class: 'md' });
  for (const block of md.split(/\n{2,}/)) {
    const lines = block.split('\n');
    const rows = lines.filter(l => l.startsWith('|'));
    const head = lines.filter(l => !l.startsWith('|') && !l.startsWith('- '));
    const bullets = lines.filter(l => l.startsWith('- '));
    const parts = [];
    if (head.length) parts.push(h('div', null, head.flatMap((l, i) => [i ? h('br') : null, inline(l)])));
    if (bullets.length) parts.push(h('ul', null, bullets.map(l => h('li', null, inline(l.slice(2))))));
    if (rows.length >= 2) {
      const cells = l => l.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
      const align = cells(rows[1]).map(c => c.endsWith(':'));
      const th = cells(rows[0]).map((c, i) => h('th', { class: align[i] ? 'r' : null }, inline(c)));
      const body = rows.slice(2).map(r => h('tr', null, cells(r).map((c, i) => h('td', { class: align[i] ? 'r' : null }, inline(c)))));
      parts.push(h('table', null, h('thead', null, h('tr', null, th)), h('tbody', null, body)));
    }
    box.append(...parts);
  }
  return box;
}

function renderHtmlCard(html) {
  const frame = h('iframe', { class: 'preview-frame', sandbox: 'allow-same-origin', srcdoc: html, title: 'Web card preview' });
  frame.addEventListener('load', () => {
    try { frame.style.height = Math.max(80, frame.contentDocument.body.scrollHeight + 4) + 'px'; } catch { /* keep default height */ }
  });
  return frame;
}

/* ---------- API ---------- */

async function getJson(path) {
  let res;
  try { res = await fetch(path); } catch { throw new Error('Cannot reach the UI Planner server. Check that "python server.py" is still running.'); }
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  return res.json();
}

async function streamRun(body, onEvent, signal) {
  let res;
  try {
    res = await fetch('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal });
  } catch (e) {
    if (e.name === 'AbortError') throw e;
    throw new Error('Cannot reach the UI Planner server. Check that "python server.py" is still running.');
  }
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try { msg = (await res.json()).error || msg; } catch { /* body was not JSON */ }
    throw new Error(msg);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
}

/* ---------- app ---------- */

const DEFAULT_PROMPT = 'Show escalated cases as a table with customer name and priority';
const PREVIEWS = [
  { id: 'teams', label: 'Teams', hint: 'Adaptive Card 1.5 for Microsoft Teams' },
  { id: 'web', label: 'Web', hint: 'HTML page' },
  { id: 'chat', label: 'Chat', hint: 'Markdown for chat apps' },
];
const RIGHT_TABS = [
  { id: 'structure', label: 'Structure' },
  { id: 'data', label: 'Data' },
];

const S = {
  prompt: store.get('uip.prompt', DEFAULT_PROMPT),
  input: store.get('uip.input', ''),
  model: store.get('uip.model', ''),
  previewTab: store.get('uip.preview', 'teams'),
  rightTab: 'structure',
  models: { loading: true, reachable: false, list: [] },
  running: false,
  statusText: '',
  startedAt: 0,
  timer: null,
  controller: null,
  result: null,
  error: null,
  sample: '',
  warm: '',
  summary: '',
};
let refs = {};

function fmtSecs(s) { return s < 0.001 ? '<1 ms' : s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`; }

/* Ask the server to load the chosen model into RAM and pre-fill its prompt cache for this data, so the
   first real request does not pay the 30-50 s cold start. Fire and forget; runs queue behind it. */
async function warmModel() {
  const model = S.model;
  if (!model || model === 'heuristic' || !S.models.reachable) { S.warm = ''; fillModels(); return; }
  const parsed = parseInput();
  S.warm = 'warming'; fillModels();
  try {
    const res = await fetch('/api/warm', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backend: 'ollama', model: model.replace(/^ollama:/, ''), data: parsed.value }) });
    const j = await res.json();
    if (S.model === model) S.warm = j.warmed ? 'ready' : 'failed';
  } catch { if (S.model === model) S.warm = 'failed'; }
  fillModels();
}

function modelOptions() {
  const opts = S.models.list.map(m => ({ value: 'ollama:' + m.name, label: m.name }));
  opts.push({ value: 'heuristic', label: 'Keyword baseline (no model)' });
  return opts;
}

function pickModel() {
  const opts = modelOptions();
  if (!opts.some(o => o.value === S.model)) {
    const pref = ['ollama:llama3.1:8b', 'ollama:llama3.2:3b'].find(v => opts.some(o => o.value === v));
    S.model = pref || (opts.find(o => o.value.startsWith('ollama:')) || opts[opts.length - 1]).value;
  }
}

function fillModels() {
  if (!refs.model) return;
  pickModel();
  refs.model.replaceChildren(...modelOptions().map(o => h('option', { value: o.value, text: o.label })));
  refs.model.value = S.model;
  refs.modelHint.textContent = S.models.loading ? 'Looking for local models…'
    : !S.models.reachable ? 'Ollama is not running, so only the keyword baseline is available.'
      : S.model === 'heuristic' ? 'Keyword matching only; useful as a baseline.'
        : S.warm === 'warming' ? 'Loading the model into memory…'
          : S.warm === 'ready' ? 'Model loaded and ready.'
            : S.warm === 'failed' ? 'Could not pre-load the model; the first run may be slow.' : '';
}

/* segmented tab switcher (like the platform switcher in the HXL playground) */
function segmented(label, items, current, onChange) {
  const wrap = h('div', { class: 'seg', role: 'tablist', 'aria-label': label });
  const buttons = items.map(it => h('button', { type: 'button', role: 'tab', 'data-id': it.id, text: it.label, onclick: () => select(it.id) }));
  function select(id, focus) {
    current = id;
    for (const b of buttons) { const on = b.dataset.id === id; b.setAttribute('aria-selected', on); b.tabIndex = on ? 0 : -1; if (on && focus) b.focus(); }
    onChange(id);
  }
  wrap.addEventListener('keydown', e => {
    const i = items.findIndex(it => it.id === current);
    const next = e.key === 'ArrowRight' ? (i + 1) % items.length : e.key === 'ArrowLeft' ? (i - 1 + items.length) % items.length : -1;
    if (next >= 0) { e.preventDefault(); select(items[next].id, true); }
  });
  wrap.append(...buttons);
  for (const b of buttons) { const on = b.dataset.id === current; b.setAttribute('aria-selected', on); b.tabIndex = on ? 0 : -1; }
  return wrap;
}

function build(root) {
  const prompt = h('textarea', { id: 'prompt', class: 'textarea', rows: 2, maxlength: 500, placeholder: 'e.g. Show the high priority cases', 'aria-describedby': 'prompt-err' });
  prompt.value = S.prompt;
  const promptErr = h('div', { id: 'prompt-err', class: 'field-error', hidden: true });
  prompt.addEventListener('input', () => { S.prompt = prompt.value; store.set('uip.prompt', S.prompt); promptErr.hidden = true; prompt.removeAttribute('aria-invalid'); });
  prompt.addEventListener('keydown', e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); run(); } });

  const model = h('select', { id: 'model', class: 'select', 'aria-label': 'Model' });
  model.addEventListener('change', () => { S.model = model.value; store.set('uip.model', S.model); S.warm = ''; fillModels(); warmModel(); });
  const modelHint = h('div', { class: 'hint', style: 'margin-top:4px' });
  const runBtn = h('button', { type: 'button', class: 'btn btn-primary', onclick: run }, icon('play'), 'Generate');
  const cancelBtn = h('button', { type: 'button', class: 'btn', hidden: true, onclick: () => S.controller && S.controller.abort() }, 'Cancel');
  const status = h('div', { class: 'status', role: 'status' });

  const input = h('textarea', { id: 'input', class: 'code', spellcheck: 'false', 'aria-label': 'API response JSON', 'aria-describedby': 'input-err' });
  input.value = S.input;
  const inputErr = h('div', { id: 'input-err', class: 'field-error', style: 'padding:6px 14px', hidden: true });
  input.addEventListener('input', () => { S.input = input.value; if (S.input.length < 200000) store.set('uip.input', S.input); checkInput(); });

  const left = { bar: h('div', { class: 'pane-bar' }), notice: h('div'), body: h('div', { class: 'pane-body' }) };
  const right = { bar: h('div', { class: 'pane-bar' }), body: h('div', { class: 'pane-body' }) };
  refs = { prompt, promptErr, model, modelHint, runBtn, cancelBtn, status, input, inputErr, left, right };

  const previewTabs = segmented('Card type', PREVIEWS, S.previewTab, id => { S.previewTab = id; store.set('uip.preview', id); drawLeft(); });
  const rightTabs = segmented('Result or input', RIGHT_TABS, S.rightTab, id => { S.rightTab = id; drawRight(); });
  refs.rightTabs = rightTabs;

  root.append(h('div', { class: 'stack' },
    h('section', { class: 'card', 'aria-label': 'Prompt' }, h('div', { class: 'card-body' },
      h('div', { class: 'prompt-row' },
        h('div', null, h('label', { class: 'label', for: 'prompt', text: 'Prompt' }), prompt, promptErr),
        h('div', null, h('div', { class: 'prompt-controls' }, model, runBtn, cancelBtn), modelHint)),
      status)),
    h('div', { class: 'split' },
      h('section', { class: 'pane', 'aria-label': 'Preview' },
        h('div', { class: 'pane-head' }, h('h2', { text: 'Preview' }), previewTabs),
        left.bar, left.notice, left.body),
      h('section', { class: 'pane', 'aria-label': 'Structure' },
        h('div', { class: 'pane-head' }, h('h2', { text: 'Structure' }), rightTabs),
        right.bar, right.body))));

  fillModels();
  checkInput();
  draw();
}

function parseInput() {
  const t = S.input.trim();
  if (!t) return { error: 'Paste the API response as JSON.' };
  try {
    const v = JSON.parse(t);
    if (!v || typeof v !== 'object' || Array.isArray(v)) return { error: 'The top-level value must be a JSON object.' };
    return { value: v };
  } catch (e) { return { error: 'Invalid JSON: ' + e.message }; }
}

function checkInput() {
  const { error } = S.input.trim() ? parseInput() : {};
  refs.inputErr.hidden = !error;
  refs.inputErr.textContent = error || '';
  if (error) refs.input.setAttribute('aria-invalid', 'true'); else refs.input.removeAttribute('aria-invalid');
}

async function loadSample() {
  try {
    if (!S.sample) S.sample = JSON.stringify(await getJson('/api/datasets/sample_api'), null, 2);
    S.input = S.sample; refs.input.value = S.sample; store.set('uip.input', S.sample); checkInput();
  } catch (e) { toast(e.message); }
}

function showRightTab(id) {
  const btn = refs.rightTabs.querySelector(`[data-id="${id}"]`);
  if (btn) btn.click();
}

const STATUS = {
  start: 'Reading the API response…',
  catalog: 'Summarizing the data…',
  cache_hit: 'Found a saved plan for this request…',
  fast_path: 'Matched without the model…',
  fast_path_declined: 'Not a clear match, asking the model…',
  model_start: d => `The model is choosing what to show (attempt ${d.attempt} of ${d.max_attempts})…`,
  model_done: 'Checking the plan…',
  validate: d => d.errors.length && d.will_retry ? 'The plan had errors. Asking the model to fix them…' : 'Building the cards…',
  compile: 'Rendering the cards…',
};

function onEvent(ev) {
  if (ev.event === 'error') throw new Error(ev.data.message);
  if (ev.event === 'result') { S.result = ev.data; return; }
  const s = STATUS[ev.event];
  if (s) { S.statusText = typeof s === 'function' ? s(ev.data) : s; drawStatus(); }
}

function drawStatus() {
  const el = refs.status;
  if (!S.running) {
    el.replaceChildren(...[S.summary && h('span', { text: S.summary }), S.rewriteNote && h('span', { class: 'hint', text: S.rewriteNote })].filter(Boolean));
    return;
  }
  const secs = Math.round((performance.now() - S.startedAt) / 1000);
  el.replaceChildren(h('span', { class: 'spinner', 'aria-hidden': 'true' }), h('span', { text: `${S.statusText} ${secs}s` }));
}

async function run() {
  if (S.running) return;
  let bad = false;
  if (!S.prompt.trim()) {
    refs.promptErr.textContent = 'Enter a prompt first.'; refs.promptErr.hidden = false;
    refs.prompt.setAttribute('aria-invalid', 'true'); refs.prompt.focus(); bad = true;
  }
  const parsed = parseInput();
  if (parsed.error) {
    showRightTab('data');
    refs.inputErr.textContent = parsed.error; refs.inputErr.hidden = false; refs.input.setAttribute('aria-invalid', 'true');
    if (!bad) refs.input.focus();
    bad = true;
  }
  if (bad) return;

  const body = { prompt: S.prompt.trim(), data: parsed.value, retries: 2 };
  if (S.model === 'heuristic') body.backend = 'heuristic';
  else { body.backend = 'ollama'; body.model = S.model.replace(/^ollama:/, ''); }

  S.controller = new AbortController();
  S.running = true; S.result = null; S.error = null; S.summary = ''; S.rewriteNote = ''; S.statusText = 'Starting…'; S.startedAt = performance.now();
  S.timer = setInterval(drawStatus, 1000);
  refs.runBtn.disabled = true; refs.cancelBtn.hidden = false;
  drawStatus(); draw();
  try {
    await streamRun(body, onEvent, S.controller.signal);
    if (!S.result) throw new Error('The run ended without a result.');
  } catch (e) {
    S.error = e.name === 'AbortError' ? { kind: 'info', message: 'Cancelled.' } : { kind: 'err', message: e.message };
  } finally {
    clearInterval(S.timer); S.running = false; S.controller = null;
    refs.runBtn.disabled = false; refs.cancelBtn.hidden = true;
    const wall = (performance.now() - S.startedAt) / 1000;
    const r = S.result;
    const modelSecs = r ? r.attempts.reduce((a, x) => a + (x.seconds || 0), 0) : 0;
    S.summary = !r ? ''
      : r.source === 'cache' ? `Seen before: served from the plan cache in ${fmtSecs(wall)}, no model used.`
        : r.source === 'fast_path' ? `Answered by the fast path in ${fmtSecs(wall)}, no model used.`
          : `Done in ${fmtSecs(wall)}${modelSecs ? ` (model ${fmtSecs(modelSecs)})` : ''}.`;
    const steps = (r && r.rewrite && r.rewrite.steps) || [];
    S.rewriteNote = steps.length
      ? `Understood as “${r.rewrite.text}” (${[...new Set(steps.map(s => `${s.from} → ${s.to}`))].join(', ')})`
      : '';
    drawStatus(); draw();
    if (S.result && S.rightTab === 'data') showRightTab('structure');
  }
}

/* ---------- drawing ---------- */

function placeholder(text) { return h('div', { class: 'placeholder', text }); }
function barText(text) { return h('span', { text }); }

function codeView(text) {
  const lines = text.split('\n').length;
  const gutter = h('div', { class: 'gutter', 'aria-hidden': 'true', text: Array.from({ length: lines }, (_, i) => i + 1).join('\n') });
  const code = h('pre', { class: 'codepre' });
  code.append(text.length < 60000 ? jsonNodes(text) : text);
  return h('div', { class: 'codeview', tabindex: '0', role: 'region', 'aria-label': 'Result JSON' }, gutter, code);
}

function drawLeft() {
  const r = S.result;
  const { bar, notice, body } = refs.left;
  const meta = PREVIEWS.find(p => p.id === S.previewTab);
  notice.replaceChildren();
  if (S.error) {
    notice.append(h('div', { class: `notice notice-${S.error.kind}`, role: S.error.kind === 'err' ? 'alert' : null },
      h('strong', { text: S.error.kind === 'err' ? 'Something went wrong. ' : '' }), S.error.message));
  } else if (r && r.fallback) {
    notice.append(h('div', { class: 'notice notice-warn' },
      h('strong', { text: 'Showing a default view. ' }), 'The model did not produce a valid plan, so this does not answer your prompt.',
      r.errors.length ? h('ul', null, r.errors.slice(0, 3).map(e => h('li', { text: e }))) : null));
  }

  if (!r) {
    bar.replaceChildren(barText(meta.hint));
    body.replaceChildren(placeholder(S.running ? 'Working…' : 'Run a prompt to see the preview.'));
    return;
  }
  const copyData = { teams: () => JSON.stringify(r.renders.card, null, 2), web: () => r.renders.html, chat: () => r.renders.markdown }[S.previewTab];
  bar.replaceChildren(barText(meta.hint), copyBtn(`Copy ${meta.label} output`, copyData));
  const view = { teams: () => renderAdaptiveCard(r.renders.card), web: () => renderHtmlCard(r.renders.html), chat: () => renderMarkdown(r.renders.markdown) }[S.previewTab]();
  body.replaceChildren(h('div', { class: 'canvas' }, view));
}

function drawRight() {
  const r = S.result;
  const { bar, body } = refs.right;
  if (S.rightTab === 'data') {
    bar.replaceChildren(barText('API response (editable)'), h('button', { type: 'button', class: 'btn btn-sm', onclick: loadSample }, 'Reset to sample'));
    body.replaceChildren(h('div', { class: 'data-wrap' }, refs.input, refs.inputErr));
    return;
  }
  if (!r) {
    bar.replaceChildren(barText('HXL widget JSON'));
    body.replaceChildren(placeholder(S.running ? 'Working…' : 'Run a prompt to see the generated HXL widget JSON.'));
    return;
  }
  const text = JSON.stringify(r.renders.hxl, null, 2);
  bar.replaceChildren(barText('HXL widget JSON · tile/widget, data bound as $data'), copyBtn('Copy HXL widget JSON', () => text));
  body.replaceChildren(codeView(text));
}

function draw() { drawLeft(); drawRight(); }

async function boot() {
  build($('#app'));
  if (!S.input.trim()) await loadSample();
  try {
    const m = await getJson('/api/models');
    S.models = { loading: false, reachable: m.reachable, list: m.models };
  } catch { S.models = { loading: false, reachable: false, list: [] }; }
  fillModels();
  warmModel();
}
boot();
