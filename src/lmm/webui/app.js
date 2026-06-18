// src/lmm/webui/app.js
import { api, openStream, hasToken, setToken } from "/api.js";

const root = document.getElementById("app");
let selected = null;   // recommend() result for the currently selected model
let servers = [];      // [{port, model, status, ...}]
let models = [];       // [{name, ...}]
let wsState = "disconnected";
let logLines = [];     // capped at 500 lines
let busy = false;      // an action (start/switch/stop) is in flight
let bound = { bound: false };   // is Hermes currently bound to the running server?

// ── Token gate ────────────────────────────────────────────────────────
function tokenGate() {
  root.innerHTML = `
    <div class="topbar"><span class="title">local-model-manager</span></div>
    <div class="gate">
      <div class="modal">
        <h3>Enter daemon token</h3>
        <p style="font-size:13px;color:#666;margin-bottom:12px;">
          Get it with: <code style="font-family:monospace">lmm token</code>
        </p>
        <div class="field">
          <input id="tok" type="password" placeholder="paste token here" autocomplete="off"/>
        </div>
        <div class="modal-actions">
          <button id="go" class="btn">Connect</button>
        </div>
      </div>
    </div>`;
  const inp = root.querySelector("#tok");
  const go = () => {
    const v = inp.value.trim();
    if (!v) return;
    setToken(v);
    location.reload();
  };
  root.querySelector("#go").onclick = go;
  inp.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
}

// ── Data refresh ──────────────────────────────────────────────────────
async function refresh() {
  const [modelsResp, serversResp] = await Promise.all([api.models(), api.servers()]);
  models = modelsResp.models;
  servers = serversResp.servers;
  await refreshBindStatus();
  // On first load, if a model is already running, select it so its config +
  // connect state show immediately — no extra click. The `!selected` guard
  // means this never overrides a selection you've made.
  if (!selected && servers[0]) {
    try { selected = await api.recommend(servers[0].model); }
    catch (e) { /* running model not resolvable in roots — leave unselected */ }
  }
  paint();
}

async function refreshBindStatus() {
  try { bound = await api.bindStatus(); }
  catch (e) { bound = { bound: false }; }
}

// ── Top status bar ────────────────────────────────────────────────────
function renderTopbar() {
  const running = servers[0];
  let statusHtml = "";
  if (running) {
    // Reach the model server via the same host the UI is served from (works
    // locally and over the LAN), on the running server's port.
    const openUrl = `${location.protocol}//${location.hostname}:${running.port}/`;
    const boundBadge = bound.bound
      ? `<span class="badge" title="Your Hermes is pointed at this server">✓ Hermes bound</span>`
      : "";
    statusHtml = `
      <span class="status-label">running:</span>
      <span class="status-value">${esc(running.model)} :${running.port}</span>
      ${boundBadge}
      <button class="btn ghost" id="btn-open-server" data-url="${esc(openUrl)}"
         title="Open the model server's built-in page (${esc(openUrl)})"
         style="padding:4px 10px;font-size:12px">Open server ↗</button>
      <button class="btn danger" id="btn-stop" style="padding:4px 10px;font-size:12px">Stop</button>`;
  } else {
    statusHtml = `<span class="status-label">no server running</span>`;
  }
  return `<div class="topbar">
    <span class="title">local-model-manager</span>
    <span class="sep"></span>
    ${statusHtml}
  </div>`;
}

// ── Sidebar ───────────────────────────────────────────────────────────
function renderSidebar() {
  const runningModel = servers[0] ? servers[0].model : null;
  const items = models.map((m) => {
    const isSel = selected && selected.model === m.name;
    const isRun = runningModel === m.name;
    return `<div class="item${isSel ? " sel" : ""}" data-name="${esc(m.name)}">
      <span class="name" title="${esc(m.name)}">${esc(m.name)}</span>
      ${isRun ? '<span class="badge">live</span>' : ""}
    </div>`;
  }).join("");
  return `<div class="side">
    <h2>Models</h2>
    ${items || '<div style="padding:10px 12px;font-size:12px;color:#999">No models found</div>'}
  </div>`;
}

// ── Fit gauge ─────────────────────────────────────────────────────────
function fitWidth(level) {
  return level === "comfortable" ? "85%" : level === "tight" ? "55%" : "20%";
}

// Group a flat llama-server argv (["-m", path, "-ngl", "999", ...]) into one
// line per flag: a "-flag" followed by a non-flag token is its value, so they
// pair up; bare/boolean flags stand alone.
function flagsToLines(flags) {
  const lines = [];
  for (let i = 0; i < flags.length; i++) {
    const tok = String(flags[i]);
    const next = i + 1 < flags.length ? String(flags[i + 1]) : null;
    if (tok.startsWith("-") && next !== null && !next.startsWith("-")) {
      lines.push(tok + " " + next);
      i++;  // consumed the value
    } else {
      lines.push(tok);
    }
  }
  return lines;
}

// Parse a flag list into {key, value} rows for the current-run-params table.
// Keys are the arg names without leading dashes. Drops -m (the model file path)
// and --alias (the model name, already in the title) — neither is a tuning knob.
const _RUNPARAM_HIDE = new Set(["m", "alias", "host", "port"]);
function flagsToKV(flags) {
  const rows = [];
  for (let i = 0; i < flags.length; i++) {
    const tok = String(flags[i]);
    if (!tok.startsWith("-")) continue;        // stray value (already consumed)
    const next = i + 1 < flags.length ? String(flags[i + 1]) : null;
    let value = "";
    if (next !== null && !next.startsWith("-")) { value = next; i++; }
    const key = tok.replace(/^-+/, "");
    if (_RUNPARAM_HIDE.has(key)) continue;
    if (key === "c") {  // context: annotate with the friendly size, e.g. 262144 (256K)
      const n = Number(value);
      const friendly = Number.isFinite(n) ? fmtCtx(n) : null;
      rows.push({ key: "c (context)",
                  value: friendly && friendly !== String(n) ? `${value} (${friendly})` : value });
      continue;
    }
    rows.push({ key, value });
  }
  return rows;
}

// Friendly context label: 262144 → "256K", 1048576 → "1M".
function fmtCtx(n) {
  if (n % 1048576 === 0) return (n / 1048576) + "M";
  if (n % 1024 === 0) return (n / 1024) + "K";
  return String(n);
}

// ── Detail panel ──────────────────────────────────────────────────────
function renderDetail() {
  if (!selected) {
    return `<div class="main" style="display:flex;align-items:center;justify-content:center;color:#999">
      Select a model to see its recommendation
    </div>`;
  }
  const rec = selected;
  const fit = rec.fit || {};
  const flagsText = flagsToLines(rec.flags || []).join("\n");
  const warningsHtml = (rec.warnings || []).length
    ? `<div class="warnings">${rec.warnings.map(w => `<div class="warn-item">⚠ ${esc(w)}</div>`).join("")}</div>`
    : "";

  // Intrinsic model metadata (from /api/models), shown above the launch recommendation.
  const meta = models.find((x) => x.name === rec.model) || {};
  const row = (label, html) => html ? `<div class="detail-row"><span class="lbl">${label}</span><span>${html}</span></div>` : "";
  const quant = meta.quant ? esc(meta.quant) + (meta.quantized_by ? ` (by ${esc(meta.quantized_by)})` : "") : "";
  // Direct card link when the GGUF embeds the repo; otherwise a reliable HF
  // search for the model name (never a fabricated/guessed direct URL).
  const card = meta.hf_base_repo
    ? `<a href="${esc(meta.hf_base_repo)}" target="_blank" rel="noopener">${esc(meta.hf_base_repo.replace("https://huggingface.co/", ""))} ↗</a>`
    : `<a href="https://huggingface.co/models?search=${encodeURIComponent(meta.display_name || rec.model)}" target="_blank" rel="noopener">search Hugging Face ↗</a>`;
  const metaHtml =
    row("Architecture", esc(meta.arch || "")) +
    row("Parameters", esc(meta.size_label || "")) +
    row("Quantization", quant) +
    row("Author", esc(meta.author || "")) +
    row("Max context", meta.context_length ? meta.context_length.toLocaleString() + " tokens" : "") +
    (meta.has_mtp ? row("Speculative", "draft-mtp (built-in draft head)") : "") +
    (meta.has_chat_template ? row("Chat template", "embedded") : "") +
    row("License", esc(meta.license || "")) +
    row("Model card", card);

  // Context dropdown: ladder values up to the model's max, default = recommended.
  const maxCtx = meta.context_length || rec.context || 8192;
  const ctxLadder = [262144, 131072, 65536, 32768, 16384, 8192].filter((c) => c <= maxCtx);
  if (!ctxLadder.includes(maxCtx)) ctxLadder.push(maxCtx);
  ctxLadder.sort((a, b) => b - a);
  const ctxOptions = ctxLadder.map((c) =>
    `<option value="${c}"${c === rec.context ? " selected" : ""}>${fmtCtx(c)} (${c.toLocaleString()} tokens)</option>`).join("");

  const runningServer = servers[0];
  const anyRunning = !!runningServer;
  const isLive = anyRunning && runningServer.model === rec.model;  // this model is the running one
  const isBound = isLive && bound.bound;  // bound ⇒ Hermes points at the running model (#2)
  // What the running server was ACTUALLY launched with (read-only; covers adopted
  // servers too). Shown only for the live model, distinct from the editable
  // "Recommended launch" flags above.
  const liveKV = isLive && Array.isArray(runningServer.flags) ? flagsToKV(runningServer.flags) : [];
  const currentParamsHtml = liveKV.length
    ? `<div class="col">
         <div class="col-head">Current run params</div>
         ${liveKV.map((r) => `<div class="detail-row"><span class="lbl">${esc(r.key)}</span><span>${esc(r.value || "✓")}</span></div>`).join("")}
       </div>`
    : "";

  let actionsHtml;
  if (isLive) {
    // The running model is the only state where an agent can be connected.
    const connectLabel = isBound ? "✓ Connected to Hermes" : "Connect an agent…";
    const connectTitle = isBound
      ? "Hermes is pointed at this model — click to re-bind"
      : "Bind Hermes (or any OpenAI-compatible app) to this running model";
    // Keep the launch action first and Connect second, matching the non-live
    // layout ([Start/Switch] [Connect]) so the buttons don't jump between states.
    actionsHtml = `<button id="btn-reload" class="btn" title="Restart this model with the launch config above (applies any edited flags / context)">Reload</button>
       <button id="btn-connect" class="btn ghost" title="${esc(connectTitle)}">${connectLabel}</button>`;
  } else {
    // Not running: must start/switch to it first; connecting is disabled until then.
    const startLabel = anyRunning ? "Switch" : "Start";
    actionsHtml =
      `<button id="btn-start" class="btn">${startLabel}</button>
       <button id="btn-connect" class="btn ghost" disabled
         title="Available only on the model that is currently live.">Connect an agent…</button>`;
  }

  return `<div class="main">
    <h1>${esc(rec.model)}</h1>
    <div class="detail-cols">
      <div class="col">${metaHtml}</div>
      ${currentParamsHtml}
    </div>
    <div class="section-label">Recommended launch</div>
    <div class="detail-row"><span class="lbl">Context</span><span><select id="ctx-select" class="ctx-select">${ctxOptions}</select></span></div>
    <div class="detail-row"><span class="lbl">Cache type</span><span>${esc(rec.cache_type || "—")}</span></div>
    <div class="gauge-wrap">
      <div class="gauge-label">RAM fit — ${esc(fit.level || "unknown")}</div>
      <div class="gauge"><div class="gauge-fill ${esc(fit.level || "")}" style="width:${fitWidth(fit.level)}"></div></div>
      ${fit.message ? `<div class="fit-msg">${esc(fit.message)}</div>` : ""}
    </div>
    ${warningsHtml}
    ${flagsText ? `<div class="detail-row"><span class="lbl">Flags</span></div>
      <textarea class="flags" id="flags-edit" spellcheck="false">${esc(flagsText)}</textarea>
      <div class="flags-hint">Editable — changes apply to the next Start/Switch (the RAM fit-check won't re-run). <a id="flags-reset">Reset to recommended</a></div>` : ""}
  </div>
  <div class="actions">${actionsHtml}</div>`;
}

// ── Log drawer ────────────────────────────────────────────────────────
function renderDrawer() {
  const stateClass = wsState === "connected" ? "connected" : wsState === "reconnecting" ? "reconnecting" : "";
  const linesHtml = logLines.map((l) => {
    const isErr = /error|fatal|fail/i.test(l);
    return `<div class="log-line${isErr ? " err" : ""}">${esc(l)}</div>`;
  }).join("");
  return `<div class="drawer">
    <div class="drawer-header">
      <span class="drawer-title">Live Logs</span>
      <span class="ws-state ${stateClass}">${wsState}</span>
      <button class="log-clear" id="btn-clear-logs"
        title="Clear the log view — new messages keep streaming; the server's log file is untouched">Clear</button>
    </div>
    <div class="logs" id="log-view">${linesHtml}</div>
  </div>`;
}

// ── Full paint ────────────────────────────────────────────────────────
function paint() {
  root.innerHTML =
    renderTopbar() +
    `<div class="layout">` +
    renderSidebar() +
    `<div class="content-col">` +
    renderDetail() +
    renderDrawer() +
    `</div>` +
    `</div>`;
  wireEvents();
  scrollLogsToBottom();  // a full repaint re-renders the log view — keep it tailed
}

function scrollLogsToBottom() {
  const logView = document.getElementById("log-view");
  if (logView) logView.scrollTop = logView.scrollHeight;
}

// Clears the client-side log buffer + view. Used on switch/start (so only the
// new model's logs show — the daemon truncates server-<port>.log on switch) AND
// by the drawer's "Clear" button (a purely visual reset). Either way the
// server's log file is untouched and new streamed lines keep appending.
function clearLogs() {
  logLines = [];
  const logView = document.getElementById("log-view");
  if (logView) logView.innerHTML = "";
}

// Wires the topbar's buttons. Called from wireEvents (full paint) AND from the
// partial status update that rebuilds only the topbar — both must wire the same
// set, so they share this helper (a divergence here once dropped Open server's
// handler after the first WS status frame).
function wireTopbar() {
  const stopBtn = root.querySelector("#btn-stop");
  if (stopBtn) stopBtn.onclick = doStop;
  // Open the model server's built-in page. When the server is LAN-exposed it
  // requires an API key, but its built-in page can't read that key from a URL —
  // so first hand the user the key + how to paste it (see openServer).
  const openBtn = root.querySelector("#btn-open-server");
  if (openBtn) openBtn.onclick = () => openServer(openBtn.dataset.url);
}

function wireEvents() {
  // Sidebar item clicks
  root.querySelectorAll(".item[data-name]").forEach((el) => {
    el.onclick = () => selectModel(el.dataset.name);
  });
  // Topbar buttons (also rewired by the partial status update — keep in sync
  // via the shared helper).
  wireTopbar();
  // Connect-an-agent button (detail pane, enabled only for the running model)
  const connectBtn = root.querySelector("#btn-connect");
  if (connectBtn) connectBtn.onclick = showConnect;
  // Clear the log view (visual only — see clearLogs). The drawer survives the
  // partial status update, so wiring it on full paint is enough.
  const clearBtn = root.querySelector("#btn-clear-logs");
  if (clearBtn) clearBtn.onclick = clearLogs;
  // Start/Switch button (and Reload, which is the same path: switch the running
  // server to the edited launch config)
  const startBtn = root.querySelector("#btn-start");
  if (startBtn) startBtn.onclick = doStart;
  const reloadBtn = root.querySelector("#btn-reload");
  if (reloadBtn) reloadBtn.onclick = doStart;
  // Reset edited flags back to the recommended config
  const flagsReset = root.querySelector("#flags-reset");
  if (flagsReset) flagsReset.onclick = () => {
    const ta = document.getElementById("flags-edit");
    if (ta && selected) ta.value = flagsToLines(selected.flags || []).join("\n");
    const sel = document.getElementById("ctx-select");
    if (sel && selected) sel.value = String(selected.context);
  };
  // Context dropdown writes the chosen -c into the flags textarea (single source
  // of truth = the flags); Start/Switch then sends it as an override.
  const ctxSel = root.querySelector("#ctx-select");
  if (ctxSel) ctxSel.onchange = () => {
    const ta = document.getElementById("flags-edit");
    if (!ta) return;
    const toks = ta.value.replace(/\\\s*\n/g, " ").trim().split(/\s+/).filter(Boolean);
    const i = toks.indexOf("-c");
    if (i >= 0 && i + 1 < toks.length) toks[i + 1] = ctxSel.value;
    else toks.push("-c", ctxSel.value);
    ta.value = flagsToLines(toks).join("\n");
  };
}

// ── Model selection ───────────────────────────────────────────────────
async function selectModel(name) {
  try {
    selected = await api.recommend(name);
    paint();
  } catch (e) {
    handleError(e);
  }
}

// Read user-edited flags as a token list, or null if unchanged from the
// recommendation. Strips the "\<newline>" line continuations, then splits on
// whitespace (paths with spaces would need quoting — uncommon for model dirs).
function readFlagOverride() {
  const ta = document.getElementById("flags-edit");
  if (!ta || !selected) return null;
  const original = flagsToLines(selected.flags || []).join("\n");
  if (ta.value.trim() === original.trim()) return null;
  const toks = ta.value.replace(/\\\s*\n/g, " ").trim().split(/\s+/).filter(Boolean);
  return toks.length ? toks : null;
}

// ── Start / Switch ────────────────────────────────────────────────────
async function doStart() {
  if (busy || !selected) return;
  const fit = selected.fit || {};
  if (fit.level === "wont_load") {
    if (!confirm(`This model will not fit in RAM (${fit.message || "insufficient memory"}). Start anyway?`)) return;
  } else if (fit.level === "tight") {
    if (!confirm(`This model is a tight fit (${fit.message || "may be slow"}). Start anyway?`)) return;
  }

  const port = (servers[0] && servers[0].port) || 8080;
  const isRunning = servers.length > 0;
  const isReload = isRunning && servers[0].model === selected.model;  // restart the live model
  const override = readFlagOverride();  // null unless the user edited the flags

  // Reset the log pane so it shows the incoming model's logs, not the previous
  // model's tail mixed in (the daemon truncates the log file on switch).
  clearLogs();
  // Loading a model is slow (the daemon waits for /health + a smoke test before
  // returning). Show progress and lock the controls; live logs stream below.
  setBusy(isReload ? `Reloading ${selected.model}…`
          : isRunning ? `Switching to ${selected.model}…` : `Starting ${selected.model}…`);
  try {
    if (isRunning) {
      await api.switch(selected.model, port, override);
    } else {
      await api.start(selected.model, 8080, override);
    }
    await refresh();
  } catch (e) {
    handleError(e);
  } finally {
    clearBusy();
  }
}

// ── Stop ──────────────────────────────────────────────────────────────
async function doStop() {
  if (busy || !servers[0]) return;
  const port = servers[0].port;
  setBusy("Stopping…");
  try {
    await api.stop(port);
    await refresh();
  } catch (e) {
    handleError(e);
  } finally {
    clearBusy();
  }
}

// ── Busy / progress indicator ─────────────────────────────────────────
function setBusy(label) {
  busy = true;
  root.classList.add("busy");
  let bar = document.getElementById("busy-bar");
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "busy-bar";
    bar.className = "busy-bar";
    bar.innerHTML = `<div class="busy-label"></div><div class="track"></div>`;
    document.body.appendChild(bar);
  }
  bar.querySelector(".busy-label").textContent = label;
  // surface the streaming logs as live progress
  const logView = document.getElementById("log-view");
  if (logView) logView.scrollTop = logView.scrollHeight;
}

function clearBusy() {
  busy = false;
  root.classList.remove("busy");
  const bar = document.getElementById("busy-bar");
  if (bar) bar.remove();
}

// ── "Connect an agent" modal ──────────────────────────────────────────
// Opens the model server's built-in page. A keyless (loopback) server opens
// straight away. A LAN-exposed server is protected by an API key, but its
// built-in page authenticates from its OWN localStorage (set via its Settings
// dialog) and ignores any key in the URL — so we can't pre-authenticate it from
// here. Instead we hand the user the key + a one-time "paste it into Settings"
// step, then open the page.
async function openServer(url) {
  let info;
  try {
    info = await api.connectionInfo();
  } catch (e) {
    // Connection-info is best-effort here; if it fails, just open the page.
    window.open(url, "_blank", "noopener");
    return;
  }
  const key = info.inference_key || "";
  if (!key) {  // keyless (loopback) server — open directly, as before
    window.open(url, "_blank", "noopener");
    return;
  }

  const keyId = "open-key-input";
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.innerHTML = `
    <div class="modal">
      <h3>This server needs its API key</h3>
      <p class="modal-intro">The model server is exposed on your LAN, so it's
        protected by an API key. Its built-in page can't read that key
        automatically — paste it in once:</p>
      <ol class="modal-steps">
        <li>Copy the key below.</li>
        <li>Click <b>Open server</b>, then in that page open
          <b>⚙ Settings → API Key</b>, paste, and save.</li>
      </ol>
      <div class="field"><label>API key</label>
        <div style="display:flex;gap:6px;align-items:center">
          <input id="${keyId}" type="password" value="${esc(key)}" readonly style="flex:1"/>
          <button class="btn ghost" id="btn-open-reveal" style="padding:5px 10px;font-size:12px">Reveal</button>
          <button class="btn ghost" id="btn-open-copy" style="padding:5px 10px;font-size:12px">Copy</button>
        </div>
      </div>
      <div class="modal-actions">
        <button class="btn" id="btn-open-go">Open server ↗</button>
        <button class="btn ghost" id="btn-open-cancel">Close</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);

  const close = () => overlay.remove();
  overlay.querySelector("#btn-open-cancel").onclick = close;
  overlay.onclick = (e) => { if (e.target === overlay) close(); };
  overlay.querySelector("#btn-open-reveal").onclick = () => {
    const inp = overlay.querySelector(`#${keyId}`);
    inp.type = inp.type === "password" ? "text" : "password";
    overlay.querySelector("#btn-open-reveal").textContent = inp.type === "password" ? "Reveal" : "Hide";
  };
  overlay.querySelector("#btn-open-copy").onclick = () => {
    copyToClipboard(key);
    overlay.querySelector("#btn-open-copy").textContent = "Copied!";
    setTimeout(() => { overlay.querySelector("#btn-open-copy").textContent = "Copy"; }, 1500);
  };
  overlay.querySelector("#btn-open-go").onclick = () => {
    copyToClipboard(key);  // copy on open too, for convenience
    window.open(url, "_blank", "noopener");
    close();
  };
}

// Connects an agent/app to the RUNNING model. On the host (loopback) the daemon
// runs as the user and binds Hermes in one click; for a remote machine it can't
// write that machine's config, so we hand over the command (Hermes) and the raw
// settings (any OpenAI-compatible app) to use locally.
async function showConnect() {
  const running = servers[0];
  if (!running) return;  // button only appears while a server is running
  let info;
  try {
    info = await api.connectionInfo();
  } catch (e) {
    handleError(e);
    return;
  }
  // The operator's Hermes profiles, so bind can target a specific one (e.g.
  // qwen-herm) instead of always the active config. Loopback-only; empty for
  // remote clients (they bind via the command below).
  let profiles = [];
  try { profiles = (await api.hermesProfiles()).profiles || []; } catch (e) { /* best-effort */ }

  const modelId = info.model_id || running.model;
  // Build the remote-bind command + Base URL from the host the browser actually
  // used to reach the daemon (location.hostname) — NOT 127.0.0.1, which is wrong
  // when this command is run on "another machine". When the server is LAN-exposed
  // it requires the inference key, so include it so the command works as-is.
  const host = location.hostname;
  const baseUrl = `http://${host}:${running.port}/v1`;
  const keyArg = info.inference_key ? ` --api-key ${info.inference_key}` : "";
  // The remote command targets the SAME profile name picked in the dropdown
  // (resolved on that machine's own ~/.hermes). Rebuilt when the picker changes.
  const remoteCmd = (prof) =>
    `lmm bind ${running.model} --host ${host} --port ${running.port}${keyArg}` +
    (prof && prof !== "default" ? ` --profile ${prof}` : "");
  let bindCmd = remoteCmd(profiles[0] ? profiles[0].name : "default");
  // Hermes already points at this running model (#2) → re-binding is a no-op,
  // so show it as done and disabled rather than an active button.
  // Profile picker: bind targets the chosen Hermes profile (e.g. qwen-herm).
  const profileOptions = profiles.map((p) =>
    `<option value="${esc(p.path)}">${esc(p.name)}</option>`).join("");
  const profileSelect = profiles.length
    ? `<select id="bind-profile" class="ctx-select" style="min-width:130px">${profileOptions}</select>`
    : "";
  // In-app bind writes the SERVER host's ~/.hermes (the daemon runs there), so
  // it's shown everywhere — over the LAN it configures the server, which is what
  // you want when your Hermes runs on the server. The one picker drives both this
  // and the remote command.
  const hostBindHtml =
    `<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
       ${profiles.length ? `<label class="modal-sub" style="margin:0">Profile</label>${profileSelect}` : ""}
       <button class="btn" id="btn-bind-now">Bind</button>
     </div>`;
  const keyId = "conn-key-input";
  const hasKey = !!info.inference_key;
  const keyFieldHtml = hasKey
    ? `<div class="field"><label>API key</label>
         <div style="display:flex;gap:6px;align-items:center">
           <input id="${keyId}" type="password" value="${esc(info.inference_key)}" readonly style="flex:1"/>
           <button class="btn ghost" id="btn-reveal" style="padding:5px 10px;font-size:12px">Reveal</button>
           <button class="btn ghost" id="btn-copy-key" style="padding:5px 10px;font-size:12px">Copy</button>
         </div>
       </div>`
    : `<div class="field"><label>API key</label><code>none required (local server)</code></div>`;

  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.innerHTML = `
    <div class="modal">
      <h3>Connect an agent to this model</h3>
      <p class="modal-intro">Point an AI agent or any OpenAI-compatible app at
        <code>${esc(modelId)}</code>, running on this host. Choose your path:</p>

      <div class="connect-section">
        <h4>Using Hermes</h4>
        <p class="modal-sub">Pick a profile, then bind it on the <b>server</b>
          (this daemon's host) to the running model:</p>
        ${hostBindHtml}
        <p class="modal-sub" style="margin-top:10px">Or, to bind a <b>different</b>
          machine's Hermes, run this once on that machine
          (a profile of that name must exist there):</p>
        <code class="block" id="conn-cmd">${esc(bindCmd)}</code>
        <button class="btn ghost" id="btn-copy-cmd">Copy command</button>
      </div>

      <div class="connect-section">
        <h4>Any OpenAI-compatible app</h4>
        <p class="modal-sub">Enter these in the app's API settings:</p>
        <div class="field"><label>Base URL</label><code>${esc(baseUrl)}</code></div>
        <div class="field"><label>Model</label><code>${esc(modelId)}</code></div>
        ${keyFieldHtml}
      </div>

      <div class="modal-actions">
        <button class="btn" id="btn-close-modal">Close</button>
      </div>
    </div>`;

  document.body.appendChild(overlay);

  overlay.querySelector("#btn-close-modal").onclick = () => overlay.remove();
  overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };

  if (hasKey) {
    overlay.querySelector("#btn-reveal").onclick = () => {
      const inp = overlay.querySelector(`#${keyId}`);
      inp.type = inp.type === "password" ? "text" : "password";
      overlay.querySelector("#btn-reveal").textContent = inp.type === "password" ? "Reveal" : "Hide";
    };
    overlay.querySelector("#btn-copy-key").onclick = () => {
      copyToClipboard(info.inference_key || "");
      overlay.querySelector("#btn-copy-key").textContent = "Copied!";
      setTimeout(() => { overlay.querySelector("#btn-copy-key").textContent = "Copy"; }, 1500);
    };
  }

  overlay.querySelector("#btn-copy-cmd").onclick = () => {
    copyToClipboard(bindCmd);
    overlay.querySelector("#btn-copy-cmd").textContent = "Copied!";
    setTimeout(() => { overlay.querySelector("#btn-copy-cmd").textContent = "Copy command"; }, 1500);
  };

  // Picking a profile updates both the host bind target and the remote command.
  const profSel = overlay.querySelector("#bind-profile");
  if (profSel) profSel.onchange = () => {
    bindCmd = remoteCmd(profSel.options[profSel.selectedIndex].text);
    const cmdEl = overlay.querySelector("#conn-cmd");
    if (cmdEl) cmdEl.textContent = bindCmd;
  };

  const bindNowBtn = overlay.querySelector("#btn-bind-now");
  if (bindNowBtn) bindNowBtn.onclick = async () => {
    const sel = overlay.querySelector("#bind-profile");
    const cfgPath = sel ? sel.value : null;
    const profName = sel ? sel.options[sel.selectedIndex].text : "Hermes";
    bindNowBtn.disabled = true;
    bindNowBtn.textContent = "Binding…";
    try {
      await api.bind(cfgPath ? { hermes_config: cfgPath } : {});
      bindNowBtn.textContent = `✓ Bound ${profName}`;
      await refreshBindStatus();
      paint();  // refresh the "✓ Hermes bound" badge (modal lives on <body>, survives)
      setTimeout(() => { bindNowBtn.disabled = false; bindNowBtn.textContent = "Bind"; }, 1800);
    } catch (e) {
      bindNowBtn.disabled = false;
      bindNowBtn.textContent = "Bind";
      showBanner(e && e.message ? "Bind failed: " + e.message : "Bind failed");
    }
  };
}

// ── WebSocket stream handler ──────────────────────────────────────────
function onStream(msg) {
  if (msg.type === "log") {
    logLines.push(msg.line || "");
    if (logLines.length > 500) logLines.splice(0, logLines.length - 500);
    // Append line directly to the log view if present (avoid full repaint)
    const logView = document.getElementById("log-view");
    if (logView) {
      // Follow the tail only if already near the bottom, so scrolling up to
      // read history isn't yanked back down by incoming lines.
      const nearBottom =
        logView.scrollHeight - logView.scrollTop - logView.clientHeight < 40;
      const div = document.createElement("div");
      const line = msg.line || "";
      div.className = "log-line" + (/error|fatal|fail/i.test(line) ? " err" : "");
      div.textContent = line;
      logView.appendChild(div);
      // Cap DOM lines at 500
      while (logView.children.length > 500) logView.removeChild(logView.firstChild);
      if (nearBottom) logView.scrollTop = logView.scrollHeight;
    }
  } else if (msg.type === "status") {
    servers = msg.servers || [];
    // Update topbar and sidebar without rebuilding the detail panel or logs
    const topbarEl = root.querySelector(".topbar");
    const sideEl = root.querySelector(".side");
    if (topbarEl) topbarEl.outerHTML = renderTopbar();
    if (sideEl) sideEl.outerHTML = renderSidebar();
    // Rewire events for the rebuilt topbar + sidebar (Connect lives in the
    // detail pane, which this partial update leaves intact — no rewire needed).
    wireTopbar();
    root.querySelectorAll(".item[data-name]").forEach((el) => {
      el.onclick = () => selectModel(el.dataset.name);
    });
  }
}

function onStreamState(state) {
  wsState = state;
  // Update the drawer header ws-state indicator
  const stateEl = root.querySelector(".ws-state");
  if (stateEl) {
    stateEl.className = "ws-state" + (state === "connected" ? " connected" : state === "reconnecting" ? " reconnecting" : "");
    stateEl.textContent = state;
  }
  // Show / hide reconnecting banner
  let banner = document.getElementById("reconnecting-banner");
  if (state === "reconnecting") {
    if (!banner) {
      banner = document.createElement("div");
      banner.id = "reconnecting-banner";
      banner.className = "reconnecting-banner";
      banner.textContent = "reconnecting…";
      document.body.appendChild(banner);
    }
  } else if (banner) {
    banner.remove();
  }
}

// ── Error banner ──────────────────────────────────────────────────────
let _bannerTimer = null;
function showBanner(msg, autoHide = true) {
  let b = document.getElementById("error-banner");
  if (!b) {
    b = document.createElement("div");
    b.id = "error-banner";
    b.className = "banner";
    b.title = "click to dismiss";
    b.onclick = hideBanner;
    document.body.appendChild(b);
  }
  b.textContent = msg;
  if (_bannerTimer) clearTimeout(_bannerTimer);
  if (autoHide) _bannerTimer = setTimeout(hideBanner, 8000);
}
function hideBanner() {
  const b = document.getElementById("error-banner");
  if (b) b.remove();
}

function handleError(e) {
  if (e && e.code === 401) {
    localStorage.removeItem("lmm_token");
    tokenGate();
    return;
  }
  // Event-driven callers (button clicks) are not inside main()'s try, so surface
  // the error directly rather than throwing into an unhandled rejection.
  showBanner((e && e.message) ? `Error: ${e.message}` : "Something went wrong");
}

// ── HTML escape ───────────────────────────────────────────────────────
function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Copies text to the clipboard, surviving insecure contexts. navigator.clipboard
// only exists over HTTPS or on localhost — over plain HTTP on a LAN IP (how the
// UI is reached from another machine) it's UNDEFINED, so `navigator.clipboard.
// writeText(...)` throws synchronously (a `.catch` can't help). We feature-detect
// and fall back to the legacy execCommand path. Best-effort: never throws, so
// callers (e.g. the Open-server button) keep working even if the copy fails.
function copyToClipboard(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).catch(() => legacyCopy(text));
      return;
    }
  } catch (e) { /* fall through to legacy */ }
  legacyCopy(text);
}

function legacyCopy(text) {
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    document.execCommand("copy");
    ta.remove();
  } catch (e) { /* give up silently — the value is still shown for manual copy */ }
}

// Esc closes the topmost open modal. All modals share `.modal-overlay`, so this
// one listener covers Connect/Open-server and any modal added later. Registered
// once at module load. (The token gate isn't a modal-overlay, so it's unaffected.)
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  const overlays = document.querySelectorAll(".modal-overlay");
  if (overlays.length) overlays[overlays.length - 1].remove();
});

// ── Entry point ───────────────────────────────────────────────────────
async function main() {
  if (!hasToken()) { tokenGate(); return; }
  try {
    await refresh();
    hideBanner();
    openStream(onStream, onStreamState);
  } catch (e) {
    if (e && e.code === 401) {
      localStorage.removeItem("lmm_token");
      tokenGate();
      return;
    }
    // Daemon unreachable — show banner and retry
    root.innerHTML = `
      <div class="topbar"><span class="title">local-model-manager</span></div>
      <div class="layout" style="align-items:center;justify-content:center;width:100%">
        <div style="color:#999;font-size:14px">Connecting to daemon…</div>
      </div>`;
    showBanner("daemon unreachable — retrying…");
    setTimeout(main, 2000);
  }
}

main();
