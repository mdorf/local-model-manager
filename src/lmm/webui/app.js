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
      <a class="btn ghost" href="${esc(openUrl)}" target="_blank" rel="noopener"
         title="Open the model server's built-in page (${esc(openUrl)})"
         style="padding:4px 10px;font-size:12px;text-decoration:none">Open server ↗</a>
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

// ── Detail panel ──────────────────────────────────────────────────────
function renderDetail() {
  if (!selected) {
    return `<div class="main" style="display:flex;align-items:center;justify-content:center;color:#999">
      Select a model to see its recommendation
    </div>`;
  }
  const rec = selected;
  const fit = rec.fit || {};
  const flagsText = flagsToLines(rec.flags || []).join(" \\\n  ");
  const warningsHtml = (rec.warnings || []).length
    ? `<div class="warnings">${rec.warnings.map(w => `<div class="warn-item">⚠ ${esc(w)}</div>`).join("")}</div>`
    : "";

  const runningServer = servers[0];
  const anyRunning = !!runningServer;
  const isLive = anyRunning && runningServer.model === rec.model;  // this model is the running one
  const isBound = isLive && bound.bound;  // bound ⇒ Hermes points at the running model (#2)

  let actionsHtml;
  if (isLive) {
    // The running model is the only state where an agent can be connected.
    const connectLabel = isBound ? "✓ Connected to Hermes" : "Connect an agent…";
    const connectTitle = isBound
      ? "Hermes is pointed at this model — click to re-bind"
      : "Bind Hermes (or any OpenAI-compatible app) to this running model";
    actionsHtml = `<button id="btn-connect" class="btn" title="${esc(connectTitle)}">${connectLabel}</button>`;
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
    <div class="detail-row"><span class="lbl">Context</span><span>${rec.context ? rec.context.toLocaleString() + " tokens" : "—"}</span></div>
    <div class="detail-row"><span class="lbl">Cache type</span><span>${esc(rec.cache_type || "—")}</span></div>
    <div class="gauge-wrap">
      <div class="gauge-label">RAM fit — ${esc(fit.level || "unknown")}</div>
      <div class="gauge"><div class="gauge-fill ${esc(fit.level || "")}" style="width:${fitWidth(fit.level)}"></div></div>
      ${fit.message ? `<div class="fit-msg">${esc(fit.message)}</div>` : ""}
    </div>
    ${warningsHtml}
    ${flagsText ? `<div class="detail-row"><span class="lbl">Flags</span></div><pre class="flags">${esc(flagsText)}</pre>` : ""}
    <div class="actions">${actionsHtml}</div>
  </div>`;
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
    renderDetail() +
    `</div>` +
    renderDrawer();
  wireEvents();
  scrollLogsToBottom();  // a full repaint re-renders the log view — keep it tailed
}

function scrollLogsToBottom() {
  const logView = document.getElementById("log-view");
  if (logView) logView.scrollTop = logView.scrollHeight;
}

// Reset the log view so a switch/start shows only the new model's logs (the
// daemon truncates server-<port>.log on switch; the WS then re-streams it).
function clearLogs() {
  logLines = [];
  const logView = document.getElementById("log-view");
  if (logView) logView.innerHTML = "";
}

function wireEvents() {
  // Sidebar item clicks
  root.querySelectorAll(".item[data-name]").forEach((el) => {
    el.onclick = () => selectModel(el.dataset.name);
  });
  // Stop button in topbar
  const stopBtn = root.querySelector("#btn-stop");
  if (stopBtn) stopBtn.onclick = doStop;
  // Connect-an-agent button (detail pane, enabled only for the running model)
  const connectBtn = root.querySelector("#btn-connect");
  if (connectBtn) connectBtn.onclick = showConnect;
  // Start/Switch button
  const startBtn = root.querySelector("#btn-start");
  if (startBtn) startBtn.onclick = doStart;
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

  // Reset the log pane so it shows the incoming model's logs, not the previous
  // model's tail mixed in (the daemon truncates the log file on switch).
  clearLogs();
  // Loading a model is slow (the daemon waits for /health + a smoke test before
  // returning). Show progress and lock the controls; live logs stream below.
  setBusy(isRunning ? `Switching to ${selected.model}…` : `Starting ${selected.model}…`);
  try {
    if (isRunning) {
      await api.switch(selected.model, port);
    } else {
      await api.start(selected.model, 8080);
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

  const modelId = info.model_id || running.model;
  // Build the remote-bind command + Base URL from the host the browser actually
  // used to reach the daemon (location.hostname) — NOT 127.0.0.1, which is wrong
  // when this command is run on "another machine". When the server is LAN-exposed
  // it requires the inference key, so include it so the command works as-is.
  const host = location.hostname;
  const baseUrl = `http://${host}:${running.port}/v1`;
  const keyArg = info.inference_key ? ` --api-key ${info.inference_key}` : "";
  const bindCmd = `lmm bind ${running.model} --host ${host} --port ${running.port}${keyArg}`;
  // Hermes already points at this running model (#2) → re-binding is a no-op,
  // so show it as done and disabled rather than an active button.
  const alreadyBound = !!bound.bound;
  const bindBtnHtml = alreadyBound
    ? `<button class="btn" id="btn-bind-now" disabled
         title="Hermes already points at this model">✓ Hermes is bound to this model</button>`
    : `<button class="btn" id="btn-bind-now">Bind Hermes on this host</button>`;
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
        <p class="modal-sub">On <b>this host</b>, bind in one click:</p>
        ${bindBtnHtml}
        <p class="modal-sub" style="margin-top:10px">On another machine, run this once
          (updates that machine's <code>~/.hermes/config.yaml</code>):</p>
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
      navigator.clipboard.writeText(info.inference_key || "").catch(() => {});
      overlay.querySelector("#btn-copy-key").textContent = "Copied!";
      setTimeout(() => { overlay.querySelector("#btn-copy-key").textContent = "Copy"; }, 1500);
    };
  }

  overlay.querySelector("#btn-copy-cmd").onclick = () => {
    navigator.clipboard.writeText(bindCmd).catch(() => {});
    overlay.querySelector("#btn-copy-cmd").textContent = "Copied!";
    setTimeout(() => { overlay.querySelector("#btn-copy-cmd").textContent = "Copy command"; }, 1500);
  };

  const bindNowBtn = overlay.querySelector("#btn-bind-now");
  if (bindNowBtn && !alreadyBound) bindNowBtn.onclick = async () => {
    const btn = bindNowBtn;
    btn.disabled = true;
    btn.textContent = "Binding…";
    try {
      const res = await api.bind({});
      btn.textContent = "✓ Bound Hermes to " + (res.model || modelId);
      await refreshBindStatus();
      paint();  // surfaces the "✓ Hermes bound" badge (modal lives on <body>, survives)
    } catch (e) {
      btn.disabled = false;
      if (e && e.code === 403) {
        btn.textContent = "Only available on the host — use the command ↓";
      } else {
        btn.textContent = "Bind Hermes on this host";
        showBanner(e && e.message ? "Bind failed: " + e.message : "Bind failed");
      }
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
    const stopBtn = root.querySelector("#btn-stop");
    if (stopBtn) stopBtn.onclick = doStop;
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
