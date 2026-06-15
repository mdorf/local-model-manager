// src/lmm/webui/api.js
function getToken() {
  if (window.LMM_TOKEN) return window.LMM_TOKEN;        // loopback-injected
  return localStorage.getItem("lmm_token") || "";
}
export function setToken(t) { localStorage.setItem("lmm_token", t); }
export function hasToken() { return !!getToken(); }

async function req(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: { "Authorization": "Bearer " + getToken(),
               ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = "";
    try { detail = await res.text(); } catch (e) { /* ignore */ }
    const err = new Error(detail || res.statusText);
    err.code = res.status;  // 401/403/409/… so callers can branch
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  models: () => req("GET", "/api/models"),
  recommend: (name) => req("GET", `/api/models/${encodeURIComponent(name)}/recommend`),
  servers: () => req("GET", "/api/servers"),
  start: (model, port) => req("POST", "/api/servers", { model, port }),
  switch: (model, port) => req("POST", "/api/servers/switch", { model, port }),
  stop: (port) => req("DELETE", `/api/servers/${port}`),
  connectionInfo: () => req("GET", "/api/connection-info"),
  bind: (body) => req("POST", "/api/bind", body || {}),
};

export function openStream(onMsg, onState) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  let ws, backoff = 500;
  const connect = () => {
    ws = new WebSocket(`${proto}://${location.host}/api/stream`, ["lmm.bearer." + getToken()]);
    ws.onopen = () => { backoff = 500; onState && onState("connected"); };
    ws.onmessage = (e) => onMsg(JSON.parse(e.data));
    ws.onclose = () => { onState && onState("reconnecting");
      setTimeout(connect, backoff); backoff = Math.min(backoff * 2, 8000); };
  };
  connect();
  return () => ws && ws.close();
}
