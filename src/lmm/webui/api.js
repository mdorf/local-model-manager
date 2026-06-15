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
  if (res.status === 401) { const e = new Error("unauthorized"); e.code = 401; throw e; }
  if (!res.ok) throw new Error((await res.text()) || res.statusText);
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
