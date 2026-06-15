# local-model-manager — Architecture

> Status: **design draft** (no implementation yet). This is the canonical design doc.
> A condensed, decision-log version lives in the maintainer's session memory; this doc is the public/repo-facing source of truth.

## 1. What this is

`local-model-manager` (LMM) is an open-source app to **manage local LLM servers** (initially [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`) on a shared host and **bind agent clients to them**. It lets a user:

- See every local model available on the host (parsed from disk, not from a manifest).
- Get a **hardware-aware, model-aware recommended launch configuration** for any model — with one-click overrides.
- **Start / stop / switch** the running model seamlessly (kill → free → relaunch → readiness-gate).
- **Bind an agent** (first target: [Hermes Agent](https://hermes-agent.nousresearch.com/)) to the running server.

### Reference deployment (the design's driving use case)

A Mac Mini (Apple Silicon, 64 GB unified RAM) serves local models to two people on one LAN. Only **one model fits in RAM at a time**, so a single `llama-server` is shared by multiple agent clients across machines/accounts. LMM is the control plane that makes "which model is loaded, and who can change it" coherent.

## 2. Goals / Non-goals

**Goals**
- Make switching the active local model **trivial and safe** (ultra user-friendly).
- Remove hand-typed `llama-server` flag soup; recommend optimal configs per model + hardware.
- Be a **self-contained app** anyone can install/run; usable from a browser or (later) a native shell.
- Treat the model server as a **shared network resource** with multiple trusted clients.

**Non-goals (explicit scope lines)**
- ❌ Request-based **auto model-swapping / routing / autoscaling / load-balancing**. On-demand swap behind one port is already solved by [`llama-swap`](https://github.com/mostlygeek/llama-swap) — **integrate, don't reinvent**.
- ❌ Per-user roles / RBAC. Trust model is **all clients trusted** (gated by a shared token, see §7).
- ❌ Training/fine-tuning, model conversion, or quantization.

## 3. Trust & topology model

- One **host** runs the models + `llama-server` + the LMM **control daemon**. The daemon is the only component that spawns/stops server processes (it must run on the host).
- **Clients** (browser today; native shell later) on **any machine on the LAN** drive the host through the daemon's API. All clients are **trusted** (no roles); access is gated by a **shared token**.
- The same distributable is **dual-role**: *host mode* (manages servers, advertises on the LAN) and *client mode* (connect to and drive a host).
- **Design for multiple hosts** (domain = `hosts → servers → models`); **ship with one host**.

```
                    LAN
 ┌──────────────┐        ┌─────────────────────── Host (e.g. Mac Mini) ──────────────────────┐
 │ Browser /    │  HTTPS │  LMM control daemon (FastAPI, system LaunchDaemon)                  │
 │ native shell │◀──────▶│   :8770  (token-gated control API + WebSocket + static web UI)      │
 │ (client)     │        │     │ spawns / supervises                                            │
 └──────────────┘        │     ▼                                                                │
                         │  llama-server  :8080 (+8081 …)  ──  /v1  (api-key gated inference)    │
 ┌──────────────┐  /v1   │     ▲                                                                │
 │ Hermes (any  │◀───────┼─────┘  (inference traffic, separate from control plane)              │
 │ machine/acct)│        │  Models on disk: <root>/<family>/<size>/<gguf>  (host-side only)      │
 └──────────────┘        └────────────────────────────────────────────────────────────────────┘
```

Key separation: the **control plane** (`:8770`, can spawn processes — high privilege) is distinct from the **inference plane** (`:8080+`, just model calls). They get distinct secrets (see §7).

## 4. Components

1. **Control daemon** (FastAPI, Python — matches Hermes' Python 3.11; may reuse its config helpers).
   - Installed as a **system `LaunchDaemon`** → starts at boot, independent of GUI login (required for always-on sharing). Runs under a chosen account (reference deployment: the owning user; a dedicated `_lmm` service account is an option — must have read access to the models dir).
   - Owns: discovery, GGUF introspection, recommendation engine, fit-check, server lifecycle + crash supervision, run-history/telemetry, agent binding, log/metrics streaming, serving the web UI, mDNS advertisement.
2. **Web UI** — served by the daemon as static assets; the only "client" needed in v1. **Discipline: the UI talks ONLY to the daemon API** — never the filesystem or processes directly. This is what makes the future Tauri/Rust shell *additive* rather than a rewrite.
3. **Native shell (later)** — Tauri wraps the same web UI and can bundle/launch the daemon.

### Deployment modes (friendliness without compromising the secure default)

- **Try-it / dev mode:** `lmm serve` runs the daemon in the foreground as the **current user** — no account, no privileges, no plist. "Clone and run" in seconds, for evaluation and development.
- **Installed mode:** a **guided installer** sets up the system service on the secure default (dedicated service account + ACL + boot-start + firewall allow) under **one admin prompt**, with **no manual commands**. This is the path to always-on sharing.

Same daemon binary in both; install mode only adds the privilege/lifecycle plumbing.

## 5. Domain model (design for many, default to one)

```
Host
  └─ ServerInstance { id, model_ref, port, flags[], status, started_by, started_at, pid }
Model (discovered)
  └─ { path, family, size, quant, arch, gguf_meta, hf_mapping, shards[], sidecars[] }
LaunchProfile  (per model; auto-recommended + user overrides kept SEPARATE)
RunRecord      (telemetry: (model,config) → loaded?, peak_ram, tok/s, ttft, timestamp)
```

- "**One model at a time**" is a **policy/mode** (auto-stop others on start), not a hardcoded limit. v1 ships a single-model UX; the data model is already plural so multi-model is later additive.
- A **RAM fit-check** sums projected memory across running instances before starting another → self-governs. On a 64 GB box this naturally prevents two 27B-Q8 models without any special-case rule.
- **Ports auto-assigned** (8080, 8081, …); users never pick.

## 6. Model discovery (metadata-first, format-agnostic)

- **Configurable list of root dirs.** Default **`~/models`** (visible). NOT `~/.models` — dotfolders are for config/state, not tens-of-GB blobs. App config/state lives in the platform config dir (`~/Library/Application Support/local-model-manager` on macOS; XDG on Linux). Honor an env/XDG override.
- Recursively scan roots for `*.gguf`; **classify from the GGUF header**, never from folder names/depth. If a file's location disagrees with its metadata, metadata wins (and the UI can flag the mismatch).
- Collapse **multi-shard** models (`*-NNNNN-of-NNNNN.gguf`) into one logical entry; pick up `mmproj`/`.jinja` sidecars; dedupe; degrade gracefully on unreadable files; decide symlink-following policy.
- **First-run auto-detect** existing libraries (Ollama `~/.ollama/models`, LM Studio, HF cache `~/.cache/huggingface/hub`) and offer to add as roots.
- **Live updates** via a filesystem watcher + a manual "Rescan".
- **Folder layout is a write-side human convention only** (e.g. a future Model Downloader writes `<root>/<family>/<size>/<gguf>` with a `_profile.yaml` at the size level). Discovery never depends on it.

### GGUF introspection (proven feasible)

The daemon reads the GGUF header directly (no extra deps needed) to extract architecture, block count, attention/KV dims, context length, quant, embedded sampler defaults, and **homepage breadcrumbs** (`general.license.link` → base repo; `general.quantized_by` / `general.repo_url` → quant repo) to **auto-suggest the HF page**, with an optional manual override.

## 7. Security

> **v1 update (2026-06-15) — daemon runs as the owning user, not `_lmm`.** §14's "dedicated service account by default" decision was reversed in v1: the daemon runs as the installing user (`$SUDO_USER` / `--user`). Rationale: it lets the daemon **bind the operator's `~/.hermes` in one click** and read the user's models without a service account or ACLs (which also retired a class of installer bugs). The security cost is contained because the daemon now binds **loopback by default** — it's only network-exposed if you opt into `--host 0.0.0.0`, and that mode is where the run-as-user tradeoff matters (a compromise would carry the user's privileges). One-click bind is **loopback-only** (the daemon can't write a remote machine's config); remote clients use the `lmm bind` command. A dedicated service account may return as an opt-in for hardened LAN deployments.

- **Control plane** (`:8770`) is LAN-bound and **gated by a shared token**. Rationale: it spawns processes with user-supplied flags — the threat isn't the trusted users, it's everything *else* on the WiFi (guests/IoT/compromised devices). "My-LAN, not the-whole-network."
- **Inference plane** (`:8080+`) gated by `llama-server --api-key`. **Use a secret distinct from the control token** so inference access can be shared without granting control.
- Secrets stored in the host's app config dir, not world-readable; **never committed**. Clients store their copy locally.
- **macOS app firewall** must allow the daemon's incoming connections; first-run setup detects and guides.
- Default bind posture is a setup choice (loopback-only vs LAN); sharing requires LAN bind + token.

## 8. Lifecycle: the "seamless switch" state machine

```
select model+profile → fit-check (warn, never block) →
  stop current (SIGTERM; wait for port free AND RAM released) →
  start (spawn with profile flags; --alias <id>) →
  poll /health until ready →
  smoke-test (one tiny completion — confirms it actually answers) →
  [optional] rebind operator's agent →
  record RunRecord (loaded?, peak RAM, tok/s, ttft)
```

- **Crash supervision:** detect `llama-server` exit, surface the error, optional auto-restart.
- **Adopt externally-running servers:** detect a server already on a port (e.g. a hand-launched `tmux` session) and show/adopt it rather than colliding.
- **Coordination:** *free-for-all + read-only presence*. Any token-holding client can switch (last-action-wins, no locks); the UI shows what's loaded, which clients are connected, and whether it's busy — so switches are informed, not blind. (Full locking deferred.) Given `llama-server -np 1`, a switch interrupts any in-flight request — presence makes that visible.

## 9. Recommendation engine + fit-check

- **Deterministic core (offline, exact):** RAM / KV-cache / max-context math straight from GGUF metadata + hardware. Auto-enable MTP (`--spec-type draft-mtp`) when the model has a NextN head. **Must be architecture-aware** — e.g. hybrid attention/SSM models (Qwen3.6 `qwen35`, `full_attention_interval`) have far smaller KV than a naive all-layers estimate (~4× overestimate otherwise).
- **Validate every recommended flag against the installed `llama-server --help`** — flag names drift between builds (e.g. `--draft-max/--draft-min` removed in favor of `--spec-draft-n-max`/`--spec-draft-p-min`).
- **Soft LLM layer (optional):** read the HF model card for chat-template / launch quirks / recommended settings. **Use a remote model** (e.g. OpenRouter) — the local model isn't running yet (chicken/egg).
- **Fit-check: warn, never hard-stop.** Graduated: fits comfortably → silent; tight (may swap/slow) → gentle warning; weights alone exceed RAM → stronger warning + "start anyway?" confirm. **Actionable** — name the culprit (often context/KV, not weights) and propose the fix ("reduce context to ~64K to fit").
- **Self-correcting:** `RunRecord` peak-RAM observations refine future estimates; configs with a known-good history on this host suppress/soften warnings.
- **Auto-recommendation and user overrides are stored separately** so re-analysis never clobbers a tweak.

## 10. Agent binding (Hermes first)

- Register a **custom provider** in Hermes' `providers:` map with `base_url: http://<host>:8080/v1` (the `/v1` matters) + the inference `api_key`; set `model.provider custom:<name>` and `model.default <id>`. Launch `llama-server --alias <id>` so `/v1/models` matches `model.default`.
  - ⚠️ Do **not** use `hermes config set model.provider custom:http://…` — `custom:` is named-provider syntax, not a URL; it sets no `base_url`.
- Set Hermes **`fallback_providers`** (e.g. OpenRouter/Nous) so a client still works when the host is idle/down.
- **Gateway hot-reload (verified, v0.16.0):** the gateway does an **mtime-keyed fresh read of `config.yaml` each turn** to resolve model/provider (and reloads `.env` per-turn), so a config rewrite is applied on the **next turn with no restart**. The app should rewrite config atomically (so mtime bumps) and need not bounce the gateway. In-flight turns complete on the old model.
- The binding layer is provider-agnostic in design so non-Hermes clients can be added later.

## 11. Config & state layout (proposed)

- **Models:** `<root>/…` (host-side; default `~/models`; reference host uses `/Users/Shared/models`).
- **App config/state:** platform config dir — `daemon.yaml` (bind addrs, ports, **separate control token + inference api-key** [redacted in any export], roots, run-as=`_lmm`), `hosts.json` (client-side known hosts), `runs.db` (telemetry), and the **live launch profiles** keyed by GGUF-header hash + size + name.
- **Launch profiles live in app state** (sole writer = the daemon; models dir stays read-only to `_lmm`). A `_profile.yaml` **beside the model is an optional import/export** format (portable seed; future community sharing) — read if present, written only on explicit export.
- **Launch profile (`_profile.yaml`) — DRAFT schema (open):**
  ```yaml
  model: Qwen3.6-27B-Q8_0.gguf
  hf_homepage: https://huggingface.co/Qwen/Qwen3.6-27B   # auto + optional override
  recommended:        # engine output — regenerated on re-analysis
    context: 131072
    flags: [-ngl, 999, -fa, on, --cache-type-k, q8_0, --cache-type-v, q8_0,
            --spec-type, draft-mtp, --spec-draft-n-max, 2, -t, 10]
  overrides: {}       # user tweaks — NEVER clobbered by re-analysis
  ```

## 12. Daemon API surface — DRAFT (open for design)

REST + a WebSocket for streaming logs/metrics/presence. All control endpoints require the shared token.

```
GET    /api/hosts                      # client-side: discovered + known hosts
GET    /api/models                     # discovered models (+ gguf meta, hf mapping, fit estimate)
POST   /api/models/rescan
GET    /api/models/{id}/profile        # recommended + overrides
PUT    /api/models/{id}/profile        # set overrides
GET    /api/models/{id}/recommendation # (re)run engine
GET    /api/servers                    # running instances + status
POST   /api/servers                    # start { model_id, profile|flags, port? } → fit-check gate
DELETE /api/servers/{id}               # stop
POST   /api/servers/switch             # atomic stop+start (the seamless path)
GET    /api/servers/{id}/health        # /health + smoke-test result
POST   /api/bind                       # bind an agent (Hermes) to a server
GET    /api/presence                   # what's loaded / clients / busy
WS     /api/stream                     # logs, metrics (/props,/metrics), presence, status
```

## 13. Roadmap

Canonical, fully-detailed release plan lives in **[ROADMAP.md](ROADMAP.md)** (every feature/idea mapped to a release, with anti-goals and deferred decisions). Summary: **v1** single-host/trusted-multi-client/single-model UX · **v1.1** richer multi-user · **v1.2** multi-model exposed · **v2** Model Downloader + deeper intelligence · **v2+** multi-host/multi-backend/benchmarking/community profiles. Keep ROADMAP.md as the source of truth; update it (not this section) when plans change.

## 14. Open questions — RESOLVED

- **Secrets:** ✅ **separate** control-plane token and inference `--api-key`.
- **mDNS:** ✅ **`_lmm._tcp`** in `local.`, instance `"LMM @ <host>"`, TXT: `version`, `control_port=8770`, `api=v1`, `auth=token`.
- **`_profile.yaml` location:** ✅ **app state is the live store** (keyed by GGUF-header hash + size + name), with a **`_profile.yaml` beside the model as optional import/export**. Daemon writes only its own state → models dir stays read-only to the service account; clients never write files (sole writer = the daemon).
- **Hermes gateway hot-reload:** ✅ **YES, no restart.** Gateway does an **mtime-keyed fresh read of `config.yaml` each turn** to resolve model/provider (and reloads `.env` per-turn). A rewrite is picked up on the next turn. Caveats: keys on mtime (write must bump mtime); in-flight turn finishes on the old model.
- **Daemon run-as:** ⚠️ *Superseded in v1 — see the §7 v1 update: the daemon runs as the owning user, loopback-by-default. The dedicated-service-account design below is retained as a future hardened-LAN opt-in.* ~~**dedicated service account by default, created automatically by the guided installer**~~ (macOS: hidden `_lmm` role account, UID <500, no shell + models-dir read **ACL**, ownership stays curated; Linux: system user + systemd unit + group/ACL). Least-privilege-by-default is the right posture for a network-exposed, process-spawning daemon that *anyone* installs — and the installer absorbs all the `dscl`/ACL/plist work under the **single admin prompt** the service install needs anyway (no manual commands). **Run-as is configurable** (escape hatch for packagers / existing accounts), but the secure account is the default. Binds >1024 → **no root at runtime**.
