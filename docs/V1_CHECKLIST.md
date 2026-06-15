# v1 Checklist — local-model-manager

> Scope: **single host, trusted multi-client over LAN, single-model UX.** See [ARCHITECTURE.md](ARCHITECTURE.md).
> Unchecked = not started (design phase). This tracks v1 only; later releases live in the roadmap.

## Foundations
- [ ] Repo scaffolding: FastAPI service, web UI shell, packaging, license (MIT?), README, CI.
- [ ] App config/state dir (platform-aware; XDG/env override). `daemon.yaml` schema.
- [ ] Domain model: `Host / ServerInstance / Model / LaunchProfile / RunRecord` (plural by design).

## Daemon (control plane)
- [ ] FastAPI service + REST surface (see ARCHITECTURE §12) + WebSocket stream.
- [ ] **Try-it mode:** `lmm serve` (foreground, current user, no privileges) for clone-and-run.
- [ ] **Guided installer (installed mode):** auto-create dedicated service account + models-dir read ACL + `LaunchDaemon`/systemd unit + boot-start + firewall allow, under one admin prompt (no manual commands). Run-as configurable; secure account is default. Uninstall path too.
- [ ] LAN bind + **shared control token** auth middleware.
- [ ] Bonjour/mDNS advertisement (`_lmm._tcp`); client-side discovery.

## Model discovery & introspection
- [ ] Configurable **list of roots**; default `~/models` (create if absent); env/XDG override.
- [ ] Recursive `.gguf` scan; **GGUF header parser** (arch, blocks, KV dims, ctx, quant, sampler, HF breadcrumbs, MTP/NextN detect).
- [ ] Shard collapsing; `mmproj`/`.jinja` sidecars; symlink policy; graceful errors.
- [ ] First-run auto-detect Ollama / LM Studio / HF caches → offer as roots.
- [ ] Filesystem watcher + manual rescan.
- [ ] HF homepage auto-suggest (+ optional manual override).

## Recommendation engine + fit-check
- [ ] Deterministic core: RAM / **architecture-aware** KV-cache / max-context math from GGUF + hardware.
- [ ] MTP auto-enable when NextN head present.
- [ ] **Validate flags against installed `llama-server --help`** (build-aware).
- [ ] Fit-check: graduated, **warn-never-block**, **actionable** (name culprit + propose fix).
- [ ] `_profile.yaml`: auto-recommendation + **separate** user overrides.
- [ ] (Optional) soft LLM card-reader via a **remote** model.

## Server lifecycle
- [ ] Start / stop / **switch** (stop→wait for port+RAM free→start→`/health` poll→**smoke test**).
- [ ] Auto-assign ports; RAM fit-check gate before start.
- [ ] Crash detection + optional auto-restart.
- [ ] **Adopt an already-running server** (e.g. hand-launched tmux on :8080).
- [ ] Live log pane; status/metrics via `/props` + `/metrics`.
- [ ] `RunRecord` capture (loaded?, peak RAM, tok/s, TTFT) → feeds fit-check self-correction.

## Coordination
- [ ] Read-only **presence** (what's loaded / clients connected / busy); last-action-wins (no locks).

## Agent binding (Hermes)
- [ ] Register `providers:` custom entry → `http://<host>:8080/v1` + inference `api_key`; set `model.provider`/`model.default`; launch `--alias <id>`.
- [ ] Set `fallback_providers`.
- [ ] **Verify** gateway hot-reload vs bounce after switch.

## UX / onboarding
- [ ] First-run setup wizard (detect llama.cpp / offer `brew install`; set roots; detect model libs; detect Hermes installs).
- [ ] One/two-click switch flow; "Advanced" disclosure for raw flags.
- [ ] "Why this config" explainer text.

## Out of scope for v1 (do NOT build)
- Multi-host management UI · multi-model concurrent UI · Model Downloader · auto-swap/routing/load-balancing · per-user roles · non-Hermes agents.

## Pre-existing maintainer action (reference deployment)
- [ ] (user, in Terminal) `sudo chown misha:staff /Users/Shared/models && sudo chmod 2750 /Users/Shared/models` — model *files* already `640 misha:staff`; directory still pending.
