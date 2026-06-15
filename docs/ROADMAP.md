# local-model-manager — Roadmap

> Canonical release-planning doc. Captures **every** feature/idea raised in design, mapped to a release.
> Companion to [ARCHITECTURE.md](ARCHITECTURE.md) (design) and [V1_CHECKLIST.md](V1_CHECKLIST.md) (v1 tasks).
> Tags: ⭐ = differentiator/novel · 🔒 = security · 🧠 = intelligence/recommendation · 🤝 = multi-user/host.

## Guiding principles (apply to every release)

- **Open-source for anyone.** User-friendly means *the installer/app does the hard part* — never drop a sound feature for simplicity. Strike the balance with sound architecture.
- **Design for many, default to one.** Plural data model (servers, models, hosts); simple default UX. Applies to multi-model, multi-host, multi-client.
- **Metadata is ground truth.** Classify models from the GGUF header, never from folder names/paths. Validate flags against the *installed* binary.
- **Secure by default.** Least-privilege service account, token-gated control plane, separate inference key — all made painless by the installer.
- **UI talks only to the daemon API.** Never reach around to the FS/processes — this keeps the future native shell additive.

---

## v1 — "confidently switch one model and point Hermes at it"

Scope: single host, trusted multi-client over LAN, single-model UX. Full task list in [V1_CHECKLIST.md](V1_CHECKLIST.md).

**Core**
- Metadata-first **model discovery** (recursive scan, GGUF introspection, shard collapse, `mmproj`/`.jinja` sidecars, fs-watcher + rescan).
- Configurable **model roots** (default `~/models`; auto-detect existing Ollama/LM Studio/HF libraries on first run). ⭐
- 🧠 **Recommendation engine** — deterministic core (architecture-aware RAM/KV/context math, MTP auto-enable, flag-validated against the installed `llama-server`); optional soft LLM card-reader via a *remote* model.
- 🧠 **Fit-check** — graduated, **warn-never-block**, **actionable** (name the culprit, propose the fix), self-correcting from run-history. ⭐
- **Seamless switch** state machine — stop → wait for port/RAM free → start → `/health` poll → **post-load smoke test** → optional agent rebind.
- **Crash detection + optional auto-restart.**
- **Adopt an already-running server** (don't fight a hand-launched one). ⭐
- **Live log pane** + real **status/metrics** (`/props`, `/metrics`) — incl. light tok/s readout.
- **Hermes binding** (`providers:` custom entry + `/v1` + inference key + `--alias`; set `fallback_providers`). Gateway hot-reloads config — no restart.
- 🤝 **Read-only presence** (what's loaded, clients connected, busy) — informed switches, no locks.
- **HF homepage** auto-suggest from GGUF metadata + optional manual override.
- **Multi-model-ready data model**, single-model UX (one-at-a-time as a *policy*, enforced by the fit-check).

**Platform / security / install**
- 🔒 Control daemon (FastAPI) — **LAN bind + shared token**; inference via **separate `--api-key`**.
- 🔒 **Dedicated service account by default**, auto-created by a **guided installer** (one admin prompt, no manual commands); run-as configurable.
- **Two deployment modes:** `lmm serve` (try-it/dev, current user, no privileges) + guided installer (always-on service). ⭐
- **mDNS/Bonjour** host advertisement (`_lmm._tcp`) for auto-discovery.
- **First-run setup wizard** (detect llama.cpp / offer `brew install`; set roots; detect model libs; detect Hermes installs).

**UX**
- One/two-click switch; "Advanced" disclosure for raw flags.
- ⭐ **"Why this config" explainer** — plain-language rationale per recommended flag (doubles as onboarding/teaching aid).

---

## v1.1 — richer multi-user 🤝

- **Notifications** (model ready / crashed / switched-by-another-client).
- **Per-user model roots** (shared host root + optional private roots).
- **Telemetry maturity** — run-history-driven estimate refinement, history views.
- **Deeper presence** — who's connected, active sessions, last-active.
- Onboarding flow for a second user (e.g. configuring their Hermes client + fallback). ⭐

---

## v1.2 — multi-model exposed

- **Concurrent multi-model UI** — run N servers on auto-assigned ports, **gated by the fit-check** (safe on big machines; self-limits on small ones).
- **Favorites / recently-used** models.
- **Launch-profile export/import** (`_profile.yaml`) — groundwork for community sharing.
- Coordination escalation option: **soft guard / opt-in locks** beyond v1's free-for-all + presence.

---

## v2 — Model Downloader & deeper intelligence

- ⭐ **Model Downloader** — fetch from HF into the `<root>/<family>/<size>/` layout (the only writer of that convention).
- 🧠 **Deeper LLM card analysis** — chat templates, launch quirks, recommended sampler/rope from the model card.
- **Chat-template (`.jinja`) management** (incl. `--jinja` wiring).

---

## v2+ — scale & ecosystem

- 🤝 **Multi-host management** — manage servers across several machines (domain already `hosts → servers → models`).
- **Multi-backend** — MLX, Ollama, vLLM behind the same control plane.
- **Remote/SSH model hosts** (aligns with Hermes' own SSH/Modal sandbox model).
- ⭐ **Benchmarking suite** — tok/s, time-to-first-token per (model, config); compare variants empirically.
- **Optional `llama-swap` integration** — on-demand swap behind one port (integrate, don't reinvent).
- ⭐ **Community launch-profile sharing** — "optimal config for *this model* on *this class of hardware*," contributed by users. Open-source flywheel; nobody's nailed this for llama.cpp.

---

## Cross-cutting / smaller niceties (slot opportunistically)

- **Folder/metadata mismatch flag** — when a `.gguf`'s location disagrees with its metadata, surface it gently (most useful once the Downloader exists).
- **RAM-fit guardrail visualization** (weights vs KV-cache breakdown, headroom).
- **Provider-agnostic binding** — design the binding layer so non-Hermes agents can be added later.

---

## Explicitly OUT of scope (anti-goals)

- Request-based **auto-swap / routing / autoscaling / load-balancing** (note `llama-swap` for on-demand swap).
- **Per-user roles / RBAC** (trust model: all clients trusted, token-gated).
- **Training / fine-tuning / quantization / model conversion.**

---

## Deferred decisions (revisit at the relevant release)

- v1.1+: promote run-as to a system service account on shared hosts (already the installer default; revisit for non-macOS specifics).
- v1.2: coordination model beyond free-for-all + presence (soft guard vs locks).
- v2: where shared `_profile.yaml` lives when multiple *hosts* share a models mount (per-host state vs shared file + locking).
