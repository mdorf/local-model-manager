# local-model-manager (`lmm`)

Manage local [llama.cpp](https://github.com/ggml-org/llama.cpp) model servers on a shared host and bind agent clients to them — with hardware-aware, model-aware launch configs and a web UI.

`lmm` lets you:

- **Discover** every local model on disk, classified from its GGUF header (not from file names).
- Get a **recommended `llama-server` configuration** for any model. `lmm` reads the model's GGUF metadata *and* profiles the machine it runs on — CPU/core layout, total and usable RAM, Metal/GPU availability — then computes a launch config (context length, KV-cache quantization, GPU layers, threads, speculative decoding) tuned to *that* hardware, with a **fit-check** that warns when a model won't comfortably fit in RAM and names the culprit.
- **Start / stop / switch** the running model from a **web UI** or the CLI (spawn → readiness-gate → smoke-test → supervise).
- **Connect an agent** to the running model — one click on the host.
- Run as an always-on **system daemon** with a token-gated HTTP control plane, drivable from any machine on your LAN.

**Binding [Hermes](https://hermes-agent.nousresearch.com/).** `lmm`'s first-class agent integration is **Hermes Agent** (Nous Research). "Connect an agent" repoints Hermes at the running model — it registers the local server as a custom OpenAI-compatible provider (`base_url` + model id) and sets it as Hermes's default — so your agent talks to the local model instead of a cloud API, and follows along when you switch models. On the host that's one click in the UI; from another machine it's a single `lmm bind` command. (Any OpenAI-compatible app works too — the UI shows the base URL and model id to paste.)

> **⚠️ `lmm` must run on the same machine as llama.cpp and your models.** The control daemon spawns and supervises `llama-server` and reads the GGUF files off local disk, so it has to live on the model host — it cannot be pointed at a remote llama.cpp. Browsers, the `lmm` CLI in client mode, and agents like Hermes can run on *any* machine on the LAN; only the daemon is pinned to the host. See [How it runs](#how-it-runs-topology).

> **Status:** backend, CLI, control daemon, and web UI are implemented and tested. Interfaces are pre-1.0 and may change. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and [ROADMAP.md](ROADMAP.md) for the plan.

## Requirements

- **Python ≥ 3.11** and [`uv`](https://github.com/astral-sh/uv)
- **llama.cpp** with `llama-server` on your `PATH` (e.g. `brew install llama.cpp`)
- One or more directories of `.gguf` models
- macOS for the always-on system installer (`lmm install` uses launchd). The CLI and daemon themselves are cross-platform.

## Getting started

> Run these on the machine that has your models and llama.cpp (see the co-location note above).

### 1. Install the dependencies

**[`uv`](https://github.com/astral-sh/uv)** — manages Python ≥ 3.11 for you:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**llama.cpp** — provides `llama-server`, which must be on your `PATH`. On macOS the simplest route is via [Homebrew](https://brew.sh) (install Homebrew first if you don't have it):

```bash
brew install llama.cpp
```

On other platforms, install llama.cpp any way you like — a package manager, or [build from source](https://github.com/ggml-org/llama.cpp) — `lmm` only needs the resulting `llama-server` binary on your `PATH`.

### 2. Get `lmm` and start the daemon

```bash
git clone https://github.com/mdorf/local-model-manager.git
cd local-model-manager
```

If your models live in `~/models`, just start the daemon:

```bash
uv run lmm daemon
```

Otherwise point it at your models directory (persisted after the first run):

```bash
LMM_MODELS_DIR=/path/to/models uv run lmm daemon
```

`uv` builds the virtualenv from `uv.lock` on first run.

### 3. Open the web UI

**→ http://127.0.0.1:8770**

On the local host the daemon injects its auth token automatically, so it just loads — pick a model, see its recommended config and RAM fit, and **Start** it. Use **Switch** to change models, **Stop** to unload, and **Connect an agent** to point Hermes at the running model in one click.

The daemon runs in the foreground here (Ctrl-C to stop it) — ideal for trying it out. To run it always-on without a terminal, see [Run as an always-on service](#run-as-an-always-on-service-macos).

## Run as an always-on service (macOS)

Getting started runs the daemon in the **foreground** (Ctrl-C to stop — works on any platform). To keep it running across logout and reboot without a terminal, install it as a launchd `LaunchDaemon`: it starts at boot, restarts on crash, and **runs as you** (the user who runs `sudo lmm install`, or `--user <name>`). `sudo` is needed for the privileged steps — preview them with `--dry-run` first. (macOS only for now — a Linux/systemd installer is on the roadmap; the daemon itself is cross-platform.)

```bash
# root must be able to find uv + llama-server, hence the explicit PATH:
sudo env "PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH" \
  .venv/bin/lmm install --project-dir "$(pwd)" --models-dir /path/to/models

sudo .venv/bin/lmm install --dry-run        # preview the exact privileged steps
sudo .venv/bin/lmm install --reinstall      # rebuild in place (keeps token + state)
```

### Stop / start / restart the installed daemon

```bash
lmm service status        # installed? responding? (read-only, no sudo)
sudo lmm service stop     # stop it (the plist stays, so it reloads on next boot)
sudo lmm service start    # (re)load it
sudo lmm service restart
```

Stopping or restarting the service leaves any running model **up** — only the control plane bounces, and it re-adopts the model on restart. (Foreground daemon? Just Ctrl-C. Note `lmm stop`, by contrast, stops the *model server*, not the daemon.)

### Uninstall (remove completely)

```bash
lmm unbind                     # (optional) revert your Hermes config to its pre-bind state
sudo .venv/bin/lmm uninstall   # remove the LaunchDaemon + shared state dir (token, venv, logs)
rm -rf <path-to-your-clone>    # remove the cloned source + its .venv
```

`uninstall` never touches your model files — `lmm` only ever reads them — so after these steps nothing of `lmm` remains on the machine.

> **Upgrading from an early `_lmm`-based build?** Run that build's `sudo lmm uninstall` (or manually `sudo dscl . -delete /Users/_lmm` and strip the models-dir ACL) *before* installing this run-as-user version — the current uninstaller no longer manages the old service account.

## How it runs (topology)

`lmm` is one distributable with two roles: a **host** that manages models, and **clients** that drive it.

| Component | Where it runs |
|---|---|
| **`lmm` control daemon** (`:8770`) | **On the model host** — it spawns/supervises `llama-server`, so it must live where the models and llama.cpp are. |
| **`llama-server`** (`:8080`) | On the host, spawned by the daemon. |
| **Web UI / `lmm` client** | Any machine on the LAN — a browser pointed at the daemon, or `lmm` in client mode. |
| **Hermes (or any agent)** | Anywhere — it just needs HTTP access to the host's `:8080/v1`. |

```
                                        LAN
  ┌─────────────────┐           ┌─ Host ───────────────────────────────────┐
  │  Browser / CLI  │◀── HTTP ─▶│ lmm control daemon   :8770               │
  │     (client)    │           │      │  spawns / supervises              │
  └─────────────────┘           │      ▼                                   │
  ┌─────────────────┐           │ llama-server   :8080   (/v1)             │
  │      Hermes     │─── /v1 ──▶│      (inference)                         │
  │  (any LAN box)  │           │ models on disk — host-side only          │
  └─────────────────┘           └──────────────────────────────────────────┘
```

By default the daemon binds **loopback** (`127.0.0.1`); pass `--host 0.0.0.0` to expose it on the LAN so other machines can connect (see [Security](#security)).

## CLI

Everything in the UI is also available on the CLI. With a daemon running, `serve` / `stop` / `status` / `switch` are routed through it; otherwise they act locally.

```bash
# Discover + inspect
uv run lmm models --root /path/to/models                 # list discovered models
uv run lmm recommend Qwen3.6-27B-Q8_0 --root /path/to/models   # tuned llama-server config + fit-check

# Run a model (accepts the model's name or its bare stem)
uv run lmm serve Qwen3.6-27B-Q8_0 --root /path/to/models # start (default :8080); waits for /health + smoke test
uv run lmm status                                        # show managed servers
uv run lmm switch Other-Model --root /path/to/models     # stop current, start another
uv run lmm stop --port 8080                              # stop the model server

# Connect Hermes to the running server
uv run lmm bind Qwen3.6-27B-Q8_0 --port 8080             # point ~/.hermes/config.yaml at it
uv run lmm bind --host other-host.local --port 8080      # bind to a model on another host (omit model to auto-detect)
uv run lmm unbind                                        # revert from the pre-bind backup
```

`recommend` does the RAM/KV-cache/context math from the GGUF header and your hardware, auto-enables speculative decoding when the model supports it, and validates flags against your installed `llama-server`. `bind` registers a custom provider and sets the default model, preserving your config's comments and other keys (reasoning models like Qwen3.6 want a generous `max_tokens` — `bind` prints a reminder).

> **Note on `stop`:** `lmm stop` stops the **model server**, not the `lmm` daemon. To control the daemon itself, see [`lmm service`](#run-as-an-always-on-service-macos).

## Control daemon (HTTP API)

The daemon owns server lifecycle and serves both the web UI and a token-gated API:

```bash
uv run lmm daemon --host 127.0.0.1 --port 8770   # --host 0.0.0.0 to expose on the LAN
uv run lmm token                                  # print the bearer token (for remote/API use)
```

All endpoints require `Authorization: Bearer <token>` except `/api/health`. On loopback the web UI gets the token injected automatically; remote clients paste it once.

| Method & path | Purpose |
|---|---|
| `GET /api/health` | liveness (open, no auth) |
| `GET /api/models` | list discovered models |
| `GET /api/models/{name}/recommend` | recommended config + fit for a model |
| `GET /api/servers` | list running/managed servers |
| `POST /api/servers` | start a server — body `{ "model": "...", "port": 8080 }` |
| `POST /api/servers/switch` | switch the running model |
| `DELETE /api/servers/{port}` | stop the server on a port |
| `GET /api/connection-info` | base URL / model / inference key for connecting an agent |
| `POST /api/bind` | bind the host's Hermes to the running model (loopback only) |
| `GET /api/bind-status` | whether the host's Hermes points at the running model |
| `WS /api/stream` | live server logs + status (subprotocol `lmm.bearer.<token>`) |

The daemon detects an **already-running** `llama-server` on startup (e.g. one launched manually, or that outlived a daemon restart) and reflects it in the UI. Stopping or restarting the daemon does **not** stop the model.

> The control API is **pre-1.0 and evolving** — treat it as unstable.

## Security

The daemon runs as **your user**, so it can read your models without extra setup and **bind Hermes for you in one click**. It binds **loopback by default** (not on the network), which keeps it low-risk for personal use.

If you expose it to the LAN (`--host 0.0.0.0`, to share with other machines):

- The **control plane** (`:8770`) is gated by a **shared bearer token** — it spawns processes, so the threat model is everything *else* on the network, not your trusted clients.
- The **inference plane** (`:8080`) is gated by `llama-server --api-key` (a secret distinct from the control token).
- A compromise of the network-facing daemon would carry your account's privileges — a deliberate tradeoff for one-click binding and zero-setup model access.

**One-click "Connect an agent" is loopback-only** — the daemon can only write the host's own `~/.hermes`. To connect an agent on a *different* machine, run `lmm bind --host <host> --port 8080 ...` on that machine (the UI shows you the exact command, pre-filled with the host you reached it through).

## Development

```bash
uv run pytest -q        # run the test suite
uv run ruff check .     # lint
```

## License

[MIT](LICENSE) © 2026 Misha Dorf
