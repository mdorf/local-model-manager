# local-model-manager (`lmm`)

Manage local [llama.cpp](https://github.com/ggml-org/llama.cpp) model servers on a shared host and bind agent clients to them — with hardware-aware, model-aware launch configs.

`lmm` lets you:

- **Discover** every local model on disk, classified from its GGUF header (not from file names).
- Get a **hardware-aware recommended `llama-server` configuration** for any model, with a fit-check.
- **Start / stop / switch** the running model (spawn → readiness-gate → smoke-test → supervise).
- **Bind an agent** (first target: [Hermes](https://hermes-agent.nousresearch.com/)) to the running server.
- Optionally run as an always-on **system daemon** with a token-gated HTTP control plane over the LAN.

> **Status:** the backend + CLI are implemented and tested. A browser UI is in progress (see [ROADMAP.md](ROADMAP.md)). Interfaces are pre-1.0 and may change.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and [ROADMAP.md](ROADMAP.md) for the release plan.

## Requirements

- **Python ≥ 3.11** and [`uv`](https://github.com/astral-sh/uv)
- **llama.cpp** with `llama-server` on your `PATH` (e.g. `brew install llama.cpp`)
- One or more directories of `.gguf` models (default model root: `~/models`)
- macOS for the system installer (`lmm install` uses launchd). The CLI itself is cross-platform.

## Quick start (from source)

```bash
git clone <your-repo-url> local-model-manager
cd local-model-manager
uv run lmm --help          # uv creates the venv from uv.lock on first run
```

All commands below are shown as `uv run lmm …`; after a system install the daemon runs the same `lmm`.

### Discover and inspect models

```bash
uv run lmm models --root /path/to/models     # list discovered models (repeat --root for several)
uv run lmm recommend Qwen3.6-27B-Q8_0 --root /path/to/models   # print a tuned llama-server command + fit-check
```

`recommend` does the RAM/KV-cache/context math from the GGUF header and your hardware, auto-enables speculative decoding when the model supports it, and validates flags against your installed `llama-server`.

### Run a model

```bash
uv run lmm serve Qwen3.6-27B-Q8_0 --root /path/to/models   # start (default port 8080); waits for /health + smoke test
uv run lmm status                                          # show managed servers
uv run lmm switch Qwen3.6-27B-NEO-CODE --root /path/to/models   # stop current, start another
uv run lmm stop --port 8080                                # stop a server
```

### Bind Hermes to the running server

```bash
uv run lmm bind Qwen3.6-27B-Q8_0 --port 8080               # point ~/.hermes/config.yaml at the local server
uv run lmm bind Qwen3.6-27B-Q8_0 --hermes-config /path/to/other/config.yaml   # bind a second user's Hermes
uv run lmm unbind                                          # revert from the pre-bind backup
```

`bind` registers a custom provider (`base_url: http://<host>:<port>/v1`) and sets the default model, preserving your config's comments and other keys. Reasoning models (e.g. Qwen3.6) need a generous `max_tokens` — `bind` prints a reminder.

## Control daemon (HTTP API)

Run a control plane that owns server lifecycle and can be driven by remote clients:

```bash
uv run lmm daemon --host 127.0.0.1 --port 8770   # use --host 0.0.0.0 to expose on the LAN
uv run lmm token                                  # print the bearer token for API calls
```

All endpoints require `Authorization: Bearer <token>` except `/api/health`.

| Method & path | Purpose |
|---|---|
| `GET /api/health` | liveness (open, no auth) |
| `GET /api/models` | list discovered models |
| `GET /api/models/{name}/recommend` | recommended config + fit for a model |
| `GET /api/servers` | list running/managed servers |
| `POST /api/servers` | start a server — body `{ "model": "...", "port": 8080 }` |
| `POST /api/servers/switch` | switch the running model — body `{ "model": "...", "port": 8080 }` |
| `DELETE /api/servers/{port}` | stop the server on a port |

> The control API is **pre-1.0 and evolving** — request bodies will be formalized (Pydantic) and log/metrics streaming (WebSocket) added alongside the web UI. Treat it as unstable for now.

## Install as an always-on system daemon (macOS)

The guided installer sets up a launchd `LaunchDaemon` that starts at boot and restarts on crash, **running as you** (the user who runs `sudo lmm install`, or `--user <name>`). It needs `sudo` for the privileged steps (preview them first with `--dry-run`).

```bash
# root must be able to find uv + llama-server, hence the explicit PATH:
sudo env "PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH" \
  .venv/bin/lmm install --project-dir "$(pwd)" --models-dir /path/to/models

sudo .venv/bin/lmm install --dry-run        # preview the exact privileged steps
sudo .venv/bin/lmm install --reinstall      # rebuild in place (keeps token + state)
sudo .venv/bin/lmm uninstall                # remove the daemon + its state dir
```

Verify: `sudo launchctl print system/com.local-model-manager.daemon` and `curl http://127.0.0.1:8770/api/health`.

**Security note:** the daemon runs as your user — so it can read your models without ACL setup and **bind Hermes for you in one click** from the web UI. It binds **loopback by default** (not exposed to the network), which keeps this low-risk for personal use. If you expose it to the LAN (`--host 0.0.0.0`, to share with other machines), note that a compromise of the network-facing daemon would carry your account's privileges; in that mode the inference port is API-key-gated. One-click bind is **loopback-only** (the daemon can only write the local host's `~/.hermes`); remote machines bind with the `lmm bind` command the UI shows.

> **Upgrading from an early `_lmm`-based build?** Run that build's `sudo lmm uninstall` (or manually `sudo dscl . -delete /Users/_lmm` and strip the models-dir ACL) *before* installing this run-as-user version — the current uninstaller no longer manages the old service account.

## Development

```bash
uv run pytest -q        # run the test suite
uv run ruff check .     # lint
```

## License

[MIT](LICENSE) © 2026 Misha Dorf
