"""Generate the launchd plist and the privileged install/uninstall steps.

Pure generators. Nothing here runs privileged commands; `lmm install` executes
the returned steps only when run as root. The daemon runs as the **owning user**
(not a dedicated service account), so no dscl account or models-dir ACL is
needed — the user already owns/reads their own models, and the daemon can write
the user's own `~/.hermes` for one-click binding.
"""

from __future__ import annotations

import os
import plistlib
import shlex

LABEL = "com.local-model-manager.daemon"
_PLIST_PATH = f"/Library/LaunchDaemons/{LABEL}.plist"
_LOG_DIR = "/Library/Logs/local-model-manager"


def plist_install_path() -> str:
    return _PLIST_PATH


def launchd_plist(*, exec_path: str, host: str, port: int, user: str,
                  env: dict | None = None) -> str:
    data = {
        "Label": LABEL,
        "ProgramArguments": [exec_path, "daemon", "--host", host, "--port", str(port)],
        "UserName": user,
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": f"{_LOG_DIR}/daemon.out.log",
        "StandardErrorPath": f"{_LOG_DIR}/daemon.err.log",
        "ProcessType": "Background",
    }
    if env:
        data["EnvironmentVariables"] = dict(env)
    return plistlib.dumps(data).decode()


def shared_setup_steps(*, user: str, shared_dir: str) -> list[str]:
    d = shlex.quote(shared_dir)
    return [f"mkdir -p {d}", f"chown {shlex.quote(user)}:staff {d}", f"chmod 2770 {d}"]


def shared_venv_exec(shared_dir: str) -> str:
    return f"{shared_dir}/venv/bin/lmm"


def shared_venv_steps(*, shared_dir: str, project_dir: str, user: str,
                      clear: bool = False) -> list[str]:
    sd = shlex.quote(shared_dir)
    py_dir = f"{shared_dir}/python"
    py_dir_q = shlex.quote(py_dir)
    venv = shlex.quote(f"{shared_dir}/venv")
    venv_py = shlex.quote(f"{shared_dir}/venv/bin/python")
    # These uv commands run as root (install is sudo-gated). Pin the cache INTO
    # the shared tree so root never writes to the invoking user's ~/.cache/uv —
    # otherwise it leaves root-owned files there that break the user's later
    # unprivileged `uv` runs (e.g. `uv tool install .`) with EACCES. The final
    # chown -R hands the cache to the owning user along with the rest of the tree.
    cache = shlex.quote(f"{shared_dir}/uv-cache")
    uv_env = f"UV_CACHE_DIR={cache}"
    # --clear lets a reinstall replace an existing venv (uv venv errors otherwise).
    clear_flag = " --clear" if clear else ""
    venv_cmd = (f"{uv_env} UV_PYTHON_INSTALL_DIR={py_dir_q} uv venv --managed-python "
                f"--python 3.11{clear_flag} {venv}")
    # Install a uv-managed Python INTO the shared tree (built as root) so the
    # interpreter the venv links to is readable after we chown the tree to the
    # owning user — avoids depending on root's/anyone-else's Python location.
    return [
        # --no-bin: don't drop a python3.11 shim into a bin dir under sudo.
        f"{uv_env} UV_PYTHON_INSTALL_DIR={py_dir_q} uv python install --no-bin 3.11",
        venv_cmd,
        f"{uv_env} uv pip install --python {venv_py} {shlex.quote(project_dir)}",
        f"chown -R {shlex.quote(user)}:staff {sd}",
    ]


def plist_steps(*, user: str) -> list[str]:
    return [
        f"mkdir -p {_LOG_DIR}",
        # -R so pre-existing log files (e.g. from a prior install as a different
        # user) are owned by the run-as user — else launchd can't open the
        # StandardError/Out paths and the job dies with EX_CONFIG (78).
        f"chown -R {shlex.quote(user)} {_LOG_DIR}",
        f"chown root:wheel {_PLIST_PATH}",
        f"chmod 644 {_PLIST_PATH}",
        f"launchctl bootstrap system {_PLIST_PATH}",
    ]


def firewall_steps(*, exec_path: str) -> list[str]:
    fw = "/usr/libexec/ApplicationFirewall/socketfilterfw"
    return [f"{fw} --add {shlex.quote(exec_path)}", f"{fw} --unblockapp {shlex.quote(exec_path)}"]


def install_steps(*, user: str, host: str, port: int, shared_dir: str,
                  project_dir: str, reinstall: bool = False) -> list[str]:
    exec_path = shared_venv_exec(shared_dir)
    steps: list[str] = []
    if reinstall:
        # stop the running job first so the plist bootstrap can re-load it
        steps.append(f"launchctl bootout system {_PLIST_PATH}")
    steps += [
        *shared_setup_steps(user=user, shared_dir=shared_dir),
        *shared_venv_steps(shared_dir=shared_dir, project_dir=project_dir,
                           user=user, clear=reinstall),
        *plist_steps(user=user),
        *firewall_steps(exec_path=exec_path),
    ]
    return steps


def uninstall_steps(*, shared_dir: str | None = None) -> list[str]:
    steps = [
        f"launchctl bootout system {_PLIST_PATH}",
        f"rm -f {_PLIST_PATH}",
        # plist_steps mkdir's _LOG_DIR at install — remove it too so uninstall
        # truly leaves nothing behind (the README promises a complete removal).
        f"rm -rf {_LOG_DIR}",
    ]
    if shared_dir:
        # drop the firewall rule the installer added for the shared-venv binary
        # (harmless if it isn't present — run() uses check=False).
        fw = "/usr/libexec/ApplicationFirewall/socketfilterfw"
        steps.append(f"{fw} --remove {shlex.quote(shared_venv_exec(shared_dir))}")
        steps.append(f"rm -rf {shlex.quote(shared_dir)}")
    return steps


def service_stop_steps() -> list[str]:
    """Stop the running daemon without uninstalling (plist stays → reloads at boot)."""
    return [f"launchctl bootout system {_PLIST_PATH}"]


def service_start_steps() -> list[str]:
    """(Re)load the installed daemon from its plist."""
    return [f"launchctl bootstrap system {_PLIST_PATH}"]


def service_restart_steps() -> list[str]:
    return [*service_stop_steps(), *service_start_steps()]


def existing_install_artifacts(*, shared_dir: str) -> list[str]:
    """Read-only: which install artifacts already exist (for the re-run guard)."""
    found: list[str] = []
    if os.path.exists(_PLIST_PATH):
        found.append("LaunchDaemon plist")
    if os.path.exists(f"{shared_dir}/venv"):
        found.append("shared venv")
    return found
