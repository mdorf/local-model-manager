"""Generate the launchd plist and the privileged install/uninstall steps.

Pure generators + a read-only free-UID finder. Nothing here runs privileged
commands; `lmm install` executes the returned steps only when run as root.
"""

from __future__ import annotations

import plistlib
import shlex
import subprocess

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


def account_steps(*, user: str, uid: int) -> list[str]:
    base = f"dscl . -create /Users/{user}"
    return [
        base,
        f"{base} UserShell /usr/bin/false",
        f'{base} RealName "Local Model Manager service"',
        f"{base} UniqueID {uid}",
        f"{base} PrimaryGroupID 1",
        f"{base} NFSHomeDirectory /var/empty",
        f"dscl . -create /Users/{user} IsHidden 1",
        f"{base} Password '*'",
    ]


def acl_steps(*, user: str, models_dir: str) -> list[str]:
    perms = ("read,execute,readattr,readextattr,readsecurity,list,search,"
             "file_inherit,directory_inherit")
    return [f'chmod -R +a "{user} allow {perms}" {shlex.quote(models_dir)}']


def acl_remove_steps(*, user: str, models_dir: str) -> list[str]:
    perms = ("read,execute,readattr,readextattr,readsecurity,list,search,"
             "file_inherit,directory_inherit")
    return [f'chmod -R -a "{user} allow {perms}" {shlex.quote(models_dir)}']


def shared_setup_steps(*, user: str, shared_dir: str) -> list[str]:
    d = shlex.quote(shared_dir)
    return [f"mkdir -p {d}", f"chown {shlex.quote(user)}:staff {d}", f"chmod 2770 {d}"]


def shared_venv_exec(shared_dir: str) -> str:
    return f"{shared_dir}/venv/bin/lmm"


def shared_venv_steps(*, shared_dir: str, project_dir: str, user: str) -> list[str]:
    venv = shlex.quote(f"{shared_dir}/venv")
    return [
        f"uv venv {venv}",
        f"uv pip install --python {shlex.quote(shared_dir + '/venv/bin/python')} {shlex.quote(project_dir)}",
        f"chown -R {shlex.quote(user)}:staff {venv}",
    ]


def plist_steps(*, user: str) -> list[str]:
    return [
        f"mkdir -p {_LOG_DIR}",
        f"chown {shlex.quote(user)} {_LOG_DIR}",
        f"chown root:wheel {_PLIST_PATH}",
        f"chmod 644 {_PLIST_PATH}",
        f"launchctl bootstrap system {_PLIST_PATH}",
    ]


def firewall_steps(*, exec_path: str) -> list[str]:
    fw = "/usr/libexec/ApplicationFirewall/socketfilterfw"
    return [f"{fw} --add {shlex.quote(exec_path)}", f"{fw} --unblockapp {shlex.quote(exec_path)}"]


def install_steps(*, user: str, uid: int, host: str, port: int,
                  models_dir: str, shared_dir: str, project_dir: str) -> list[str]:
    exec_path = shared_venv_exec(shared_dir)
    return [
        # account must exist before anything chowns to it
        *account_steps(user=user, uid=uid),
        *shared_setup_steps(user=user, shared_dir=shared_dir),
        *acl_steps(user=user, models_dir=models_dir),
        *shared_venv_steps(shared_dir=shared_dir, project_dir=project_dir, user=user),
        *plist_steps(user=user),
        *firewall_steps(exec_path=exec_path),
    ]


def uninstall_steps(*, user: str, models_dir: str | None = None,
                    shared_dir: str | None = None) -> list[str]:
    steps = [
        f"launchctl bootout system {_PLIST_PATH}",
        f"rm -f {_PLIST_PATH}",
    ]
    # Remove the models-dir ACL while the account still resolves; deleting the
    # account first would leave an orphaned-UUID ACL on the dir + files.
    if models_dir:
        steps.extend(acl_remove_steps(user=user, models_dir=models_dir))
    steps.append(f"dscl . -delete /Users/{user}")
    if shared_dir:
        steps.append(f"rm -rf {shlex.quote(shared_dir)}")
    return steps


def find_free_service_uid(low: int = 250, high: int = 499) -> int:
    """Return an unused UID in [low, high], scanning dscl read-only."""
    used: set[int] = set()
    try:
        out = subprocess.run(["dscl", ".", "-list", "/Users", "UniqueID"],
                             capture_output=True, text=True, timeout=10)
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[-1].lstrip("-").isdigit():
                used.add(int(parts[-1]))
    except (OSError, subprocess.SubprocessError):
        pass
    for uid in range(low, high + 1):
        if uid not in used:
            return uid
    raise RuntimeError(f"no free service UID in [{low}, {high}]")
