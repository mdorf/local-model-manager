import plistlib

from lmm.deploy import (
    LABEL,
    existing_install_artifacts,
    install_steps,
    launchd_plist,
    plist_install_path,
    plist_steps,
    shared_setup_steps,
    shared_venv_exec,
    shared_venv_steps,
    uninstall_steps,
)


def test_plist_steps_chowns_log_dir_recursively():
    # -R so a prior install's log files become writable by the run-as user
    # (else launchd can't open StandardError/OutPath → EX_CONFIG).
    joined = "\n".join(plist_steps(user="misha"))
    assert "chown -R misha" in joined
    assert "launchctl bootstrap" in joined


def test_plist_has_required_keys():
    xml = launchd_plist(exec_path="/usr/local/bin/lmm", host="127.0.0.1", port=8770, user="misha")
    data = plistlib.loads(xml.encode())
    assert data["Label"] == LABEL
    assert data["UserName"] == "misha"
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] is True
    assert data["ProgramArguments"][0] == "/usr/local/bin/lmm"
    assert "daemon" in data["ProgramArguments"]
    assert "127.0.0.1" in data["ProgramArguments"]


def test_plist_env_has_state_home_path():
    xml = launchd_plist(exec_path="/s/venv/bin/lmm", host="127.0.0.1", port=8770, user="misha",
                        env={"LMM_STATE_DIR": "/Users/Shared/local-model-manager",
                             "HOME": "/Users/misha", "PATH": "/opt/homebrew/bin:/usr/bin"})
    data = plistlib.loads(xml.encode())
    e = data["EnvironmentVariables"]
    assert e["LMM_STATE_DIR"] == "/Users/Shared/local-model-manager"
    assert e["HOME"] == "/Users/misha"
    assert "/opt/homebrew/bin" in e["PATH"]


def test_shared_setup_steps():
    joined = "\n".join(shared_setup_steps(user="misha", shared_dir="/Users/Shared/local-model-manager"))
    assert "mkdir -p" in joined and "/Users/Shared/local-model-manager" in joined
    assert "chown" in joined and "misha:staff" in joined
    assert "2770" in joined


def test_shared_venv_steps():
    steps = shared_venv_steps(shared_dir="/Users/Shared/local-model-manager",
                              project_dir="/proj", user="misha")
    joined = "\n".join(steps)
    assert "uv venv" in joined
    assert "uv pip install" in joined and "/proj" in joined
    assert shared_venv_exec("/Users/Shared/local-model-manager").endswith("venv/bin/lmm")
    assert "chown -R" in joined and "misha:staff" in joined


def test_shared_venv_uses_readable_python():
    shared = "/Users/Shared/local-model-manager"
    steps = shared_venv_steps(shared_dir=shared, project_dir="/proj", user="misha")
    joined = "\n".join(steps)
    assert "uv python install" in joined and "3.11" in joined and "--no-bin" in joined
    assert f"{shared}/python" in joined
    venv_step = next(s for s in steps if "uv venv" in s)
    assert "--managed-python" in venv_step and "UV_PYTHON_INSTALL_DIR" in venv_step


def test_shared_venv_clear_flag():
    plain = "\n".join(shared_venv_steps(shared_dir="/s", project_dir="/p", user="misha"))
    assert "--clear" not in plain
    cleared = shared_venv_steps(shared_dir="/s", project_dir="/p", user="misha", clear=True)
    venv_step = next(s for s in cleared if "uv venv" in s)
    assert "--clear" in venv_step


def test_install_steps_run_as_user_no_account_or_acl():
    steps = install_steps(user="misha", host="127.0.0.1", port=8770,
                          shared_dir="/Users/Shared/local-model-manager", project_dir="/proj")
    joined = "\n".join(steps)
    assert "uv venv" in joined
    assert plist_install_path() in joined
    assert "launchctl bootstrap" in joined
    assert "socketfilterfw" in joined
    assert "mkdir -p" in joined
    # run-as-user: no service account, no models-dir ACL
    assert "dscl" not in joined
    assert "+a " not in joined


def test_install_steps_reinstall_adds_bootout_and_clear():
    plain = "\n".join(install_steps(user="misha", host="127.0.0.1", port=8770,
                                    shared_dir="/s", project_dir="/p"))
    assert "launchctl bootout" not in plain
    reinst = install_steps(user="misha", host="127.0.0.1", port=8770,
                           shared_dir="/s", project_dir="/p", reinstall=True)
    joined = "\n".join(reinst)
    assert "launchctl bootout" in joined and "--clear" in joined
    bootout_idx = next(i for i, s in enumerate(reinst) if "bootout" in s)
    bootstrap_idx = next(i for i, s in enumerate(reinst) if "bootstrap" in s)
    assert bootout_idx < bootstrap_idx


def test_uninstall_steps_no_account_or_acl():
    joined = "\n".join(uninstall_steps(shared_dir="/Users/Shared/local-model-manager"))
    assert "launchctl bootout" in joined
    assert plist_install_path() in joined
    assert "rm -rf /Users/Shared/local-model-manager" in joined
    # run-as-user: nothing to delete or de-ACL
    assert "dscl" not in joined
    assert "chmod" not in joined


def test_existing_install_artifacts_detects_present(monkeypatch, tmp_path):
    monkeypatch.setattr("os.path.exists", lambda p: False)
    assert existing_install_artifacts(shared_dir=str(tmp_path)) == []
    monkeypatch.setattr("os.path.exists", lambda p: True)
    found = existing_install_artifacts(shared_dir=str(tmp_path))
    assert any("plist" in f for f in found) and any("venv" in f for f in found)
