import pytest

import json

from lmm.cli import (
    _install_child_path,
    _install_user,
    _resolve_project_dir,
    _write_daemon_config,
    build_parser,
    cmd_install,
    cmd_uninstall,
)


def test_write_daemon_config_syncs_host_preserves_secrets(tmp_path):
    # A --reinstall that changes --host must refresh daemon.json's host (it decides
    # the model's bind), while keeping the token/inference_key so clients keep working.
    p = tmp_path / "daemon.json"
    p.write_text(json.dumps({"host": "127.0.0.1", "port": 8770, "token": "KEEPTOK",
                             "inference_key": "KEEPKEY", "roots": ["/old"]}))
    _write_daemon_config(p, host="0.0.0.0", port=8770, models_dir="/new")
    d = json.loads(p.read_text())
    assert d["host"] == "0.0.0.0"          # synced to this install (the fix)
    assert d["token"] == "KEEPTOK"          # secrets preserved across reinstall
    assert d["inference_key"] == "KEEPKEY"
    assert d["roots"] == ["/new"]
    # fresh install (no existing file) generates secrets
    p2 = tmp_path / "fresh.json"
    _write_daemon_config(p2, host="0.0.0.0", port=8770, models_dir="/m")
    d2 = json.loads(p2.read_text())
    assert d2["host"] == "0.0.0.0" and len(d2["token"]) >= 16


@pytest.fixture
def proj(tmp_path):
    """A directory that looks like a real source checkout (has pyproject.toml)."""
    d = tmp_path / "clone"
    d.mkdir()
    (d / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    return str(d)


def test_parser_has_install_uninstall():
    p = build_parser()
    a = p.parse_args(["install", "--dry-run"])
    assert a.func is cmd_install
    assert a.dry_run is True
    b = p.parse_args(["uninstall", "--dry-run"])
    assert b.func is cmd_uninstall


def test_install_user_resolution(monkeypatch):
    monkeypatch.setenv("SUDO_USER", "misha")
    # explicit --user wins
    assert _install_user(build_parser().parse_args(["install", "--user", "alice"])) == "alice"
    # else the sudo invoker
    assert _install_user(build_parser().parse_args(["install"])) == "misha"


def test_resolve_project_dir_prefers_explicit_arg():
    args = build_parser().parse_args(["install", "--project-dir", "/some/where"])
    assert _resolve_project_dir(args) == "/some/where"


def test_install_child_path_prepends_user_bins():
    # sudo strips PATH; install-time uv/llama-server must still be findable.
    p = _install_child_path("misha")
    parts = p.split(":")
    assert parts[0] == "/Users/misha/.local/bin"   # uv lives here
    assert "/opt/homebrew/bin" in parts            # llama-server (brew)


def test_install_errors_when_project_dir_not_a_project(monkeypatch, tmp_path, capsys):
    # A uv-tool-installed CLI can't guess its source; a bad/missing path must
    # fail loudly (not hand uv a non-project dir like the old default did).
    monkeypatch.setenv("SUDO_USER", "misha")
    missing = str(tmp_path / "nope")
    rc = cmd_install(build_parser().parse_args(["install", "--project-dir", missing]))
    out = capsys.readouterr().out.lower()
    assert rc == 1
    assert "could not locate" in out and "--project-dir" in out


def test_install_dry_run_runs_as_user_no_account(monkeypatch, tmp_path, capsys, proj):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setenv("SUDO_USER", "misha")
    rc = cmd_install(build_parser().parse_args(["install", "--dry-run", "--project-dir", proj]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "misha" in out  # plist UserName / "runs as user"
    assert "/Users/Shared/local-model-manager" in out
    assert "launchctl bootstrap" in out
    assert "daemon.json" in out
    assert "dscl" not in out  # run-as-user: no service account


def test_install_without_root_refuses(monkeypatch, tmp_path, capsys, proj):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setenv("SUDO_USER", "misha")
    monkeypatch.setattr("os.geteuid", lambda: 1000)
    rc = cmd_install(build_parser().parse_args(["install", "--project-dir", proj]))
    out = capsys.readouterr().out.lower()
    assert rc == 1
    assert "must run as root" in out


def test_uninstall_dry_run(capsys):
    rc = cmd_uninstall(build_parser().parse_args(["uninstall", "--dry-run"]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "launchctl bootout" in out
    assert "rm -rf" in out


def test_install_refuses_root_user(monkeypatch, tmp_path, capsys, proj):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setattr("os.geteuid", lambda: 0)
    rc = cmd_install(build_parser().parse_args(
        ["install", "--user", "root", "--project-dir", proj]))
    assert rc == 1
    assert "root" in capsys.readouterr().out.lower()


def test_install_refuses_nonexistent_user(monkeypatch, tmp_path, capsys, proj):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setattr("os.geteuid", lambda: 0)
    rc = cmd_install(build_parser().parse_args(
        ["install", "--user", "no_such_user_xyz_123", "--project-dir", proj]))
    assert rc == 1
    assert "does not exist" in capsys.readouterr().out.lower()


def test_install_refuses_when_already_installed(monkeypatch, tmp_path, capsys, proj):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setenv("SUDO_USER", "misha")
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.setattr("lmm.deploy.existing_install_artifacts", lambda **kw: ["shared venv"])
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr("pathlib.Path.write_text", lambda *a, **k: calls.append("write"))
    rc = cmd_install(build_parser().parse_args(["install", "--project-dir", proj]))
    out = capsys.readouterr().out.lower()
    assert rc == 1
    assert "already installed" in out
    assert calls == []  # guard returns before any mutation
