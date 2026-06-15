from lmm.cli import _install_user, build_parser, cmd_install, cmd_uninstall


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


def test_install_dry_run_runs_as_user_no_account(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setenv("SUDO_USER", "misha")
    rc = cmd_install(build_parser().parse_args(["install", "--dry-run", "--project-dir", "/proj"]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "misha" in out  # plist UserName / "runs as user"
    assert "/Users/Shared/local-model-manager" in out
    assert "launchctl bootstrap" in out
    assert "daemon.json" in out
    assert "dscl" not in out  # run-as-user: no service account


def test_install_without_root_refuses(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setenv("SUDO_USER", "misha")
    monkeypatch.setattr("os.geteuid", lambda: 1000)
    rc = cmd_install(build_parser().parse_args(["install", "--project-dir", "/proj"]))
    out = capsys.readouterr().out.lower()
    assert rc == 1
    assert "sudo" in out or "root" in out


def test_uninstall_dry_run(capsys):
    rc = cmd_uninstall(build_parser().parse_args(["uninstall", "--dry-run"]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "launchctl bootout" in out
    assert "rm -rf" in out


def test_install_refuses_root_user(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setattr("os.geteuid", lambda: 0)
    rc = cmd_install(build_parser().parse_args(
        ["install", "--user", "root", "--project-dir", "/proj"]))
    assert rc == 1
    assert "root" in capsys.readouterr().out.lower()


def test_install_refuses_nonexistent_user(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setattr("os.geteuid", lambda: 0)
    rc = cmd_install(build_parser().parse_args(
        ["install", "--user", "no_such_user_xyz_123", "--project-dir", "/proj"]))
    assert rc == 1
    assert "does not exist" in capsys.readouterr().out.lower()


def test_install_refuses_when_already_installed(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setenv("SUDO_USER", "misha")
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.setattr("lmm.deploy.existing_install_artifacts", lambda **kw: ["shared venv"])
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr("pathlib.Path.write_text", lambda *a, **k: calls.append("write"))
    rc = cmd_install(build_parser().parse_args(["install", "--project-dir", "/proj"]))
    out = capsys.readouterr().out.lower()
    assert rc == 1
    assert "already installed" in out
    assert calls == []  # guard returns before any mutation
