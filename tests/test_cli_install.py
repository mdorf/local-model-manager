from lmm.cli import build_parser, cmd_install, cmd_uninstall


def test_parser_has_install_uninstall():
    p = build_parser()
    a = p.parse_args(["install", "--dry-run", "--user", "_lmm"])
    assert a.func is cmd_install
    assert a.dry_run is True
    b = p.parse_args(["uninstall", "--dry-run"])
    assert b.func is cmd_uninstall


def test_install_dry_run_shows_shared_scheme(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    rc = cmd_install(build_parser().parse_args(
        ["install", "--dry-run", "--project-dir", "/proj"]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "/Users/Shared/local-model-manager" in out
    assert "uv venv" in out
    assert "LMM_STATE_DIR" in out
    assert "launchctl bootstrap" in out
    assert "daemon.json" in out


def test_install_without_root_refuses(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setattr("os.geteuid", lambda: 1000)
    rc = cmd_install(build_parser().parse_args(["install", "--project-dir", "/proj"]))
    out = capsys.readouterr().out.lower()
    assert rc == 1
    assert "sudo" in out or "root" in out


def test_uninstall_dry_run(monkeypatch, tmp_path, capsys):
    rc = cmd_uninstall(build_parser().parse_args(["uninstall", "--dry-run"]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "launchctl bootout" in out


def test_install_reinstall_flag_parsed():
    assert build_parser().parse_args(["install", "--reinstall"]).reinstall is True
    assert build_parser().parse_args(["install", "--force"]).reinstall is True
    assert build_parser().parse_args(["install"]).reinstall is False


def test_install_refuses_when_already_installed(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.setattr("lmm.deploy.existing_install_artifacts",
                        lambda **kw: ["shared venv"])
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr("pathlib.Path.write_text", lambda *a, **k: calls.append("write"))
    rc = cmd_install(build_parser().parse_args(
        ["install", "--user", "_lmm", "--uid", "250", "--project-dir", "/proj"]))
    out = capsys.readouterr().out.lower()
    assert rc == 1
    assert "already installed" in out
    assert "reinstall" in out
    assert calls == []  # guard returned before any mutation


def test_install_reinstall_skips_account_and_boots_out_first(monkeypatch, tmp_path):
    # regression: the UID-reassignment fix lives in cmd_install's imperative
    # path (skip account_steps when the account already exists). Pin it: on
    # --reinstall with an existing account, NO `dscl . -create` is issued, and
    # the running job is booted out before the plist is rewritten/bootstrapped.
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.setattr("lmm.deploy.existing_install_artifacts",
                        lambda **kw: ["service account _lmm (uid 250)"])
    monkeypatch.setattr("lmm.deploy.account_uid", lambda user: 250)
    events = []

    class _R:
        returncode = 0

    monkeypatch.setattr("subprocess.run",
                        lambda cmd, *a, **k: events.append(("run", cmd)) or _R())
    monkeypatch.setattr("pathlib.Path.write_text",
                        lambda self, *a, **k: events.append(("write", str(self))))
    rc = cmd_install(build_parser().parse_args(
        ["install", "--reinstall", "--user", "_lmm", "--uid", "250",
         "--project-dir", "/proj"]))
    assert rc == 0
    # account reused, never recreated -> UID can't be reassigned
    assert not any(kind == "run" and "dscl . -create" in cmd for kind, cmd in events)
    bootout_i = next(i for i, (k, c) in enumerate(events) if k == "run" and "bootout" in c)
    write_i = next(i for i, (k, c) in enumerate(events) if k == "write" and c.endswith(".plist"))
    bootstrap_i = next(i for i, (k, c) in enumerate(events) if k == "run" and "bootstrap" in c)
    assert bootout_i < write_i < bootstrap_i
