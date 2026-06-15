from lmm.cli import build_parser, cmd_install, cmd_uninstall


def test_parser_has_install_uninstall():
    p = build_parser()
    a = p.parse_args(["install", "--dry-run", "--user", "_lmm"])
    assert a.func is cmd_install
    assert a.dry_run is True
    b = p.parse_args(["uninstall", "--dry-run"])
    assert b.func is cmd_uninstall


def test_install_dry_run_prints_steps_without_executing(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    rc = cmd_install(build_parser().parse_args(
        ["install", "--dry-run", "--exec", "/usr/local/bin/lmm",
         "--models-dir", "/Users/Shared/models"]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "dscl . -create /Users/_lmm" in out
    assert "launchctl bootstrap" in out
    assert "would run" in out.lower() or "dry" in out.lower()


def test_install_without_root_refuses(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setattr("os.geteuid", lambda: 1000)
    rc = cmd_install(build_parser().parse_args(
        ["install", "--exec", "/usr/local/bin/lmm", "--models-dir", "/tmp/m"]))
    out = capsys.readouterr().out.lower()
    assert rc == 1
    assert "sudo" in out or "root" in out
