from lmm.cli import build_parser, cmd_service


def test_parser_routes_service():
    a = build_parser().parse_args(["service", "status"])
    assert a.func is cmd_service and a.action == "status"
    b = build_parser().parse_args(["service", "restart", "--dry-run"])
    assert b.action == "restart" and b.dry_run is True


def test_service_stop_dry_run(capsys):
    rc = cmd_service(build_parser().parse_args(["service", "stop", "--dry-run"]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "launchctl bootout" in out


def test_service_start_dry_run(capsys):
    rc = cmd_service(build_parser().parse_args(["service", "start", "--dry-run"]))
    assert rc == 0
    assert "launchctl bootstrap" in capsys.readouterr().out


def test_service_restart_dry_run_shows_both(capsys):
    rc = cmd_service(build_parser().parse_args(["service", "restart", "--dry-run"]))
    out = capsys.readouterr().out
    assert rc == 0
    assert "launchctl bootout" in out and "launchctl bootstrap" in out


def test_service_stop_without_root_refuses(monkeypatch, capsys):
    monkeypatch.setattr("os.geteuid", lambda: 1000)
    rc = cmd_service(build_parser().parse_args(["service", "stop"]))
    out = capsys.readouterr().out.lower()
    assert rc == 1
    assert "root" in out or "sudo" in out


def test_service_status_not_installed(monkeypatch, tmp_path, capsys):
    # Point the plist path at a file that doesn't exist → "not installed".
    monkeypatch.setattr("lmm.deploy.plist_install_path", lambda: str(tmp_path / "nope.plist"))
    rc = cmd_service(build_parser().parse_args(["service", "status"]))
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "not installed" in out
    assert "lmm daemon" in out  # points at the foreground option


def test_service_status_installed_not_responding(monkeypatch, tmp_path, capsys):
    plist = tmp_path / "svc.plist"
    plist.write_text("<plist/>")
    monkeypatch.setattr("lmm.deploy.plist_install_path", lambda: str(plist))
    monkeypatch.setattr("lmm.cli.SHARED_DIR", str(tmp_path))  # no daemon.json → defaults
    monkeypatch.setattr("lmm.cli._probe_health", lambda host, port: False)
    rc = cmd_service(build_parser().parse_args(["service", "status"]))
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "installed" in out
    assert "not responding" in out
    assert "sudo lmm service start" in out
