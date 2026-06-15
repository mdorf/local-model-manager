from lmm.daemonconfig import DaemonConfig, load_or_create_config


def test_creates_config_with_token(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    cfg = load_or_create_config()
    assert isinstance(cfg, DaemonConfig)
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8770
    assert len(cfg.token) >= 16
    assert isinstance(cfg.roots, list) and cfg.roots


def test_config_is_stable_across_loads(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    first = load_or_create_config()
    second = load_or_create_config()
    assert first.token == second.token


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    monkeypatch.setenv("LMM_MODELS_DIR", str(tmp_path / "models"))
    cfg = load_or_create_config()
    assert str(tmp_path / "models") in cfg.roots
