import lmm.state as state
from lmm.state import state_dir


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "env"))
    monkeypatch.setattr(state, "SHARED_STATE_DIR", tmp_path / "shared")
    assert state_dir() == tmp_path / "env"


def test_shared_used_when_daemon_json_present(monkeypatch, tmp_path):
    monkeypatch.delenv("LMM_STATE_DIR", raising=False)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "daemon.json").write_text("{}")
    monkeypatch.setattr(state, "SHARED_STATE_DIR", shared)
    assert state_dir() == shared


def test_home_used_when_no_shared(monkeypatch, tmp_path):
    monkeypatch.delenv("LMM_STATE_DIR", raising=False)
    monkeypatch.setattr(state, "SHARED_STATE_DIR", tmp_path / "absent")
    d = state_dir()
    assert d != tmp_path / "absent"
    assert "local-model-manager" in str(d)
