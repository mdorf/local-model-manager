from lmm.state import InstanceRecord, load_instances, save_instances, state_dir


def test_state_dir_honors_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    assert state_dir() == tmp_path / "st"


def test_homepage_override_set_get_clear(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    from lmm import state
    url = "https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF"
    assert state.load_homepages() == {}
    state.set_homepage("ornith-1.0-35b-Q8_0.gguf", url)
    assert state.load_homepages() == {"ornith-1.0-35b-Q8_0.gguf": url}
    state.set_homepage("ornith-1.0-35b-Q8_0.gguf", "")  # empty clears
    assert state.load_homepages() == {}


def test_homepage_override_ignores_non_http(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    from lmm import state
    state.set_homepage("m.gguf", "javascript:alert(1)")  # not http(s) → rejected
    assert state.load_homepages() == {}


def test_save_and_load_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    recs = [InstanceRecord(port=8080, pid=123, model_path="/m/a.gguf",
                           started_at=1.0, external=False),
            InstanceRecord(port=8081, pid=456, model_path="/m/b.gguf",
                           started_at=2.0, external=True)]
    save_instances(recs)
    loaded = load_instances()
    assert loaded == recs


def test_load_returns_empty_when_no_file(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "none"))
    assert load_instances() == []


def test_load_tolerates_corrupt_file(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    d = state_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "instances.json").write_text("{not json")
    assert load_instances() == []
