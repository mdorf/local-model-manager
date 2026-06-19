from lmm.hermes import list_profiles, profile_config_path, profiles_bound_to


def _hermes(tmp_path):
    hd = tmp_path / ".hermes"
    (hd / "profiles" / "qwen-herm").mkdir(parents=True)
    (hd / "profiles" / "kimi-herm").mkdir(parents=True)
    return hd


def test_profiles_bound_to_reports_only_local_ones(tmp_path):
    hd = _hermes(tmp_path)
    # default → cloud (not connected)
    (hd / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  base_url: https://openrouter.ai/api/v1\n")
    # qwen-herm → local via custom provider on :8080 (CONNECTED)
    (hd / "profiles" / "qwen-herm" / "config.yaml").write_text(
        "model:\n  provider: custom:local\n  default: stale-old-name\n"
        "providers:\n  local:\n    base_url: http://127.0.0.1:8080/v1\n")
    # kimi-herm → local but a DIFFERENT port (not connected to :8080)
    (hd / "profiles" / "kimi-herm" / "config.yaml").write_text(
        "model:\n  provider: custom:local\n  base_url: http://127.0.0.1:9999/v1\n")

    assert profiles_bound_to(8080, hd) == ["qwen-herm"]


def test_profiles_bound_to_matches_by_url_not_model_label(tmp_path):
    # A stale model.default must NOT exclude a profile that still points at :8080
    # (llama-server ignores the model field). Detection is base_url/port-based.
    hd = _hermes(tmp_path)
    (hd / "config.yaml").write_text(
        "model:\n  provider: custom:local\n  default: WHATEVER\n  base_url: http://localhost:8080/v1\n")
    assert "default" in profiles_bound_to(8080, hd)


def test_profile_config_path_resolution(tmp_path):
    hd = tmp_path / ".hermes"
    # default / None / "" all resolve to the root config
    for name in (None, "", "default"):
        assert profile_config_path(name, hd) == hd / "config.yaml"
    # a named profile resolves under profiles/<name>/
    assert profile_config_path("qwen-herm", hd) == hd / "profiles" / "qwen-herm" / "config.yaml"


def test_lists_default_root_and_named_profiles(tmp_path):
    hd = tmp_path / ".hermes"
    (hd / "profiles" / "qwen-herm").mkdir(parents=True)
    (hd / "profiles" / "gpt-herm").mkdir(parents=True)
    (hd / "config.yaml").write_text("model: {}\n")
    (hd / "profiles" / "qwen-herm" / "config.yaml").write_text("model: {}\n")
    (hd / "profiles" / "gpt-herm" / "config.yaml").write_text("model: {}\n")

    profs = list_profiles(hd)
    names = [p["name"] for p in profs]
    # default (the root config) first, then named profiles sorted
    assert names == ["default", "gpt-herm", "qwen-herm"]
    assert profs[0]["path"] == str(hd / "config.yaml")
    assert profs[2]["path"] == str(hd / "profiles" / "qwen-herm" / "config.yaml")


def test_skips_profile_dirs_without_a_config(tmp_path):
    hd = tmp_path / ".hermes"
    (hd / "profiles" / "empty").mkdir(parents=True)  # no config.yaml inside
    (hd / "config.yaml").write_text("model: {}\n")
    names = [p["name"] for p in list_profiles(hd)]
    assert names == ["default"]


def test_empty_when_no_hermes_dir(tmp_path):
    assert list_profiles(tmp_path / "nonexistent") == []
