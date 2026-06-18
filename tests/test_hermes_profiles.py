from lmm.hermes import list_profiles, profile_config_path


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
