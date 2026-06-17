from lmm.hardware import HardwareInfo, _metal_working_set_bytes, detect_hardware


def test_detect_hardware_returns_plausible_values():
    hw = detect_hardware()
    assert isinstance(hw, HardwareInfo)
    assert hw.total_ram_bytes > 1 * 1024**3          # > 1 GiB
    assert hw.logical_cores >= 1
    assert hw.perf_cores >= 1
    assert hw.perf_cores <= hw.logical_cores
    assert 0.0 < hw.usable_fraction <= 1.0
    assert hw.usable_ram_bytes == int(hw.total_ram_bytes * hw.usable_fraction)


def test_usable_ram_scales_with_fraction():
    hw = HardwareInfo(total_ram_bytes=64 * 1024**3, logical_cores=14,
                      perf_cores=10, platform="Darwin", has_metal=True,
                      usable_fraction=0.85)
    assert hw.usable_ram_bytes == int(64 * 1024**3 * 0.85)


def test_metal_working_set_honors_wired_limit_override(monkeypatch):
    total = 64 * 1024**3
    # explicit iogpu.wired_limit_mb (e.g. raised by a power user) is respected
    monkeypatch.setattr("lmm.hardware._sysctl_int", lambda k: 57344)  # 56 GiB in MB
    assert _metal_working_set_bytes(total) == 57344 * 1024 * 1024
    # 0/unset → macOS default ~75% of RAM
    monkeypatch.setattr("lmm.hardware._sysctl_int", lambda k: 0)
    assert _metal_working_set_bytes(total) == int(total * 0.75)


def test_detect_hardware_sets_gpu_working_set_on_metal():
    hw = detect_hardware()
    if hw.has_metal:
        assert 0 < hw.gpu_working_set_bytes <= hw.total_ram_bytes


def test_detect_hardware_works_without_usr_sbin_on_path(monkeypatch):
    # regression: a launchd daemon's PATH lacks /usr/sbin, so a bare `sysctl`
    # wasn't found and RAM read as 0 (everything → wont_load). With the absolute
    # path, detection must still work regardless of PATH.
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    hw = detect_hardware()
    if hw.platform == "Darwin":
        assert hw.total_ram_bytes > 1 * 1024**3
        assert hw.usable_ram_bytes > 0
