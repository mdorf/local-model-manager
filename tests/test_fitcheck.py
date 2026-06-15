from lmm.fitcheck import assess_fit


def test_comfortable_when_well_under_budget():
    r = assess_fit(total_bytes=10, weights_bytes=5, usable_ram_bytes=100)
    assert r.level == "comfortable"
    assert r.fits is True


def test_tight_when_near_budget():
    r = assess_fit(total_bytes=90, weights_bytes=40, usable_ram_bytes=100)
    assert r.level == "tight"
    assert r.fits is True
    assert "tight" in r.message.lower() or "swap" in r.message.lower()


def test_wont_load_when_total_exceeds_budget():
    r = assess_fit(total_bytes=120, weights_bytes=40, usable_ram_bytes=100)
    assert r.level == "wont_load"
    assert r.fits is False


def test_wont_load_when_weights_alone_exceed_budget():
    r = assess_fit(total_bytes=120, weights_bytes=110, usable_ram_bytes=100)
    assert r.level == "wont_load"
    assert "weights" in r.message.lower()


def test_message_is_human_readable_with_gib():
    r = assess_fit(total_bytes=50 * 1024**3, weights_bytes=30 * 1024**3,
                   usable_ram_bytes=54 * 1024**3)
    assert "GiB" in r.message
