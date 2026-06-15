from lmm.net import is_loopback


def test_is_loopback():
    assert is_loopback("127.0.0.1")
    assert is_loopback("::1")
    assert is_loopback("localhost")
    assert is_loopback("")
    assert is_loopback(None)
    assert is_loopback("  127.0.0.1  ")  # whitespace-tolerant
    assert not is_loopback("0.0.0.0")
    assert not is_loopback("192.168.1.10")
    assert not is_loopback("mishmacmini.local")
