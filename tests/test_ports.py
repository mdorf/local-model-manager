import socket

from lmm.ports import is_port_in_use, pick_free_port


def test_pick_free_port_returns_unused_port():
    port = pick_free_port(start=49500)
    assert 49500 <= port < 65536
    assert is_port_in_use(port) is False


def test_is_port_in_use_detects_a_bound_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    bound = s.getsockname()[1]
    try:
        assert is_port_in_use(bound) is True
    finally:
        s.close()


def test_pick_free_port_skips_in_use(monkeypatch):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    bound = s.getsockname()[1]
    try:
        got = pick_free_port(start=bound)
        assert got != bound
        assert got > bound
    finally:
        s.close()
