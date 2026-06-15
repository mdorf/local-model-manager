from lmm.api import _default_command_builder
from lmm.daemonconfig import DaemonConfig


def test_command_builder_omits_api_key_on_loopback(qwen_like):
    # Loopback daemon → inference server bound to 127.0.0.1, no --api-key
    # (so llama-server's own UI works locally without a key).
    build = _default_command_builder(
        DaemonConfig(host="127.0.0.1", inference_key="sek", roots=[str(qwen_like.parent)]))
    cmd, _path = build(qwen_like.name, 8080)
    assert "--api-key" not in cmd
    assert "--host" in cmd and "127.0.0.1" in cmd


def test_command_builder_adds_api_key_on_lan(qwen_like):
    # LAN-exposed daemon → inference server bound to 0.0.0.0 + --api-key.
    build = _default_command_builder(
        DaemonConfig(host="0.0.0.0", inference_key="sek", roots=[str(qwen_like.parent)]))
    cmd, _path = build(qwen_like.name, 8080)
    assert "--api-key" in cmd and "sek" in cmd
    assert "0.0.0.0" in cmd
