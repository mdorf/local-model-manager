from fastapi.testclient import TestClient

from lmm.api import _instance_dict, create_app
from lmm.daemonconfig import DaemonConfig
from lmm.server import ServerInstance
from tests.conftest import write_minimal_gguf


def test_instance_dict_exposes_launch_flags():
    inst = ServerInstance(port=8080, pid=1, model_path="/m/x.gguf", started_at=0.0,
                          status="ready",
                          command=["llama-server", "-m", "/m/x.gguf", "-c", "8192"])
    assert _instance_dict(inst)["flags"] == ["-m", "/m/x.gguf", "-c", "8192"]
    # unknown command (e.g. adopted server, pid lost) → flags is None
    bare = ServerInstance(port=8080, pid=1, model_path="/m/x.gguf", started_at=0.0,
                          status="ready")
    assert _instance_dict(bare)["flags"] is None

_META = {
    "general.architecture": "qwen35", "general.name": "Qwen3.6-27B",
    "general.basename": "Qwen3.6-27B", "general.size_label": "27B",
    "general.file_type": 7, "qwen35.block_count": 65,
    "qwen35.context_length": 262144, "qwen35.full_attention_interval": 4,
    "qwen35.attention.head_count_kv": 4, "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256, "qwen35.nextn_predict_layers": 1,
}
_H = {"Authorization": "Bearer t"}


def _client(root):
    cfg = DaemonConfig(host="127.0.0.1", port=8770, token="t", roots=[str(root)])
    return TestClient(create_app(cfg))


def test_list_models(tmp_path):
    write_minimal_gguf(tmp_path / "Qwen3.6-27B-Q8_0.gguf", _META,
                       ["blk.64.nextn.eh_proj.weight"])
    r = _client(tmp_path).get("/api/models", headers=_H)
    assert r.status_code == 200
    models = r.json()["models"]
    assert len(models) == 1
    assert models[0]["arch"] == "qwen35"
    assert models[0]["has_mtp"] is True


def test_recommend_endpoint(tmp_path):
    write_minimal_gguf(tmp_path / "Qwen3.6-27B-Q8_0.gguf", _META,
                       ["blk.64.nextn.eh_proj.weight"])
    r = _client(tmp_path).get("/api/models/Qwen3.6-27B-Q8_0.gguf/recommend", headers=_H)
    assert r.status_code == 200
    body = r.json()
    assert "flags" in body and "-c" in body["flags"]
    assert body["fit"]["level"] in ("comfortable", "tight", "wont_load")


def test_recommend_unknown_model_404(tmp_path):
    r = _client(tmp_path).get("/api/models/nope.gguf/recommend", headers=_H)
    assert r.status_code == 404
