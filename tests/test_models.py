from pathlib import Path

from lmm.gguf import GGUFInfo, read_gguf
from lmm.models import classify, quant_from_file_type


def test_quant_mapping():
    assert quant_from_file_type(7) == "Q8_0"
    assert quant_from_file_type(0) == "F32"
    assert quant_from_file_type(99999) == "unknown"
    assert quant_from_file_type(6) == "unknown"


def test_classify_tolerates_array_marker_for_scalar_fields():
    info = GGUFInfo(
        version=3,
        metadata={
            "general.architecture": "qwen35",
            "general.file_type": {"__array__": True, "elem_type": 8, "count": 2},
            "qwen35.nextn_predict_layers": {"__array__": True, "elem_type": 8, "count": 2},
        },
        tensor_names=[],
    )
    m = classify(info, "/tmp/x.gguf")
    assert m.quant == "unknown"
    assert m.has_mtp is False


def test_classify_core_fields(qwen_like):
    m = classify(read_gguf(qwen_like), qwen_like)
    assert m.arch == "qwen35"
    assert m.size_label == "27B"
    assert m.quant == "Q8_0"
    assert m.block_count == 65
    assert m.context_length == 262144
    assert m.family == "qwen3.6"          # basename minus size, lowercased


def test_classify_detects_mtp(qwen_like):
    m = classify(read_gguf(qwen_like), qwen_like)
    assert m.has_mtp is True


def test_classify_extracts_hf_base_repo(qwen_like):
    m = classify(read_gguf(qwen_like), qwen_like)
    assert m.hf_base_repo == "https://huggingface.co/Qwen/Qwen3.6-27B"


def test_classify_passes_through_shards_and_sidecars(qwen_like):
    shards = [qwen_like]
    sidecars = [Path("/models/mmproj.gguf")]
    m = classify(read_gguf(qwen_like), qwen_like, shards=shards, sidecars=sidecars)
    assert m.shards == shards
    assert m.sidecars == sidecars
