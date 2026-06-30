from pathlib import Path

from lmm.gguf import GGUFInfo, read_gguf
from lmm.models import Model, classify, quant_from_file_type


def _mk(path):
    return Model(path=Path(path), arch="qwen35", name="Qwen3.6-27B", family="qwen3.6",
                 size_label="27B", quant="Q8_0", block_count=65, context_length=262144,
                 has_mtp=True, hf_base_repo=None)


def test_model_matches_filename_path_and_stem():
    m = _mk("/Users/Shared/models/Qwen3.6-27B-Q8_0.gguf")
    assert m.matches("Qwen3.6-27B-Q8_0.gguf")                          # filename
    assert m.matches("/Users/Shared/models/Qwen3.6-27B-Q8_0.gguf")     # full path
    assert m.matches("Qwen3.6-27B-Q8_0")                               # bare stem (README form)
    assert not m.matches("Qwen3.6-27B-Other")


def test_model_mmproj_from_sidecars():
    m = Model(path=Path("/m/x.gguf"), arch="q", name="x", family="x", size_label="35B",
              quant="Q8_0", block_count=40, context_length=262144, has_mtp=False,
              hf_base_repo=None,
              sidecars=[Path("/m/template.jinja"), Path("/m/mmproj-F16.gguf")])
    assert m.mmproj == "/m/mmproj-F16.gguf"
    assert _mk("/m/x.gguf").mmproj is None   # no sidecars → text-only


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


def test_classify_extracts_sampling_defaults():
    info = GGUFInfo(version=3, metadata={
        "general.architecture": "qwen35moe",
        "general.sampling.temp": 1.0,
        "general.sampling.top_k": 20,
        "general.sampling.top_p": 0.95,
    }, tensor_names=[])
    m = classify(info, "/tmp/x.gguf")
    assert m.sampling == {"temp": 1.0, "top_k": 20, "top_p": 0.95}


def test_classify_sampling_none_when_absent():
    info = GGUFInfo(version=3, metadata={"general.architecture": "x"}, tensor_names=[])
    assert classify(info, "/tmp/x.gguf").sampling is None


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


def test_classify_extracts_license_quantizer_chat_template_and_card():
    info = GGUFInfo(
        version=3,
        metadata={
            "general.architecture": "qwen35",
            "general.license": "apache-2.0",
            "general.quantized_by": "Unsloth",
            "general.author": "HauhauCS",
            "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.6-27B",
            "tokenizer.chat_template": "{{ template }}",
        },
        tensor_names=[],
    )
    m = classify(info, "/tmp/x.gguf")
    assert m.license == "apache-2.0"
    assert m.quantized_by == "Unsloth"
    assert m.author == "HauhauCS"
    assert m.has_chat_template is True
    assert m.hf_base_repo == "https://huggingface.co/Qwen/Qwen3.6-27B"  # base_model fallback


def test_classify_optional_metadata_absent():
    info = GGUFInfo(version=3, metadata={"general.architecture": "qwen35"}, tensor_names=[])
    m = classify(info, "/tmp/x.gguf")
    assert m.license is None and m.quantized_by is None
    assert m.has_chat_template is False and m.hf_base_repo is None


def test_classify_derives_hf_repo_from_author_and_name():
    # No embedded repo URL, but author + name → best-effort <author>/<name> card.
    info = GGUFInfo(version=3, metadata={
        "general.architecture": "qwen35",
        "general.author": "HauhauCS",
        "general.name": "Qwen3.6-27B-Uncensored-HauhauCS-Balanced",
    }, tensor_names=[])
    m = classify(info, "/tmp/x.gguf")
    assert m.hf_base_repo == "https://huggingface.co/HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced"


def test_classify_embedded_link_beats_derived():
    info = GGUFInfo(version=3, metadata={
        "general.architecture": "qwen35",
        "general.author": "HauhauCS",
        "general.name": "Some-Name",
        "general.base_model.0.repo_url": "https://huggingface.co/Qwen/Qwen3.6-27B",
    }, tensor_names=[])
    m = classify(info, "/tmp/x.gguf")
    assert m.hf_base_repo == "https://huggingface.co/Qwen/Qwen3.6-27B"  # embedded wins
