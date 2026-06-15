from lmm.llama import supported_flags

_HELP = """
usage: llama-server [options]

----- common params -----
-h,    --help, --usage          print usage and exit
-ngl,  --n-gpu-layers N         number of layers to store in VRAM
-fa,   --flash-attn [on|off]    enable flash attention
       --cache-type-k TYPE      KV cache type for K
--spec-type none,draft-mtp,...  speculative decoding types
-t,    --threads N              number of threads
"""


def test_supported_flags_extracts_short_and_long():
    flags = supported_flags(_HELP)
    assert "-ngl" in flags
    assert "--n-gpu-layers" in flags
    assert "-fa" in flags
    assert "--flash-attn" in flags
    assert "--cache-type-k" in flags
    assert "--spec-type" in flags
    assert "-t" in flags
    assert "--threads" in flags


def test_supported_flags_are_all_dashed():
    flags = supported_flags(_HELP)
    assert "usage:" not in flags
    assert all(f.startswith("-") for f in flags)
