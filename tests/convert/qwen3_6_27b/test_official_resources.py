from __future__ import annotations

import pytest

from tools.convert.qwen3_6.common.official_resources import (
    OFFICIAL_RESOURCE_SHA256,
    validate_official_resource_hashes,
)


UNSLOTH_TOKENIZER_SHA256 = (
    "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
)


def test_unsloth_tokenizer_hash_is_rejected():
    hashes = dict(OFFICIAL_RESOURCE_SHA256)
    hashes["frontend/tokenizer.json"] = UNSLOTH_TOKENIZER_SHA256

    with pytest.raises(
        ValueError,
        match=(
            "tokenizer.json.*expected "
            + OFFICIAL_RESOURCE_SHA256["frontend/tokenizer.json"]
            + ".*got "
            + UNSLOTH_TOKENIZER_SHA256
        ),
    ):
        validate_official_resource_hashes(hashes)
