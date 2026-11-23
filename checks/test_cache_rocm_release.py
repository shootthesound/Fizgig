"""#389: the latent cache hands the allocator's cache back after each picture on AMD (ROCm), where each bucket shape's
convolution buffers otherwise pile up through a recache; NVIDIA keeps its cache."""
import inspect
from unittest.mock import patch

from fizgig.families import cache


def test_released_on_rocm_only():
    for hip, expected in (("7.15", 1), (None, 0)):
        with patch.object(cache.torch.version, "hip", hip), patch.object(cache.torch.cuda, "is_available", lambda: True), \
                patch.object(cache.torch.cuda, "empty_cache") as empty:
            cache.release_between_pictures()
        assert empty.call_count == expected


def test_latent_stage_releases_after_each_batch():
    src = inspect.getsource(cache)
    stage = src[src.index("def encode(batch):"):src.index("encode_datasets(datasets, encode, args)")]
    assert "release_between_pictures()" in stage
