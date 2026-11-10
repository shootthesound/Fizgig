"""#316: write_rocm_env.py keeps BNB_ROCM_VERSION on the bitsandbytes ROCm library that matches PyTorch's ROCm, unless
that library lacks the 4-bit kernels (cgemm_4bit_*) - then the nearest version that has them, newer first. Fake
library files stand in for bitsandbytes' (only their export names matter)."""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("write_rocm_env", os.path.join(ROOT, "write_rocm_env.py"))
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)

FULL = b"junk\0cgemm_4bit_bf16\0cgemm_4bit_fp16\0cgemm_4bit_fp32\0more"
NONE = b"junk\0kgemm_4bit_inference_naive_bf16\0more"


@pytest.fixture
def libs(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_bitsandbytes_lib_dir", lambda: tmp_path)
    monkeypatch.setattr(W, "_bnb_lib_ext", lambda: ".dll")

    def make(**versions):
        for v, full in versions.items():
            (tmp_path / f"libbitsandbytes_rocm{v[1:]}.dll").write_bytes(FULL if full else NONE)
    return make


def test_the_match_is_kept_when_it_has_the_kernels(libs):
    libs(v715=True, v716=True)
    assert W._with_4bit_gemm("715", W._bitsandbytes_rocm_lib("715"))[0] == "715"


def test_316_the_match_without_them_moves_to_the_nearest_newer_one(libs):
    libs(v713=True, v714=True, v715=False, v716=True, v717=True)
    v, lib = W._with_4bit_gemm("715", W._bitsandbytes_rocm_lib("715"))
    assert v == "716" and lib.name == "libbitsandbytes_rocm716.dll"


def test_no_newer_one_takes_the_nearest_older(libs):
    libs(v713=True, v714=True, v715=False)
    assert W._with_4bit_gemm("715", W._bitsandbytes_rocm_lib("715"))[0] == "714"


def test_none_has_them_keeps_the_match(libs):
    libs(v714=False, v715=False, v716=False)
    assert W._with_4bit_gemm("715", W._bitsandbytes_rocm_lib("715"))[0] == "715"


def test_no_library_found_is_left_alone(libs):
    libs(v716=True)
    assert W._with_4bit_gemm("715", None) == ("715", None)
