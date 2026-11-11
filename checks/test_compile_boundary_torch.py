"""MOSS-Image's gradient checkpoint goes inside the compiled blocks only on PyTorch 2.15+: on 2.10 that placement keeps
7-10 activations a block and a run spilled into system RAM (#324); older PyTorch keeps it outside."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.families.driver import FamilyDriver  # noqa: E402
from fizgig.families.moss_image import MOSS_IMAGE  # noqa: E402


def _driver():
    d = FamilyDriver.__new__(FamilyDriver)
    d.description = MOSS_IMAGE
    return d


def test_inside_on_new_torch(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.15.0+cu134")
    assert _driver()._torch_boundary() == ("inside", "")
    assert _driver().compile_plan("on", 1000, "int8", 0)[0] == "inside"


def test_outside_on_old_torch(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.10.0+cu128")
    b, why = _driver()._torch_boundary()
    assert b == "outside" and "2.15" in why
    b, why = _driver().compile_plan("on", 1000, "int8", 0)
    assert b == "outside" and "PyTorch 2.10.0" in why
