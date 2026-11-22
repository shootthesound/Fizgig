"""A text encoder's memory is freed when it's unloaded, even when the model is a reference cycle (transformers'
modules often are): Flux 3's Qwen3-VL-4B kept 6.8 GB on the card through a whole run, until Python's collector ran -
on a 16 GB card its bf16 DiT couldn't load beside it."""
import gc
import os
import weakref

import pytest

torch = pytest.importorskip("torch")


class _Cyclic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)
        self.me = [self]                        # a reference cycle, as many models have


class _TE:
    def __init__(self):
        self.model = _Cyclic()

    def unload(self):
        self.model = None                       # what Flux 3's / Anima's / ACE-Step's encoders do


def test_every_driver_unload_frees_a_cyclic_encoder():
    from fizgig.families.driver import FamilyDriver

    class Drv(FamilyDriver):
        def unload_text_encoder(self, te):
            te.unload()

    gc.disable()                                # the collector must not happen to run in between
    try:
        te = _TE()
        ref = weakref.ref(te.model.lin.weight)
        Drv().unload_text_encoder(te)
        assert ref() is None, "the encoder's weights outlived its unload"
    finally:
        gc.enable()


def test_train_releases_after_dropping_the_encoder():
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "families", "train.py"),
               encoding="utf-8").read()
    assert src.count("driver.unload_text_encoder(te)") == src.count("release_memory()       # the encoder's cycle")
