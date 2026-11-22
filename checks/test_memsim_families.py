"""Simulated GPU memory (memsim.py), every training family: its default launch - Auto's precision and block swap,
previews on, 1 MP (ACE-Step: two 4-minute songs) - on simulated 24 and 16 GB cards (what a Windows desktop leaves of
them), with the margin (memsim.margin) to spare. A change that makes a family's plan run out on a card it fits today
fails here.

Several minutes for all of them, so they run when FIZGIG_MEMSIM=1 (the nightly sets it); Klein's quicker set in
test_memsim_klein.py runs every time. Needs a torch built with CUDA and an NVIDIA card present (nothing runs on it)."""
import os

import pytest

from memsim import CAN_RUN, HEADERS, run_launch

pytestmark = [
    pytest.mark.skipif(not CAN_RUN, reason="the simulated card needs a torch built with CUDA and an NVIDIA card "
                                           "present (nothing runs on it)"),
    pytest.mark.skipif(os.environ.get("FIZGIG_MEMSIM") != "1",
                       reason="every family's simulated launch takes several minutes: set FIZGIG_MEMSIM=1")]

DESKTOP_GB = 1.3


def _families():
    from fizgig.families.registry import training_families
    return [d.key for d in training_families()
            if any(n.startswith(f"{d.key}_") for n in os.listdir(HEADERS))]


def _say(r):
    stages = ", ".join(f"{k} {a:.2f} / {v:.2f} GB" for k, (a, v) in r.stages.items())
    return f"{stages} (allocated / reserved, of {r.usable:.2f}); error: {r.error!r}"


def _helpers_missing(family):
    """ACE-Step's tokenizer and silence latent come from Hugging Face at run time (no download in a check)."""
    if family != "acestep15xl":
        return False
    from huggingface_hub import try_to_load_from_cache
    from fizgig.acestep.driver import HELPER, SILENCE
    return not isinstance(try_to_load_from_cache(HELPER, SILENCE), str)


@pytest.mark.parametrize("card", [24, 16])
@pytest.mark.parametrize("family", _families())
def test_default_launch_fits(tmp_path, family, card):
    if _helpers_missing(family):
        pytest.skip("ACE-Step's helper files (tokenizer, silence latent) aren't in the Hugging Face cache")
    r = run_launch(family, tmp_path, {"SAMPLE_ENABLED": True}, card_gb=card, others_gb=DESKTOP_GB)
    if any("nothing fits" in ln for ln in r.lines):
        pytest.skip(f"Auto's own plan says nothing fits a {card} GB card")
    assert r.error is None and r.ooms == 0, _say(r)
    assert not [ln for ln in r.lines if "preview failed" in ln], _say(r)
