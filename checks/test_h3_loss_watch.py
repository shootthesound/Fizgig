"""MiniMax H3 video has Adaptive LR and the per-image loss watch (18 Nov 2026), and a caption the watch re-encodes
mid-run (a Problem Images edit, auto-recaption) is written in H3's own text cache layout - the hidden_states its
batch_cond reads - not the shared cond__ file, which would leave the next step without its text."""
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from fizgig.families import form
from fizgig.families.registry import get
from fizgig.minimax.driver import MiniMaxDriver


def test_h3_offers_adaptive_lr_and_the_loss_watch():
    desc = get("minimax")
    assert desc.adaptive_lr and desc.loss_watch
    keys = {r["key"] for s in form.form(desc)["sections"] for r in s["rows"]}
    assert {"ADAPTIVE_LR", "KREA2_LOSS_WATCH", "KREA2_PER_IMAGE_LR"} <= keys


def test_a_reencoded_caption_keeps_h3s_layout(tmp_path):
    item = SimpleNamespace(text_encoder_output_cache_path=str(tmp_path / "clip_minimaxh3_te.safetensors"),
                           caption="a new caption", item_key="clip")
    hidden = torch.randn(7, 16, dtype=torch.bfloat16)
    MiniMaxDriver.save_text_cache(None, item, {"hidden_states": hidden})
    sd = load_file(item.text_encoder_output_cache_path)
    assert set(sd) == {"hidden_states", "attention_mask"}
    assert torch.equal(sd["hidden_states"], hidden) and bool(sd["attention_mask"].all())
    batch = {k: v[None] for k, v in sd.items()}
    driver = MiniMaxDriver.__new__(MiniMaxDriver)
    driver._uncond, driver.options = None, {}
    assert torch.equal(driver.batch_cond(batch, "cpu")["hidden_states"][0], hidden)   # what training reads
