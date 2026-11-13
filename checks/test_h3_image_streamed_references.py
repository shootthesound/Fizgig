"""MiniMax H3 Image, an edit's before-pictures on the streamed text encoder (a card too small to hold it whole): the
encode goes through the encoder's own layer-by-layer forward. Calling the model whole met layers still in system RAM
and failed on the first picture."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.minimax_image.driver import MiniMaxH3ImageDriver  # noqa: E402


class _Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [7] * max(1, len(text.split()))}

    def convert_tokens_to_ids(self, token):
        return {"<|vision_start|>": 1, "<|vision_end|>": 2, "<|image_pad|>": 3}[token]


class _Whole:
    def __call__(self, **kw):
        raise AssertionError("the streamed encoder's model was called whole")


class _Streamed:
    device = "cpu"

    def __init__(self):
        self.tokenizer, self.model, self.calls = _Tok(), _Whole(), []

    def _reference_forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        self.calls.append((input_ids.shape, None if pixel_values is None else pixel_values.shape, image_grid_thw))
        return torch.zeros(1, input_ids.shape[1], 8)


def test_streamed_encoder_takes_its_own_forward():
    drv, te = MiniMaxH3ImageDriver.__new__(MiniMaxH3ImageDriver), _Streamed()
    picture = np.zeros((64, 96, 3), dtype=np.uint8)
    out = drv.encode_text_with_references(te, ["a red hat", "plain"], [[picture], []])
    (ids, pix, grid), (ids2, pix2, grid2) = te.calls
    pads = int(grid[0].prod()) // 4
    assert ids[1] == out[0]["hidden_states"].shape[0] == out[0]["token_tags"].shape[0]
    assert pix[0] == int(grid[0].prod()) and int((out[0]["token_tags"] != out[0]["token_tags"][0]).sum()) == pads + 2
    assert pix2 is None and grid2 is None and ids2[1] == 1            # no picture: the caption alone, same forward
