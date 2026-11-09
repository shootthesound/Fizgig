"""Trigger-word attention for the training maps: how much of each picture token's attention goes to the caption's
trigger word, averaged over heads and blocks.

A model with the `trigger_attention` ability (families/description.py) calls two hooks, both no-ops unless a probe is
active (a training-maps capture, run eagerly):
  * layout(img, txt, total, valid, kv_total) - in its forward, once the joint sequence is built: the picture's query
    rows [img[0], img[1]), where the text's keys start, the joint length its attention sees, and how many keys are
    real (a padded tail past `valid` is ignored). Cross-attention (the picture reads a separate text sequence: Fibo
    2.0's injection blocks, Anima, SDXL) gives kv_total, the keys' own length, or -1 for any;
  * see(q, k, key_mask) - in its attention, with q and k as attention uses them ([B, H, L, D], after QK-norm and RoPE;
    k may have fewer heads, GQA). Calls whose length isn't the joint one (a text-only refiner) are skipped.

The probe works out softmax(q k^T / sqrt(d)) itself, over every key (the trigger's share of each row needs the whole
row's normaliser), a few heads at a time, and keeps only the trigger's columns, summed. The driver says where the
trigger sits in its text (driver.trigger_text: the prompt as its encoder builds it, and the part it drops).
"""
import re
from contextlib import contextmanager

import torch

_ACTIVE = None
HEADS_PER_CHUNK = 4


class TriggerProbe:
    """positions: a list (one trigger), or {trigger word: positions} (several, each kept on its own as well as all
    together under "")."""

    def __init__(self, positions):
        groups = dict(positions) if isinstance(positions, dict) else {"": positions}
        self.groups = {w: sorted({int(p) for p in ps}) for w, ps in groups.items()}
        if "" not in self.groups:
            self.groups[""] = sorted({p for ps in self.groups.values() for p in ps})
        self.pos = self.groups[""]
        self.layout = None
        self.acc = None
        self.n = 0

    def see(self, q, k, key_mask=None):
        if self.layout is None or not self.pos:
            return
        (a, b), t0, total, valid, kv_total = self.layout
        if q.shape[-2] != total or (kv_total != -1 and k.shape[-2] != (total if kv_total is None else kv_total)):
            return
        if kv_total is not None:                    # cross-attention: every key is the text's
            valid = k.shape[-2]
        cols = {w: [t0 + p for p in ps if t0 + p < valid] for w, ps in self.groups.items()}
        if not cols[""]:
            return
        qi = q[0, :, a:b].float()                                   # [H, Ni, D]
        kk = k[0, :, :valid].float()                                # [Hk, valid, D]
        if kk.shape[0] != qi.shape[0]:                              # GQA: each key head serves a run of query heads
            kk = kk.repeat_interleave(qi.shape[0] // kk.shape[0], dim=0)
        km = None
        if key_mask is not None and key_mask.dtype == torch.bool:
            km = key_mask.reshape(key_mask.shape[0], -1)[0, :valid]
        scale = qi.shape[-1] ** -0.5
        out = {w: torch.zeros(qi.shape[1], device=qi.device) for w, c in cols.items() if c}
        for h in range(0, qi.shape[0], HEADS_PER_CHUNK):
            s = qi[h:h + HEADS_PER_CHUNK] @ kk[h:h + HEADS_PER_CHUNK].transpose(-1, -2) * scale     # [h, Ni, valid]
            if km is not None:
                s.masked_fill_(~km, float("-inf"))
            lse = torch.logsumexp(s, dim=-1, keepdim=True)
            for w in out:
                out[w] += (s[..., cols[w]] - lse).exp().sum(-1).sum(0)
            del s, lse
        for w in out:
            out[w] /= qi.shape[0]
        if self.acc is None:
            self.acc = out
        else:
            for w in out:
                self.acc[w] = self.acc[w] + out[w] if w in self.acc else out[w]
        self.n += 1

    def result(self):
        """{trigger word: its mean share of each picture token's attention over the blocks seen} with "" for all
        of them together, or None."""
        return None if self.acc is None or not self.n else {w: a / self.n for w, a in self.acc.items()}


def layout(img, txt, total, valid=None, kv_total=None):
    p = _ACTIVE
    if p is not None:
        p.layout = ((int(img[0]), int(img[1])), int(txt), int(total), int(total if valid is None else valid),
                    None if kv_total is None else int(kv_total))


def see(q, k, key_mask=None):
    p = _ACTIVE
    if p is not None:
        p.see(q, k, key_mask)


@contextmanager
def capturing(positions):
    """A probe for the trigger's text positions, active for the forward inside. Run it eagerly: a compiled block
    doesn't call back into Python."""
    global _ACTIVE
    probe = TriggerProbe(positions)
    _ACTIVE = probe
    try:
        yield probe
    finally:
        _ACTIVE = None


def trigger_positions(tokenizer, full, dropped, caption, trigger):
    """The trigger's token positions in the text the model sees: `full` is the prompt as the encoder builds it (with
    the caption inside), `dropped` the leading part the encoder throws away (its tokens don't reach the model).
    Every whole-word occurrence of each trigger (a comma list is several) counts. [] when it isn't in the caption."""
    if not caption or not trigger:
        return []
    at = full.find(caption)
    if at < 0:
        return []
    spans = []
    for word in [w.strip() for w in str(trigger).split(",") if w.strip()]:
        for m in re.finditer(r"(?<!\w)" + re.escape(word) + r"(?!\w)", caption, flags=re.IGNORECASE):
            spans.append((at + m.start(), at + m.end()))
    if not spans:
        return []
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    drop = len(tokenizer(dropped, add_special_tokens=False)["input_ids"]) if dropped else 0
    out = []
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if i >= drop and e > s and any(e > a and s < b for a, b in spans):
            out.append(i - drop)
    return out
