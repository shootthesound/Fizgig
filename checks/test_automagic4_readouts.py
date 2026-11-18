"""#349 / to-do #11: Automagic v4's braked share reaches the run card (the bar's brk=, parsed like lr= and gn=) and the
step log, and the epoch log line names each group's sensitivity reference. CPU only, a tiny model."""
import json

import torch

from fizgig.training import optimizers as O
from fizgig.training.progress import _POSTFIX_RE
from fizgig.training.step_log import StepLog




def test_braked_share_and_references():
    from fizgig.training.automagic4 import Automagic4
    torch.manual_seed(0)
    a, b = torch.nn.Linear(16, 16), torch.nn.Linear(16, 8)
    opt = Automagic4([{"params": list(a.parameters()), "lr": 2.5e-4, "family": "attn"},
                      {"params": list(b.parameters()), "lr": 2.5e-4, "family": "mlp"}], lr=2.5e-4)
    assert O.automagic4_braked(opt) is None                     # nothing read yet
    for _ in range(12):
        opt.zero_grad()
        (a(torch.randn(4, 16)).pow(2).mean() + b(torch.randn(4, 16)).pow(2).mean()).backward()
        opt.step()
    brakes = O.automagic4_brakes(opt)
    share = O.automagic4_braked(opt)
    assert share is not None and 0.0 <= share <= 1.0
    assert abs(share - sum(s for _b, s in brakes) / len(brakes)) < 1e-9     # the cached reading, no second pass
    refs = O.automagic4_refs(opt)
    assert refs is not None and len(refs) == 2
    line = O.automagic4_detail(opt)
    assert "% of weights braked" in line and (("sensitivity references attn" in line) == (refs[0] is not None))


def test_other_optimizers_show_nothing():
    p = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([p], lr=1e-4)
    assert O.automagic4_braked(opt) is None and O.automagic4_refs(opt) is None and O.automagic4_detail(opt) == ""


def test_bar_and_step_log_carry_it(tmp_path):
    assert dict(_POSTFIX_RE.findall(" avr_loss=0.1, lr=2.50e-04, gn=0.31, brk=63]")) == \
        {"lr": "2.50e-04", "gn": "0.31", "brk": "63"}
    log = StepLog(str(tmp_path), "run")
    log.add(20, 1, 0.5, 2.5e-4, 0.63)
    log.add(21, 1, 0.4)
    log.flush()
    recs = [json.loads(x) for x in open(log.path, encoding="utf-8")]
    assert recs[0]["b"] == 0.63 and "b" not in recs[1]
