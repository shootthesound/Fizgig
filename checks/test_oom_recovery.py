"""Out-of-memory recovery (web/oom.py): what ran out is named from the exit code and the log; a GPU run with a save
carries on once, a repeat at about the same point goes one step lighter (block swap, then a precision down only from
Auto), no save restarts once lighter, a system-RAM kill with a save gets one Continue, at most MAX_TRIES tries; and a
failed run doesn't hold the queue."""
from types import SimpleNamespace

from fizgig.families import launch
from fizgig.web import oom
from fizgig.web import queue as webqueue


def _watch(*lines):
    w = oom.Watch()
    for ln in lines:
        w.feed(ln)
    return w


def test_what_ran_out():
    assert oom.classify(1, _watch("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")) == "gpu"
    assert oom.classify(1, _watch("RuntimeError: HIP out of memory")) == "gpu"
    assert oom.classify(1, _watch("RuntimeError: DefaultCPUAllocator: not enough memory: you tried to allocate")) \
        == "ram"
    assert oom.classify(1, _watch("OSError: [WinError 1455] The paging file is too small for this operation")) == "ram"
    assert oom.classify(1, _watch("MemoryError")) == "ram"
    assert oom.classify(-9, _watch("steps: 40%"), platform="linux") == "ram"          # the OOM killer: no message
    assert oom.classify(137, _watch(), platform="linux") == "ram"
    assert oom.classify(-9, _watch(), platform="win32") is None
    assert oom.classify(1, _watch("KeyError: 'x'")) is None
    assert oom.classify(0, _watch("CUDA out of memory")) is None                    # a preview's, survived
    assert oom.classify(-9, _watch("CUDA out of memory"), stopped=True) is None      # you stopped it


def test_the_plan_the_trainer_used():
    w = _watch("INFO:x:[precision] Auto plan: int8, block swap 6 (int8 fits); asked ('auto', -1)")
    assert w.plan == ("int8", 6)
    assert oom.used_plan("[precision] Auto plan: bf16, block swap 0 (x)\n[precision] Auto plan: nf4, block swap 3") \
        == ("nf4", 3)


DESC = SimpleNamespace(precisions=("bf16", "int8", "nf4"), precision_labels={})
LABELS = launch.PRECISION_LABELS


def _step(values, plan, max_swap=26):
    return oom.step_down(DESC, values, plan, max_swap, launch.precision_key, LABELS)


def test_one_step_lighter():
    auto = {"FAMILY_PRECISION": LABELS["auto"], "BLOCKS_SWAP": "Auto (detect from GPU)"}
    v, why = _step(auto, ("int8", 6))
    assert v["BLOCKS_SWAP"] == "10" and v["FAMILY_PRECISION"] == LABELS["int8"] and why == "block swap 6 -> 10"
    v, why = _step({**auto, "BLOCKS_SWAP": "26"}, ("int8", 26))                       # swap at its most: precision
    assert v["FAMILY_PRECISION"] == LABELS["nf4"] and why == "precision int8 -> nf4"
    assert _step(auto, ("nf4", 26))[0] is None                                       # nothing lighter
    v, why = _step(auto, ("nf4", 0))                                                 # NF4 can't stream: no swap step
    assert v is None and "can't swap blocks" in why
    mine = {"FAMILY_PRECISION": LABELS["bf16"], "BLOCKS_SWAP": "26"}
    v, why = _step(mine, None)
    assert v is None and "your own choice" in why                                    # a precision you picked stays
    v, why = _step({"FAMILY_PRECISION": LABELS["bf16"], "BLOCKS_SWAP": "4"}, None, max_swap=16)
    assert v["BLOCKS_SWAP"] == "6" and v["FAMILY_PRECISION"] == LABELS["bf16"]       # step 16/8 = 2


def _decide(kind, tries=(), save=True, step=100, per_epoch=20, ft=False, light=("v", "block swap 6 -> 10")):
    return oom.decide(kind, list(tries), save, step, per_epoch, ft, lambda: light)


def test_what_it_does():
    assert _decide("gpu")["do"] == "continue" and not _decide("gpu")["lighter"]      # fresh memory first
    near = [{"kind": "gpu", "step": 95}]
    assert _decide("gpu", near) == {"do": "continue", "lighter": True,
                                     "why": "out again at about the same point: one step lighter"}
    far = [{"kind": "gpu", "step": 40}]
    assert _decide("gpu", far)["do"] == "continue" and not _decide("gpu", far)["lighter"]
    assert _decide("gpu", save=False)["do"] == "restart" and _decide("gpu", save=False)["lighter"]
    assert _decide("gpu", [{"kind": "gpu", "step": 3}], save=False)["do"] == "stop"
    assert _decide("gpu", [{"kind": "gpu"}] * oom.MAX_TRIES)["do"] == "stop"          # at most MAX_TRIES
    assert _decide("ram")["do"] == "continue" and not _decide("ram")["lighter"]      # one Continue, same settings
    assert _decide("ram", [{"kind": "ram", "step": 100}])["do"] == "stop"
    assert _decide("ram", save=False)["do"] == "stop"
    assert _decide("gpu", near, ft=True)["do"] == "stop"                              # a fine-tune: no swap to raise
    assert _decide(None)["do"] == "stop"


def test_advice_and_summary():
    assert "fewer blocks" in oom.advice("ram", True, "epoch 4") and "epoch 4" in oom.advice("ram", True, "epoch 4")
    assert "Continue" in oom.advice("gpu", True, "epoch 4") and "smaller picture size" in oom.advice("gpu", False)
    s = oom.summary([{"kind": "gpu", "step": 100, "action": "Continue (same settings)"},
                     {"kind": "gpu", "step": 104, "action": "Continue (block swap 6 -> 10)"}])
    assert "1. GPU ran out at step 100" in s and "2. GPU ran out at step 104: Continue (block swap 6 -> 10)" in s
    assert "ran out before training's first step" in oom.summary([{"kind": "gpu", "step": None, "action": "x"}])


def test_a_failed_run_moves_the_queue_on(monkeypatch):
    calls = []
    monkeypatch.setattr(webqueue, "schedule", lambda delay, starter: calls.append(delay))
    monkeypatch.setitem(webqueue._q, "failed", [])
    webqueue.on_run_ended("failed", "pixie_a", lambda item: (True, ""))
    assert calls == [10] and "pixie_a failed" in webqueue.status()["failed"]
    assert not webqueue.status()["held"]
