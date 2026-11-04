"""The loss graph's trend views: epoch averages and the kohya-style running average."""
import json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.training.step_log import StepLog, read, loss_path, epoch_means, running_avg  # noqa: E402


def _rows(losses_by_epoch):
    rows, s = [], 0
    for e, ls in enumerate(losses_by_epoch, 1):
        for l in ls:
            s += 1
            rows.append([s, e, l, None, None])
    return rows


def test_epoch_means():
    assert epoch_means(_rows([[1, 3], [2, 2, 2]])) == [[2, 1, 2.0], [5, 2, 2.0]]


def test_running_avg_kohya():
    r = running_avg(_rows([[4, 2], [1, 1]]))
    # epoch 1: cumulative 4, 3; epoch 2 replaces position 0 then 1: (1+2)/2, (1+1)/2
    assert [x[1] for x in r] == [4.0, 3.0, 1.5, 1.0]


def test_step_lines_and_old_logs(tmp_path):
    lg = StepLog(str(tmp_path), "x")
    for i in range(1, 5):
        lg.add(i, 1, 0.1 * i, lr=1e-4)
    lg.flush()
    d = read(loss_path(str(tmp_path), "x"))
    assert len(d["points"]) == 4 and len(d["points"][0]) == 4 and "val" not in d
    assert "t" not in json.loads(open(loss_path(str(tmp_path), "x")).readline())   # step lines as before
    p = loss_path(str(tmp_path), "old")
    open(p, "w").write('{"s": 1, "e": 1, "l": 0.08, "lr": 0.0001}\n{"s": 2, "e": 1, "l": 0.07}\n')
    assert len(read(p)["epoch_mean"]) == 1
