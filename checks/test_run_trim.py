"""Delete run's "Keep the newest 5 epochs": the older epochs' checkpoints and saved states go; the finished LoRA, the
newest 5 epochs, the newest saved state, a Save now's checkpoint and the previews stay. Fewer than 5: nothing goes."""
import os

import pytest

server = pytest.importorskip("fizgig.web.server")


def _run(tmp_path, epochs, states):
    for e in range(1, epochs + 1):
        (tmp_path / f"r-{e:06d}.safetensors").write_bytes(b"0" * 10)
    for e in states:
        (tmp_path / f"r-{e:06d}-state").mkdir()
        (tmp_path / f"r-{e:06d}-state" / "lora.safetensors").write_bytes(b"0")
    (tmp_path / "r.safetensors").write_bytes(b"0")
    (tmp_path / "r-step0000123.safetensors").write_bytes(b"0")
    (tmp_path / "rr-000001.safetensors").write_bytes(b"0")          # another run's
    return server.run_outputs(str(tmp_path), "r")


def test_keeps_the_newest_five(tmp_path):
    plan = server.trim_plan(_run(tmp_path, 8, [6, 7, 8]))
    assert sorted(os.path.basename(p) for p in plan["checkpoints"]) == [f"r-{e:06d}.safetensors" for e in (1, 2, 3)]
    assert sorted(os.path.basename(p) for p in plan["states"]) == ["r-000006-state", "r-000007-state"]
    assert plan["epochs"] == 8 and plan["bytes"] == 32


def test_fewer_than_five_is_nothing(tmp_path):
    plan = server.trim_plan(_run(tmp_path, 3, [3]))
    assert plan["checkpoints"] == [] and plan["states"] == [] and plan["epochs"] == 3
