"""Delete run's "Delete, keep the newest 5 epochs": everything Delete takes except the newest 5 epochs (all of them
when fewer are left) - their checkpoints and previews stay, with the finished LoRA, a Save now's checkpoint and the
previews from before training; older epochs' checkpoints and previews and every saved state go."""
import os

import pytest

server = pytest.importorskip("fizgig.web.server")


def _run(tmp_path, epochs, states):
    (tmp_path / "sample").mkdir()
    for e in range(1, epochs + 1):
        (tmp_path / f"r-{e:06d}.safetensors").write_bytes(b"0" * 10)
        (tmp_path / "sample" / f"r_e{e:06d}_00_20261119120000_42.png").write_bytes(b"0")
    (tmp_path / "sample" / "r_e000000_00_20261119110000_42.png").write_bytes(b"0")      # before training
    for e in states:
        (tmp_path / f"r-{e:06d}-state").mkdir()
        (tmp_path / f"r-{e:06d}-state" / "lora.safetensors").write_bytes(b"0")
    (tmp_path / "r.safetensors").write_bytes(b"0")
    (tmp_path / "r-step0000123.safetensors").write_bytes(b"0")
    (tmp_path / "rr-000001.safetensors").write_bytes(b"0")          # another run's
    return server.run_outputs(str(tmp_path), "r")


names = lambda ps: sorted(os.path.basename(p) for p in ps)


def test_keeps_the_newest_five(tmp_path):
    plan = server.trim_plan(_run(tmp_path, 8, [6, 7, 8]))
    assert names(plan["checkpoints"]) == [f"r-{e:06d}.safetensors" for e in (1, 2, 3)]
    assert names(plan["states"]) == ["r-000006-state", "r-000007-state", "r-000008-state"]
    assert names(plan["previews"]) == [f"r_e{e:06d}_00_20261119120000_42.png" for e in (1, 2, 3)]
    assert plan["epochs"] == 8 and plan["kept"] == 5


def test_fewer_than_five_keeps_them_all_and_still_tidies(tmp_path):
    plan = server.trim_plan(_run(tmp_path, 3, [3]))
    assert plan["checkpoints"] == [] and plan["previews"] == [] and plan["kept"] == 3
    assert names(plan["states"]) == ["r-000003-state"]


def test_one_epoch_left_deletes_the_run_and_keeps_its_lora(tmp_path, monkeypatch):
    _run(tmp_path, 1, [])
    removed = []
    monkeypatch.setattr(server.history, "get", lambda rid: {"output_dir": str(tmp_path), "lora": "r"})
    monkeypatch.setattr(server.history, "remove", lambda rid: removed.append(rid))
    assert server.trim_run("x", confirm=True) == {"ok": True}
    assert removed == ["x"] and (tmp_path / "r-000001.safetensors").exists() and (tmp_path / "r.safetensors").exists()
