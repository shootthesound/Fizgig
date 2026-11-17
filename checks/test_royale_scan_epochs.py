"""LoRA Royale's scanner reads the epoch a Save now snapshot or a run's last checkpoint was saved at (the trainer's
ss_epoch). The last checkpoint holds the final epoch: when that epoch's own file is there it is the same weights and
is left out, not rendered twice and always picked by Max renders' "last one in"; when the run didn't save that epoch
it is listed as that epoch. A snapshot sits between the epochs it was taken in, in time order."""
import json
import struct

from fizgig.lora_royale.scan import scan_checkpoints
from fizgig.web.royale import promote, scan


def ckpt(folder, name, epoch=None):
    """A checkpoint file with only a header: no tensors, the trainer's ss_epoch when given."""
    head = json.dumps({"__metadata__": {"ss_epoch": str(epoch)}} if epoch else {}).encode()
    (folder / name).write_bytes(struct.pack("<Q", len(head)) + head)


def test_the_last_checkpoint_of_a_finished_run_is_its_final_epoch_once(tmp_path):
    for e in (1, 2, 3):
        ckpt(tmp_path, f"mylora-{e:06d}.safetensors", e)
    ckpt(tmp_path, "mylora.safetensors", 3)
    assert [l for l, _ in scan_checkpoints(str(tmp_path))] == [1, 2, 3]
    assert scan(str(tmp_path))["summary"] == "3 checkpoints found (epochs 1–3)."


def test_a_final_epoch_not_saved_on_its_own_is_listed_as_that_epoch(tmp_path):
    for e in (5, 10):
        ckpt(tmp_path, f"mylora-{e:06d}.safetensors", e)
    ckpt(tmp_path, "mylora.safetensors", 12)
    got = scan_checkpoints(str(tmp_path))
    assert [l for l, _ in got] == [5, 10, 12] and got[-1][1].endswith("mylora.safetensors")


def test_a_snapshot_sits_between_the_epochs_it_was_taken_in(tmp_path):
    for e in (1, 2, 3, 4):
        ckpt(tmp_path, f"mylora-{e:06d}.safetensors", e)
    ckpt(tmp_path, "mylora-step0000500.safetensors", 3)          # Save now during epoch 3
    assert [l for l, _ in scan_checkpoints(str(tmp_path))] == [1, 2, "step 500", 3, 4]
    assert scan(str(tmp_path))["summary"] == "5 checkpoints found (epochs 1–4, plus 1 Save now snapshot)."


def test_a_promoted_snapshot_keeps_the_run_name(tmp_path):
    ckpt(tmp_path, "mylora-step0000500.safetensors", 3)
    out = promote(str(tmp_path / "mylora-step0000500.safetensors"), "step 500", "mylora", str(tmp_path / "picks"))
    assert out.replace("\\", "/").endswith("picks/mylora-step500-pick.safetensors")
