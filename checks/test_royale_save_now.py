"""LoRA Royale's scanner against the web UI's "Save now" and the trainer's last checkpoint: a snapshot of the
run, or its bare "<run>.safetensors", is one of its checkpoints - not a stray that demotes the whole folder
to filename labelling. A snapshot of some *other* run, or a file named after no run here, still demotes it.
"""
from fizgig.lora_royale.scan import scan_checkpoints
from fizgig.web.royale import scan


def write(folder, *names):
    for n in names:
        (folder / n).write_bytes(b"")
    return str(folder)


def test_a_snapshot_joins_its_run(tmp_path):
    d = write(tmp_path, "mylora-000001.safetensors", "mylora-000002.safetensors", "mylora-000003.safetensors",
              "mylora-step0004064.safetensors")
    got = scan_checkpoints(d)
    assert [l for l, _ in got] == [1, 2, 3, "step 4064"]
    assert got[-1][1].endswith("mylora-step0004064.safetensors")


def test_epochs_alone_are_unchanged(tmp_path):
    d = write(tmp_path, "mylora-000003.safetensors", "mylora-000001.safetensors", "mylora-000002.safetensors")
    assert [l for l, _ in scan_checkpoints(d)] == [1, 2, 3]


def test_several_snapshots_come_after_their_epochs_in_order(tmp_path):
    d = write(tmp_path, "mylora-000001.safetensors", "mylora-step0004064.safetensors",
              "mylora-step0000128.safetensors")
    assert [l for l, _ in scan_checkpoints(d)] == [1, "step 128", "step 4064"]



def test_the_last_checkpoint_joins_its_run(tmp_path):
    d = write(tmp_path, "mylora-000001.safetensors", "mylora-000002.safetensors", "mylora-000003.safetensors",
              "mylora.safetensors")
    got = scan_checkpoints(d)
    assert [l for l, _ in got] == [1, 2, 3, "last"]
    assert got[-1][1].endswith("mylora.safetensors")


def test_the_last_checkpoint_follows_the_snapshots(tmp_path):
    d = write(tmp_path, "mylora-000001.safetensors", "mylora-000002.safetensors",
              "mylora-step0004064.safetensors", "mylora.safetensors")
    assert [l for l, _ in scan_checkpoints(d)] == [1, 2, "step 4064", "last"]


def test_a_bare_file_of_no_run_here_still_demotes(tmp_path):
    d = write(tmp_path, "mylora-000001.safetensors", "mylora-000002.safetensors", "other.safetensors")
    assert [l for l, _ in scan_checkpoints(d)] == ["mylora-000001", "mylora-000002", "other"]


def test_a_stray_still_demotes_the_run(tmp_path):
    d = write(tmp_path, "mylora-000001.safetensors", "mylora-000002.safetensors", "someone-else.safetensors")
    assert [l for l, _ in scan_checkpoints(d)] == ["mylora-000001", "mylora-000002", "someone-else"]


def test_a_snapshot_of_another_run_still_demotes(tmp_path):
    d = write(tmp_path, "mylora-000001.safetensors", "mylora-000002.safetensors", "other-step0001000.safetensors")
    assert [l for l, _ in scan_checkpoints(d)] == ["mylora-000001", "mylora-000002", "other-step0001000"]


def test_the_summary_names_the_snapshots(tmp_path):
    d = write(tmp_path, "mylora-000001.safetensors", "mylora-000002.safetensors", "mylora-step0004064.safetensors")
    r = scan(d)
    assert r["summary"] == "3 checkpoints found (epochs 1–2, plus 1 Save now snapshot)."
    assert r["run"] == "mylora"


def test_the_summary_counts_the_last_checkpoint(tmp_path):
    d = write(tmp_path, "mylora-000001.safetensors", "mylora-000002.safetensors", "mylora.safetensors")
    r = scan(d)
    assert r["summary"] == "3 checkpoints found (epochs 1–2, plus 1 last checkpoint)."
    assert r["run"] == "mylora"


def test_the_summary_counts_both_extras(tmp_path):
    d = write(tmp_path, "mylora-000001.safetensors", "mylora-000002.safetensors",
              "mylora-step0004064.safetensors", "mylora.safetensors")
    assert scan(d)["summary"] == "4 checkpoints found (epochs 1–2, plus 1 Save now snapshot and 1 last checkpoint)."


def test_the_summary_is_unchanged_without_snapshots(tmp_path):
    d = write(tmp_path, "mylora-000001.safetensors", "mylora-000060.safetensors")
    assert scan(d)["summary"] == "2 checkpoints found (epochs 1–60)."


def test_an_unclassified_folder_still_reads_as_one(tmp_path):
    d = write(tmp_path, "style-a.safetensors", "style-b.safetensors")
    assert scan(d)["summary"] == "2 LoRAs found — compared by filename."
