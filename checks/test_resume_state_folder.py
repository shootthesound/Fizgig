"""#352: Resume Training must name the state folder itself, and the LoRA output folder must not be a state folder -
both are refused at Start with which folder to pick, instead of the trainer failing on "not a saved training state"."""
import os

from fizgig.families import checks


def _state(d, name):
    s = d / name
    s.mkdir()
    for f in ("lora.safetensors", "optimizer.pt", "training_state.json"):
        (s / f).write_bytes(b"x")
    return s


def _run(v):
    """Only the resume / output folder problems (the rest of checks.run wants a whole form)."""
    out = checks.run(v, blocks_swap=0, swap_max=10, arch_label="x", name_error=None)
    return [p for p in out if "Resume" in p or "output folder" in p]


def test_the_folder_holding_the_states_names_the_newest(tmp_path):
    old = _state(tmp_path, "my_lora-000040-state")
    new = _state(tmp_path, "my_lora-000050-state")
    os.utime(old, (1, 1))
    out = _run({"LORA_OUTPUT_DIR": str(tmp_path), "RESUME_TRAINING": str(tmp_path)})
    assert len(out) == 1 and "not a state itself" in out[0] and new.name in out[0]
    assert _run({"LORA_OUTPUT_DIR": str(tmp_path), "RESUME_TRAINING": str(new)}) == []


def test_a_folder_with_no_states(tmp_path):
    out = _run({"LORA_OUTPUT_DIR": str(tmp_path), "RESUME_TRAINING": str(tmp_path)})
    assert len(out) == 1 and "isn't a saved training state" in out[0]


def test_a_state_folder_as_the_output_folder(tmp_path):
    s = _state(tmp_path, "my_lora-000050-state")
    out = _run({"LORA_OUTPUT_DIR": str(s), "RESUME_TRAINING": str(tmp_path)})
    assert any("output folder is a saved training state" in p and str(tmp_path) in p for p in out)


def test_a_fine_tune_checkpoint_file_and_an_empty_box_pass(tmp_path):
    ck = tmp_path / "ft-000002.safetensors"
    ck.write_bytes(b"x")
    assert _run({"LORA_OUTPUT_DIR": str(tmp_path), "RESUME_TRAINING": str(ck)}) == []
    assert _run({"LORA_OUTPUT_DIR": str(tmp_path), "RESUME_TRAINING": ""}) == []
