"""A run's checkpoints in a folder of thousands of LoRAs (a ComfyUI loras folder): only that run's headers are read
(the Runs page's Previews & checkpoints sat on Loading for minutes), the rest counted; a file's kind is kept for the
session."""
import pytest

server = pytest.importorskip("fizgig.web.server")


class H:
    def json(self, obj, code=200):
        return obj


def test_only_the_runs_files_are_read(tmp_path, monkeypatch):
    for n in ("mine-000001.safetensors", "mine.safetensors", "other-000002.safetensors", "third.safetensors"):
        (tmp_path / n).write_bytes(b"\x00" * 8)
    import os, time
    old = time.time() - 60
    for f in tmp_path.iterdir():
        os.utime(f, (old, old))
    read = []
    monkeypatch.setattr(server, "_read_checkpoint_kind", lambda p: read.append(os.path.basename(p)) or "lora")
    r = server.Handler._checkpoints(H(), str(tmp_path), "mine")
    assert sorted(i["name"] for i in r["items"]) == ["mine-000001.safetensors", "mine.safetensors"]
    assert r["others"] == 2 and sorted(read) == ["mine-000001.safetensors", "mine.safetensors"]
    server.Handler._checkpoints(H(), str(tmp_path), "mine")
    assert len(read) == 2                                  # kept: not read again
