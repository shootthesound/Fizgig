"""To-do #13 (#308): a training folder downloaded onto a pod from a link - which links it takes, the disk check, and a
zip fetched over HTTP (resumed from a half-downloaded .part), unpacked and removed. CPU only, a local web server."""
import functools
import http.server
import os
import subprocess
import sys
import threading
import zipfile

import pytest

from fizgig.scripts.fetch_dataset import parse
from fizgig.web.pod_dataset import _name, disk_problem


def test_links():
    assert parse("https://huggingface.co/datasets/org/set") == ("hf", ("org/set", ""))
    assert parse("https://huggingface.co/datasets/org/set/tree/main/photos/a") == ("hf", ("org/set", "photos/a"))
    assert parse("hf:org/set/photos") == ("hf", ("org/set", "photos"))
    assert parse("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view?usp=sharing") == ("drive", "1AbCdEfGhIjKlMnOp")
    assert parse("https://example.com/data/faces.tar.gz?x=1") == ("url", "https://example.com/data/faces.tar.gz?x=1")
    with pytest.raises(ValueError, match="zip the folder"):
        parse("https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOp")
    with pytest.raises(ValueError):
        parse("not a link")
    assert _name("https://example.com/data/faces.tar.gz") == "faces"
    assert _name("https://huggingface.co/datasets/org/set/tree/main/photos") == "photos"
    assert _name("hf:org/set") == "set"


def test_disk_check():
    assert disk_problem(10e9, False, 20e9) == ""
    assert "unpacking it takes as much again" in disk_problem(10e9, True, 15e9)
    assert disk_problem(None, True, 1) == ""          # unknown size (Drive): no refusal


def test_zip_over_http_resumes_and_unpacks(tmp_path):
    src = tmp_path / "srv"
    src.mkdir()
    with zipfile.ZipFile(src / "set.zip", "w") as z:
        for i in range(3):
            z.writestr(f"set/pic{i}.txt", f"caption {i}" * 200)
    data = (src / "set.zip").read_bytes()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(src))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "set.zip.part").write_bytes(data[:100])   # a download that stopped part way
        url = f"http://127.0.0.1:{srv.server_address[1]}/set.zip"
        r = subprocess.run([sys.executable, "-m", "fizgig.scripts.fetch_dataset", url, str(dest)],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.path.abspath("src")})
        assert r.returncode == 0, r.stdout + r.stderr
        assert sorted(os.listdir(dest)) == ["pic0.txt", "pic1.txt", "pic2.txt"]   # one top folder: its contents
        assert (dest / "pic2.txt").read_text() == "caption 2" * 200
    finally:
        srv.shutdown()
