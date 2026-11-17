"""#366: a caption model still downloading or loading can be stopped, and the Captions tab says which it is doing.
Stop only acted between pictures and Unload was greyed while the model loaded, so a load that took long (a first
download, or 19 GB going into a small card) held its VRAM with no way out; and one message ("Loading..., the first
time it downloads") covered both the download and the load, so a finished download still read as unfinished. The
worker now says when it downloads, and Stop / Unload end a load at once."""
import subprocess
import sys
import threading
import time

from fizgig.web import captions


def _wait(cond, s=5.0):
    end = time.time() + s
    while time.time() < end and not cond():
        time.sleep(0.05)
    return cond()


def test_a_loading_caption_model_says_so_and_stops_at_once():
    fake = ("import time; print('INFO: Downloading JoyCaption Beta One (the first time only)...', flush=True); "
            "time.sleep(120)")                       # a worker that never gets to READY
    proc = subprocess.Popen([sys.executable, "-c", fake], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.PIPE, text=True, bufsize=1)
    try:
        with captions._lock:
            captions._st.update(proc=proc, key=("test",), state="loading", model="JoyCaption Beta One",
                                ready=threading.Event(), phase="loading", job="captions")
        threading.Thread(target=captions._reader, args=(proc,), daemon=True).start()
        assert _wait(lambda: captions.status()["phase"] == "downloading")
        t = time.time()
        assert captions.stop() is True               # Stop during a load cancels it ...
        assert time.time() - t < 5                   # ... at once, not after QUIT's 20 s wait
        assert proc.wait(timeout=5) is not None
        assert _wait(lambda: captions.status()["state"] == "idle")
        assert not captions.status()["loaded"]
    finally:
        if proc.poll() is None:
            proc.kill()
