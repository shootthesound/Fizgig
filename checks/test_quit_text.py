"""The window's question on quitting while a run trains says what stops, what is kept and lost, what Fizgig asks at
its next start, and which button does what."""
from fizgig.web.app_window import quit_text


def test_saved_with_a_queue():
    t = quit_text({"lora": "sixM", "saved": 12, "of": 40, "queued": 2})
    assert "saved at epoch 12 of 40" in t and "carry sixM on from epoch 12" in t
    assert "2 queued runs" in t and "clear the queue" in t
    assert "OK: quit Fizgig and stop the run." in t and "Cancel: keep Fizgig open" in t and "Pause" in t


def test_saved_no_queue():
    t = quit_text({"lora": "sixM", "saved": 3, "of": 0, "queued": 0})
    assert "saved at epoch 3:" in t and "or start fresh" in t and "queued" not in t


def test_nothing_saved():
    assert "not saved an epoch yet" in quit_text({"lora": "sixM", "queued": 1})
    assert "1 queued run " in quit_text({"lora": "sixM", "queued": 1})
    assert "opens as usual" in quit_text({"lora": "sixM"})
