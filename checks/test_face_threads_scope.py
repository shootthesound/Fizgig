"""PR #382 fix-up: the face scorer's CPU thread limit stays in its own sessions. Importing fizgig must not put
OMP_NUM_THREADS into the environment, which every process the server starts (training, caching, captioning, the
Studio worker) would inherit."""
import importlib

import fizgig


def test_import_leaves_omp_alone(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    importlib.reload(fizgig)
    import os
    assert "OMP_NUM_THREADS" not in os.environ
