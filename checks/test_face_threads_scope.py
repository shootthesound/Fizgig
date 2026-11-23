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


def test_ort_session_swap_is_scoped():
    """The thread-capping InferenceSession is in place only inside limited_ort_sessions() (InsightFace's setup), and a
    thread count a caller set below the cap is kept."""
    import pytest
    ort = pytest.importorskip("onnxruntime")
    from fizgig import face_utils
    base = ort.InferenceSession
    with face_utils.limited_ort_sessions():
        inside = ort.InferenceSession
    assert inside is not base and issubclass(inside, base)
    assert ort.InferenceSession is base
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    assert face_utils.limit_ort_session_options(so).intra_op_num_threads == 2
    so.intra_op_num_threads = 0
    assert face_utils.limit_ort_session_options(so).intra_op_num_threads == face_utils.FACE_CPU_THREADS
