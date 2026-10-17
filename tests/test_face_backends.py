"""Face backends: detector selection / auto fallback, YOLO ONNX decoding + NMS, letterbox mapping, identity embedder
thresholds and cache invalidation, likeness + loss-watch integration. Heavy models are mocked; the real-ONNX tests
run when the exported models / test images are present (FIZGIG_FACE_ONNX_DIR, FIZGIG_FACE_IMG_DIR)."""

import json
import math
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fizgig import face_utils as F  # noqa: E402

ONNX_DIR = os.environ.get("FIZGIG_FACE_ONNX_DIR", os.path.join(ROOT, "user_data", "models", "face"))
IMG_DIR = os.environ.get("FIZGIG_FACE_IMG_DIR", "")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for k in ("FIZGIG_FACE_DETECTOR", "FIZGIG_FACE_YOLO_MODEL", "FIZGIG_FACE_IDENTITY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(F, "PREFS_FILE", str(tmp_path / "prefs.json"))    # never the real prefs.json
    monkeypatch.setattr(F, "_HERE", str(tmp_path / "fizgig"))
    monkeypatch.setattr(F, "LOCAL_YOLO_DEFAULT",                         # nor a real user_data/models/face model
                        str(tmp_path / "fizgig" / "user_data" / "models" / "face" / "fdetailer_v11.onnx"))


# --------------------------------------------------------------------------------------------- settings / selection
def test_settings_defaults_prefs_and_env(monkeypatch, tmp_path):
    st = F.load_face_settings({})
    assert st["face_detector"] == "auto" and st["face_identity"] == "auto" and st["face_yolo_model"] == ""
    assert st["face_yolo_conf"] == pytest.approx(0.35)
    (tmp_path / "prefs.json").write_text(json.dumps({"face_detector": "yolo", "face_yolo_model": ' "C:/m/f.onnx" ',
                                                     "face_yolo_conf": "0.5", "face_identity": "CLIP"}))
    st = F.load_face_settings()
    assert (st["face_detector"], st["face_yolo_model"], st["face_yolo_conf"], st["face_identity"]) == \
        ("yolo", "C:/m/f.onnx", 0.5, "clip")
    monkeypatch.setenv("FIZGIG_FACE_DETECTOR", "insightface")
    monkeypatch.setenv("FIZGIG_FACE_IDENTITY", "dinov2")
    st = F.load_face_settings()
    assert st["face_detector"] == "insightface" and st["face_identity"] == "dinov2"
    monkeypatch.delenv("FIZGIG_FACE_DETECTOR")
    monkeypatch.delenv("FIZGIG_FACE_IDENTITY")
    assert F.load_face_settings({"face_detector": "bogus", "face_identity": "nope", "face_yolo_conf": "x"}) == \
        dict(F.FACE_SETTING_DEFAULTS, face_yolo_conf=0.35)


def test_local_fdetailer_is_the_default_yolo_model_when_present(tmp_path, monkeypatch):
    p = tmp_path / "user_data" / "models" / "face" / "fdetailer_v11.onnx"
    monkeypatch.setattr(F, "LOCAL_YOLO_DEFAULT", str(p))
    assert F.load_face_settings({})["face_yolo_model"] == ""                     # absent -> deepghs download
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    assert F.load_face_settings({})["face_yolo_model"] == str(p)
    assert F.load_face_settings({"face_yolo_model": "other.onnx"})["face_yolo_model"] == "other.onnx"
    e = F.FaceEmbedder(settings={"face_detector": "yolo"})
    assert e.cache_id == "dinov2@yolo:fdetailer_v11.onnx"


def test_resolve_identity_auto():
    assert F.resolve_identity({"face_detector": "yolo", "face_identity": "auto"}) == "dinov2"
    assert F.resolve_identity({"face_detector": "auto", "face_identity": "auto"}) == "arcface"
    assert F.resolve_identity({"face_detector": "insightface", "face_identity": "auto"}) == "arcface"
    assert F.resolve_identity({"face_detector": "insightface", "face_identity": "clip"}) == "clip"


def _fi(x1, y1, x2, y2, gender="unknown", src="yolo"):
    return F._face_info(x1, y1, x2, y2, gender, 0.9, src)


def _detector(backend, insight_faces, yolo_faces, insight_ok=True, yolo_ok=True):
    d = F.FaceDetector(backend=backend, settings={})
    calls = []
    d._insight = types.SimpleNamespace(available=lambda: insight_ok, load=lambda: None, app=None,
                                       detect_bgr=lambda bgr: calls.append("insightface") or list(insight_faces))
    d._yolo = types.SimpleNamespace(available=lambda: yolo_ok, load=lambda: None, label="m.onnx", model_path="m.onnx",
                                    detect_bgr=lambda bgr: calls.append("yolo") or list(yolo_faces))
    return d, calls


def test_auto_runs_both_and_merges_preferring_insightface_on_human_faces():
    human = _fi(100, 100, 140, 150, "female", "insightface")._replace(score=0.85)
    head_on_human = _fi(90, 80, 150, 160)                 # YOLO head box around the same face -> dropped
    furry = _fi(300, 40, 420, 170)                        # a face InsightFace can't see -> kept
    d, calls = _detector("auto", [human], [head_on_human, furry])
    faces = d.detect_bgr(np.zeros((64, 64, 3), np.uint8))
    assert calls == ["insightface", "yolo"] and d.last_source == "insightface+yolo"
    assert faces == [human, furry] and d.has_gender
    assert d.get_largest_by_gender(faces, "female") == human and d.get_largest(faces) == furry


def test_auto_merge_drops_weak_insightface_boxes_on_yolo_faces():
    weak_on_suit = _fi(10, 10, 50, 50, "male", "insightface")._replace(score=0.54)   # RetinaFace on a fursuit
    weak_elsewhere = _fi(200, 200, 230, 230, "male", "insightface")._replace(score=0.55)
    suit = _fi(0, 0, 60, 60)
    merged = F.merge_faces([weak_on_suit, weak_elsewhere], [suit])
    assert merged == [suit, weak_elsewhere]


def test_auto_without_insightface_uses_yolo_and_survives_a_broken_yolo(capsys):
    d, calls = _detector("auto", [], [_fi(1, 2, 30, 40)], insight_ok=False)
    assert d.available and d.detect_bgr(np.zeros((8, 8, 3), np.uint8)) and calls == ["yolo"]
    d, calls = _detector("auto", [_fi(0, 0, 9, 9, "male", "insightface")], [], insight_ok=True)
    d._yolo.detect_bgr = lambda bgr: (_ for _ in ()).throw(OSError("offline"))
    assert len(d.detect_bgr(np.zeros((8, 8, 3), np.uint8))) == 1          # human-photo prep never breaks
    assert len(d.detect_bgr(np.zeros((8, 8, 3), np.uint8))) == 1
    assert capsys.readouterr().out.count("YOLO model unavailable") == 1    # logged once, then skipped


def test_detect_all_is_cached_per_file_and_settings(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (16, 16)).save(img)
    d, calls = _detector("auto", [], [_fi(1, 1, 9, 9)])
    assert d.detect_all(str(img)) == d.detect_all(str(img)) and calls == ["insightface", "yolo"]
    os.utime(img, (1, 1))                                                  # changed file -> detected again
    d.detect_all(str(img))
    assert calls == ["insightface", "yolo"] * 2


def test_yolo_mode_has_no_gender_and_gender_selection_falls_back():
    d, calls = _detector("yolo", [_fi(0, 0, 5, 5, "male", "insightface")], [_fi(0, 0, 9, 9), _fi(0, 0, 20, 20)])
    faces = d.detect_bgr(np.zeros((32, 32, 3), np.uint8))
    assert calls == ["yolo"] and not d.has_gender
    f = d.get_largest_by_gender(faces, "male")
    assert f.area == 400 and f.gender == "unknown"
    assert d.get_largest_by_gender(faces, "male", fallback_to_any=False) is None


def test_insightface_mode_never_runs_yolo():
    d, calls = _detector("insightface", [], [_fi(0, 0, 9, 9)])
    assert d.detect_bgr(np.zeros((8, 8, 3), np.uint8)) == [] and calls == ["insightface"]


def test_settings_key_tracks_model_and_conf():
    a = F.FaceDetector(settings={"face_detector": "yolo", "face_yolo_model": "a.onnx"})
    b = F.FaceDetector(settings={"face_detector": "yolo", "face_yolo_model": "b.onnx"})
    c = F.FaceDetector(settings={"face_detector": "yolo", "face_yolo_model": "a.onnx", "face_yolo_conf": 0.5})
    assert len({a.settings_key, b.settings_key, c.settings_key}) == 3
    assert "a.onnx" in a.describe()


def test_pt_model_is_refused_and_missing_model_unavailable(tmp_path):
    pt = tmp_path / "face.pt"
    pt.write_bytes(b"\x80\x02fake pickle")
    y = F.YoloOnnxBackend(str(pt))
    assert not y.available()
    with pytest.raises(ValueError, match="convert_yolo_to_onnx"):
        y.resolve_path()
    assert not F.YoloOnnxBackend(str(tmp_path / "missing.onnx")).available()
    d = F.FaceDetector(settings={"face_detector": "yolo", "face_yolo_model": str(pt)})
    with pytest.raises(ImportError):
        d.detect_bgr(np.zeros((8, 8, 3), np.uint8))


def test_relative_model_path_resolves_against_the_fizgig_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "_HERE", str(tmp_path))
    (tmp_path / "models" / "face").mkdir(parents=True)
    (tmp_path / "models" / "face" / "m.onnx").write_bytes(b"x")
    y = F.YoloOnnxBackend(os.path.join("models", "face", "m.onnx"))
    assert y.model_path == str(tmp_path / "models" / "face" / "m.onnx") and y.available()


# --------------------------------------------------------------------------------------------- YOLO decoding
def _det_table(rows, nc, extra=0, n=50):
    """(1, 4+nc+extra, n) ultralytics detect layout with `rows` = [(cx, cy, w, h, cls, score)] and zero padding."""
    t = np.zeros((4 + nc + extra, n), np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(rows):
        t[:4, i] = (cx, cy, w, h)
        t[4 + c, i] = s
        if extra:
            t[4 + nc:, i] = np.random.default_rng(i).normal(size=extra)     # mask coefficients: ignored
    return t[None]


ROWS = [(100, 100, 40, 40, 0, 0.9), (102, 101, 40, 42, 0, 0.8),     # overlapping faces -> one after NMS
        (300, 200, 60, 80, 0, 0.6),                                  # a second face
        (100, 100, 40, 40, 1, 0.95),                                 # another class on the same spot
        (50, 50, 10, 10, 0, 0.1)]                                    # below conf


def test_decode_detect_nms_conf_and_class_filter():
    out = F.decode_yolo([_det_table(ROWS, 2)], conf=0.35, iou=0.5, class_ids=[0], n_classes=2)
    assert [round(o[4], 2) for o in out] == [0.9, 0.6]
    x1, y1, x2, y2, s, c = out[0]
    assert (x1, y1, x2, y2, c) == (80, 80, 120, 120, 0)
    allc = F.decode_yolo([_det_table(ROWS, 2)], conf=0.35, iou=0.5, class_ids=None, n_classes=2)
    assert sorted(o[5] for o in allc) == [0, 0, 1]                  # NMS is per class
    assert F.decode_yolo([_det_table(ROWS, 2)], conf=0.35, iou=0.5, class_ids=[5], n_classes=2) == []


def test_decode_transposed_and_without_metadata():
    t = _det_table(ROWS, 2, n=2100)                 # real heads have thousands of anchors (vs <=1000 end-to-end rows)
    a = F.decode_yolo([t], 0.35, 0.5, [0], n_classes=2)
    b = F.decode_yolo([np.transpose(t, (0, 2, 1))], 0.35, 0.5, [0], n_classes=None)
    assert a == b


def test_decode_segment_layout_ignores_mask_coefficients():
    nc = 6                                                           # Fdetailer: face + 5 other classes
    rows = [(200, 150, 50, 60, 0, 0.88), (400, 300, 30, 30, 3, 0.99)]
    seg = _det_table(rows, nc, extra=32, n=80)
    protos = np.zeros((1, 32, 160, 160), np.float32)
    with_meta = F.decode_yolo([seg, protos], 0.35, 0.5, [0], n_classes=nc)
    no_meta = F.decode_yolo([seg, protos], 0.35, 0.5, [0], n_classes=None)  # 4+6+32 channels, nc inferred
    assert len(with_meta) == 1 and with_meta == no_meta
    assert with_meta[0][:4] == (175, 120, 225, 180)


def test_decode_end_to_end_layout():
    e2e = np.zeros((1, 300, 6), np.float32)
    e2e[0, 0] = (10, 20, 50, 80, 0.9, 0)
    e2e[0, 1] = (11, 21, 51, 81, 0.7, 1)
    e2e[0, 2] = (0, 0, 5, 5, 0.2, 0)
    out = F.decode_yolo([e2e], 0.35, 0.5, [0], n_classes=2)
    assert out == [(10.0, 20.0, 50.0, 80.0, pytest.approx(0.9), 0)]


def test_letterbox_and_box_mapping_back_to_the_image():
    rgb = np.zeros((300, 600, 3), np.uint8)
    canvas, r, (left, top) = F.letterbox(rgb, 640)
    assert canvas.shape == (640, 640, 3) and r == pytest.approx(640 / 600) and left == 0 and top == 160
    assert canvas[0, 0, 0] == 114 and canvas[320, 320, 0] == 0

    class Sess:                      # a face at image (100..200, 50..150) seen in letterbox space
        def run(self, _o, feeds):
            x = feeds["images"]
            assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32 and x.max() <= 1.0
            cx, cy = (150 * r + left), (100 * r + top)
            return [_det_table([(cx, cy, 100 * r, 100 * r, 0, 0.9)], 1)]
    y = F.YoloOnnxBackend("x.onnx")
    y._sess, y._input, y.size, y.names, y.class_ids = Sess(), "images", 640, {0: "face"}, [0]
    f = y.detect_rgb(rgb)[0]
    assert np.allclose(f.bbox, [100, 50, 200, 150], atol=1) and f.gender == "unknown" and f.source == "yolo"


def test_pick_classes():
    y = F.YoloOnnxBackend("x.onnx")
    y.names = {0: "pussy", 1: "Face", 2: "head"}
    assert y._pick_classes() == [1]
    y.names = {0: "anything"}
    assert y._pick_classes() == [0]                                   # one-class model taken as is
    y.names = {0: "hand", 1: "eye"}
    with pytest.raises(ValueError):
        y._pick_classes()
    assert F._parse_names({"names": "{0: 'face', 1: 'head'}"}) == {0: "face", 1: "head"}
    assert F._parse_names({}) is None


# --------------------------------------------------------------------------------------------- identity embedder
def test_thresholds_per_backend_and_verdicts():
    th = {k: F.FaceEmbedder(backend=k, settings={}).thresholds for k in ("arcface", "dinov2", "clip")}
    assert th["arcface"] == {"match": 0.45, "borderline": 0.30, "floor": 0.25}     # the old hard-coded values
    for t in th.values():
        assert t["match"] > t["borderline"] > t["floor"]
    assert th["dinov2"]["match"] > th["arcface"]["match"] and th["clip"]["floor"] > th["dinov2"]["match"]
    e = F.FaceEmbedder(backend="dinov2", settings={"face_detector": "yolo"})
    assert [e.verdict(s) for s in (0.7, 0.5, 0.3, None, float("nan"))] == \
        ["match", "borderline", "drift", "no face", "no face"]
    a = F.FaceEmbedder(backend="arcface", settings={})
    assert a.verdict(0.5) == "match" and a.verdict(0.35) == "borderline"


def test_look_cutoff_uses_the_embedder_floor():
    scores = [0.9, 0.88, 0.86, 0.85, 0.3, 0.84, 0.87]
    assert F.look_cutoff(scores[:3], 0.25) is None
    arc = F.FaceEmbedder(backend="arcface", settings={}).cutoff(scores)
    dino = F.FaceEmbedder(backend="dinov2", settings={}).cutoff([s - 0.4 for s in scores])
    assert arc == pytest.approx(0.86 - 1.5 * (0.88 - 0.84))
    assert dino == pytest.approx(F.IDENTITY_THRESHOLDS["dinov2"]["floor"])   # fence below the floor -> floor


def test_cache_id_and_settings_key_invalidate_across_backends():
    arc = F.FaceEmbedder(settings={"face_detector": "insightface"})
    arc2 = F.FaceEmbedder(settings={"face_detector": "auto"})
    dino = F.FaceEmbedder(settings={"face_detector": "yolo", "face_yolo_model": "C:/m/fdetailer_v11.onnx"})
    dino_default = F.FaceEmbedder(settings={"face_detector": "yolo"})
    clip = F.FaceEmbedder(settings={"face_detector": "yolo", "face_identity": "clip"})
    assert arc.cache_id == arc2.cache_id == "arcface" and arc.settings_key == arc2.settings_key
    assert dino.cache_id == "dinov2@yolo:fdetailer_v11.onnx" and dino_default.cache_id == "dinov2@yolo:default"
    assert len({arc.cache_id, dino.cache_id, dino_default.cache_id, clip.cache_id}) == 4
    assert F.same_embedder(None, "arcface") and F.same_embedder("arcface", "arcface")   # legacy files = ArcFace
    assert not F.same_embedder(None, dino.cache_id) and not F.same_embedder("arcface", dino.cache_id)


class _FakeDetector:
    def __init__(self, faces):
        self.faces, self.available = faces, True

    def detect_bgr(self, bgr):
        return list(self.faces)

    def get_largest(self, faces):
        return max(faces, key=lambda f: f.area) if faces else None


def test_crop_embedder_crops_the_largest_face_and_normalises(monkeypatch):
    e = F.FaceEmbedder(backend="dinov2", settings={"face_detector": "yolo"})
    seen = []
    e._impl.detector = _FakeDetector([_fi(10, 10, 20, 20), _fi(100, 60, 140, 100)])
    monkeypatch.setattr(e._impl, "_features", lambda crop: seen.append(crop.size) or np.array([3.0, 4.0]))
    monkeypatch.setattr(F, "_importable", lambda *m: True)
    emb, bbox = e.embed_pil(Image.new("RGB", (200, 160)))
    assert np.allclose(emb, [0.6, 0.8]) and bbox == (100, 60, 140, 100)
    assert seen == [(60, 60)]                       # 40 px face, 25% margin each side, square
    e._impl.detector = _FakeDetector([])
    assert e.embed_pil(Image.new("RGB", (64, 64))) == (None, None)


def test_crop_clamps_to_the_image():
    c = F._CropEmbedder.crop(Image.new("RGB", (100, 80)), (0, 0, 60, 70))
    assert c.size[0] <= 100 and c.size[1] <= 80


def test_arcface_bbox_dropped_for_padded_detections():
    face = types.SimpleNamespace(bbox=np.array([5, 5, 40, 50]), normed_embedding=np.ones(4) / 2)
    a = F._ArcFace({})
    a.backend = types.SimpleNamespace(raw_faces=lambda bgr: ([face], (0, 0)), available=lambda: True)
    emb, bbox = a.embed_bgr(np.zeros((60, 60, 3), np.uint8))
    assert bbox == (5, 5, 40, 50) and emb.dtype == np.float32
    a.backend = types.SimpleNamespace(raw_faces=lambda bgr: ([face], (15, 15)), available=lambda: True)
    assert a.embed_bgr(np.zeros((60, 60, 3), np.uint8))[1] is None


def test_embedder_unavailable_raises_cleanly(monkeypatch):
    e = F.FaceEmbedder(backend="dinov2", settings={"face_detector": "yolo"})
    monkeypatch.setattr(e._impl, "available", lambda: False)
    with pytest.raises(ImportError, match="DINOv2"):
        e.embed_image(Image.new("RGB", (8, 8)))


# --------------------------------------------------------------------------------------------- Auto look match
class _ProbeDetector:
    """detect_all by file name: h* = human photo, f* = furry face only, n* = no face, m* = human + furry."""
    def __init__(self):
        self.calls = 0

    def detect_all(self, p):
        self.calls += 1
        k = os.path.basename(p)[0]
        human = _fi(0, 0, 50, 50, "male", "insightface")._replace(score=0.8)
        return {"h": [human], "f": [_fi(0, 0, 80, 80)], "n": [], "m": [human, _fi(100, 0, 180, 80)],
                "w": [_fi(0, 0, 50, 50, "male", "insightface")._replace(score=0.52)]}[k]


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(F.InsightFaceBackend, "available", staticmethod(lambda: True))
    F._DECISIONS.clear()

    def make(**counts):
        d = tmp_path / ("ds_" + "_".join(f"{k}{v}" for k, v in counts.items()))
        d.mkdir()
        for k, v in counts.items():
            for i in range(v):
                (d / f"{k}{i:03d}.png").write_bytes(b"x")
        return sorted(str(p) for p in d.iterdir())
    return make


AUTO = {"face_detector": "auto", "face_identity": "auto"}


def test_look_match_zero_human_faces_is_dinov2(dataset):
    det = _ProbeDetector()
    d = F.decide_look_match(dataset(f=20, n=4), AUTO, detector=det, log=None)
    assert (d.identity, d.kind, d.human, d.yolo_only, d.probed) == ("dinov2", "furry", 0, 20, 24)
    assert d.settings["face_identity"] == "dinov2" and d.settings["face_detector"] == "auto"
    assert d.embedder().cache_id.startswith("dinov2@auto:")
    assert "0/24" in d.message and "DINOv2" in d.message


def test_look_match_a_few_human_faces_still_dinov2(dataset):
    d = F.decide_look_match(dataset(h=3, f=18, w=3), AUTO, detector=_ProbeDetector(), log=None)
    assert d.identity == "dinov2" and d.human == 3          # bystanders / weak RetinaFace hits don't block it


def test_look_match_mostly_human_is_arcface(dataset):
    msgs = []
    d = F.decide_look_match(dataset(h=21, n=2, f=1), AUTO, detector=_ProbeDetector(), log=msgs.append)
    assert (d.identity, d.kind) == ("arcface", "human") and d.embedder().cache_id == "arcface"
    assert msgs and "21/24" in msgs[0] and "ArcFace" in msgs[0]


def test_look_match_mixed_set_uses_dinov2_for_all(dataset):
    d = F.decide_look_match(dataset(h=13, f=8, m=3), AUTO, detector=_ProbeDetector(), log=None)
    assert (d.identity, d.kind) == ("dinov2", "mixed")


def test_look_match_explicit_settings_bypass_the_probe(dataset):
    det = _ProbeDetector()
    paths = dataset(h=24)
    assert F.decide_look_match(paths, {"face_detector": "auto", "face_identity": "clip"}, detector=det,
                               log=None).identity == "clip"
    assert F.decide_look_match(paths, {"face_detector": "yolo"}, detector=det, log=None).identity == "dinov2"
    assert F.decide_look_match(paths, {"face_detector": "insightface"}, detector=det, log=None).identity == "arcface"
    assert det.calls == 0


def test_look_match_is_cached_and_deterministic(dataset, monkeypatch):
    det = _ProbeDetector()
    paths = dataset(h=40, f=60)
    a = F.decide_look_match(paths, AUTO, detector=det, log=None)
    assert det.calls == 24
    b = F.decide_look_match(list(reversed(paths)), AUTO, detector=det, log=None)
    assert b is a and det.calls == 24                       # same folder -> same sample -> cached decision
    os.utime(F.probe_sample(paths)[0], (1, 1))              # a probed file changed -> decided again
    F.decide_look_match(paths, AUTO, detector=det, log=None)
    assert det.calls == 48


def test_look_match_threshold_knob(dataset, monkeypatch):
    paths = dataset(h=15, n=9)                              # 62% human, no furry faces
    assert F.decide_look_match(paths, AUTO, detector=_ProbeDetector(), log=None).identity == "arcface"
    monkeypatch.setenv("FIZGIG_FACE_AUTO_THRESHOLD", "0.8")
    d = F.decide_look_match(paths, AUTO, detector=_ProbeDetector(), log=None)
    assert d.identity == "dinov2" and "too few human faces" in d.message


def test_probe_sample_spreads_and_caps(tmp_path):
    for i in range(100):
        (tmp_path / f"{i:03d}.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    s = F.probe_sample([str(p) for p in tmp_path.iterdir()], 24)
    names = [os.path.basename(p) for p in s]
    assert len(s) == 24 and names[0] == "000.png" and names[-1] == "099.png" and len(set(names)) == 24


def test_look_match_with_no_insightface_is_dinov2(dataset, monkeypatch):
    paths = dataset(h=5)
    monkeypatch.setattr(F.InsightFaceBackend, "available", staticmethod(lambda: False))
    assert F.decide_look_match(paths, AUTO, detector=_ProbeDetector(), log=None).identity == "dinov2"


# --------------------------------------------------------------------------------------------- integrations
def test_likeness_wraps_the_embedder(monkeypatch, tmp_path):
    from fizgig.lora_royale import likeness
    ref = tmp_path / "ref.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(ref)

    class Emb:
        settings_key, label, available = ("fake",), "Fake", True

        def embed(self, path):
            return np.array([1.0, 0.0])

        def embed_image(self, img):
            return None if img.getpixel((0, 0)) == (0, 0, 0) else np.array([0.6, 0.8])
    monkeypatch.setattr(likeness, "get_embedder", lambda settings=None: Emb())
    monkeypatch.setattr(F, "decide_look_match", lambda paths, settings=None, **k: types.SimpleNamespace(settings={}))
    out = likeness.score_renders(str(ref), [(1, Image.new("RGB", (8, 8), (9, 9, 9))),
                                            (2, Image.new("RGB", (8, 8), (0, 0, 0)))])
    assert out[0] == (1, pytest.approx(0.6)) and out[1][0] == 2 and math.isnan(out[1][1])
    assert likeness.available() and likeness.embedder_name() == "Fake" and likeness.last_embedder_label == "Fake"
    assert math.isnan(likeness.cosine(None, np.ones(2)))


def test_likeness_rebuilds_the_embedder_when_settings_change():
    from fizgig.lora_royale import likeness
    a = likeness.get_embedder({"face_detector": "yolo", "face_identity": "dinov2"})
    assert likeness.get_embedder({"face_detector": "yolo", "face_identity": "dinov2"}) is a
    b = likeness.get_embedder({"face_detector": "insightface"})
    assert b is not a and b.name == "arcface"


def _warmup(tmp_path, payload):
    from fizgig.families import loss_watch
    (tmp_path / "fizgig_look_scores.json").write_text(json.dumps(payload))
    got = {}
    obj = types.SimpleNamespace(image_dir=str(tmp_path),
                                watch=types.SimpleNamespace(set_warmup_keys=lambda k: got.setdefault("k", set(k))))
    loss_watch.Watch._warmup(obj, {"a", "b", "c"})
    return got.get("k")


def test_loss_watch_warmup_with_embedder_scores(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO)
    keys = _warmup(tmp_path, {"embedder": "dinov2@yolo:fdetailer_v11.onnx", "embedder_label": "DINOv2",
                              "thresholds": F.IDENTITY_THRESHOLDS["dinov2"], "cutoff": 0.5,
                              "scores": {"a": 0.8, "b": 0.45, "c": None, "gone": 0.1}})
    assert keys == {"b"} and "DINOv2" in caplog.text
    assert _warmup(tmp_path, {"cutoff": 0.25, "scores": {"a": 0.2, "b": 0.6}}) == {"a"}      # legacy ArcFace file


def test_block_profile_report_names_the_embedder():
    import inspect
    from fizgig.families import block_profile
    src = inspect.getsource(block_profile)
    assert "abl.get('embedder')" in src and "face recognition (ArcFace)" not in src


def test_no_caller_reaches_into_insightface_internals():
    for rel in ("src/fizgig/web/prep_jobs.py", "src/fizgig/web/studio.py", "src/fizgig/web/gallery_likeness.py",
                "src/fizgig/scripts/profile_run.py", "src/fizgig/dataset/prep.py"):
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        assert "._app.get(" not in src and "_detect_with_pad_retry" not in src, rel
        assert "s >= 0.45" not in src and "1.5 * (q3 - q1), 0.25)" not in src, rel     # no fixed ArcFace bands / floor


# --------------------------------------------------------------------------------------------- Image Prep + web UI
def test_prep_verdicts_and_cutoff_follow_the_embedder():
    from fizgig.dataset import prep
    dino = F.IDENTITY_THRESHOLDS["dinov2"]
    assert prep.verdict(0.5)[0] == "match" and prep.verdict(0.5, dino)[0] == "borderline"
    assert prep.verdict(None, dino)[0] == "no face" and prep.verdict(0.3, dino)[0] == "drift"
    assert prep.ARCFACE == F.IDENTITY_THRESHOLDS["arcface"]
    scores = {str(i): s for i, s in enumerate([0.9, 0.88, 0.87, 0.86, 0.1])}
    assert prep.cutoff(scores)[1] == pytest.approx(max(0.87 - 1.5 * 0.02, 0.25))
    low = {str(i): s for i, s in enumerate([0.5, 0.3, 0.45, 0.2, 0.1])}
    assert prep.cutoff(low, dino["floor"])[1] == pytest.approx(0.40)
    assert prep.suggest(low, set(), dino["floor"]) == {"1", "3", "4"}


def test_prep_saves_scores_with_their_embedder(tmp_path):
    from fizgig.dataset import prep
    scores = {str(tmp_path / f"{i}.png"): s for i, s in enumerate([0.5, 0.45, 0.47, 0.44, 0.1])}
    prep.save_scores(str(tmp_path), [str(tmp_path / "0.png")], scores)
    old = json.loads((tmp_path / "fizgig_look_scores.json").read_text())
    assert "embedder" not in old and old["cutoff"] == pytest.approx(prep.cutoff(scores)[1])   # unchanged format
    emb = F.FaceEmbedder(settings={"face_detector": "yolo", "face_identity": "dinov2"})
    prep.save_scores(str(tmp_path), [str(tmp_path / "0.png")], scores, emb)
    d = json.loads((tmp_path / "fizgig_look_scores.json").read_text())
    assert d["embedder"] == emb.cache_id and d["embedder_label"] == emb.label
    assert d["cutoff"] == pytest.approx(prep.cutoff(scores, 0.40)[1]) and set(d["scores"]) == {"0", "1", "2", "3", "4"}


def test_prep_notes_a_gender_choice_on_a_yolo_face():
    from fizgig.dataset import prep
    det = F.FaceDetector(settings={"face_detector": "auto"})
    furry = F._face_info(0, 0, 50, 50, "unknown", 0.8, "yolo")
    woman = F._face_info(0, 0, 20, 20, "female", 0.9)
    job = prep.Prep(lambda t: None, det)
    assert job.select_face([furry], "largest_male") == (furry, "  Note: the face finder can't tell gender on this "
                                                               "face, using largest face\n")
    assert job.select_face([furry, woman], "largest_female") == (woman, "")
    assert job.select_face([woman], "largest_male")[1] == "  Note: No male face, using largest face\n"


def test_web_face_settings_validate_and_report(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from fizgig.web import prep_jobs
    model = tmp_path / "fizgig" / "m.onnx"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"x")
    assert prep_jobs.save_face_settings({"detector": "yolo", "identity": "dinov2", "yolo_model": "m.onnx",
                                         "auto_threshold": "0.4"}) == \
        {"face_detector": "yolo", "face_identity": "dinov2", "face_yolo_model": "m.onnx", "face_auto_threshold": 0.4}
    assert prep_jobs.save_face_settings({"yolo_model": ""}) == {"face_yolo_model": ""}
    for bad in ({"detector": "x"}, {"identity": "arc"}, {"yolo_model": "nope.onnx"}, {"yolo_model": str(model) + ".pt"},
                {"auto_threshold": "2"}):
        with pytest.raises(ValueError):
            prep_jobs.save_face_settings(bad)
    (tmp_path / "prefs.json").write_text(json.dumps({"face_detector": "yolo", "face_yolo_model": "m.onnx"}))
    f = prep_jobs.face_settings()
    assert f["detector"] == "yolo" and f["yolo_model"] == "m.onnx" and f["has_gender"] is False
    assert [k for k, _ in f["detectors"]] == list(F.DETECTORS) and [k for k, _ in f["identities"]] == list(F.IDENTITIES)


def test_web_look_filter_uses_one_embedder_per_session(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from fizgig.web import prep_jobs

    class Emb:
        label, thresholds, cache_id = "Fake DINOv2", dict(F.IDENTITY_THRESHOLDS["dinov2"]), "dinov2@auto:default"
        settings_key = ("fake",)

        def embed(self, path):
            v = np.array([1.0, 0.0]) if "a" in os.path.basename(path) else np.array([0.6, 0.8])
            return v

    files = []
    for n in ("a1", "a2", "a3", "b1", "a4"):
        p = tmp_path / f"{n}.png"
        Image.new("RGB", (8, 8)).save(p)
        files.append(str(p))
    monkeypatch.setattr(prep_jobs, "session_embedder", lambda paths, log=None: (Emb(), "[faces] Auto look match: test"))
    prep_jobs._look.update(state="idle")
    prep_jobs.look_scan(str(tmp_path), files[:3])
    for _ in range(200):
        if prep_jobs._look["state"] != "running":
            break
        import time
        time.sleep(0.01)
    st = prep_jobs.look_status()
    assert st["state"] == "done" and st["embedder"] == "Fake DINOv2" and st["decision"].endswith("test")
    by = {r["name"]: r for r in st["rows"]}
    assert by["b1.png"]["score"] == pytest.approx(0.6) and by["b1.png"]["verdict"] == "borderline"   # DINOv2 bands
    assert json.loads((tmp_path / "fizgig_look_scores.json").read_text())["embedder"] == "dinov2@auto:default"


def test_fetch_models_lists_the_face_fallbacks():
    from fizgig.scripts import fetch_models as fm
    ids = [t[0] for t in fm.TOOLS]
    head = [i for i in ids if "anime_head_detection" in i][0]
    assert F.DEFAULT_YOLO["revision"] in head and F.DEFAULT_YOLO["file"] in head
    assert any(i.startswith("hf-model:facebook/dinov2-base@" + F._DinoV2.revision) for i in ids)


# --------------------------------------------------------------------------------------------- real ONNX (CPU)
def _onnx(name):
    p = os.path.join(ONNX_DIR, name)
    if not os.path.isfile(p):
        pytest.skip(f"{p} not present")
    return p


def _img(name):
    if not IMG_DIR:
        pytest.skip("set FIZGIG_FACE_IMG_DIR to a folder of test images")
    p = os.path.join(IMG_DIR, name)
    if not os.path.isfile(p):
        pytest.skip(f"{p} not present")
    return p


@pytest.mark.parametrize("model,image,min_faces", [
    ("FacesV1.onnx", "suit_2023_09_20_Hexagon_Husky_Photoshoot_05_j.jpg", 1),
    ("dghs_head_v2.0s.onnx", "draw_Harvett_August_2026_png.jpg", 1),
    ("bingsu_face_yolov8n.onnx", "ctrl_human_zidane.jpg", 1),
    ("fdetailer_v11.onnx", "suit_2023_09_20_Hexagon_Husky_Photoshoot_05_j.jpg", 1),   # seg model, face class of 6
    ("fdetailer_v11.onnx", "draw_Harvett_Fox_png.jpg", 1),
])
def test_real_onnx_detection(model, image, min_faces):
    pytest.importorskip("onnxruntime")
    d = F.FaceDetector(settings={"face_detector": "yolo", "face_yolo_model": _onnx(model), "face_yolo_conf": 0.35})
    faces = d.detect_all(_img(image))
    W, H = Image.open(_img(image)).size
    assert len(faces) >= min_faces
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H and 0.35 <= f.score <= 1 and f.gender == "unknown"


def test_real_fdetailer_picks_the_face_class():
    pytest.importorskip("onnxruntime")
    y = F.YoloOnnxBackend(_onnx("fdetailer_v11.onnx"))
    y.load()
    assert y.names[0] == "face" and len(y.names) == 6 and y.class_ids == [0] and y.size == 640


def test_real_seg_onnx_detection():
    """A YOLO *segment* export (as Fdetailer is) through onnxruntime."""
    pytest.importorskip("onnxruntime")
    d = F.FaceDetector(settings={"face_detector": "yolo", "face_yolo_model": _onnx("person_yolov8n_seg.onnx")})
    d._yolo.classes = ("person",)
    faces = d.detect_all(_img("ctrl_human_bus.jpg"))
    assert len(faces) >= 3
