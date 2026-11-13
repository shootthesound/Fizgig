"""Element boost needs Target Megapixels 0.75 or more (launch.BOOST_MIN_MP; Peter, 12 Nov 2026): below that a small face
is a patch or two wide and boosting it teaches pixelation. With Start at a smaller size it's the stage size that counts
(the boost runs early in the run). Below the limit the tick is greyed with a note and nothing is sent; no preset turns
it on below the limit."""
from fizgig.families import form, launch
from fizgig.families.registry import FAMILIES, get


def _sent(desc, v):
    return launch.face_weight(desc, {**v, "megapixels": v.get("DATASET_MEGAPIXELS")})


def test_the_limit_and_the_stage_size():
    d = get("minimax_image")
    for mp, stage, ok in (("0.25", "Off", False), ("0.5", "Off", False), ("0.75", "Off", True), ("1.0", "Off", True),
                          ("1.0", "0.5", False), ("1.0", "0.75", True)):
        v = {"FACE_BOOST": True, "DATASET_MEGAPIXELS": mp, "STAGE_MP": stage}
        st = form.state(d, v)
        assert ("FACE_BOOST" in st["disabled"]) is not ok, (mp, stage)
        assert (_sent(d, v) is not None) is ok, (mp, stage)
        if not ok:
            assert "0.75" in st["notes"]["FACE_BOOST"] and "FACE_BOOST_AREA" in st["hidden"]


def test_no_preset_turns_it_on_below_the_limit():
    for d in FAMILIES.values():
        for name, p in d.presets or ():
            if p.get("FACE_BOOST") and d.element_boost:
                assert not launch.boost_too_small(d, p), (d.key, name)
