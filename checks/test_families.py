"""Every model family's description is consistent: it validates, its files and its abilities agree."""
import pytest

from fizgig.families import registry

FAMILIES = sorted(registry.FAMILIES)


@pytest.mark.parametrize("key", FAMILIES)
def test_description_validates(key):
    assert registry.get(key).validate() == []


@pytest.mark.parametrize("key", FAMILIES)
def test_model_files_have_unique_prefs(key):
    keys = [m.pref_key for m in registry.get(key).model_files]
    assert len(keys) == len(set(keys)), f"{key}: a Preferences key is used twice"


@pytest.mark.parametrize("key", FAMILIES)
def test_preview_speed_lora_has_a_file_row(key):
    d = registry.get(key)
    sp = d.preview_speed()
    if sp is not None and sp.pref_key:
        assert sp.pref_key in {m.pref_key for m in d.model_files}, f"{key}: {sp.name} has no Preferences row"


@pytest.mark.parametrize("key", FAMILIES)
def test_presets_are_named_and_distinct(key):
    names = [n for n, _ in registry.get(key).presets]
    assert len(names) == len(set(names)), f"{key}: two presets share a name"


@pytest.mark.parametrize("key", FAMILIES)
def test_text_encoder_builds_are_complete(key):
    """Every listed text encoder build has its file, a full SHA-256, a size and a measured cosine, and none repeats
    the family's own file (docs/drivers/README.md, "Text encoder builds")."""
    import re
    d = registry.get(key)
    own = {(m.repo, m.path) for m in d.model_files if m.role == "text_encoder"}
    seen = set()
    for b in d.text_encoder_builds:
        assert b.label and b.repo and b.path.endswith(".safetensors"), b
        assert re.fullmatch(r"[0-9a-f]{64}", b.sha256), (b.label, b.sha256)
        assert b.size_gb > 0 and 0 < b.cosine <= 1, b
        assert (b.repo, b.path) not in own and b.sha256 not in seen, b
        seen.add(b.sha256)


@pytest.mark.parametrize("key", FAMILIES)
def test_no_checkpointing_figures(key):
    """train_memory_nogc: offered precisions only, rising points, never nf4; a swapped or uncompiled run, or one
    the figure doesn't fit, keeps gradient checkpointing on."""
    from fizgig.families import quant
    d = registry.get(key)
    monkey = pytest.MonkeyPatch(); monkey.setattr(quant, "card_gb", lambda: 96.0)   # behave like a big card
    for k, pts in d.train_memory_nogc.items():
        p, _, kind = k.partition(":")
        assert p in d.precisions, f"{key}: {k}"
        assert list(pts) == sorted(pts) and all(b[1] >= a[1] for a, b in zip(pts, pts[1:])), f"{key}: {k} points"
        mp = pts[0][0]
        assert quant.plan_checkpointing(d, p, 0, mp, free_gb=1000, kind=kind)[0] is False
        assert quant.plan_checkpointing(d, p, 4, mp, free_gb=1000, kind=kind)[0] is True
        # uncompiled: on for a family whose figures are compiled runs; off for one that never compiles (H3) or whose
        # figures were measured uncompiled (nogc_uncompiled)
        assert quant.plan_checkpointing(d, p, 0, mp, free_gb=1000, compiled=False, kind=kind)[0] is \
            bool(d.compiles and not d.nogc_uncompiled)
        assert quant.plan_checkpointing(d, p, 0, mp, free_gb=pts[0][1], kind=kind)[0] is True
        assert quant.plan_checkpointing(d, p, 0, pts[-1][0] + 0.5, free_gb=1000, kind=kind)[0] is True   # past the figures
    if d.compiles:          # Krea 2's nf4 keeps its dequantised weights for backward: never without checkpointing
        assert quant.plan_checkpointing(d, "nf4", 0, 0.25, free_gb=1000)[0] is True
    monkey.undo()


def test_h3_clip_figures_need_tread_and_the_default_window(monkeypatch):
    """H3's clip figures were measured with TREAD and the Default clip window (20-49); a clip run without either keeps
    every block checkpointed (PR #282: TREAD off ran out at 88 GB on a plan of ~64), stills are unaffected."""
    from fizgig.families import quant
    from fizgig.minimax import common as T
    from fizgig.minimax.driver import MiniMaxDriver
    d = registry.get("minimax")
    monkeypatch.setattr(quant, "card_gb", lambda: 96.0)
    drv = MiniMaxDriver()
    drv.description = d
    monkeypatch.setattr(T, "_max_effective_mp", lambda group: (6.39, 27))
    for opts, measured in (({"tread": "0.5@2-47", "clip_blocks": "20-49"}, True), ({"clip_blocks": "20-49"}, False),
                           ({"tread": "0.5@2-47"}, False), ({"tread": "0.5@2-47", "clip_blocks": "6-49"}, False)):
        drv.set_options(opts)
        kind, mp = drv.nogc_load(None)
        assert (kind == "clips") is measured, opts
        assert quant.plan_checkpointing(d, "int8", 0, mp, free_gb=1000, kind=kind)[0] is (not measured), opts
        assert quant.plan_partial_checkpointing(d, "int8", 0, mp, free_gb=1000, kind=kind)[0] == 0 or measured, opts


def test_small_cards_always_checkpoint(monkeypatch):
    from fizgig.families import quant
    monkeypatch.setattr(quant, "card_gb", lambda: 32.0)
    for key in FAMILIES:
        d = registry.get(key)
        for p in d.train_memory_nogc:
            assert quant.plan_checkpointing(d, p, 0, 0.25, free_gb=1000)[0] is True


def test_ranks_past_the_measured_ones_checkpoint(monkeypatch):
    from fizgig.families import quant
    monkeypatch.setattr(quant, "card_gb", lambda: 96.0)
    d = registry.get("qwen_image21")
    assert quant.plan_checkpointing(d, "int8", 0, 1.0, free_gb=90, rank=32)[0] is False
    assert quant.plan_checkpointing(d, "int8", 0, 1.0, free_gb=90, rank=64)[0] is True


def test_partial_checkpointing_fits_with_the_margin(monkeypatch):
    """Partial checkpointing: big cards only, never past the figures, never all blocks, the planned peak plus the
    margin inside free VRAM; every block on without figures, on a small card, under swap or past rank 32."""
    from fizgig.families import quant
    for key in FAMILIES:
        d = registry.get(key)
        for k, (blocks, pts) in d.train_memory_partial.items():
            p, _, kind = k.partition(":")
            assert p in d.precisions and blocks > 1, f"{key}: {k}"
            assert list(pts) == sorted(pts) and all(b[1] >= a[1] and b[2] >= a[2] > 0 for a, b in zip(pts, pts[1:])), k
            monkeypatch.setattr(quant, "card_gb", lambda: 48.0)
            for mp, on_gb, per in pts:
                for free in (38.9, 44.5, 47.1, 79.0):
                    n, _ = quant.plan_partial_checkpointing(d, p, 0, mp, free_gb=free, kind=kind)
                    assert 0 <= n < blocks and (n == 0 or on_gb + n * per + quant.NOGC_MARGIN_GB <= free), (k, mp, free)
                assert quant.plan_partial_checkpointing(d, p, 2, mp, free_gb=90, kind=kind)[0] == 0
                assert quant.plan_partial_checkpointing(d, p, 0, mp, free_gb=90, kind=kind, rank=64)[0] == 0
            assert quant.plan_partial_checkpointing(d, p, 0, pts[-1][0] + 0.5, free_gb=90, kind=kind)[0] == 0
            monkeypatch.setattr(quant, "card_gb", lambda: 32.0)
            assert quant.plan_partial_checkpointing(d, p, 0, pts[0][0], free_gb=1000, kind=kind)[0] == 0


def test_share_shortfall_names_a_base_that_cannot_fit():
    """A helper's share at the lead's precision: a base that can't swap (4-bit) and doesn't fit is named before the
    model loads; one that fits with swap, or a family without figures, isn't."""
    from fizgig.families import quant
    d = registry.get("krea2")
    assert quant.share_shortfall(d, None, "nf4", 0, 1.0, free_gb=14.5).startswith("needs ~15.0 GB")
    assert quant.share_shortfall(d, None, "nf4", 0, 1.0, free_gb=22.0) == ""
    assert quant.share_shortfall(d, None, "int8", 15, 1.0, free_gb=14.5) == ""
    assert quant.share_shortfall(d, None, "fp8", 0, 1.0, free_gb=1.0) == ""
