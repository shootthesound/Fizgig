"""A prompt-pair slider has no dataset: memory and compile planning read its practice pictures' size, not a 0.25 MP
guess (Krea 2's compile boundary differs between 0.25 and 1 MP)."""
from fizgig.families import train


def test_slider_bank_megapixels():
    bank = train._SliderBank([None] * 4)
    bank.megapixels = 1008 * 1008 / 1e6
    assert abs(train._largest_bucket_mp(bank) - 1.016064) < 1e-6


def test_no_size_known_is_none():
    assert train._largest_bucket_mp(train._SliderBank([None])) is None


def test_compile_plan_reads_it():
    src = open(train.__file__, encoding="utf-8").read()
    assert "mp_max = _largest_bucket_mp(group) or 0.25" in src


def test_vimoo_plan_reads_it(monkeypatch):
    from fizgig.families import quant
    from fizgig.families.registry import get
    seen = {}
    monkeypatch.setattr(quant, "plan", lambda desc, drv, p, s, megapixels, pictures, **k: seen.setdefault("mp", megapixels))
    bank = train._SliderBank([None] * 4)
    bank.megapixels = 1.0
    get("vimoo").load_driver().plan_run("auto", -1, group=bank, run={})
    assert seen["mp"] == 1.0
