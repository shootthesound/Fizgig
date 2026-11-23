"""#401: the comparison sheet's Every epoch mode takes the epochs listed in its Epochs box from the training folder's
own checkpoints, so "5,11,22,38" works without rendering those epochs at the top first, and an epoch the run doesn't
have is named instead of silently giving every column."""
import pytest

from fizgig.web import royale

RUN = [[str(e), f"/run/lora-{e:06d}.safetensors"] for e in range(1, 41)]


def test_listed_epochs_come_from_the_run_with_nothing_rendered():
    cols = royale.sheet_columns([], RUN, "5,11,22,38")
    assert [c[0] for c in cols] == ["5", "11", "22", "38"]


def test_listed_epochs_ignore_what_was_rendered():
    rendered = [r for r in RUN if int(r[0]) % 10 == 0]          # Max renders picked 10, 20, 30, 40
    assert [c[0] for c in royale.sheet_columns(rendered, RUN, "5 11")] == ["5", "11"]


def test_an_epoch_the_run_lacks_is_named():
    with pytest.raises(ValueError, match="no 55"):
        royale.sheet_columns([], RUN, "5,55")


def test_blank_and_every_n_use_the_rendered_epochs_then_the_run():
    rendered = RUN[:4]
    assert royale.sheet_columns(rendered, RUN, "") == rendered
    assert [c[0] for c in royale.sheet_columns([], RUN, "every 10")] == ["10", "20", "30", "40"]
    with pytest.raises(ValueError):
        royale.sheet_columns([], [], "")
