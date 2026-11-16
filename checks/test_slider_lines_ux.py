"""A prompt slider's boxes of lines with the by-epoch switch on (app.js promptLines), as a photographer meets
them: Pair +1 and -1 lines switched on the first time pairs only lines on just the same epochs, changing none; Undo
after a change that takes words or lines away; Backspace joins only lines that can join; lines on their own lettered
over both boxes, with named Pair / Unpair buttons; an empty pair line's hint; Paste several lines under the lines;
▲ ▼ big enough to hit; and the note under an end whose lines are on together (form._slider_plan)."""
import os
import re

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family
from test_slider_paired import _inputs, _page

CSS = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "style.css")
# the Undo button by a box's buttons, and whether it shows
UNDO = r"""
const U = (w) => w.querySelector(".prompt-undo"), shown = (w) => U(w).style.display !== "none";
const bs = (it) => { const b = box(it); b.setSelectionRange(0, 0);
  return b.fire("keydown", {key: "Backspace"}).defaultPrevented; };
"""


def test_the_first_switch_on_pairs_only_lines_on_the_same_epochs():
    # switched on with no pairs kept: each +1 line with the first -1 line on just the same epochs (no epochs: the whole
    # run), in order; every other line on its own - no -1 line's epochs change and no empty line is added
    pos, neg = "smiling @1-4\nlaughing @5-\ngrinning", "frowning @1-4\npouting @2-3\ncrying @5-8\nsulking"
    got = _page(UNDO + r"""
      note("off", nums(P()), nums(N()));
      flip(true);
      note("on", nums(P()), nums(N()), JSON.parse(S.values.FAMILY_SLIDER_PAIRS).length,
        [...P(), ...N()].map((it) => { const b = it.querySelector(".prompt-chain");
          return [b.style.display !== "none", b.textContent, !!b.title]; }));
    """, FAMILY_SLIDER_PAIRED=False, FAMILY_SLIDER_POS=pos, FAMILY_SLIDER_NEG=neg)
    assert got[0] == ["off", pos, neg, "1 2 3", "1 2 3 4"]
    what, plus, minus, p_nums, m_nums, pairs, chains = got[1]
    assert plus == pos and sorted(minus.split("\n")) == sorted(neg.split("\n"))   # every line as it was, none added
    # smiling-frowning (1-4) and grinning-sulking (the whole run) pair; laughing @5- and crying @5-8 aren't on the same
    # epochs: on their own, lettered as one run over both boxes - A in the +1 box, B and C in the -1 box
    assert (minus, p_nums, m_nums, pairs) == ("frowning @1-4\nsulking\npouting @2-3\ncrying @5-8", "1 A 2", "1 2 B C",
                                              2)
    # every line's chain button shows, with its name and a tooltip
    assert chains == [[True, "Unpair", True], [True, "Pair", True], [True, "Unpair", True], [True, "Unpair", True],
                      [True, "Unpair", True], [True, "Pair", True], [True, "Pair", True]]


def test_backspace_joins_only_lines_on_the_same_epochs_whose_partner_has_no_words():
    got = _page(UNDO + r"""
      const b = box(P()[0]); b.value = "smiling wide"; b.setSelectionRange(7, 7); b.fire("keydown", {key: "Enter"});
      typed(N()[1], "scowling");                      // the new pair's -1 line has words: in the +1 box too, no join
      note("partner has words", bs(P()[1]));
      typed(N()[1], "");
      note("partner empty", bs(P()[1]));              // joined, its empty partner gone with it
      note("other epochs", bs(P()[1]));               // "laughing @5-" under a line on 1-4: nothing happens
      flip(false); note("off, other epochs", bs(P()[1]));
      const c = box(P()[0]); c.setSelectionRange(7, 7); c.fire("keydown", {key: "Enter"});
      note("off, same epochs", bs(P()[1]));
    """)
    assert got == [
        ["partner has words", "smiling @1-4\n wide @1-4\nlaughing @5-", "frowning @1-4\nscowling @1-4\ncrying @5-",
         False],
        ["partner empty", "smiling wide @1-4\nlaughing @5-", "frowning @1-4\ncrying @5-", True],
        ["other epochs", "smiling wide @1-4\nlaughing @5-", "frowning @1-4\ncrying @5-", False],
        ["off, other epochs", "smiling wide @1-4\nlaughing @5-", "frowning @1-4\ncrying @5-", False],
        ["off, same epochs", "smiling wide @1-4\nlaughing @5-", "frowning @1-4\ncrying @5-", True]]


def test_undo_puts_both_boxes_and_their_pairs_back_until_the_next_change():
    got = _page(UNDO + r"""
      const all = () => JSON.stringify([S.values.FAMILY_SLIDER_POS, S.values.FAMILY_SLIDER_NEG,
        S.values.FAMILY_SLIDER_PAIRS]), was = all();
      P()[1].querySelector(".prompt-del").fire("click");      // x on a pair's +1 line: the pair goes from both boxes
      note("x", shown(POS), shown(NEG));
      U(POS).fire("click"); await tick(); note("undone", nums(P()), nums(N()), all() === was, shown(POS));
      N()[1].querySelector(".prompt-del").fire("click"); note("x -1", shown(POS), shown(NEG));
      typed(N()[0], "frowning hard"); note("typed: gone", shown(NEG));
      const b = box(P()[0]); b.value = "smiling wide"; b.setSelectionRange(7, 7); b.fire("keydown", {key: "Enter"});
      bs(P()[1]); note("joined", shown(POS));                  // Backspace joining lines
      U(POS).fire("click"); await tick(); note("split again", nums(P()), nums(N()));
      chain(P()[1]); note("unpaired", nums(P()), nums(N()), shown(POS));   // its -1 line had no words: it went
      U(POS).fire("click"); await tick(); note("paired again", nums(P()), nums(N()));
      // Paste several lines in the -1 box: its words fill pair 2's empty -1 line, then one more adds a pair
      const q = NEG.querySelector(".prompt-quick"); q.value = "glaring\nsulking @5-8";
      NEG.querySelector(".prompt-quickbox button").fire("click"); note("pasted", nums(P()), nums(N()), shown(NEG));
      U(NEG).fire("click"); await tick(); note("unpasted", nums(P()), nums(N()));
      const qp = POS.querySelector(".prompt-quick"); qp.value = "beaming";       // only added: no Undo
      qp.fire("keydown", {key: "Enter", ctrlKey: true}); note("added", shown(POS), shown(NEG));
    """)
    assert got == [
        ["x", "smiling @1-4", "frowning @1-4", True, False],
        ["undone", "smiling @1-4\nlaughing @5-", "frowning @1-4\ncrying @5-", "1 2", "1 2", True, False],
        ["x -1", "smiling @1-4", "frowning @1-4", False, True],
        ["typed: gone", "smiling @1-4", "frowning hard @1-4", False],
        ["joined", "smiling wide @1-4", "frowning hard @1-4", True],
        ["split again", "smiling @1-4\n wide @1-4", "frowning hard @1-4\n @1-4", "1 2", "1 2"],
        ["unpaired", "smiling @1-4\n wide @1-4", "frowning hard @1-4", "1 A", "1", True],
        ["paired again", "smiling @1-4\n wide @1-4", "frowning hard @1-4\n @1-4", "1 2", "1 2"],
        ["pasted", "smiling @1-4\n wide @1-4\n @5-8", "frowning hard @1-4\nglaring @1-4\nsulking @5-8", "1 2 3",
         "1 2 3", True],
        ["unpasted", "smiling @1-4\n wide @1-4", "frowning hard @1-4\n @1-4", "1 2", "1 2"],
        ["added", "smiling @1-4\n wide @1-4\nbeaming", "frowning hard @1-4\n @1-4\n", False, False]]


def test_paste_several_lines_sits_under_the_lines_and_an_empty_pair_line_says_what_its_pair_trains():
    got = _page(UNDO + r"""
      const order = (w) => w.children.map((c) => c.className.split(" ")[0] || c.tagName.toLowerCase());
      const text = (sel) => POS.querySelector(sel).textContent;
      note("layout", order(POS), order(NEG), text(".prompt-quick-title"), text(".prompt-quickbox button"),
        text(".prompt-actions button"));
      POS.querySelector(".prompt-actions button").fire("mousedown", {button: 0});   // + Add a line: a new pair
      note("new pair", box(P()[2]).placeholder, box(N()[2]).placeholder);
      typed(P()[2], "beaming"); note("+1 words", box(N()[2]).placeholder);
      chain(P()[1]); note("on its own", box(P()[1]).placeholder, box(N()[1]).placeholder);
    """)
    layout = got[0]
    for kids in layout[3:5]:            # the lines, what's wrong with them, Paste several lines, then the buttons
        assert kids.index("prompt-lines") < kids.index("field-hint") < kids.index("prompt-quickbox") < kids.index(
            "prompt-actions"), kids
    assert layout[5:] == ["Paste several lines", "Add", "+ Add a line"]
    assert got[1][3:] == ["No words yet in either box: pair 3 trains nothing"] * 2
    assert got[2][3:] == ["No words yet: pair 3 trains only its +1 words"]
    assert got[3][3:] == ["Type this line's prompt…"] * 2          # a line on its own: as before


def test_lines_pasted_in_the_plus_box_stay_as_pairs():
    # paired, several lines pasted in a +1 line (or in its Paste several lines, the undo check's "added") come as pairs,
    # each -1 line with no words yet - they used to go as the boxes sorted, the -1 box not having them yet
    got = _page(r"""
      const b = box(P()[0]); b.setSelectionRange(b.value.length, b.value.length);
      b.fire("paste", {clipboardData: {getData: () => "\ngrinning"}}); note("pasted", nums(P()), nums(N()));
    """)
    assert got == [["pasted", "smiling @1-4\ngrinning\nlaughing @5-", "frowning @1-4\n\ncrying @5-", "1 2 3", "1 2 3"]]


def test_the_move_buttons_are_big_enough_to_hit():
    rule = re.search(r"\.prompt-move button \{(.*?)\}", open(CSS, encoding="utf-8").read(), re.S).group(1)
    size = dict(re.findall(r"(?<![\w-])((?:min-)?(?:width|height)):\s*(\d+)px", rule))
    assert max(int(size.get("width", 0)), int(size.get("min-width", 0))) >= 20, rule
    assert max(int(size.get("height", 0)), int(size.get("min-height", 0))) >= 20, rule


def test_an_ends_lines_on_together_say_what_they_train_as(tmp_path):
    # as the picture's note: the lines on together, the first time, and the one prompt they make - and when
    desc = get_family("krea2")
    v, _inp = _inputs(tmp_path, desc, MAX_TRAIN_EPOCHS="10", FAMILY_SLIDER_POS="smiling\ngrinning",
                      FAMILY_SLIDER_NEG="frowning @1-4\ncrying @3-\nsulking @9-9")
    notes = form.state(desc, v, "")["notes"]
    assert notes["FAMILY_SLIDER_POS"] == ('"smiling" and "grinning" are on together at epochs 1–10 and train as one '
                                          'prompt: "smiling, grinning". For a change, give them epochs of their own.')
    assert notes["FAMILY_SLIDER_NEG"] == ('"frowning" and "crying" are on together at epochs 3–4 and train as one '
                                          'prompt: "frowning, crying". For a change, give them epochs of their own.')
    one = form.state(desc, {**v, "FAMILY_SLIDER_NEG": "frowning @1-5\ncrying @5-"}, "")["notes"]
    assert one["FAMILY_SLIDER_NEG"].startswith('"frowning" and "crying" are on together at epoch 5 and train as one ')
    alone = form.state(desc, {**v, "FAMILY_SLIDER_POS": "smiling @1-4\nlaughing @5-"}, "")["notes"]
    assert "FAMILY_SLIDER_POS" not in alone                          # none on together: no note
    off = form.state(desc, {**v, "FAMILY_SLIDER_BY_EPOCH": False}, "")["notes"]
    assert off["FAMILY_SLIDER_POS"].startswith(f'Holds 2 lines from "{launch.BY_EPOCH}"')     # the tick off: as before
