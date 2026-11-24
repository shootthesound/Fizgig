"""Captions > Search: the words that matched are washed behind, not just listed. A caption line is a plain textarea
and a textarea paints its own text, so the wash lives in a copy of the line held behind it. That works only while
the copy stays exactly where the box puts its text: ahead of the box in the DOM, pinned to its row so it re-wraps
where the box does, and never styled in any way that moves the words it sits behind.
"""
import os
import re

STATIC = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static")


def _js():
    return open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()


def _css():
    return open(os.path.join(STATIC, "style.css"), encoding="utf-8").read()


def _rule(css, selector):
    """One rule's declarations, so an assertion can name a property without pinning the order they're written in.
    Anchored to the start of a line: ".caption-lines .prompt-line textarea" is also a suffix of the big view's
    ".lightbox-cap .caption-lines .prompt-line textarea", and matching that instead would test the wrong rule."""
    found = re.search(r"(?m)^" + re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert found is not None, selector
    return found.group(1)


def test_filter_and_highlight_agree_on_the_query():
    """Which files show and where a hit is painted are the same match, read from one place - otherwise a line can
    be shown with its hit left unpainted, or the colour left behind on a line that no longer matches."""
    js = _js()
    assert re.search(r"const capQuery = \(\) => \$\(\"capSearch\"\)\.value\.trim\(\)\.toLowerCase\(\);", js)
    assert re.search(r"const q = capQuery\(\);", js)
    readers = [l for l in js.splitlines() if "capSearch" in l and ".value" in l]
    assert len(readers) == 1, readers


def test_only_a_line_with_a_hit_gets_a_copy():
    """Nothing is built when the box is empty or this line holds no hit: an ordinary render is exactly as it was,
    and the copy is dropped again the moment the words stop matching. Only caption lines get one - a search from
    the Captions tab must not wash the prompt boxes elsewhere on the page."""
    js = _js()
    start = js.index("const capHlPaint = (ta) => {")
    paint = js[start:js.index("\n};", start)]
    assert re.search(r'ta\.tagName !== "TEXTAREA" \|\| !ta\.closest\("\.caption-lines"\)', paint)
    assert re.search(r"if \(!q \|\| lower\.indexOf\(q\) < 0\)", paint)
    assert paint.index("if (!q ||") < paint.index('el("div", {class: "cap-hl"')


def test_the_copy_never_replaces_the_words():
    """It goes in ahead of the textarea, and both are positioned, so the box comes last and paints over the
    colour. aria-hidden because the copy holds the same words: read once, by the textarea, not twice."""
    js, css = _js(), _css()
    assert re.search(r"ta\.parentElement\.insertBefore\(hl, ta\)", js)
    assert re.search(r'"aria-hidden": "true"', js)
    assert "position: relative" in _rule(css, ".caption-lines .prompt-line")
    assert "position: relative" in _rule(css, ".caption-lines .prompt-line textarea")


def test_the_copy_is_pinned_to_its_row_and_measured_off_the_box():
    """Pinned on three sides, so it follows every resize and re-wraps where the textarea does - a width copied in
    pixels was wrong the moment the column moved. Only its left edge is set from JS, and the gutter that moves it
    (the line number, which arrives with the second caption) is re-applied wherever lines are counted. Font,
    padding and wrapping come from the textarea's own computed style: the big view sets its own font size and the
    table its own width, so a hardcoded copy drifts behind whichever one it forgot."""
    js, css = _js(), _css()
    hl = _rule(css, ".cap-hl")
    for decl in ["position: absolute", "top: 0", "bottom: 0", "right: 0", "white-space: pre-wrap",
                 "pointer-events: none", "user-select: none"]:
        assert decl in hl, decl
    assert re.search(r"const capHlPlace = \(ta\) => \{\n\s*if \(ta\._capHl\) ta\._capHl\.style\.left = ta\.offsetLeft", js)
    assert re.search(r"boxes\(\)\.forEach\(capHlPlace\);", js)
    assert js.index("capHlPaint(ta);") < js.index("if (grow && ta.scrollHeight <= ta.clientHeight) return;")
    assert re.search(r"const cs = getComputedStyle\(ta\);", js)
    for prop in ["fontFamily", "fontSize", "lineHeight", "paddingLeft", "whiteSpace", "overflowWrap", "wordBreak"]:
        assert f'"{prop}"' in js, prop


def test_the_wash_is_colour_only():
    """Padding, a border, a radius or a margin on the span moves the text it sits behind and the copy stops lining
    up - the whole failure mode of this trick. And the wash stays a wash: a flat mix with the theme's warning hue,
    never an opaque band."""
    css = _css()
    hit = _rule(css, ".cap-hl-hit")
    assert "background-color: var(--search-hit)" in hit and "color: inherit" in hit
    assert not re.search(r"padding|border|radius|margin|text-decoration", hit)
    assert re.search(r"--search-hit: color-mix\(in srgb, var\(--warning\) \d+%, transparent\);", css)
