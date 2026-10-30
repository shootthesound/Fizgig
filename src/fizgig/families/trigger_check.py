"""Check a trigger word against a model's own tokenizer, before captioning: does the word Fizgig LoRA Isolation joins
it into ("zwxem, a woman" -> zwxemwoman) keep the class word whole, how many pieces is it, does it share a piece or a
first letter with the other subjects' words - and a few clean alternatives. Tokenizer only (CPU, no weights).

The caption's template (system prompt, chat markup) doesn't change how a word splits: these tokenizers split on
spaces before merging pieces, so the word is checked at a caption's start and mid-caption, as it appears."""
import random
import threading

from fizgig.families.multi_person import CLASS_WORDS

# family key -> what its text encoder tokenizes with: (loader, repo or None for the bundled Qwen3-VL copy, subfolder)
_SPECS = {
    "krea2": ("auto", None, None),
    "flux3": ("auto", None, None),
    "minimax": ("auto", None, None),
    "minimax_refmod": ("auto", None, None),          # H3's own text encoder
    "qwen_image21": ("auto", "Qwen/Qwen-Image-2.1", "processor"),
    "klein": ("qwen2", "Qwen/Qwen3-8B", None),
    "zimage": ("auto", "Tongyi-MAI/Z-Image-Turbo", "tokenizer"),
    "zimage_base": ("auto", "Tongyi-MAI/Z-Image", "tokenizer"),
    "anima": ("auto", "circlestone-labs/Anima-Base-v1.0-Diffusers", "tokenizer"),
    "sdxl": ("clip", "stabilityai/stable-diffusion-xl-base-1.0", "tokenizer"),
    "moss_image": ("auto", "OpenMOSS-Team/MOSS-Image-Base", "tokenizer"),
}
_TOK, _LOCK = {}, threading.Lock()


def supported(family):
    return family in _SPECS


def _tokenizer(family):
    with _LOCK:
        if family not in _TOK:
            kind, repo, sub = _SPECS[family]
            from fizgig.utils.hf_cache import from_pretrained_cache_first
            kw = {"subfolder": sub} if sub else {}
            if repo is None:
                from transformers import AutoTokenizer
                from fizgig.krea2.embedder import _bundled_tokenizer_dir
                _TOK[family] = AutoTokenizer.from_pretrained(_bundled_tokenizer_dir())
            elif kind == "qwen2":
                from transformers import Qwen2Tokenizer
                _TOK[family] = from_pretrained_cache_first(Qwen2Tokenizer, repo, **kw)
            elif kind == "clip":
                from transformers import CLIPTokenizer
                _TOK[family] = from_pretrained_cache_first(CLIPTokenizer, repo, **kw)
            else:
                from transformers import AutoTokenizer
                _TOK[family] = from_pretrained_cache_first(AutoTokenizer, repo, **kw)
        return _TOK[family]


def _pieces(tok, text):
    """The word's pieces as plain text (no space markers), as its caption would tokenize it."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return [tok.decode([i]).replace("</w>", "") for i in ids]


def _splits_clean(pieces, trigger, cls):
    """The pieces fall exactly at the trigger / class border: some leading pieces spell the trigger, the rest the
    class word."""
    text = ""
    for i, p in enumerate(pieces):
        text += p.strip()
        if text.lower() == trigger.lower():
            return "".join(q.strip() for q in pieces[i + 1:]).lower() == cls.lower()
        if len(text) > len(trigger):
            return False
    return False


def _verdict(tok, trigger, cls, others):
    """{ok, joined, pieces, mid_pieces, notes}: ok when the joined word splits cleanly at both positions and shares
    nothing with the other words."""
    trigger, cls = trigger.strip(), cls.strip()
    joined = trigger + cls if cls else trigger
    start, mid = _pieces(tok, joined), [p.strip() for p in _pieces(tok, " " + joined)]
    alone = [p.strip() for p in _pieces(tok, " " + trigger)]
    notes, ok = [], True
    if cls and not (_splits_clean(start, trigger, cls) and _splits_clean(mid, trigger, cls)):
        ok = False
        tip = ("Words ending in a vowel often do this; one ending in a consonant usually joins cleanly."
               if trigger[-1:].lower() in "aeiouy" else
               f"Its ending runs into '{cls}'. Try a different last letter, or one of the words below.")
        notes.append(f"breaks '{cls}': the model reads {joined} as {' | '.join(mid)}, so '{cls}' isn't a whole "
                     f"word any more. {tip}")
    if len(alone) == 1 and len(trigger) > 2:
        notes.append(f"'{trigger}' is a single piece the text encoder already knows - it may carry a meaning of its own. "
                     "A made-up word in two or three pieces is safer.")
    for o in (x.strip() for x in others or () if x.strip()):
        if o.lower() == trigger.lower():
            continue
        if o[:1].lower() == trigger[:1].lower():
            ok = False
            notes.append(f"starts with the same letter as '{o}' - the two subjects can blend. Pick a different first letter.")
        shared = {p.lower() for p in alone if len(p) > 1} & {p.strip().lower() for p in _pieces(tok, " " + o) if len(p.strip()) > 1}
        if shared:
            ok = False
            notes.append(f"shares the piece{'s' if len(shared) > 1 else ''} {', '.join(sorted(shared))} with '{o}'.")
    return {"ok": ok, "joined": joined, "pieces": mid, "start_pieces": [p.strip() for p in start], "notes": notes}


# made-up words shaped like the guide's good ones (dlor, gwelt, knurv, zwxem): an onset, a vowel, a consonant end
_ONSET = ("dl", "zw", "vl", "vr", "kv", "sv", "gw", "dv", "zv", "kn", "thr", "skr", "bl", "gr", "tr", "j", "v", "z", "k")
_VOWEL = ("a", "e", "i", "o", "u", "y")
_CODA = ("rv", "lk", "vk", "kv", "xm", "rx", "lv", "zk", "x", "sk", "rk", "nk", "lt", "rd")
# letter pairs that never sit together in an English word: every suggestion holds one, so none can be a real word
_NOT_ENGLISH = {"zw", "dl", "vl", "vr", "kv", "vk", "sv", "zv", "dv", "zk", "kz", "xm", "rx", "wx", "xk", "vz", "jv", "xv",
                "vx", "qz", "zx", "fv", "vf", "pz", "jz", "zj", "bx", "xb", "mx", "dx",
                "vd", "tx", "kx", "xz", "gx", "xg", "lx", "xl", "nx", "vm", "vn", "vp", "vb", "zr", "lz"}


def _not_english(w):
    w = w.lower()
    return any(w[i:i + 2] in _NOT_ENGLISH for i in range(len(w) - 1))


def _suggest(tok, cls, others, avoid, n=5, seed=0):
    rnd = random.Random(seed)
    used = {o[:1].lower() for o in others if o}
    out, tries = [], 0
    while len(out) < n and tries < 4000:
        tries += 1
        w = rnd.choice(_ONSET) + rnd.choice(_VOWEL) + rnd.choice(_CODA)
        if not 4 <= len(w) <= 6 or w[0] in used or w in out or w == avoid or not _not_english(w):
            continue
        v = _verdict(tok, w, cls, list(others) + out)
        if v["ok"] and len(v["pieces"]) >= 2 and not v["notes"]:
            out.append(w)
            used.add(w[0])
    return out


def check(family, trigger, cls="woman", others=()):
    """The tester's answer for one trigger word on one model's tokenizer."""
    if family not in _SPECS:
        return {"error": "This model's text encoder isn't covered by the trigger checker."}
    trigger = (trigger or "").strip().strip(",")
    cls = (cls or "").strip().lower()
    if not trigger or " " in trigger:
        return {"error": "Type one trigger word (no spaces)."}
    if cls and cls not in CLASS_WORDS:
        cls = cls.split()[-1]
    tok = _tokenizer(family)
    v = _verdict(tok, trigger, cls, others)
    v["suggestions"] = [] if v["ok"] and not v["notes"] else _suggest(tok, cls, [*others, trigger], trigger,
                                                                   seed=sum(map(ord, trigger)))
    v["trigger"], v["cls"] = trigger, cls
    from fizgig.families.registry import get as get_family
    d = get_family("minimax" if family == "minimax_refmod" else family)
    v["encoder"] = (f"{d.display_name}'s text encoder" + (f" ({d.text_encoder_label})" if getattr(d, "text_encoder_label", "")
                                                         else "")) if d else family
    return v
