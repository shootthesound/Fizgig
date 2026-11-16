"""Add a Chinese translation to every caption in a folder (English + Chinese bilingual captions), in its own process.

The web UI's Captions tab runs this; the desktop app's Bilingual card does the same in-process. Same model
(Helsinki-NLP/opus-mt-en-zh, ~300 MB, downloaded on first use), same format: `<trigger>, <english> - <chinese>`, the
trigger word (the first comma-separated part) kept as it is.

    python -m fizgig.scripts.translate_captions <folder> [--skip_existing] [--ext .txt] [--stop_file PATH]

Stdout (flushed): INFO / PROGRESS <current> <total> / OK <name> / SKIP <name> (<why>) / FAIL <name> (<why>) /
STOPPED / DONE <translated> <skipped> <failed>.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_ID = "Helsinki-NLP/opus-mt-en-zh"
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def _say(line):
    print(line, flush=True)


def _load():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from fizgig.utils.hf_cache import from_pretrained_cache_first
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = from_pretrained_cache_first(AutoTokenizer, MODEL_ID)
    model = from_pretrained_cache_first(AutoModelForSeq2SeqLM, MODEL_ID,
                                        torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    model.eval()
    return tok, model, device


def _translate(tok, model, device, english):
    import torch
    if not english.strip():
        return ""
    inputs = tok(english.strip(), return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, num_beams=4, early_stopping=True)
    text = tok.decode(out[0], skip_special_tokens=True).strip().strip('"').strip("'").strip()
    return " ".join(text.split())


def main():
    p = argparse.ArgumentParser(description="Add a Chinese translation to every caption in a folder")
    p.add_argument("folder")
    p.add_argument("--list_file", default="", help="Explicit UTF-8 list of caption paths")
    p.add_argument("--skip_existing", action="store_true", help="Leave captions that already contain Chinese")
    p.add_argument("--ext", default=".txt", help="The caption files' extension (.txt, .caption)")
    p.add_argument("--stop_file", default="")
    a = p.parse_args()
    # captions only: a song's <name>.lyrics.txt are the words sung, not a description to translate
    if a.list_file:
        with open(a.list_file, encoding="utf-8") as f:
            candidates = [line.strip() for line in f if line.strip()]
    else:
        candidates = glob.glob(os.path.join(glob.escape(a.folder), "*" + a.ext))
    files = sorted(set(f for f in candidates if os.path.isfile(f)
                       and f.lower().endswith(a.ext.lower()) and not f.lower().endswith(".lyrics.txt")))
    if not files:
        _say(f"INFO: no {a.ext} files in {a.folder}")
        _say("DONE 0 0 0")
        return 0
    _say(f"INFO: loading the translator ({MODEL_ID}); the first time it downloads ~300 MB")
    try:
        tok, model, device = _load()
    except Exception as e:
        _say(f"FAIL: translator ({e})")
        return 1
    _say(f"INFO: translating {len(files)} captions on {device}")
    done = skipped = failed = 0
    for i, fp in enumerate(files, 1):
        if a.stop_file and os.path.exists(a.stop_file):
            _say("STOPPED")
            break
        _say(f"PROGRESS: {i} {len(files)}")
        name = os.path.basename(fp)
        try:
            with open(fp, encoding="utf-8") as f:
                original = f.read().strip()
            if not original:
                skipped += 1
                _say(f"SKIP: {name} (empty)")
                continue
            if a.skip_existing and _CJK.search(original):
                skipped += 1
                _say(f"SKIP: {name} (already has Chinese)")
                continue
            trigger, rest = original.split(", ", 1) if ", " in original else ("", original)
            chinese = _translate(tok, model, device, rest or original)
            if not chinese:
                failed += 1
                _say(f"FAIL: {name} (empty translation, kept as it was)")
                continue
            with open(fp, "w", encoding="utf-8") as f:
                f.write(f"{trigger}, {rest} - {chinese}" if trigger else f"{original} - {chinese}")
            done += 1
            _say(f"OK: {name}")
        except Exception as e:
            failed += 1
            _say(f"FAIL: {name} ({type(e).__name__}: {e})")
    _say(f"DONE {done} {skipped} {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
