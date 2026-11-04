"""The Detail crops stage (families/launch.py, before caching): each size the run trains at gets its crops
(dataset/detail_crops.py). A picture with no crop caption at any size is captioned once, at its largest crop, with
the Qwen3-VL captioner and the Captions tab's Style instruction (trigger word first); each picture's caption at its
largest captioned size is then written over its crop at every other size.

    python -m fizgig.scripts.detail_crops --folder <training folder> --mp 1.0 [--mp 0.25] --captioner <file>
        --instruction_file <file> [--trigger zwxem] [--ext .txt]"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fizgig.dataset import detail_crops  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--folder", required=True)
    ap.add_argument("--mp", type=float, action="append", required=True)
    ap.add_argument("--captioner", default="")
    ap.add_argument("--instruction_file", default="")
    ap.add_argument("--trigger", default="")
    ap.add_argument("--ext", default=".txt")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    a = ap.parse_args()
    for mp in a.mp:
        detail_crops.make(a.folder, mp, log=lambda t: print(t, flush=True), ext=a.ext)
    # one caption per picture: copied from another size when one has it, else written for its largest crop
    n = detail_crops.share_captions(a.folder, a.ext)
    if n:
        print(f"[detail crops] {n} caption(s) copied from the same crops at another size", flush=True)
    todo = detail_crops.to_caption(a.folder, a.ext)
    if not todo:
        print("[detail crops] every crop has its caption", flush=True)
        return 0
    if not (a.captioner and os.path.isfile(a.captioner) and a.instruction_file):
        print(f"[detail crops] {len(todo)} crop(s) have no caption and no captioner is set: set the Krea 2 text "
              "encoder (or a secondary captioning model) in Preferences", flush=True)
        return 1
    import torch
    from fizgig.scripts import batch_caption as bc
    with open(a.instruction_file, encoding="utf-8") as f:
        instruction = f.read().strip()
    print(f"[detail crops] captioning {len(todo)} crop(s) with Qwen3-VL", flush=True)
    state = bc._load_state_from_config({"backend": "qwen", "text_encoder": a.captioner},
                                       "cuda" if torch.cuda.is_available() else "cpu")
    trig = a.trigger if a.trigger and a.trigger.lower() != "trigger_word" else ""
    for i, p in enumerate(todo, 1):
        try:
            cap = bc._caption_qwen(state.model, p, instruction, a.max_new_tokens)
            if cap:
                bc._write_caption(p, cap, trig, a.ext)
            print(f"[cache] captions {i}/{len(todo)}", flush=True)       # the run card's count
        except Exception as e:
            print(f"[detail crops] {os.path.basename(p)}: no caption ({e})", flush=True)
    del state.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    n = detail_crops.share_captions(a.folder, a.ext)
    if n:
        print(f"[detail crops] {n} caption(s) copied to the same crops at the other sizes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
