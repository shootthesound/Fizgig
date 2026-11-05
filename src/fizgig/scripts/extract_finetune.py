"""A full fine-tune as ordinary LoRAs (the web Extract tab's From a fine-tune; was the Checkpoint to LoRA app): the
difference between the trained checkpoint and the base it started from, one LoRA per rank from a single SVD pass
(fizgig.extraction.model_diff). Progress as "SVD: i/n" lines, each file written as "@@OUT {path}".

    python -m fizgig.scripts.extract_finetune --base BASE --tuned TRAINED --out-dir DIR --name PREFIX --ranks 32,64
"""
import argparse
import json
import os
import sys


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.abspath(os.path.join(here, "..", "..")), os.path.abspath(os.path.join(here, "..", "..", ".."))):
        if p not in sys.path:
            sys.path.insert(0, p)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="extracted")
    ap.add_argument("--ranks", default="32,64")
    ap.add_argument("--identity", action="store_true",
                    help="Identity mode: the model's identity blocks keep 4x each rank (MOSS-Image, Qwen Image 2.1)")
    a = ap.parse_args(argv)
    from fizgig.extraction.model_diff import extract_diff_loras
    last = [-1]

    def progress(done, total, _key):
        pct = done * 100 // max(1, total)
        if pct != last[0]:                      # one line a percent, not one a layer
            last[0] = pct
            print(f"SVD: {done}/{total}", flush=True)

    paths = extract_diff_loras(a.base, a.tuned, a.out_dir, [int(r) for r in a.ranks.split(",") if r.strip()],
                               name=a.name, progress=progress, log=lambda m: print(m, flush=True),
                               identity=a.identity)
    for p in paths:
        print("@@OUT " + json.dumps(p), flush=True)
    return 0 if paths else 1


if __name__ == "__main__":
    sys.exit(main())
