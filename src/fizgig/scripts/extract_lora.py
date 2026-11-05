"""CLI: reduce a LoRA to a lower rank - exact weight SVD straight from the file, no model loaded, in the family's own
key format. --preset keeps one of the family's block groups (Klein: Identity, Style+Composition, Details); --blocks
picks blocks by id.

Usage:
    python src/fizgig/scripts/extract_lora.py --family klein --source my_lora.safetensors \
        --output my_lora_r4.safetensors --rank 4 [--preset Identity | --blocks single_1,single_2]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    from fizgig.families.registry import FAMILIES, get
    parser = argparse.ArgumentParser(description="Reduce a LoRA's rank by exact weight SVD")
    parser.add_argument("--family", type=str, required=True, choices=sorted(FAMILIES),
                        help="The model family the LoRA was trained for")
    parser.add_argument("--source", type=str, required=True, help="The LoRA to reduce")
    parser.add_argument("--output", type=str, required=True, help="Where to write the reduced LoRA")
    parser.add_argument("--rank", type=int, default=4, help="The rank to keep (default 4)")
    parser.add_argument("--preset", type=str, default="",
                        help="One of the family's extract presets (its block groups); default all blocks")
    parser.add_argument("--blocks", type=str, default="",
                        help="Comma-separated block ids to keep, at full strength (instead of a preset)")
    parser.add_argument("--identity", action="store_true",
                        help="Identity mode: the model's identity blocks keep 4x the rank (MOSS-Image, Qwen Image 2.1)")
    args = parser.parse_args()

    from fizgig.families.extract import extract_weight_only
    desc = get(args.family)
    blocks = None
    if args.blocks:
        blocks = {b.strip(): 1.0 for b in args.blocks.split(",") if b.strip()}
    elif args.preset:
        presets = dict(desc.extract_presets)
        if args.preset not in presets:
            parser.error(f"{desc.display_name} presets: {', '.join(presets) or 'none (all blocks only)'}")
        blocks = dict(presets[args.preset]) or None
    if args.identity and not desc.identity_blocks:
        print(f"{desc.display_name} has no identity blocks: identity mode is off", flush=True)
    r = extract_weight_only(desc, args.source, args.output, args.rank, blocks=blocks, identity=args.identity,
                            progress=lambda stage, i, n: print(f"\r{stage}: {i + 1}/{n}", end="", flush=True))
    print(f"\nWrote {r['output']}: {r['layers']} layers at rank {args.rank}, "
          f"{100 * r['energy']:.1f}% of the change kept (mean), {r['seconds']:.1f}s")


if __name__ == "__main__":
    main()
