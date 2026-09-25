"""One-shot GUI launch metadata; never run matrix kernels or load bitsandbytes."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    from fizgig.utils.capabilities import detect
    print(json.dumps(asdict(detect())), flush=True)
