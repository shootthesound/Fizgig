"""One-time conversion of an ultralytics YOLO face model (.pt) to .onnx for Fizgig's YOLO face finder.

Fizgig runs YOLO face models with onnxruntime only: no ultralytics (AGPL) dependency, and no unpickling of
downloaded .pt files inside Fizgig. Converting needs ultralytics ONCE, so run this with a Python that already
has it - e.g. Forge / A1111's (ADetailer installs ultralytics), or a throwaway venv (`pip install ultralytics`).

    # Windows, Forge's bundled Python (adjust the paths):
    C:\\...\\webui_forge\\system\\python\\python.exe convert_yolo_to_onnx.py C:\\...\\fdetailerAdetailerFor_v11.pt
    # -> fdetailerAdetailerFor_v11.onnx next to the .pt (or --out PATH)

Then save it as user_data/models/face/fdetailer_v11.onnx in the Fizgig folder (used whenever the YOLO model setting
is blank), or give its path in Image Prep -> Face settings -> YOLO model.

A .pt file IS a pickle - loading it can run code. Only convert models from sources you trust (this is the same
load ADetailer already does with them). Nothing is downloaded: ultralytics auto-install/update checks are off.
"""

import argparse
import os
import shutil
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pt", help="the ultralytics YOLO .pt model (detect or segment)")
    ap.add_argument("--out", help="output .onnx path (default: next to the .pt)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="input size baked into the ONNX (default 640: as accurate as 1024 for faces on our tests, "
                         "~2x faster on CPU)")
    ap.add_argument("--opset", type=int, default=None, help="ONNX opset (default: ultralytics' choice)")
    a = ap.parse_args(argv)

    os.environ.setdefault("YOLO_AUTOINSTALL", "false")    # never pip-install into this Python
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("YOLO_VERBOSE", "false")
    src = os.path.abspath(a.pt)
    if not os.path.isfile(src):
        sys.exit(f"not found: {src}")
    with open(src, "rb") as f:
        head = f.read(64)
    if head.lstrip().startswith((b"<", b"{")):
        sys.exit(f"{src} is not a model (it starts with {head[:20]!r} - an HTML/JSON page, e.g. a download that "
                 "needed a login). Download the model again from the model page in a browser.")
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics isn't installed in this Python. Run this script with Forge/A1111's Python "
                 "(ADetailer ships ultralytics) or a throwaway venv: python -m venv yolo && yolo\\Scripts\\pip "
                 "install ultralytics onnx")
    model = YOLO(src)
    names = getattr(model, "names", {}) or {}
    print(f"[convert] {os.path.basename(src)}: task={getattr(model, 'task', '?')} classes={names}")
    exported = model.export(format="onnx", imgsz=a.imgsz, dynamic=False, simplify=False, half=False,
                            **({"opset": a.opset} if a.opset else {}))
    exported = str(exported)
    out = os.path.abspath(a.out) if a.out else os.path.splitext(src)[0] + ".onnx"
    if os.path.abspath(exported) != out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        shutil.move(exported, out)              # os.replace fails across drives
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        print(f"[convert] check: input {sess.get_inputs()[0].shape}, outputs "
              f"{[o.shape for o in sess.get_outputs()]}")
    except ImportError:
        pass
    face = [n for n in names.values() if str(n).lower() in ("face", "head")]
    print(f"[convert] wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(f"[convert] Fizgig will use class {face[0]!r}" if face else
          "[convert] no 'face'/'head' class name - Fizgig uses class 0 for one-class models, otherwise none")
    return out


if __name__ == "__main__":
    main()
