"""modal_dispatch.py — CLI bridge for Next.js to spawn detached Modal GPU jobs."""
import os
import sys
import json
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Fizgig Modal Dispatcher")
    parser.add_argument("--settings-json", default="{}")
    parser.add_argument("--toml-path", default="")
    parser.add_argument("--gpu", default="A100-40GB")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_modal_file = os.path.join(base_dir, "run_modal.py")
    modal_bin = "/home/rgukt/venv/bin/modal" if os.path.isfile("/home/rgukt/venv/bin/modal") else "modal"

    target_gpu = args.gpu.strip()
    cmd = [
        modal_bin, "run", run_modal_file,
        "--settings-json", args.settings_json,
        "--toml-path", args.toml_path,
        "--gpu", target_gpu,
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"[modal-dispatch] Dispatching job targeting GPU: {target_gpu}...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        print(proc.stdout)
        print(json.dumps({"success": True, "gpu": target_gpu, "output": proc.stdout}))
    else:
        print(f"[modal-dispatch] Error: {proc.stderr}")
        print(json.dumps({"success": False, "error": proc.stderr}))
        sys.exit(1)

if __name__ == "__main__":
    main()
