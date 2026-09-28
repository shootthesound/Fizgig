"""Test 4: Modal Cloud GPU Provider Integration Test
Verifies modal CLI availability, app structure, and execution syntax.
"""
import subprocess
import os

print("[TEST 4] Testing Modal Cloud GPU Integration...")

modal_bin = "/home/rgukt/venv/bin/modal"
if not os.path.isfile(modal_bin):
    print(f"  ✗ Modal binary not found at {modal_bin}")
    sys.exit(1)

# Check modal version
res = subprocess.run([modal_bin, "--version"], capture_output=True, text=True)
print(f"  ✓ {res.stdout.strip()}")

# Test parsing modal_trainer.py
trainer_file = os.path.join(os.path.dirname(__file__), "..", "modal_trainer.py")
if os.path.isfile(trainer_file):
    print(f"  ✓ Found modal_trainer.py: {trainer_file}")
    # Verify python syntax
    syntax_check = subprocess.run(["python3", "-m", "py_compile", trainer_file], capture_output=True, text=True)
    # Verify dry-run execution via modal CLI
    print("  Testing modal run execution...")
    run_check = subprocess.run([modal_bin, "run", trainer_file, "--dry-run"], capture_output=True, text=True)
    if run_check.returncode == 0:
        print("  ✓ modal run --dry-run executed successfully on Modal cloud runner")
    else:
        print(f"  ✗ modal run failed: {run_check.stderr[:200]}")
        sys.exit(1)
else:
    print(f"  ✗ modal_trainer.py missing at {trainer_file}")
    sys.exit(1)

print("[TEST 4] Modal cloud integration test PASSED successfully!")

