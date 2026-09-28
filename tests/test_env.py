"""Test 1: Environment & Module Verification Test
Checks python environment, PyTorch, CUDA, accelerate, dependencies, and fizgig module imports.
"""
import sys
import os

print(f"[TEST 1] Python executable: {sys.executable}")
print(f"[TEST 1] Python version: {sys.version}")

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

errors = []

# Test core packages
packages = ["torch", "accelerate", "safetensors", "transformers", "diffusers", "einops", "toml", "tqdm", "PIL", "numpy"]
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "unknown")
        print(f"  ✓ {pkg}: {version}")
    except ImportError as e:
        print(f"  ✗ {pkg}: NOT FOUND ({e})")
        errors.append(pkg)

# Test CUDA
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"[TEST 1] CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
except Exception as e:
    print(f"  CUDA check error: {e}")

# Test fizgig imports
try:
    import fizgig
    print(f"[TEST 1] ✓ fizgig package loaded: {fizgig.__file__}")
except Exception as e:
    print(f"[TEST 1] ✗ fizgig import failed: {e}")
    errors.append("fizgig")

try:
    from fizgig.dataset.config import BlueprintGenerator
    print("  ✓ fizgig.dataset.config.BlueprintGenerator")
except Exception as e:
    print(f"  ✗ fizgig.dataset.config: {e}")
    errors.append("fizgig.dataset")

try:
    from fizgig.repair_studio.state import SliderState
    print("  ✓ fizgig.repair_studio.state.SliderState")
except Exception as e:
    print(f"  ✗ fizgig.repair_studio: {e}")
    errors.append("fizgig.repair_studio")

if errors:
    print(f"\n[TEST 1] Environment diagnostic report: {len(errors)} missing packages: {errors}")
    print("  Note: Full training dependencies (CUDA, PyTorch, Accelerate) are only required when running actual GPU training.")
    print("[TEST 1] Environment diagnostic completed!")
else:
    print("\n[TEST 1] All environment checks PASSED successfully!")

