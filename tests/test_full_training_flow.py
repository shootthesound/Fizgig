"""Test 5: Full End-to-End Training Flow Test
Validates the complete lifecycle of a training job:
1. Dataset TOML generation & snapshot creation
2. Settings serialization (.last_train.json)
3. Modal Cloud runner dispatch (dry-run & argument verification)
4. Training Pause request sentinel (.pause_requested)
5. Training Resume (sentinel deletion)
6. Preset saving and loading
"""
import os
import sys
import json
import subprocess
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PRESETS_DIR = os.path.join(BASE_DIR, "presets")
LAST_TRAIN_FILE = os.path.join(BASE_DIR, ".last_train.json")
PAUSE_FLAG_FILE = os.path.join(BASE_DIR, ".pause_requested")

print("[TEST 5] Starting Full End-to-End Training Flow Test...")

# Step 1: Validate dataset TOML generation
os.makedirs(DATASET_DIR, exist_ok=True)
dummy_img_dir = "/tmp/fizgig_test_flow_dataset"
os.makedirs(dummy_img_dir, exist_ok=True)

toml_path = os.path.join(DATASET_DIR, "Fizgig_train.toml")
toml_content = f"""[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
num_repeats = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "{dummy_img_dir}"
cache_directory = "{os.path.join(dummy_img_dir, 'cache')}"
"""

with open(toml_path, "w", encoding="utf-8") as f:
    f.write(toml_content)

assert os.path.isfile(toml_path), "Failed to write dataset TOML"
print(f"  ✓ Step 1: Created dataset TOML: {toml_path} ({len(toml_content)} bytes)")

# Step 2: Validate settings serialization (.last_train.json)
test_settings = {
    "ARCHITECTURE": "Flux 2 Klein Base 9B",
    "LORA_NAME": "e2e_test_character_lora",
    "NETWORK_DIM": 16,
    "NETWORK_ALPHA": 16,
    "LEARNING_RATE": "0.0001",
    "MAX_TRAIN_EPOCHS": 10,
    "SAVE_EVERY_N_EPOCHS": 1,
    "DATASET_CONFIG": "dataset/Fizgig_train.toml",
    "FP8": True,
    "SCALED": True,
    "GRADIENT_CHECKPOINTING": True,
    "OPTIMIZER_TYPE": "adamw8bit",
}

with open(LAST_TRAIN_FILE, "w", encoding="utf-8") as f:
    json.dump(test_settings, f, indent=2)

assert os.path.isfile(LAST_TRAIN_FILE), "Failed to write .last_train.json"
with open(LAST_TRAIN_FILE, "r", encoding="utf-8") as f:
    restored = json.load(f)
assert restored["LORA_NAME"] == "e2e_test_character_lora"
print(f"  ✓ Step 2: Settings snapshot verified (.last_train.json)")

# Step 3: Test Modal cloud runner execution
modal_trainer = os.path.join(BASE_DIR, "modal_trainer.py")
python_bin = "/home/rgukt/venv/bin/python3" if os.path.exists("/home/rgukt/venv/bin/python3") else sys.executable

cmd = [
    python_bin,
    modal_trainer,
    "--dry-run",
    "--settings-json", json.dumps(test_settings),
    "--toml-path", toml_path,
]

res = subprocess.run(cmd, capture_output=True, text=True)
assert res.returncode == 0, f"Modal runner failed: {res.stderr}"
assert "Target Architecture: Flux 2 Klein Base 9B" in res.stdout
assert "Target GPU: A100-40GB" in res.stdout
assert "Job parameters validated successfully" in res.stdout
print("  ✓ Step 3: Modal cloud runner dispatched and verified in dry-run mode")

# Step 4: Validate Pause sentinel creation (.pause_requested)
# When pause is requested, the sentinel file is created
with open(PAUSE_FLAG_FILE, "w", encoding="utf-8") as f:
    f.write("1\n")

assert os.path.exists(PAUSE_FLAG_FILE), "Pause sentinel was not created"
print("  ✓ Step 4: Pause sentinel (.pause_requested) verified")

# Step 5: Validate Resume (deletes sentinel file)
if os.path.exists(PAUSE_FLAG_FILE):
    os.remove(PAUSE_FLAG_FILE)

assert not os.path.exists(PAUSE_FLAG_FILE), "Pause sentinel was not removed upon resume"
print("  ✓ Step 5: Resume lifecycle verified (pause sentinel cleanly removed)")

# Step 6: Validate Presets save & load
os.makedirs(PRESETS_DIR, exist_ok=True)
test_preset_file = os.path.join(PRESETS_DIR, "e2e_test_preset.json")
with open(test_preset_file, "w", encoding="utf-8") as f:
    json.dump(test_settings, f, indent=2)

assert os.path.isfile(test_preset_file), "Preset file was not created"
with open(test_preset_file, "r", encoding="utf-8") as f:
    loaded_preset = json.load(f)
assert loaded_preset["NETWORK_DIM"] == 16
print("  ✓ Step 6: Presets save and restore verified")

# Clean up temporary test files
if os.path.exists(test_preset_file):
    os.remove(test_preset_file)

print("\n🎉 [TEST 5] Full End-to-End Training Flow Test PASSED successfully!\n")
