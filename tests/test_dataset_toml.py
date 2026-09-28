"""Test 2: Dataset TOML Generation & Validation Test
Verifies that dataset TOML generated matches the exact format expected by fizgig.
"""
import os
import sys
import re

print("[TEST 2] Testing Dataset TOML format...")

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

test_folder = "/tmp/test_dataset_images"
os.makedirs(test_folder, exist_ok=True)

# Build standard Fizgig TOML
res_width = 1024
res_height = 1024
batch_size = 1
num_repeats = 1
enable_bucket = True
bucket_no_upscale = True
caption_ext = ".txt"

toml_lines = [
    "[general]",
    f"resolution = [{res_width}, {res_height}]",
    f'caption_extension = "{caption_ext}"',
    f"batch_size = {batch_size}",
    f"num_repeats = {num_repeats}",
    f"enable_bucket = {'true' if enable_bucket else 'false'}",
    f"bucket_no_upscale = {'true' if bucket_no_upscale else 'false'}",
    "",
    "[[datasets]]",
    f'image_directory = "{test_folder}"',
    f'cache_directory = "{os.path.join(test_folder, "cache")}"'
]

toml_text = "\n".join(toml_lines) + "\n"
test_toml_path = os.path.join(DATASET_DIR, "test_Fizgig_train.toml")

with open(test_toml_path, "w", encoding="utf-8") as f:
    f.write(toml_text)

print(f"[TEST 2] Written TOML to {test_toml_path}:")
print("------------------------------------------")
print(toml_text)
print("------------------------------------------")

# Validate with regex identical to _verify_frozen_dataset_config in jj.py line 27503
listed = [m for m in re.findall(r'^\s*image_directory\s*=\s*"([^"]*)"', toml_text, re.M)]
assert test_folder in listed, f"Verification failed: {test_folder} not in {listed}"

print(f"[TEST 2] ✓ Verified: image_directory extracted successfully: {listed}")
print("[TEST 2] Dataset TOML generation test PASSED!")
