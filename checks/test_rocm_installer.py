"""Exercise installer stages with fake git/Python, without changing the host ROCm stack."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

INSTALLER = Path(__file__).resolve().parents[1] / "install_fizgig_rocm.sh"
REPO = "https://github.com/IxMxAMAR/SageAttention-RDNA4.git"
SAGE_COMMIT = "068036781b2061d4ee89277b52749165e3f7a887"


@pytest.fixture
def install(tmp_path):
    root = tmp_path / "Fizgig with spaces"
    fake_bin = tmp_path / "bin"
    toolkit = tmp_path / "ROCm toolkit"
    for folder in (root / "venv/bin", fake_bin, toolkit):
        folder.mkdir(parents=True)
    log = tmp_path / "calls.jsonl"
    runner = f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
is_git = pathlib.Path(sys.argv[0]).name == "git"
entry = {"kind": "git" if is_git else "python", "args": args,
         "cwd": os.getcwd(), "exe": sys.argv[0],
         "env": {k: os.environ.get(k) for k in
                 ("ROCM_HOME", "HIP_PATH", "PYTORCH_ROCM_ARCH", "MAX_JOBS")}}
if args == ["-"]:
    entry["script"] = sys.stdin.read()
if not is_git and args[:4] == ["-m", "pip", "install", "."]:
    entry["setup"] = pathlib.Path("setup.py").read_text()
    entry["revision"] = pathlib.Path(".git/test-head").read_text()
with open(os.environ["SAGE_TEST_LOG"], "a") as f:
    f.write(json.dumps(entry) + "\\n")
if is_git:
    if args[0] == "clone":
        if os.environ.get("SAGE_TEST_FAIL_CLONE"):
            sys.exit(43)
        target = pathlib.Path(args[-1])
        (target / ".git").mkdir(parents=True)
        (target / ".git/origin").write_text(args[-2])
        (target / "setup.py").write_text('python_requires=">=3.12,<3.13",\\n# keep existing source\\n')
    elif args[0] == "-C":
        metadata = pathlib.Path(args[1]) / ".git"
        if args[2] == "remote":
            print((metadata / "origin").read_text())
        elif args[2] == "fetch":
            if os.environ.get("SAGE_TEST_FAIL_FETCH"):
                sys.exit(44)
        elif args[2] == "checkout":
            (metadata / "test-head").write_text(args[-1])
elif args[:4] == ["-m", "pip", "install", "."] and os.environ.get("SAGE_TEST_FAIL_BUILD"):
    sys.exit(37)
'''
    for exe in (root / "venv/bin/python", fake_bin / "git"):
        exe.write_text(runner)
        exe.chmod(0o755)
    (fake_bin / "nproc").write_text("#!/bin/sh\necho 12\n")
    (fake_bin / "nproc").chmod(0o755)
    env = {**os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
           "SAGE_TEST_ROOT": str(root), "SAGE_TEST_LOG": str(log), "ROCM_HOME": str(toolkit),
           "SAGE_TEST_ARCH": "gfx1201", "VIRTUAL_ENV": str(root / "venv")}
    env.pop("MAX_JOBS", None)

    def run(stage="install_sageattention_rdna4", **changes):
        command = 'source "$1"; FIZGIG_ROOT="$SAGE_TEST_ROOT"; ARCH="$SAGE_TEST_ARCH"; ' + stage
        return subprocess.run(["bash", "-c", command, "installer-test", str(INSTALLER)],
                              env={**env, **changes}, text=True, capture_output=True)

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return root, toolkit, run, calls


def test_clone_and_build_use_fizgig_venv_and_requested_flags(install):
    root, toolkit, run, calls = install
    result = run(VIRTUAL_ENV="/some/other/venv")
    assert result.returncode == 0, result.stderr
    records = calls()
    source = root / "user_data/src/SageAttention-RDNA4"
    assert records[0]["args"] == ["clone", "--no-checkout", "--depth", "1", REPO, str(source)]
    assert records[1]["args"] == ["-C", str(source), "fetch", "--depth", "1", "origin", SAGE_COMMIT]
    assert records[2]["args"] == ["-C", str(source), "checkout", "--detach", SAGE_COMMIT]
    build = next(r for r in records if r["args"][:4] == ["-m", "pip", "install", "."])
    assert build["exe"] == str(root / "venv/bin/python")
    assert build["cwd"] == str(source)
    assert build["args"] == ["-m", "pip", "install", ".", "--no-build-isolation", "--no-deps"]
    assert 'python_requires=">=3.12,<3.15"' in build["setup"]
    assert '<3.13' not in build["setup"]
    assert build["revision"] == SAGE_COMMIT
    assert "# keep existing source" in build["setup"]
    assert build["env"] == {"ROCM_HOME": str(toolkit), "HIP_PATH": str(toolkit),
                            "PYTORCH_ROCM_ARCH": "gfx1201", "MAX_JOBS": "12"}
    assert records[-1]["cwd"] == str(root)  # import verification outside the clone
    assert "from sageattention import sageattn" in records[-1]["script"]


def test_repeat_install_keeps_existing_clone_and_local_changes(install):
    root, _, run, calls = install
    assert run().returncode == 0
    marker = root / "user_data/src/SageAttention-RDNA4/local-change"
    marker.write_text("keep me")
    result = run(MAX_JOBS="4")
    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "keep me"
    setup = (marker.parent / "setup.py").read_text()
    assert setup.count('python_requires=">=3.12,<3.15"') == 1
    assert "# keep existing source" in setup
    assert len([r for r in calls() if r["args"][0] == "clone"]) == 1
    assert calls()[-2]["env"]["MAX_JOBS"] == "4"


@pytest.mark.parametrize("flag", ["SAGE_TEST_FAIL_BUILD", "SAGE_TEST_FAIL_CLONE", "SAGE_TEST_FAIL_FETCH"])
def test_failures_stop_the_optional_build(install, flag):
    _, _, run, calls = install
    result = run(**{flag: "1"})
    assert result.returncode != 0
    assert not any("script" in r for r in calls())  # no success/import check after a failed build


@pytest.mark.parametrize("flag", ["SAGE_TEST_FAIL_BUILD", "SAGE_TEST_FAIL_CLONE", "SAGE_TEST_FAIL_FETCH"])
def test_optional_failure_warns_and_continues_in_conditional_context(install, flag):
    _, _, run, calls = install
    # This context suppresses Bash's implicit errexit inside functions: every
    # failing command must propagate failure explicitly for the warning to work.
    result = run('if ! install_sageattention_rdna4; then echo "WARN: build failed" >&2; fi; echo continued',
                 **{flag: "1"})
    assert result.returncode == 0 and "WARN: build failed" in result.stderr
    assert "continued" in result.stdout
    assert not any("script" in r for r in calls())


def test_other_amd_cards_skip_this_hardware_specific_build(install):
    _, _, run, calls = install
    result = run(SAGE_TEST_ARCH="gfx1100")
    assert result.returncode == 0 and "Skipping" in result.stdout
    assert calls() == []


def test_existing_unrelated_directory_is_not_overwritten(install):
    root, _, run, calls = install
    source = root / "user_data/src/SageAttention-RDNA4"
    source.mkdir(parents=True)
    (source / "keep").write_text("keep")
    assert run().returncode != 0
    assert (source / "keep").read_text() == "keep"
    assert calls() == []


def test_missing_toolkit_fails_before_cloning(install):
    _, toolkit, run, calls = install
    toolkit.rmdir()
    result = run()
    assert result.returncode != 0 and "ROCm build toolkit" in result.stderr
    assert calls() == []


def test_clone_with_another_origin_is_left_untouched(install):
    root, _, run, calls = install
    source = root / "user_data/src/SageAttention-RDNA4"
    (source / ".git").mkdir(parents=True)
    (source / ".git/origin").write_text("https://example.com/something-else.git")
    result = run()
    assert result.returncode != 0 and "not the expected" in result.stderr
    assert not any(r["kind"] == "python" for r in calls())


@pytest.mark.parametrize("channel,index", [("stable", "https://stable.repo.amd.com/rocm/whl-next/"),
                                          ("nightly", "https://nightly.repo.amd.com/rocm/whl-next/")])
def test_torch_install_stage_has_no_missing_helper_calls(install, channel, index):
    _, _, run, calls = install
    result = run("install_torch_" + channel, ROCM_CHANNEL=channel)
    assert result.returncode == 0, result.stderr
    assert "command not found" not in result.stderr
    records = calls()
    assert index in records[0]["args"]
    assert sum("GPU kernel smoke test" in r.get("script", "") for r in records) == 1
    assert sum("torchvision import" in r.get("script", "") for r in records) == 1
