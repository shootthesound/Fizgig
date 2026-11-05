"""Model files: a Preferences row pointing at another model's file is found by its header and moved to the right one."""
import json
import os
import struct

from fizgig.scripts import fetch_models as fm

FLUX2 = [512, 32, 3, 3]          # Klein 9B's (and Flux 3's, Fibo 2.0's) VAE
FLUX1 = [512, 16, 3, 3]          # Z-Image's


def _vae(path, shape):
    hdr = json.dumps({"decoder.conv_in.weight": {"dtype": "F16", "shape": shape, "data_offsets": [0, 2]}}).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)) + hdr + b"\0\0")
    return str(path)


def test_klein_then_zimage_ae_is_renamed_and_zimage_fetches_its_own(tmp_path):
    """Klein downloaded first saved FLUX.2 as ae.safetensors, then Z-Image linked it: the file becomes
    flux2-vae.safetensors with Klein's row on it, and Z-Image's rows are cleared for its own download."""
    ae = _vae(tmp_path / "ae.safetensors", FLUX2)
    prefs = {"vae": ae, "zimage_vae": ae, "zimage_base_vae": ae, "other": "x"}
    assert fm.repair_links(prefs, str(tmp_path), log=lambda m: None)
    flux2 = str(tmp_path / "flux2-vae.safetensors")
    assert os.path.isfile(flux2) and not os.path.exists(ae)
    assert prefs == {"vae": flux2, "zimage_vae": "", "zimage_base_vae": "", "other": "x"}
    assert not fm.repair_links(prefs, str(tmp_path), log=lambda m: None)          # nothing left to do


def test_klein_then_zimage_with_flux2_vae_already_there(tmp_path):
    """The same with Flux 3's flux2-vae.safetensors already downloaded: Klein moves to it, and the stray ae is left
    for Z-Image's download to replace (no row points at it)."""
    ae = _vae(tmp_path / "ae.safetensors", FLUX2)
    flux2 = _vae(tmp_path / "flux2-vae.safetensors", FLUX2)
    prefs = {"vae": ae, "zimage_vae": ae, "flux3_vae": flux2}
    assert fm.repair_links(prefs, str(tmp_path), log=lambda m: None)
    assert prefs == {"vae": flux2, "zimage_vae": "", "flux3_vae": flux2}
    w = fm.Weight("zimage_vae", "r", "split_files/vae/ae.safetensors", 0.0, "")
    w.signature = ("decoder.conv_in.weight", tuple(FLUX1))
    assert fm.is_other_file(ae, w.signature)        # fetch_weight downloads over it instead of linking it


def test_zimage_then_klein_klein_moves_to_a_flux2_file(tmp_path):
    """Z-Image downloaded first saved FLUX.1 as ae.safetensors, then Klein linked it: Klein's row goes to a FLUX.2
    file another model has (else it is cleared); Z-Image's stays."""
    ae = _vae(tmp_path / "ae.safetensors", FLUX1)
    elsewhere = _vae(tmp_path / "fibo_vae.safetensors", FLUX2)
    prefs = {"vae": ae, "zimage_vae": ae, "flux3_vae": elsewhere}
    assert fm.repair_links(prefs, str(tmp_path), log=lambda m: None)
    assert prefs == {"vae": elsewhere, "zimage_vae": ae, "flux3_vae": elsewhere}
    prefs = {"vae": ae, "zimage_vae": ae}
    assert fm.repair_links(prefs, str(tmp_path), log=lambda m: None)
    assert prefs == {"vae": "", "zimage_vae": ae}


def test_files_it_cannot_tell_apart_are_left_alone(tmp_path):
    """A file without the tensor (another format), an unreadable one, or the right one: no change."""
    other = tmp_path / "diffusers_vae.safetensors"
    hdr = json.dumps({"something.else": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    other.write_bytes(struct.pack("<Q", len(hdr)) + hdr + b"\0\0")
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a model")
    right = _vae(tmp_path / "mine.safetensors", FLUX2)
    prefs = {"vae": str(other), "zimage_vae": str(junk), "flux3_vae": right}
    before = dict(prefs)
    assert not fm.repair_links(prefs, str(tmp_path), log=lambda m: None)
    assert prefs == before
