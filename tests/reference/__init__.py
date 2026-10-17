"""OpenMOSS's reference MOSS-Image pipeline, vendored unchanged for parity tests only (Apache-2.0,
OpenMOSS-Team/MOSS-Image-Base `pipeline_moss_image.py`). Fizgig's pinned diffusers predates AutoencoderKLFlux2,
which the tests never instantiate, so `load()` gives diffusers a placeholder of that name before importing."""
import importlib.util
import os


def load():
    import diffusers
    if not hasattr(diffusers, "AutoencoderKLFlux2"):
        diffusers.AutoencoderKLFlux2 = type("AutoencoderKLFlux2", (), {})
    spec = importlib.util.spec_from_file_location("moss_reference_pipeline",
                                                  os.path.join(os.path.dirname(__file__), "pipeline_moss_image.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
