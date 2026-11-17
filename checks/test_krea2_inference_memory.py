"""CPU checks for Krea 2's inference memory lifecycle."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

from fizgig.families.registry import get
from fizgig.families.workbench import WorkbenchEngine
from fizgig.krea2 import workbench as W
from fizgig.krea2.driver import Krea2Driver


class VAE(torch.nn.Module):
    def __init__(self, failures=1, error=torch.OutOfMemoryError):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.use_tiling = False
        self.failures, self.error, self.calls = failures, error, 0
        self.clear_cache = Mock()

    def decode_to_pixels(self, latents):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error("decode failed")
        return torch.zeros(1, 3, 2, 2)

    def enable_tiling(self):
        self.use_tiling = True

    def disable_tiling(self):
        self.use_tiling = False


class InferenceMemoryChecks(unittest.TestCase):
    def setUp(self):
        self.engine = get("krea2").make_workbench_engine()
        self.assertIsInstance(self.engine, W.Krea2WorkbenchEngine)
        self.enterContext(patch.object(torch.cuda, "empty_cache"))
        self.enterContext(patch.object(torch.cuda, "is_available", return_value=False))
        self.enterContext(patch.object(torch.cuda, "current_device", return_value=0))
        self.enterContext(patch.object(torch.version, "hip", "test"))

    def test_auto_reserves_more_memory_for_higher_resolution(self):
        engine = self.engine
        engine.dit = SimpleNamespace(blocks=[object() for _ in range(8)], parameters=lambda: (), buffers=lambda: ())
        engine.swapped, engine._auto_swap, engine._reserve_extra = 0, True, 0
        weight = SimpleNamespace(device=torch.device("cuda"), numel=lambda: W.GIB, element_size=lambda: 1)
        with patch.object(W, "_block_weights", return_value=[weight]), \
                patch.object(W, "_free_vram_gb", return_value=1), \
                patch.object(engine, "_set_swap") as change, \
                patch.object(engine, "_park_dit"), \
                patch.object(WorkbenchEngine, "render", return_value="image"):
            self.assertEqual(engine.render({}, 512, 512, 42), "image")
            small = change.call_args.args[0]
            engine.render({}, 2048, 2048, 42)
            large = change.call_args.args[0]
        self.assertGreater(large, small)
        self.assertLessEqual(large, 6)  # two blocks must remain on device

    def test_manual_swap_is_not_replanned(self):
        self.engine._auto_swap = False
        with patch.object(self.engine, "_set_swap") as change, \
                patch.object(WorkbenchEngine, "render", return_value="image"):
            self.engine.render({}, 2048, 2048, 42)
        change.assert_not_called()

    def test_parked_weights_get_the_same_budget_as_resident_weights(self):
        engine = self.engine
        def weight(size):
            return SimpleNamespace(device=torch.device("cpu"), numel=lambda: size * W.GIB, element_size=lambda: 1)
        blocks, embedder = [weight(1) for _ in range(8)], weight(2)
        engine.dit = SimpleNamespace(blocks=blocks, parameters=lambda: iter(blocks + [embedder]), buffers=lambda: ())
        engine.swapped, engine._auto_swap, engine._reserve_extra = 0, True, 0
        with patch.object(W, "_block_weights", side_effect=lambda b: [b]), \
                patch.object(W, "_free_vram_gb", return_value=10) as free, \
                patch.object(engine, "_set_swap") as change, patch.object(engine, "_park_dit"), \
                patch.object(WorkbenchEngine, "render", return_value="image"):
            engine.render({}, 512, 512, 42)
            parked = change.call_args.args[0]
            for w in blocks + [embedder]:
                w.device = torch.device("cuda")
            free.return_value = 0
            engine.render({}, 512, 512, 42)
            self.assertEqual(change.call_args.args[0], parked)
        self.assertEqual(parked, 3)

    def test_render_restores_before_sampling_and_parks_when_done_or_failed(self):
        engine = self.engine
        engine._auto_swap = False
        for fails in (False, True):
            with self.subTest(fails=fails):
                events = []
                def render(*a, **kw):
                    events.append("sample")
                    if fails:
                        raise torch.OutOfMemoryError("sampling failed")
                    return "image"
                with patch.object(engine, "_park_dit", side_effect=events.append), \
                        patch.object(WorkbenchEngine, "render", side_effect=render):
                    if fails:
                        with self.assertRaisesRegex(torch.OutOfMemoryError, "sampling failed"):
                            engine.render({}, 2048, 2048, 42)
                    else:
                        self.assertEqual(engine.render({}, 2048, 2048, 42), "image")
                self.assertEqual(events, ["cuda", "sample", "cpu"])

    def test_rocm_allocator_uses_available_memory_and_restores_existing_limit(self):
        engine = self.engine
        engine._auto_swap = False
        for dev, previous, fails in (("cuda", 1., False), ("cuda", .5, False),
                                     ("cuda", 1., True), ("cuda:1", 1., False)):
            with self.subTest(device=dev, previous=previous, fails=fails):
                engine.device = dev
                expected_device = torch.device("cuda:0" if dev == "cuda" else dev)
                def get_fraction(device):
                    from torch.cuda._utils import _get_device_index
                    self.assertEqual(_get_device_index(device), expected_device.index)
                    return previous
                with patch.object(torch.version, "hip", "test"), \
                        patch.object(torch.cuda, "is_available", return_value=True), \
                        patch.object(torch.cuda, "get_per_process_memory_fraction", side_effect=get_fraction), \
                        patch.object(torch.cuda, "mem_get_info", return_value=(10 * W.GIB, 16 * W.GIB)), \
                        patch.object(torch.cuda, "memory_reserved", return_value=2 * W.GIB), \
                        patch.object(torch.cuda, "set_per_process_memory_fraction") as limit, \
                        patch.object(engine, "_park_dit"), \
                        patch.object(WorkbenchEngine, "render", side_effect=ValueError("render") if fails else None,
                                     return_value="image"):
                    if fails:
                        with self.assertRaisesRegex(ValueError, "render"):
                            engine.render({}, 2048, 2048, 42)
                    else:
                        self.assertEqual(engine.render({}, 2048, 2048, 42), "image")
                self.assertEqual([c.args for c in limit.call_args_list],
                                 [(min(previous, 12 / 16), expected_device),
                                  (previous, expected_device)])

    def test_no_available_memory_fails_before_sampling(self):
        with patch.object(torch.version, "hip", "test"), \
                patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "get_per_process_memory_fraction", return_value=1.), \
                patch.object(torch.cuda, "mem_get_info", return_value=(0, 16 * W.GIB)), \
                patch.object(torch.cuda, "memory_reserved", return_value=0), \
                patch.object(torch.cuda, "set_per_process_memory_fraction"), \
                patch.object(self.engine, "_park_dit"), \
                patch.object(WorkbenchEngine, "render") as sample, \
                self.assertRaisesRegex(torch.OutOfMemoryError, "no available GPU memory"):
            self.engine.render({}, 2048, 2048, 42)
        sample.assert_not_called()

    def test_auto_swap_uses_the_allocator_limit(self):
        engine = self.engine
        blocks = [SimpleNamespace(device=torch.device("cpu"), numel=lambda: W.GIB, element_size=lambda: 1)
                  for _ in range(8)]
        embedder = SimpleNamespace(device=torch.device("cpu"), numel=lambda: 2 * W.GIB, element_size=lambda: 1)
        engine.dit = SimpleNamespace(blocks=blocks, parameters=lambda: iter(blocks + [embedder]), buffers=lambda: ())
        engine.swapped, engine._auto_swap, engine._reserve_extra = 0, True, 0
        with patch.object(torch.version, "hip", "test"), \
                patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "get_per_process_memory_fraction", return_value=1.), \
                patch.object(torch.cuda, "mem_get_info", return_value=(10 * W.GIB, 16 * W.GIB)), \
                patch.object(torch.cuda, "memory_reserved", return_value=2 * W.GIB), \
                patch.object(torch.cuda, "set_per_process_memory_fraction"), \
                patch.object(W, "_block_weights", side_effect=lambda b: [b]), \
                patch.object(W, "_free_vram_gb", return_value=10), \
                patch.object(engine, "_set_swap") as change, patch.object(engine, "_park_dit"), \
                patch.object(WorkbenchEngine, "render", return_value="image"):
            engine.render({}, 512, 512, 42)
        change.assert_called_once_with(3)

    def test_nvidia_render_does_not_change_allocator_limit(self):
        with patch.object(torch.version, "hip", None), \
                patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "set_per_process_memory_fraction") as limit, \
                patch.object(self.engine, "_park_dit") as park, \
                patch.object(WorkbenchEngine, "render", return_value="image"):
            self.engine.render({}, 2048, 2048, 42)
        limit.assert_not_called()
        park.assert_not_called()
        with patch.object(torch.version, "hip", None), patch.object(self.engine, "_park_dit") as park:
            self.engine._finish_decode()
        park.assert_called_once_with("cuda")

    def test_decode_cleanup_keeps_krea_parked_and_other_families_restore(self):
        with patch.object(self.engine, "_park_dit") as park:
            self.engine._finish_decode()
        park.assert_not_called()
        with patch.object(self.engine, "_park_dit") as park:
            WorkbenchEngine._finish_decode(self.engine)
        park.assert_called_once_with("cuda")

    def test_failed_swap_setup_is_parked_before_error_leaves_render(self):
        engine = self.engine
        engine.dit = SimpleNamespace(blocks=[object()] * 8, parameters=lambda: (), buffers=lambda: ())
        engine.swapped, engine._auto_swap, engine._reserve_extra = 0, True, 0
        with patch.object(W, "_block_weights", return_value=[]), \
                patch.object(W, "_free_vram_gb", return_value=10), \
                patch.object(engine, "_set_swap", side_effect=torch.OutOfMemoryError("swap setup")), \
                patch.object(engine, "_park_dit") as park, \
                self.assertRaisesRegex(torch.OutOfMemoryError, "swap setup"):
            engine.render({}, 2048, 2048, 42)
        park.assert_called_once_with("cpu")

    def test_unchanged_sliders_reuse_one_baseline_image(self):
        state = self.engine.default_state(2048, 2048)
        state.seed = 42
        state.prompt = "A red teapot"
        self.engine.swapped = 0
        self.assertTrue(self.engine.same_as_baseline(state))
        with patch.object(self.engine, "generate_preview", return_value="image") as render:
            self.assertEqual(self.engine.generate_baseline(state), "image")
            self.assertEqual(self.engine.generate_baseline(state), "image")
        render.assert_called_once_with(state)

    def test_initial_load_preserves_fp8_and_honours_manual_and_nf4(self):
        cases = [("auto", "auto", "cuda", "fp8", "bf16", 26, True),
                 ("auto", "auto", "cuda", "", "int8", 26, True),
                 ("nf4", "auto", "cuda", "", "nf4", 0, False),
                 ("bf16", 4, "cuda", "", "bf16", 4, False),
                 ("bf16", "auto", "cpu", "", "bf16", 0, False)]
        for precision, swap, device, layout, expected_precision, count, automatic in cases:
            with self.subTest(precision=precision, swap=swap, device=device, layout=layout):
                engine = get("krea2").make_workbench_engine()
                with patch("fizgig.krea2.utils.dit_layout", return_value=layout), \
                        patch.object(WorkbenchEngine, "ensure_pipeline") as load:
                    engine.ensure_pipeline("model", "vae", "te", precision=precision,
                                           blocks_to_swap=swap, device=device)
                self.assertEqual(load.call_args.kwargs["precision"], expected_precision)
                self.assertEqual(load.call_args.kwargs["blocks_to_swap"], count)
                self.assertEqual(engine._auto_swap, automatic)
                if device == "cuda":
                    self.assertTrue(load.call_args.kwargs["lowmem"])

    def test_planner_does_not_hold_weight_buffers_during_offload(self):
        engine = self.engine
        engine.dit = SimpleNamespace(blocks=[object() for _ in range(8)], parameters=lambda: (), buffers=lambda: ())
        engine.swapped, engine._auto_swap, engine._reserve_extra = 0, True, 0
        refs = []
        def weights(block):
            weight = torch.ones(1)
            refs.append(weakref.ref(weight))
            return [weight]
        def offload(count):
            self.assertTrue(all(ref() is None for ref in refs))
        with patch.object(W, "_block_weights", side_effect=weights), \
                patch.object(W, "_free_vram_gb", return_value=1), \
                patch.object(engine, "_set_swap", side_effect=offload), \
                patch.object(engine, "_park_dit"), \
                patch.object(WorkbenchEngine, "render", return_value="image"):
            engine.render({}, 1024, 1024, 42)

    def test_weight_budget_counts_base_once_and_excludes_lora(self):
        from fizgig.families.lora import LoRALinear
        base = torch.nn.Linear(4, 4, bias=False)
        wrapped = LoRALinear(base)
        wrapped.add("primary", rank=2, alpha=2, trainable=False)
        weights = list(W._block_weights(wrapped))
        self.assertEqual(len(weights), 1)
        self.assertIs(weights[0], base.weight)

    def test_swapped_model_parks_and_restores_between_stages(self):
        engine = self.engine
        engine.swapped = 4
        offloader = Mock()
        engine.dit = SimpleNamespace(offloader=offloader, move_to_device_except_swap_blocks=Mock())
        events = []
        offloader.set_forward_only.side_effect = lambda on: events.append("drain")
        with patch.object(W.quant, "move", side_effect=lambda model, device: events.append(str(device))), \
                patch.object(engine.driver, "block_swap_mode") as restore:
            engine._park_dit("cpu")
            self.assertEqual(events, ["drain", "cpu"])
            engine._park_dit("cuda")
            restore.assert_called_once_with(engine.dit, inference=True)
            engine.dit.move_to_device_except_swap_blocks.assert_called_once_with(torch.device("cuda"))

    def test_reconfiguration_closes_old_workers_before_enabling_new_swap(self):
        engine = self.engine
        engine.swapped = 2
        old = Mock()
        engine.dit = SimpleNamespace(offloader=old, blocks_to_swap=2)
        events = []
        old.thread_pool.shutdown.side_effect = lambda **kw: events.append("shutdown")
        with patch.object(engine, "_park_dit", side_effect=lambda d: events.append("park")), \
                patch.object(engine, "_invalidate_activation_cache"), \
                patch.object(engine.driver, "enable_block_swap", side_effect=lambda *a, **kw: events.append("enable")), \
                patch.object(engine.driver, "block_swap_mode"):
            engine._set_swap(4)
        self.assertEqual(events, ["park", "shutdown", "enable"])
        old.remove_hooks.assert_called_once()
        self.assertEqual(engine.swapped, 4)

    def test_offloader_reconfiguration_preserves_tiny_model_output(self):
        from fizgig.krea2 import driver as D, generation_attention as G
        from fizgig.krea2.model import SingleMMDiTConfig, SingleStreamDiT
        from fizgig.krea2.offloading import Offloader
        engine = self.engine
        torch.manual_seed(8)
        engine.dit = SingleStreamDiT(SingleMMDiTConfig(
            features=64, tdim=32, txtdim=24, heads=4, kvheads=2, multiplier=2,
            layers=4, patch=2, channels=16, txtlayers=3, txtheads=4, txtkvheads=2)).float().eval()
        engine.device, engine.swapped = "cpu", 0
        self.addCleanup(engine.reset)
        cond = {"hidden_states": torch.randn(5, 3, 24), "attention_mask": torch.ones(5, dtype=torch.bool)}
        recipe = engine.desc.preview_checkpoint_sampling
        def sample():
            return engine.driver.generate(engine.dit, cond, 32, 32, steps=8, seed=7,
                                          cfg=1, sigmas=recipe.sigmas)
        # CPU weights need no transfers; keep the real worker scheduling and sampler.
        with patch.object(D, "DTYPE", torch.float32), patch.object(G, "wanted", return_value=False), \
                patch.object(Offloader, "swap_weight_devices", return_value=None), \
                patch.object(torch.cuda, "current_device", return_value=0), patch.object(torch.cuda, "set_device"):
            reference = sample()
            for count in (1, 2, 0):
                engine._set_swap(count)
                torch.testing.assert_close(sample(), reference)

    def test_oom_retry_releases_failed_tensors_and_is_bounded(self):
        engine = self.engine
        engine.dit = SimpleNamespace(blocks=[torch.nn.Linear(4, 4) for _ in range(8)])
        engine.swapped, engine._auto_swap, engine._reserve_extra = 0, True, 0
        refs, counts = [], []
        def sample():
            temporary = torch.ones(10)
            refs.append(weakref.ref(temporary))
            raise torch.OutOfMemoryError("sampling failed")
        def change(count):
            self.assertTrue(all(ref() is None for ref in refs))
            counts.append(count)
            engine.swapped = count
        with patch.object(engine, "_set_swap", side_effect=change), \
                self.assertRaisesRegex(torch.OutOfMemoryError, "sampling failed"):
            engine._sample_with_memory(sample)
        self.assertEqual(counts, [2, 4, 6])
        self.assertEqual(len(refs), 4)
        self.assertGreater(engine._reserve_extra, 0)

    def test_manual_oom_and_other_errors_are_not_retried(self):
        engine = self.engine
        engine.dit = SimpleNamespace(blocks=[object()] * 8)
        engine.swapped, engine._auto_swap = 0, False
        for error in (torch.OutOfMemoryError, ValueError):
            sample = Mock(side_effect=error("failed"))
            with self.assertRaises(error):
                engine._sample_with_memory(sample)
            self.assertEqual(sample.call_count, 1)

    def test_stage_cleanup_runs_even_when_loading_vae_fails(self):
        engine = self.engine
        engine._auto_swap, engine.lowmem = False, True
        engine.device, engine.dit = "cuda", object()
        engine.vae = Mock()
        engine.vae.to.side_effect = lambda device: (_ for _ in ()).throw(torch.OutOfMemoryError("load VAE")) \
            if device == "cuda" else None
        with patch.object(engine, "sampling", return_value=(8, 1., None, ())), \
                patch.object(engine, "_act_for_render", return_value=None), \
                patch.object(engine, "_start_meter", return_value=None), \
                patch.object(engine, "_cond_to_device", side_effect=lambda c: c), \
                patch.object(engine.driver, "generate", return_value=torch.zeros(1)), \
                patch.object(engine, "_park_dit") as park, \
                self.assertRaisesRegex(torch.OutOfMemoryError, "load VAE"):
            engine.render({}, 1024, 1024, 42)
        self.assertEqual([c.args[0] for c in park.call_args_list], ["cuda", "cpu", "cpu"])
        self.assertEqual([c.args[0] for c in engine.vae.to.call_args_list], ["cuda", "cpu"])

    def test_non_rocm_pipeline_preserves_generic_arguments(self):
        engine = self.engine
        with patch.object(torch.version, "hip", None), \
                patch.object(WorkbenchEngine, "ensure_pipeline") as load:
            engine.ensure_pipeline("model", "vae", "te", precision="auto", blocks_to_swap="auto", device="cuda", lowmem=False)
        self.assertFalse(engine._auto_swap)
        self.assertFalse(engine.driver._file_backed_inference)
        self.assertEqual(load.call_args.kwargs, dict(precision="auto", blocks_to_swap="auto", device="cuda", lowmem=False))

    def test_non_rocm_parking_uses_generic_policy(self):
        with patch.object(torch.version, "hip", None), \
                patch.object(WorkbenchEngine, "_park_dit") as park:
            self.engine._park_dit("cpu")
        park.assert_called_once_with("cpu")

    def test_amd_pr_preserves_original_turbo_recipe(self):
        recipe = get("krea2").preview_checkpoint_sampling
        self.assertEqual((recipe.steps, recipe.cfg, recipe.scheduler), (8, 1., "simple"))
        self.assertFalse(recipe.sigmas)

    def test_non_rocm_worker_keeps_block_swap_preference(self):
        from fizgig.scripts import studio_worker
        with patch.object(torch.version, "hip", None), patch.object(studio_worker, "_auto_swap", return_value=12):
            self.assertEqual(studio_worker._plan_for(get("krea2"), {"blocks_to_swap": "auto"})["blocks_to_swap"], 12)

    def test_auto_survives_worker_plan_only_for_krea2(self):
        from fizgig.scripts.studio_worker import _plan_for
        from fizgig.scripts import studio_worker
        with patch.object(studio_worker, "_auto_swap", return_value=12):
            self.assertEqual(_plan_for(get("krea2"), {"blocks_to_swap": "auto"})["blocks_to_swap"], "auto")
            self.assertEqual(_plan_for(get("klein"), {"blocks_to_swap": "auto"})["blocks_to_swap"], 12)
            self.assertEqual(_plan_for(get("krea2"), {"blocks_to_swap": 0})["blocks_to_swap"], 0)

    def test_vae_oom_retries_tiled_and_restores_configuration(self):
        vae = VAE()
        image = Krea2Driver().decode(vae, torch.zeros(1, 16, 1, 1, 1), 8, 8)
        self.assertEqual(image.size, (2, 2))
        self.assertEqual(vae.calls, 2)
        self.assertFalse(vae.use_tiling)
        self.assertEqual(vae.clear_cache.call_count, 2)

    def test_vae_failure_does_not_leave_cache_or_tiling_enabled(self):
        for error in (torch.OutOfMemoryError, ValueError):
            vae = VAE(failures=2, error=error)
            with self.assertRaises(error):
                Krea2Driver().decode(vae, torch.zeros(1, 16, 1, 1, 1), 8, 8)
            self.assertFalse(vae.use_tiling)
            vae.clear_cache.assert_called()
            self.assertEqual(vae.calls, 2 if error is torch.OutOfMemoryError else 1)



if __name__ == "__main__":
    unittest.main()
