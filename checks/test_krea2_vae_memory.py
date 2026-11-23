"""Single-image VAE passes match temporal caching without retaining its features."""
import unittest
from unittest.mock import patch

import torch

from fizgig.krea2.vae import AutoencoderKLQwenImage


class VAEMemoryChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        self.vae = AutoencoderKLQwenImage(base_dim=4, z_dim=2, num_res_blocks=1).eval()




    def compare_cached(self, direction, tiled):
        vae = self.vae
        vae.tile_sample_min_height = vae.tile_sample_min_width = 32
        vae.tile_sample_stride_height = vae.tile_sample_stride_width = 24
        vae.use_tiling = tiled
        if direction == "encode":
            module, cache_name = vae.encoder, "_enc_feat_map"
            x = torch.randn(1, 3, 1, 64, 64)
            run = lambda: vae.encode(x)["latent_dist"].parameters
        else:
            module, cache_name = vae.decoder, "_feat_map"
            x = torch.randn(1, 2, 1, 8, 8)
            run = lambda: vae.decode(x)["sample"]
        forward = module.forward
        cached_bytes = []
        def cached(*args, **kwargs):
            cache = getattr(vae, cache_name)
            kwargs["feat_cache"] = cache
            result = forward(*args, **kwargs)
            cached_bytes.append(sum(t.numel() * t.element_size() for t in cache if torch.is_tensor(t)))
            return result
        with patch.object(module, "forward", side_effect=cached):
            expected = run()
        with patch.object(module, "forward", wraps=forward) as calls:
            actual = run()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if direction == "decode":
            expected_pixels = (expected.float().add(1).div(2).clamp(0, 1) * 255).byte()
            actual_pixels = (actual.float().add(1).div(2).clamp(0, 1) * 255).byte()
            self.assertTrue(torch.equal(actual_pixels, expected_pixels))
        self.assertGreater(max(cached_bytes), 0)
        self.assertTrue(all(c.kwargs["feat_cache"] is None for c in calls.call_args_list))
        self.assertTrue(all(t is None for t in getattr(vae, cache_name)))
        self.assertEqual(len(calls.call_args_list) > 1, tiled)

    @torch.no_grad()
    def test_image_encode_matches_cached_output(self):
        self.compare_cached("encode", False)

    @torch.no_grad()
    def test_image_decode_matches_cached_output(self):
        self.compare_cached("decode", False)

    @torch.no_grad()
    def test_tiled_image_encode_matches_cached_output(self):
        self.compare_cached("encode", True)

    @torch.no_grad()
    def test_tiled_image_decode_matches_cached_output(self):
        self.compare_cached("decode", True)

    @torch.no_grad()
    def test_amd_decodes_pictures_in_tiles(self):
        """#404: on AMD a whole 1024x1536 decode reserved ~30 GB on a 16 GB card; decode_to_pixels tiles it there
        (512 px tiles, 384 stride), leaves the VAE's own tiling settings as they were, and decodes whole elsewhere."""
        vae = self.vae
        vae.latents_mean, vae.latents_std = [0.0, 0.0], [1.0, 1.0]
        z = torch.randn(1, 2, 1, 96, 64)                       # a 512x768 picture
        before = (vae.tile_sample_min_height, vae.tile_sample_min_width,
                  vae.tile_sample_stride_height, vae.tile_sample_stride_width)
        for rocm in (False, True):
            with self.subTest(rocm=rocm):
                seen = []
                tiled = vae.tiled_decode
                def spy(*a, **k):
                    seen.append((vae.tile_sample_min_height, vae.tile_sample_stride_height))
                    return tiled(*a, **k)
                with patch("fizgig.modules.wide_attention.ROCM", rocm), \
                        patch.object(vae, "tiled_decode", side_effect=spy):
                    pixels = vae.decode_to_pixels(z)
                self.assertEqual(pixels.shape, (1, 3, 768, 512))
                self.assertEqual(seen, [(512, 384)] if rocm else [])
                self.assertFalse(vae.use_tiling)
                self.assertEqual((vae.tile_sample_min_height, vae.tile_sample_min_width,
                                  vae.tile_sample_stride_height, vae.tile_sample_stride_width), before)

    @torch.no_grad()
    def test_video_keeps_temporal_caches(self):
        vae = self.vae
        for tiled in (False, True):
            with self.subTest(tiled=tiled):
                vae.use_tiling = tiled
                vae.tile_sample_min_height = vae.tile_sample_min_width = 16
                vae.tile_sample_stride_height = vae.tile_sample_stride_width = 8
                with patch.object(vae.encoder, "forward", wraps=vae.encoder.forward) as enc, \
                        patch.object(vae.decoder, "forward", wraps=vae.decoder.forward) as dec:
                    z = vae.encode(torch.randn(1, 3, 5, 24, 24))["latent_dist"].mode()
                    image = vae.decode(z)["sample"]
                self.assertEqual(image.shape, (1, 3, 5, 24, 24))
                self.assertTrue(all(c.kwargs["feat_cache"] is not None for c in enc.call_args_list + dec.call_args_list))


if __name__ == "__main__":
    unittest.main()
