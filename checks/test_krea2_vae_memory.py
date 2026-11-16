"""Single-image VAE passes match temporal caching without retaining its features."""
import unittest
from unittest.mock import patch

import torch

from fizgig.krea2.vae import AutoencoderKLQwenImage, QwenImageCausalConv3d


class VAEMemoryChecks(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(torch.version, "hip", "test"))
        torch.manual_seed(19)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        self.vae = AutoencoderKLQwenImage(base_dim=4, z_dim=2, num_res_blocks=1).eval()

    @torch.no_grad()
    def test_single_frame_convolution_matches_zero_padding(self):
        for kernel, padding in ((3, 1), (1, 0), ((3, 1, 1), (1, 0, 0))):
            with self.subTest(kernel=kernel):
                conv = QwenImageCausalConv3d(3, 4, kernel, padding=padding)
                x = torch.randn(2, 3, 1, 9, 11)
                expected = torch.nn.functional.conv3d(torch.nn.functional.pad(x, conv._padding),
                                                       conv.weight, conv.bias, conv.stride)
                with patch('fizgig.krea2.vae.F.pad', side_effect=AssertionError('image padding')):
                    actual = conv(x)
                torch.testing.assert_close(actual, expected)

    @torch.no_grad()
    def test_video_and_cached_convolution_keep_temporal_weights(self):
        conv = QwenImageCausalConv3d(3, 4, 3, padding=1)
        for frames, cached in ((3, False), (1, True)):
            with self.subTest(frames=frames, cached=cached):
                x = torch.randn(1, 3, frames, 9, 11)
                cache = torch.randn(1, 3, 2, 9, 11) if cached else None
                padded = torch.nn.functional.pad(torch.cat([cache, x], dim=2) if cached else x,
                                                   (1, 1, 1, 1, 0 if cached else 2, 0))
                expected = torch.nn.functional.conv3d(padded, conv.weight, conv.bias)
                torch.testing.assert_close(conv(x, cache), expected)

    @torch.no_grad()
    def test_non_rocm_keeps_padded_convolution(self):
        conv = QwenImageCausalConv3d(3, 4, 3, padding=1)
        x = torch.randn(1, 3, 1, 9, 11)
        pad = torch.nn.functional.pad
        with patch.object(torch.version, "hip", None), \
                patch('fizgig.krea2.vae.F.pad', wraps=pad) as padded:
            actual = conv(x)
        padded.assert_called_once()
        expected = torch.nn.functional.conv3d(pad(x, conv._padding), conv.weight, conv.bias)
        torch.testing.assert_close(actual, expected)

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
