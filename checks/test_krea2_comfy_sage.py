"""ComfyUI-style Sage calls, checked on CPU with exact attention standing in for the kernel."""
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from fizgig.krea2 import attention as A
from fizgig.krea2 import generation_attention as G
from fizgig.krea2 import driver as D
from fizgig.krea2.model import KreaActivationCacheEntry, SingleMMDiTConfig, SingleStreamDiT
from fizgig.krea2.sampling import prepare


class ComfySageChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.calls = []
        self.enterContext(patch.object(torch.cuda, "is_available", return_value=False))
        self.enterContext(patch.object(A, "_TRIM_MULTIPLE", 1))
        self.enterContext(patch.object(A, "sageattn", side_effect=self.sage))
        self.enterContext(patch.object(A, "sageattn_varlen", side_effect=AssertionError("varlen called")))

    def sage(self, q, k, v, **kwargs):
        self.assertEqual(kwargs, dict(tensor_layout="HND", is_causal=False, sm_scale=None, smooth_k=False))
        self.assertEqual(q.shape[1], k.shape[1])
        self.assertEqual(k.shape, v.shape)
        self.calls.append(tuple(q.shape))
        return F.scaled_dot_product_attention(q, k, v)

    def check_lengths(self, lens):
        q = torch.randn(2, 9, 4, 8)
        k, v = torch.randn(2, 9, 2, 8), torch.randn(2, 9, 2, 8)
        mask = torch.arange(5)[None, :] < torch.tensor(lens)[:, None] - 4
        params = A.AttentionParams.create_attention_params_from_mask("sageattn", True, 4, mask)
        # No CUDA scalar reads or transfers may be repeated by individual blocks.
        with torch.no_grad(), patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host read")), \
                patch.object(torch.Tensor, "__index__", side_effect=AssertionError("scalar read")):
            got = A.attention(q, k, v, params)
            again = A.attention(q, k, v, params)
        torch.testing.assert_close(got, again)
        for i, n in enumerate(lens):
            want = F.scaled_dot_product_attention(q[i:i+1, :n].transpose(1, 2),
                                                  k[i:i+1, :n].transpose(1, 2),
                                                  v[i:i+1, :n].transpose(1, 2), enable_gqa=True)
            torch.testing.assert_close(got[i:i+1, :n], want.transpose(1, 2).flatten(2))
            self.assertEqual(torch.count_nonzero(got[i, n:]).item(), 0)

    def test_uniform_masked_batch_is_one_call_per_block(self):
        self.check_lengths([6, 6])
        self.assertEqual(self.calls, [(2, 4, 6, 8)] * 2)

    def test_ragged_batch_excludes_padding_without_per_block_host_reads(self):
        self.check_lengths([6, 8])
        self.assertEqual(self.calls, [(1, 4, 6, 8), (1, 4, 8, 8)] * 2)

    def test_unmasked_text_fusion_is_batched(self):
        q = torch.randn(12, 3, 4, 8)
        with torch.no_grad():
            got = A.attention(q, q, q, A.AttentionParams("sageattn", True))
        want = F.scaled_dot_product_attention(q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2))
        torch.testing.assert_close(got, want.transpose(1, 2).flatten(2))
        self.assertEqual(self.calls, [(12, 4, 3, 8)])


    def model(self):
        return SingleStreamDiT(SingleMMDiTConfig(
            features=64, tdim=32, txtdim=24, heads=4, kvheads=2, multiplier=2,
            layers=2, patch=2, channels=16, txtlayers=3, txtheads=4, txtkvheads=2)).float().eval()


    def test_forward_and_cached_resume_skip_shape_padding(self):
        model = self.model()
        img, pos, mask = prepare(torch.randn(2, 16, 4, 4), 5, 2, torch.ones(2, 5, dtype=torch.bool))
        args = dict(img=img, pos=pos, mask=mask, context=torch.randn(2, 5, 3, 24), t=torch.tensor([.5, .5]))
        lengths = []
        hook = model.blocks[0].register_forward_pre_hook(lambda module, inputs: lengths.append(inputs[0].shape[1]))
        self.addCleanup(hook.remove)
        with torch.no_grad():
            reference = model(**args)
            model.attn_mode, model.split_attn = "sageattn", True
            got = model(**args)
            cache = KreaActivationCacheEntry([])
            cached = model.forward_cached(**args, new_cache=cache)
            resumed = model.forward_cached(**args, cached=cache, resume_from=1)
        self.assertEqual(lengths, [256, 9, 9])
        for result in (got, cached, resumed):
            torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)

    def test_guided_generation_restores_training_backend(self):
        model = self.model()
        driver = D.Krea2Driver()
        cond = {"hidden_states": torch.randn(8, 3, 24),
                "attention_mask": torch.tensor([True, True, False, False, True, False, True, True])}
        args = dict(steps=2, seed=7, cfg=2., neg_cond=cond)
        with patch.object(D, "DTYPE", torch.float32), patch.object(G, "wanted", return_value=False):
            reference = driver.generate(model, cond, 64, 64, **args)
        with patch.object(D, "DTYPE", torch.float32), patch.object(G, "wanted", return_value=True):
            got = driver.generate(model, cond, 64, 64, **args)
        torch.testing.assert_close(got, reference, atol=2e-5, rtol=2e-5)
        self.assertEqual((model.attn_mode, model.split_attn), ("torch", False))
        model.train()
        with patch.object(D, "DTYPE", torch.float32), \
                patch.object(A, "sageattn", side_effect=AssertionError("training used Sage")):
            driver._forward(model, torch.randn(1, 16, 8, 8), torch.tensor([.5]), cond).square().mean().backward()
        self.assertIsNotNone(model.first.weight.grad)

    def test_kernel_failure_restores_backend_without_silent_fallback(self):
        model = self.model()
        @G.configured_generation
        def generate(driver, dit):
            self.assertEqual(dit.attn_mode, "sageattn")
            raise RuntimeError("kernel failed")
        with patch.object(G, "wanted", return_value=True), self.assertRaisesRegex(RuntimeError, "kernel failed"):
            generate(None, model)
        self.assertEqual((model.attn_mode, model.split_attn), ("torch", False))


if __name__ == "__main__":
    unittest.main()
