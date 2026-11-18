"""CPU checks for immutable checkpoint storage; HIP ordering still needs hardware validation."""
from contextlib import nullcontext
import hashlib
import sys
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors.torch import save_file

from fizgig.families.lora import LoRALinear
from fizgig.families.registry import get
from fizgig.krea2 import offloading as O
from fizgig.krea2 import utils as U
from fizgig.krea2 import workbench as W
from fizgig.krea2.model import SingleMMDiTConfig, SingleStreamDiT
from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen, release_mapped_pages, move_frozen_weights

# file-backed weights are a Linux ROCm feature (mmap.madvise has no Windows or macOS form): checks pass on any machine
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="file-backed weights are Linux only")


def test_mapped_fp8_load_matches_existing_loader(tmp_path):
    config = SingleMMDiTConfig(features=64, tdim=32, txtdim=24, heads=4, kvheads=2, multiplier=2,
                              layers=4, patch=2, channels=16, txtlayers=3, txtheads=4, txtkvheads=2)
    torch.manual_seed(42)
    model = SingleStreamDiT(config).bfloat16()
    state = model.state_dict()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            state[name + '.weight'] = module.weight.detach().to(torch.float8_e4m3fn)
            state[name + '.weight_scale'] = torch.ones(1)
    path = str(tmp_path/'tiny.safetensors')
    save_file(state, path)
    before = hashlib.sha256((tmp_path/'tiny.safetensors').read_bytes()).digest()
    normal = U.load_krea2_dit(path, config=config, device='cpu', mmap_cpu=False)
    engine = get('krea2').make_workbench_engine()
    engine.driver._file_backed_inference = True
    loader = U.load_krea2_dit
    with patch.object(U, 'load_krea2_dit', side_effect=lambda *a, **kw: loader(*a, config=config, **kw)):
        mapped = engine.driver.load_dit(path, 'cpu')
    assert mapped._cpu_weight_masters
    assert any(hasattr(w.untyped_storage(), '_fizgig_mmap') for w in mapped._cpu_weight_masters.values())
    release_mapped_pages(mapped._cpu_weight_masters.values())
    for name, value in normal.state_dict().items():
        other = mapped.state_dict()[name]
        assert value.dtype == other.dtype
        torch.testing.assert_close(value.float(), other.float(), rtol=0, atol=0)
    assert hashlib.sha256((tmp_path/'tiny.safetensors').read_bytes()).digest() == before


@torch.no_grad()
def test_swaps_and_parking_preserve_frozen_weights_and_editable_state(tmp_path):
    values = {'a': torch.randn(8, 8), 'b': torch.randn(8, 8)}
    path = tmp_path/'weights.safetensors'
    save_file(values, str(path))
    before = path.read_bytes()
    with MemoryEfficientSafeOpen(str(path), mmap_cpu=True) as reader:
        masters = {name: reader.get_tensor(name) for name in values}
    blocks = []
    for name in values:
        linear = torch.nn.Linear(8, 8).requires_grad_(False)
        linear.weight = torch.nn.Parameter(masters[name], requires_grad=False)
        wrapped = LoRALinear(linear)
        wrapped.add('primary', rank=2, alpha=2, trainable=False,
                    A=torch.randn(2, 8), B=torch.randn(8, 2))
        blocks.append(torch.nn.Sequential(wrapped))
    cpu_weights = {block[0].base.weight: masters[name] for block, name in zip(blocks, values)}
    off = O.ModelOffloader('test', blocks, 2, 1, False, torch.device('cpu'), cpu_weights=cpu_weights)
    off.stream = Mock()
    try:
        # A CPU clone stands in for the resident GPU buffer; masters remain file-backed.
        blocks[0][0].base.weight.data = masters['a'].clone()
        with patch.object(torch.Tensor, 'pin_memory', lambda tensor, **kw: tensor.clone()), \
                patch.object(torch.cuda, 'stream', side_effect=lambda stream: nullcontext()):
            for step in range(8):
                outgoing, incoming = blocks if step % 2 == 0 else reversed(blocks)
                a, b = outgoing[0].base, incoming[0].base
                off._swap_from_masters(torch.device('cuda'), [(a, b, a.weight.data, b.weight.data)])
                for block, name in zip(blocks, values):
                    torch.testing.assert_close(block[0].base.weight, values[name])
        assert off.staging_buffer_a is None
        assert off._copy_done.synchronize.call_count == 7
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList(blocks)
        model._cpu_weight_masters = cpu_weights
        model.offloader = off
        engine = get('krea2').make_workbench_engine()
        engine.dit = model
        blocks[0][0].base.bias.add_(.25)  # A Studio bias delta must survive parking.
        x = torch.randn(2, 8)
        expected = [b(x).clone() for b in blocks]
        # a small ROCm card: the memory policy's reach (rocm_small_card), with no GPU on the machine running the check
        with patch.object(torch.version, 'hip', 'test'), patch.object(torch.cuda, 'synchronize'), \
                patch.object(torch.cuda, 'is_available', return_value=False), \
                patch.object(W, 'rocm_small_card', return_value=True):
            engine._park_dit('cpu')
        for block, reference in zip(blocks, expected):
            torch.testing.assert_close(block(x), reference)
            assert block[0].base.weight.data_ptr() == cpu_weights[block[0].base.weight].data_ptr()
        assert path.read_bytes() == before
    finally:
        off.thread_pool.shutdown(wait=True)


def test_offloaded_preparation_keeps_master_weights_off_the_device_on_failure():
    blocks = [torch.nn.Linear(8, 8).requires_grad_(False) for _ in range(2)]
    weights = {b.weight: b.weight.detach() for b in blocks}
    off = O.ModelOffloader('test', blocks, 2, 1, False, torch.device('cpu'), cpu_weights=weights)
    def fail(*a, **kw):
        assert blocks[1].weight is None
        raise torch.OutOfMemoryError('placement failed')
    try:
        with patch.object(blocks[1], 'to', side_effect=fail):
            try:
                off.prepare_block_devices_before_forward(blocks)
            except torch.OutOfMemoryError:
                pass
            else:
                raise AssertionError('Expected placement failure')
        assert blocks[1].weight in weights
        assert blocks[1].weight.data_ptr() == weights[blocks[1].weight].data_ptr()
    finally:
        off.thread_pool.shutdown(wait=True)


def test_text_encoder_unload_skips_copy_only_in_file_backed_inference():
    driver = get('krea2').make_workbench_engine().driver
    with patch.object(torch.cuda, 'is_available', return_value=False):
        for enabled in (True, False):
            driver._file_backed_inference = enabled
            te = Mock()
            driver.unload_text_encoder(te)
            assert te.to.call_count == int(not enabled)


def test_file_backing_selection_leaves_training_and_nvidia_unchanged():
    from fizgig.families.workbench import WorkbenchEngine
    for hip, device, precision, enabled in (
            ('test', 'cuda', 'bf16', True), (None, 'cuda', 'bf16', False),
            ('test', 'cpu', 'bf16', False), ('test', 'cuda', 'int8', False)):
        engine = get('krea2').make_workbench_engine()
        # ROCm here means a small ROCm card (rocm_small_card): the check's machine has no GPU to ask
        with patch.object(torch.version, 'hip', hip), \
                patch.object(W, 'rocm_small_card',
                             side_effect=lambda device: bool(torch.version.hip) and torch.device(device).type == 'cuda'), \
                patch.object(WorkbenchEngine, 'ensure_pipeline'):
            engine.ensure_pipeline('model', 'vae', 'te', device=device, precision=precision, blocks_to_swap=4)
        assert engine.driver._file_backed_inference == enabled
    fresh_driver = get('krea2').make_workbench_engine().driver
    assert not getattr(fresh_driver, '_file_backed_inference', False)


def test_gpu_upload_uses_pinned_copy_not_the_private_mapping(tmp_path):
    path = tmp_path/'weight.safetensors'
    save_file({'w':torch.randn(8, 8)}, str(path))
    with MemoryEfficientSafeOpen(str(path), mmap_cpu=True) as reader:
        source = reader.get_tensor('w')
    module = torch.nn.Linear(8, 8).requires_grad_(False)
    module.weight = torch.nn.Parameter(source, requires_grad=False)
    bias = module.bias
    masters = {module.weight: source}
    calls = []
    def pin(tensor, **kw):
        assert tensor is source
        calls.append('pin')
        return tensor.clone()
    def upload(tensor, device):
        assert not hasattr(tensor.untyped_storage(), '_fizgig_mmap')
        calls.append('upload')
        return tensor.clone()  # A CPU stand-in for the destination GPU tensor.
    with patch.object(torch.Tensor, 'pin_memory', pin), patch.object(torch.Tensor, 'to', upload):
        move_frozen_weights(module, torch.device('cuda'), masters)
    assert calls == ['pin', 'upload']
    assert module.weight.data_ptr() != source.data_ptr()
    assert module.bias is bias
    torch.testing.assert_close(module.weight, source)
