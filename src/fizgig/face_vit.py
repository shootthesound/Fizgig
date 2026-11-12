"""AdaFace ViT-Base: Fizgig's face identity model for people (face_utils.FaceEmbedder "adaface").

The network is CVLface's ViT 'base' (github.com/mk-minchul/CVLface, MIT): patch 8, width 512, 24 blocks, 16 heads, MLP
x3 with ReLU6, a 512-d feature head, rebuilt here for inference (no dropout / drop-path / masking) so only the weights
are downloaded: minchul/cvlface_adaface_vit_base_webface4m (trained on WebFace4M, non-commercial research use - as
InsightFace's own models). Input: a 112x112 face aligned to the 5-point ArcFace template, RGB, (x - 0.5) / 0.5.
Chosen over ArcFace on 12 Nov 2026: its best epochs matched Peter's picks every time on H3 Image runs."""
import torch
import torch.nn as nn


class _Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
        return self.proj((a @ v).transpose(1, 2).reshape(b, n, d))


class _Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.ReLU6(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim, heads, ratio):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn, self.mlp = _Attn(dim, heads), _Mlp(dim, int(dim * ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _PatchEmbed(nn.Module):
    def __init__(self, dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class AdaFaceViT(nn.Module):
    def __init__(self, out_dim=512, dim=512, depth=24, heads=16, ratio=3, patch=8, size=112):
        super().__init__()
        n = (size // patch) ** 2
        self.n, self.dim = n, dim
        self.patch_embed = _PatchEmbed(dim, patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads, ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.feature = nn.Sequential(nn.Linear(dim * n, dim, bias=False), nn.BatchNorm1d(dim, eps=2e-5),
                                     nn.Linear(dim, out_dim, bias=False), nn.BatchNorm1d(out_dim, eps=2e-5))

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for b in self.blocks:
            x = b(x)
        return self.feature(self.norm(x).reshape(x.shape[0], self.n * self.dim))


def load(path):
    """The network with the weights file's tensors (CVLface's 'model.net.' keys), eval mode, CPU float32."""
    from safetensors.torch import load_file
    sd = {k[len("model.net."):]: v for k, v in load_file(path).items() if k.startswith("model.net.")}
    net = AdaFaceViT()
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(f"AdaFace weights don't fit the network (missing {missing[:4]}, unexpected {unexpected[:4]})")
    return net.eval()


def embed(net, aligned_rgb):
    """Unit 512-d embedding of a 112x112x3 uint8 RGB aligned face: the face and its mirror image, averaged."""
    x = torch.from_numpy(aligned_rgb.copy()).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)[None]
    with torch.no_grad():
        e = net(torch.cat([x, x.flip(3)])).sum(0)
    return (e / e.norm()).numpy()
