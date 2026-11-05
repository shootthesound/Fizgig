"""Weight-only rank reduction for any described family (the Extract tab's standard-layer path), in the family's own
key format (the old extractor writes kohya keys, which is the wrong format for families that approved another one).

For every LoRA pair the delta dW = (alpha/r) * B @ A has rank <= r, so its SVD is exact from two thin QR
factorisations and an r x r SVD - no dense out x in matrix, no model, fast on CPU. The top `rank` components are
kept and split evenly between the new up and down weights, with alpha = the new rank (scale 1).
"""
import time

import torch


# Identity mode (Extract, both cards; Peter, 5 Nov 2026, #301): a model's identity blocks (description.identity_blocks)
# keep this many times the rank, so a character's likeness survives a small file
IDENTITY_RANK = 4


@torch.no_grad()
def reduce_pair(A, B, scale, rank):
    """(A [r, in], B [out, r], scale) -> (down [k, in], up [out, k], kept energy fraction), k = min(rank, r)."""
    A, B = A.float(), B.float() * scale
    qb, rb = torch.linalg.qr(B)                   # B = qb rb
    qa, ra = torch.linalg.qr(A.T)                 # A^T = qa ra  ->  B A = qb (rb ra^T) qa^T
    u, s, vh = torch.linalg.svd(rb @ ra.T)
    k = min(rank, s.numel())
    root = s[:k].sqrt()
    up = (qb @ u[:, :k]) * root
    down = (root[:, None] * vh[:k]) @ qa.T
    energy = float((s[:k] ** 2).sum() / (s ** 2).sum().clamp_min(1e-30))
    return down, up, energy


def extract_weight_only(desc, source, output, rank, dtype=torch.bfloat16, progress=None, blocks=None, identity=False):
    """Write `output` at `rank`. blocks: {block id: multiplier} - only modules in those blocks are kept, each delta
    scaled by its block's multiplier (None = every module at 1). identity: the model's identity blocks keep
    IDENTITY_RANK x the rank (never more than the source has). Returns a summary dict (layers, skipped, dropped
    (outside `blocks`), mean kept energy, seconds, params)."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    from fizgig.families.lorafile import (family_keys, get_up, loha_delta, loha_modules, lokr_factors, lokr_modules,
                                          lora_pairs)
    t0 = time.time()
    sd, energies, skipped, params, dropped = {}, [], 0, 0, 0
    ids = set(desc.identity_blocks) if identity else set()
    block_of = desc.load_driver().block_of if blocks or ids else None

    def rank_of(mod):
        return rank * IDENTITY_RANK if ids and block_of(mod) in ids else rank

    def mult(mod):
        """The module's multiplier, or None when its block is not kept."""
        if blocks is None:
            return 1.0
        return blocks.get(block_of(mod))
    with safe_open(source, "pt") as f:
        metadata = dict(f.metadata() or {})
        pairs = lora_pairs(desc, f.keys())
        for i, (mod, dk, uk, ak) in enumerate(pairs):
            if progress is not None:
                progress("SVD", i, len(pairs))
            if mod is None:
                skipped += 1
                continue
            m = mult(mod)
            if m is None:
                dropped += 1
                continue
            A, B = f.get_tensor(dk), get_up(f, uk)
            alpha = float(f.get_tensor(ak).item()) if ak else float(A.shape[0])
            down, up, e = reduce_pair(A, B, m * alpha / A.shape[0], rank_of(mod))
            kd, ku, ka = family_keys(desc, mod)
            sd[kd] = down.to(dtype).contiguous()
            sd[ku] = up.to(dtype).contiguous()
            sd[ka] = torch.tensor(float(down.shape[0]))
            energies.append(e)
            params += down.numel() + up.numel()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # LoKR and LoHa: the delta is full rank - build it densely and SVD it
        dense = [("LoKR", mod, stem) for mod, stem in lokr_modules(desc, f.keys())]
        dense += [("LoHa", mod, stem) for mod, stem in loha_modules(desc, f.keys())]
        for i, (kind, mod, stem) in enumerate(dense):
            if progress is not None:
                progress(f"SVD ({kind})", i, len(dense))
            if mod is None:
                skipped += 1
                continue
            m = mult(mod)
            if m is None:
                dropped += 1
                continue
            if kind == "LoKR":
                w1, w2, scale = lokr_factors(f, stem)
                dW = torch.kron(w1.to(dev), w2.to(dev)) * (scale * m)
            else:
                dW = loha_delta(f, stem).to(dev) * m
            U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
            k = min(rank_of(mod), S.numel())
            root = S[:k].sqrt()
            down, up = (root[:, None] * Vh[:k]).cpu(), (U[:, :k] * root).cpu()
            kd, ku, ka = family_keys(desc, mod)
            sd[kd] = down.to(dtype).contiguous()
            sd[ku] = up.to(dtype).contiguous()
            sd[ka] = torch.tensor(float(k))
            energies.append(float((S[:k] ** 2).sum() / (S ** 2).sum().clamp_min(1e-30)))
            params += down.numel() + up.numel()
            del dW, U, S, Vh
    if not sd:
        raise RuntimeError(f"No {desc.display_name} LoRA modules found in {source}"
                           + (" inside the chosen blocks" if blocks else ""))
    for stale in ("sshs_model_hash", "sshs_legacy_hash", "modelspec.hash_sha256"):
        metadata.pop(stale, None)
    metadata.update({"ss_network_dim": str(rank), "ss_network_alpha": str(float(rank)),
                     "ss_fizgig_extract": f"weight-only SVD to rank {rank}"
                                          + (f" over {len(blocks)} blocks" if blocks else "")
                                          + (f", identity blocks at rank {rank * IDENTITY_RANK}" if ids else "")})
    if desc.lora.save_fused:
        sd = desc.load_driver().export_lora_state_dict(sd)
    save_file(sd, output, metadata=metadata)
    return {"layers": len(energies), "skipped": skipped, "dropped": dropped, "energy": sum(energies) / len(energies),
            "seconds": time.time() - t0, "params": params, "output": output}
