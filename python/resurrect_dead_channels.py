#!/usr/bin/env python3
"""
Resurrect dead channels in a transformer checkpoint and write a new checkpoint.

A channel is dead when no gradient can reach the weights that would make it useful. Under Muon
with weight decay such channels can end up with exactly zero weights on both sides of a
multiplication, a state that never recovers by itself. Two kinds of channel are handled:

  FFN hidden channels: hidden channel j is written by row j of ffn_linear1, and of
    ffn_linear_gate for SwiGLU, and read by column j of ffn_linear2. It is dead when that column
    is near zero, since then neither input row gets gradient. For SwiGLU it is also dead when
    both input rows are near zero, since silu(0) = 0 and the gate is 0, so neither row gets
    gradient whatever the column holds. Resurrection sets the ffn_linear1 and gate rows to a
    noisy copy of a random live channel's rows and the ffn_linear2 column to exactly zero, so the
    model's function is unchanged and the column starts receiving gradient from the new
    activation. A channel with only one of its two input rows near zero recovers on its own and
    is left alone.

  Attention q/k RoPE pairs: pair p of kv head g consists of rows [g*D + 2p, g*D + 2p + 1] of
    k_proj and the same rows of q_proj for each query head in the group. The pair is dead when
    both the q rows and the k rows are near zero. Each side only gets gradient through the
    other, so both are resurrected: the q rows become a noisy copy of a random live pair's q rows
    at full scale, the k rows a noisy copy scaled by -k-scale, so that the k side has a strong
    gradient signal while the change to the attention logits stays small.

Near zero means a norm at most -dead-frac times the tensor's reference channel norm, the same
definition as the deadrows_batch training metric (see Metrics.reference_channel_norm). Copy
sources are channels that are live on every side. Noise is additive Gaussian with standard
deviation -noise-frac times the RMS of the copied source row, and the result is rescaled to the
source row's norm, so 1.0 means the copy is about as much noise as signal at a live channel's
scale. The FFN copies are then multiplied by -copy-scale: resurrecting many channels at full
live norm inflates the model's total weight norm, which train.py's adaptive weight decay reacts
to by decaying everything harder.

Dead v/out attention channels and dead output channels of other weight tensors are reported but
not resurrected.

The optimizer state and the SWA model are dropped from the output checkpoint. train.py starts a
fresh optimizer and fresh SWA when they are absent. Train the result only with -wd-floor-frac
set, otherwise the resurrected channels die again at the next high learning rate phase.

Example:
  python resurrect_dead_channels.py -checkpoint in.ckpt -output out.ckpt -dry-run
  python resurrect_dead_channels.py -checkpoint in.ckpt -output out.ckpt
"""

import argparse
import logging
import os
import sys

import torch

from katago.train import load_model
from katago.train.metrics_pytorch import Metrics


def row_norms(w):
    return torch.linalg.vector_norm(w.float(), dim=tuple(range(1, w.dim())))


def col_norms(w):
    return torch.linalg.vector_norm(w.float(), dim=0)


def is_dead(norms, dead_frac):
    return norms <= dead_frac * Metrics.reference_channel_norm(norms)


def noisy_copy(src, noise_frac, generator):
    """src plus Gaussian noise with std noise_frac * rms(src), rescaled to the norm of src so the
    resurrected channel starts at the scale of a live one."""
    src = src.float()
    src_norm = src.norm()
    assert src_norm > 0, "copy source must be a live channel"
    rms = src_norm / (src.numel() ** 0.5)
    noise = torch.randn(src.shape, generator=generator, dtype=torch.float32) * (noise_frac * rms)
    out = src + noise
    return out * (src_norm / out.norm())


def resurrect_ffn(sd, prefix, dead_frac, noise_frac, copy_scale, generator, dry_run):
    """Returns (num_dead, num_channels, num_one_row_dead). prefix ends with '.'"""
    w1 = sd[prefix + "ffn_linear1.weight"]
    w2 = sd[prefix + "ffn_linear2.weight"]
    wg = sd.get(prefix + "ffn_linear_gate.weight")
    ffn_dim = w1.shape[0]
    assert w2.shape[1] == ffn_dim

    out_dead = is_dead(col_norms(w2), dead_frac)
    row1_dead = is_dead(row_norms(w1), dead_frac)
    if wg is not None:
        rowg_dead = is_dead(row_norms(wg), dead_frac)
        dead = out_dead | (row1_dead & rowg_dead)
        one_row_dead = (row1_dead ^ rowg_dead) & ~dead
        live = ~(out_dead | row1_dead | rowg_dead)
    else:
        dead = out_dead
        one_row_dead = row1_dead & ~dead
        live = ~(out_dead | row1_dead)

    dead_idx = torch.nonzero(dead).flatten()
    live_idx = torch.nonzero(live).flatten()
    n_dead = dead_idx.numel()
    if n_dead > 0 and live_idx.numel() == 0:
        raise Exception(f"{prefix}: no fully live FFN channel to copy from")
    if n_dead > 0 and not dry_run:
        src_idx = live_idx[torch.randint(live_idx.numel(), (n_dead,), generator=generator)]
        with torch.no_grad():
            for j, s in zip(dead_idx.tolist(), src_idx.tolist()):
                w1[j] = noisy_copy(w1[s], noise_frac, generator) * copy_scale
                if wg is not None:
                    wg[j] = noisy_copy(wg[s], noise_frac, generator) * copy_scale
                w2[:, j] = 0.0
    return n_dead, ffn_dim, int(one_row_dead.sum().item())


def resurrect_qk(sd, prefix, num_heads, num_kv_heads, dead_frac, noise_frac, k_scale, generator, dry_run):
    """Returns (num_dead_pairs, num_pairs)."""
    wq = sd[prefix + "q_proj.weight"]
    wk = sd[prefix + "k_proj.weight"]
    head_dim = wq.shape[0] // num_heads
    assert wq.shape[0] == num_heads * head_dim
    assert wk.shape[0] == num_kv_heads * head_dim
    assert head_dim % 2 == 0
    n_rep = num_heads // num_kv_heads
    num_pairs = head_dim // 2

    def q_rows(g, p):
        return [h * head_dim + 2 * p + i for h in range(g * n_rep, (g + 1) * n_rep) for i in range(2)]

    def k_rows(g, p):
        return [g * head_dim + 2 * p + i for i in range(2)]

    qn = torch.zeros(num_kv_heads, num_pairs)
    kn = torch.zeros(num_kv_heads, num_pairs)
    for g in range(num_kv_heads):
        for p in range(num_pairs):
            qn[g, p] = torch.linalg.vector_norm(wq[q_rows(g, p)].float())
            kn[g, p] = torch.linalg.vector_norm(wk[k_rows(g, p)].float())
    q_dead = is_dead(qn.flatten(), dead_frac).view_as(qn)
    k_dead = is_dead(kn.flatten(), dead_frac).view_as(kn)
    dead = q_dead & k_dead
    dead_list = torch.nonzero(dead).tolist()
    live_list = torch.nonzero(~(q_dead | k_dead)).tolist()
    if len(dead_list) > 0 and len(live_list) == 0:
        raise Exception(f"{prefix}: no fully live q/k pair to copy from")
    if len(dead_list) > 0 and not dry_run:
        src_choice = torch.randint(len(live_list), (len(dead_list),), generator=generator).tolist()
        with torch.no_grad():
            for (g, p), c in zip(dead_list, src_choice):
                sg, sp = live_list[c]
                wq[q_rows(g, p)] = noisy_copy(wq[q_rows(sg, sp)], noise_frac, generator)
                wk[k_rows(g, p)] = noisy_copy(wk[k_rows(sg, sp)], noise_frac, generator) * k_scale
    return len(dead_list), num_kv_heads * num_pairs


def count_dead_v_out(sd, prefix, dead_frac):
    """Value channels whose v_proj row and out_proj column are both near zero, the same kind of
    stuck state as a dead q/k pair. Returns (num_dead, num_channels)."""
    wv = sd[prefix + "v_proj.weight"]
    wo = sd[prefix + "out_proj.weight"]
    if wo.shape[1] != wv.shape[0]:
        # Grouped-query attention: out_proj reads the expanded heads, so columns do not map
        # one to one onto v_proj rows.
        return 0, 0
    dead = is_dead(row_norms(wv), dead_frac) & is_dead(col_norms(wo), dead_frac)
    return int(dead.sum().item()), wv.shape[0]


def main():
    parser = argparse.ArgumentParser(description="Resurrect dead FFN channels and q/k pairs in a checkpoint")
    parser.add_argument("-checkpoint", required=True, help="Input checkpoint")
    parser.add_argument("-output", required=True, help="Output checkpoint, must not already exist")
    parser.add_argument("-dead-frac", type=float, default=Metrics.DEAD_CHANNEL_NORM_FRAC, help=f"A channel side is dead when its norm is at most this fraction of the tensor's reference channel norm (default {Metrics.DEAD_CHANNEL_NORM_FRAC})")
    parser.add_argument("-noise-frac", type=float, default=1.0, help="Std of the additive noise on copied rows, relative to the source row RMS (default 1.0)")
    parser.add_argument("-copy-scale", type=float, default=1.0, help="Scale of the resurrected FFN input and gate rows relative to the copied source rows (default 1.0). Below 1 keeps the resurrected rows from inflating the model's weight norm, which train.py's adaptive weight decay would otherwise react to, at the cost of a smaller initial activation (proportional to the square of the scale for SwiGLU) and so slower recruitment")
    parser.add_argument("-k-scale", type=float, default=0.1, help="Scale of the resurrected k rows relative to the copied source (default 0.1)")
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("-dry-run", action="store_true", help="Only report what would be resurrected")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if not args.dry_run and os.path.exists(args.output):
        raise Exception(f"Output already exists: {args.output}")

    state_dict = load_model.load_checkpoint(args.checkpoint, map_location="cpu")
    if "config" not in state_dict:
        raise Exception("Checkpoint has no embedded model config, cannot determine the attention head layout")
    config = state_dict["config"]
    num_heads = config["transformer_heads"]
    num_kv_heads = config.get("transformer_kv_heads", num_heads)
    # Prefix-stripped view of the model tensors. The values are the same tensor objects as in
    # state_dict["model"], so in-place edits through this view land in the saved checkpoint.
    by_name = load_model.load_model_state_dict(state_dict)
    generator = torch.Generator().manual_seed(args.seed)

    ffn_prefixes = sorted(k[:-len("ffn_linear1.weight")] for k in by_name if k.endswith("ffn_linear1.weight"))
    attn_prefixes = sorted(k[:-len("q_proj.weight")] for k in by_name if k.endswith("q_proj.weight"))

    total_ffn_dead = total_ffn = total_one_row = 0
    for prefix in ffn_prefixes:
        n_dead, n, one_row = resurrect_ffn(by_name, prefix, args.dead_frac, args.noise_frac, args.copy_scale, generator, args.dry_run)
        logging.info(f"{prefix}: FFN channels dead {n_dead}/{n}" + (f", with one input row dead and left alone {one_row}" if one_row > 0 else ""))
        total_ffn_dead += n_dead
        total_ffn += n
        total_one_row += one_row

    total_qk_dead = total_qk = 0
    for prefix in attn_prefixes:
        n_dead, n = resurrect_qk(by_name, prefix, num_heads, num_kv_heads, args.dead_frac, args.noise_frac, args.k_scale, generator, args.dry_run)
        if n_dead > 0:
            logging.info(f"{prefix}: q/k pairs dead {n_dead}/{n}")
        total_qk_dead += n_dead
        total_qk += n
        n_vo_dead, n_vo = count_dead_v_out(by_name, prefix, args.dead_frac)
        if n_vo_dead > 0:
            logging.info(f"Note: {prefix} has {n_vo_dead}/{n_vo} dead v/out channels, which this script does not resurrect")

    for name, w in by_name.items():
        if not Metrics.is_output_channel_tensor(w):
            continue
        if any(name.startswith(p) for p in ffn_prefixes) or any(name.startswith(p) for p in attn_prefixes):
            continue
        n_dead = int(is_dead(row_norms(w), args.dead_frac).sum().item())
        if n_dead > 0:
            logging.info(f"Note: {name} has {n_dead}/{w.shape[0]} near-zero output channels, which this script does not resurrect")

    logging.info(f"Total FFN channels dead: {total_ffn_dead}/{total_ffn} ({100.0 * total_ffn_dead / max(1, total_ffn):.1f}%), with one input row dead: {total_one_row}")
    logging.info(f"Total q/k pairs dead: {total_qk_dead}/{total_qk} ({100.0 * total_qk_dead / max(1, total_qk):.2f}%)")
    if args.dry_run:
        logging.info("Dry run, not writing anything")
        return

    dropped = [k for k in ("optimizer", "swa_model") if k in state_dict]
    for k in dropped:
        del state_dict[k]
    logging.info(f"Dropped from checkpoint: {dropped}")
    torch.save(state_dict, args.output)
    logging.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
