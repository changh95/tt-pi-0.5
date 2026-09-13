# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-only (torch, no device, no ttnn tensors) proofs for the TT_FUSED=1 graph
(``opt/pi05-base-p150-megakernel``).

Every exact reformulation the fused device graph relies on is checked here against the reference
math, plus the knob plumbing (``TT_FUSED=0`` -> legacy choices; unset = fused). Run with the host python of
the model's tt-metal tree (torch 2.7.1, pytest 9):

    cd models/pi05-base-p150
    PYTHONPATH=code TT_METAL_HOME=$TREE $TREE/python_env/bin/python -m pytest \
        code/models/experimental/pi0_5/tests/test_fused_host.py -q

or as a plain script (``python test_fused_host.py``: runs every ``test_*`` with asserts).

Exactness legend used in the assertions:
    torch.equal          bit-identical (pure data movement / identities)
    fp64 rel 1e-12       the same per-row math in float64; only the BLAS blocking (summation order)
                         may differ between shapes -> max|d| <= 1e-12 * max(1, max|ref|)
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

from models.experimental.pi0_5.common.configs import GemmaConfig, SigLIPConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig, RESIDUAL_MODES
from models.experimental.pi0_5.common import fused_host as fh
from models.experimental.pi0_5.reference.torch_gemma import GemmaBlock, adarms_norm, precompute_freqs_cis, rms_norm
from models.experimental.pi0_5.reference.torch_siglip import SigLIPVisionTower

torch.manual_seed(0)
F64 = torch.float64


def _assert_close(a: torch.Tensor, b: torch.Tensor, rtol: float, what: str):
    """max|a-b| <= rtol * max(1, max|b|): fp64 identity up to BLAS summation order."""
    d = (a.double() - b.double()).abs().max().item()
    scale = max(1.0, b.double().abs().max().item())
    assert d <= rtol * scale, f"{what}: max|d| = {d:.3e} > {rtol} x {scale:.3e}"
    return d


# =========================================================================================
# 1. knob plumbing
# =========================================================================================


def test_knob_default_on_and_zero_selects_legacy():
    # default ON since the device pass (2026-09-13); TT_FUSED=0 restores the legacy path bit-for-bit
    assert FusedConfig.from_env({}).enabled is True
    assert FusedConfig.from_env({}) == FusedConfig.from_env({"TT_FUSED": "1"})
    for v in ("0", "false", "off", ""):
        cfg = FusedConfig.from_env({"TT_FUSED": v})
        assert cfg.enabled is False
        assert cfg == FusedConfig.legacy()
    # legacy choices when off, regardless of sub-knobs in the environment
    off = FusedConfig.from_env({"TT_FUSED": "0", "PI05_FUSED_RESIDUAL": "bf16", "PI05_MLP_CHUNK": "0"})
    assert off.expert_residual_weight_dtype() == "bfloat8_b"
    assert off.expert_mlp_act_dtype() == "bfloat8_b"
    assert off.action_out_weight_dtype() == "bfloat8_b"
    assert off.keep_fused_gate_up_copy() is True
    assert off.fused_residual is False
    assert off.mlp_chunk == 256  # sub-knobs are not read when off


def test_knob_on_defaults_match_the_evaluation_recipe():
    cfg = FusedConfig.from_env({"TT_FUSED": "1"})
    assert cfg.enabled and cfg.trace and cfg.trace_region_size == 160_000_000
    assert cfg.residual == "bf16" and cfg.mlp_chunk == 256 and cfg.vlm_down_pc == "mcast2d" and cfg.vlm_gateup_pc == "auto"
    assert FusedConfig.from_env({"TT_FUSED": "1", "PI05_MLP_CHUNK": "0"}).mlp_chunk == 0
    assert cfg.siglip_batched and cfg.skip_vlm_tail
    assert cfg.sdpa_vlm is None and cfg.sdpa_expert is None and cfg.sdpa_siglip is None  # legacy SDPA configs
    assert FusedConfig.from_env({"TT_FUSED": "1", "PI05_SDPA_EXPERT_CHUNKS": "legacy"}).sdpa_expert is None
    assert cfg.expert_mm == "mcast1d_fp32" and cfg.expert_mm_blocks == (1, 8, 4, 1, 4, 0, 2)  # device pass 2026-09-13
    assert FusedConfig.from_env({"TT_FUSED": "1", "PI05_EXPERT_MM": "linear"}).expert_mm == "linear"
    # measured defaults (device pass 2026-09-13): 11x2 core grid, one M tile per core, N blocks of 4 / 1 tiles
    assert cfg.dit_blocks == (1, 8, 4, 1, 4, 0, 2) and cfg.euler_dit_blocks == (1, 8, 1, 1, 1, 0, 2)
    assert cfg.expert_residual_weight_dtype() == "bfloat16"
    assert cfg.expert_mlp_act_dtype() == "bfloat16"
    assert cfg.action_out_weight_dtype() == "bfloat16"
    assert cfg.keep_fused_gate_up_copy() is False
    assert cfg.fused_residual is True


def test_knob_sub_modes_and_parsing():
    env = {
        "TT_FUSED": "1",
        "PI05_TRACE": "0",
        "PI05_TRACE_REGION_SIZE": "95000000",
        "PI05_FUSED_RESIDUAL": "mixed",
        "PI05_MLP_CHUNK": "0",
        "PI05_SIGLIP_BATCHED": "0",
        "PI05_SKIP_VLM_TAIL": "0",
        "PI05_SDPA_VLM_CHUNKS": "128,256",
        "PI05_SDPA_EXPERT_CHUNKS": "64,256",
        "PI05_SDPA_SIGLIP_CHUNKS": "64, 256",
        "PI05_DIT_BLOCKS": "4,4,4,2,2",
        "PI05_EULER_DIT_BLOCKS": "op",
    }
    cfg = FusedConfig.from_env(env)
    assert cfg.trace is False and cfg.trace_region_size == 95_000_000
    assert cfg.residual == "mixed" and cfg.expert_mlp_act_dtype() == "bfloat8_b"
    assert cfg.expert_residual_weight_dtype() == "bfloat16"  # mixed still needs bf16 weights (residual == weight format)
    assert cfg.mlp_chunk == 0 and cfg.siglip_batched is False and cfg.skip_vlm_tail is False
    assert cfg.sdpa_vlm == (128, 256) and cfg.sdpa_expert == (64, 256) and cfg.sdpa_siglip == (64, 256)
    assert cfg.dit_blocks == (4, 4, 4, 2, 2, 0, 0)  # 5 values -> full device grid
    assert cfg.euler_dit_blocks is None  # "op" -> the op's default blocks
    seven = FusedConfig.from_env({"TT_FUSED": "1", "PI05_DIT_BLOCKS": "1,8,4,1,4,11,2"})
    assert seven.dit_blocks == (1, 8, 4, 1, 4, 11, 2)
    for bad in ("1,8,4", "1,8,3,1,2", "2,8,4,1,4,0,-1", "0,8,4,1,4"):
        try:
            FusedConfig.from_env({"TT_FUSED": "1", "PI05_DIT_BLOCKS": bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"PI05_DIT_BLOCKS={bad!r} should be rejected")
    legacy_res = FusedConfig.from_env({"TT_FUSED": "1", "PI05_FUSED_RESIDUAL": "legacy"})
    assert legacy_res.fused_residual is False
    assert legacy_res.expert_residual_weight_dtype() == "bfloat8_b"
    assert legacy_res.expert_mlp_act_dtype() == "bfloat8_b"
    assert set(RESIDUAL_MODES) == {"bf16", "mixed", "legacy"}


def test_knob_legacy_sample_actions_availability():
    """The legacy expert block feeds a bf8 residual + mlp.down_proj into dit_minimal_matmul_addcmul_fused,
    whose factory requires residual format == weight format: the legacy sample_actions can only run on a
    model whose expert o_proj / down_proj stayed bf8 (TT_FUSED unset, or PI05_FUSED_RESIDUAL=legacy)."""
    assert FusedConfig.legacy().legacy_sample_actions_available is True
    assert FusedConfig.from_env({"TT_FUSED": "0"}).legacy_sample_actions_available is True
    assert FusedConfig.from_env({}).legacy_sample_actions_available is False  # default = fused, bf16 expert weights
    assert FusedConfig.from_env({"TT_FUSED": "1", "PI05_FUSED_RESIDUAL": "legacy"}).legacy_sample_actions_available is True
    for mode in ("bf16", "mixed"):
        cfg = FusedConfig.from_env({"TT_FUSED": "1", "PI05_FUSED_RESIDUAL": mode})
        assert cfg.expert_residual_weight_dtype() == "bfloat16"
        assert cfg.legacy_sample_actions_available is False
    assert FusedConfig.from_env({"TT_FUSED": "1"}).legacy_sample_actions_available is False  # default bf16
    # invariant the guard relies on: availability <=> bf8 expert residual weights
    for env in ({"TT_FUSED": "0"}, {"TT_FUSED": "1"}, {"TT_FUSED": "1", "PI05_FUSED_RESIDUAL": "mixed"}, {"TT_FUSED": "1", "PI05_FUSED_RESIDUAL": "legacy"}):
        cfg = FusedConfig.from_env(env)
        assert cfg.legacy_sample_actions_available == (cfg.expert_residual_weight_dtype() == "bfloat8_b")
    for bad in (
        {"TT_FUSED": "1", "PI05_FUSED_RESIDUAL": "fp32"},
        {"TT_FUSED": "1", "PI05_MLP_CHUNK": "100"},
        {"TT_FUSED": "1", "PI05_SDPA_VLM_CHUNKS": "100,32"},
        {"TT_FUSED": "1", "PI05_SDPA_VLM_CHUNKS": "32"},
        {"TT_FUSED": "1", "PI05_DIT_BLOCKS": "1,2,3"},
        {"TT_FUSED": "maybe"},
    ):
        try:
            FusedConfig.from_env(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


# =========================================================================================
# 2. SigLIP: host im2col == device unfold order, conv2d equivalence, pos table, concat order
# =========================================================================================


def test_im2col_matches_device_unfold_order_exactly():
    x = torch.randn(2, 3, 224, 224)
    host = fh.im2col_patches(x, 14)
    dev = fh.device_unfold_reference(x, 14)
    assert host.shape == (2, 256, 588) and dev.shape == (2, 256, 588)
    assert torch.equal(host, dev)  # same permutation of the same pixels
    padded = fh.im2col_patches(x, 14, pad_to=608)
    assert padded.shape == (2, 256, 608)
    assert torch.equal(padded[..., :588], host) and torch.count_nonzero(padded[..., 588:]) == 0


def test_im2col_commutes_with_bf16_rounding():
    """The legacy path rounded the pixels to bf16 BEFORE its device unfold; the fused path rounds the
    im2col at upload. A permutation commutes with elementwise rounding -> identical bf16 inputs."""
    x = torch.randn(2, 3, 224, 224)
    a = fh.im2col_patches(x.to(torch.bfloat16).float(), 14, pad_to=608)
    b = fh.im2col_patches(x, 14, pad_to=608).to(torch.bfloat16).float()
    assert torch.equal(a, b)


def test_patch_linear_weight_prep_and_conv2d_equivalence():
    conv_w = torch.randn(1152, 3, 14, 14, dtype=F64) * 0.02
    conv_b = torch.randn(1152, dtype=F64)
    # the legacy PatchEmbeddingTTNN.__init__ prep, transcribed
    legacy_w = conv_w.permute(0, 2, 3, 1).contiguous().view(1152, -1).T.contiguous()
    legacy_w = F.pad(legacy_w, (0, 0, 0, 608 - 588))
    w = fh.conv_weight_to_patch_linear(conv_w, 608)
    assert w.shape == (608, 1152) and torch.equal(w, legacy_w)
    x = torch.randn(2, 3, 224, 224, dtype=F64)
    ref = F.conv2d(x, conv_w, conv_b, stride=14).flatten(2).transpose(1, 2)  # [2, 256, 1152]
    got = fh.im2col_patches(x, 14, pad_to=608) @ w + conv_b
    _assert_close(got, ref, 1e-10, "im2col @ W_patch + b vs conv2d")


def test_positional_table_is_the_identity_gather():
    pos = torch.randn(256, 1152)
    table = fh.positional_table(pos, 2)
    assert table.shape == (2, 256, 1152)
    ids = torch.arange(256)
    gathered = F.embedding(ids, pos)  # legacy: ttnn.embedding(arange, pos_emb)
    assert torch.equal(table[0], gathered) and torch.equal(table[1], gathered)
    # bf16 rounding of the table == bf16 rounding of the gathered rows (same values)
    assert torch.equal(table.to(torch.bfloat16)[0], gathered.to(torch.bfloat16))


def test_prefix_concat_order_batched_equals_legacy_per_image_concat():
    img = torch.randn(2, 256, 2048)
    lang = torch.randn(1, 224, 2048)
    legacy = torch.cat([img[0:1], img[1:2], lang], dim=1)  # concat([img0, img1, lang])
    fused = torch.cat([fh.prefix_concat_batched(img), lang], dim=1)
    assert fused.shape == (1, 736, 2048) and torch.equal(fused, legacy)


def _tiny_siglip_weights(cfg: SigLIPConfig, g: torch.Generator) -> dict:
    h, ff, p = cfg.hidden_size, cfg.intermediate_size, cfg.num_patches
    r = lambda *s: torch.randn(*s, generator=g, dtype=F64) * 0.1  # noqa: E731
    w = {
        "vision_model.embeddings.patch_embedding.weight": r(h, 3, cfg.patch_size, cfg.patch_size),
        "vision_model.embeddings.patch_embedding.bias": r(h),
        "vision_model.embeddings.position_embedding.weight": r(p, h),
        "vision_model.post_layernorm.weight": 1 + r(h),
        "vision_model.post_layernorm.bias": r(h),
    }
    for i in range(cfg.num_hidden_layers):
        pre = f"vision_model.encoder.layers.{i}."
        w.update(
            {
                pre + "layer_norm1.weight": 1 + r(h),
                pre + "layer_norm1.bias": r(h),
                pre + "layer_norm2.weight": 1 + r(h),
                pre + "layer_norm2.bias": r(h),
                pre + "self_attn.q_proj.weight": r(h, h),
                pre + "self_attn.q_proj.bias": r(h),
                pre + "self_attn.k_proj.weight": r(h, h),
                pre + "self_attn.k_proj.bias": r(h),
                pre + "self_attn.v_proj.weight": r(h, h),
                pre + "self_attn.v_proj.bias": r(h),
                pre + "self_attn.out_proj.weight": r(h, h),
                pre + "self_attn.out_proj.bias": r(h),
                pre + "mlp.fc1.weight": r(ff, h),
                pre + "mlp.fc1.bias": r(ff),
                pre + "mlp.fc2.weight": r(h, ff),
                pre + "mlp.fc2.bias": r(h),
            }
        )
    return w


def test_siglip_batched_equals_per_image():
    """Batching the two cameras is per-row identical math (every op is row-/batch-independent)."""
    cfg = SigLIPConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, image_size=28, patch_size=14, intermediate_size=64)
    g = torch.Generator().manual_seed(1)
    tower = SigLIPVisionTower(cfg, _tiny_siglip_weights(cfg, g))
    x = torch.randn(2, 3, 28, 28, generator=g, dtype=F64)
    batched = tower.forward(x)
    per_image = torch.cat([tower.forward(x[0:1]), tower.forward(x[1:2])], dim=0)
    assert batched.shape == (2, cfg.num_patches, 32)
    _assert_close(batched, per_image, 1e-12, "SigLIP batched vs per-image")


def test_bias_fusion_math():
    """out_proj / fc2 bias inside the linear == linear then add (same expression; the device adds it in
    the fp32 accumulator instead of after the bf16 pack: rounding-level, noted in DEVICE_VALIDATION.md).
    fp64: identical up to BLAS summation order (1e-12 relative). bf16 emulation of the two orders
    (fused: round(acc + b); legacy: round(round(acc) + b)) differs by at most the bf16 rounding of
    the intermediate accumulator plus that of the output, i.e. <= 2^-7 (|acc| + |acc + b|): relative
    to the OUTPUT this is unbounded under cancellation (acc + b ~ 0), which is why the fusion is
    recorded as 'rounding-level' and gated by the device PCC, not claimed exact."""
    g = torch.Generator().manual_seed(7)
    x = torch.randn(2, 256, 1536, dtype=F64, generator=g)
    w = torch.randn(1152, 1536, dtype=F64, generator=g) / 1536**0.5
    b = torch.randn(1152, dtype=F64, generator=g)
    fused, separate = F.linear(x, w, b), F.linear(x, w) + b
    d = _assert_close(fused, separate, 1e-12, "bias fusion (fp64)")
    assert d < 1e-9
    # device order: fused = round_bf16(acc + b); legacy = round_bf16(round_bf16(acc) + b)
    acc = F.linear(x, w)
    fused_bf16 = fused.to(torch.bfloat16).double()
    legacy_bf16 = (acc.to(torch.bfloat16).double() + b).to(torch.bfloat16).double()
    diff = (fused_bf16 - legacy_bf16).abs()
    bound = 2.0**-7 * (acc.abs() + fused.abs())  # one bf16 ULP of the accumulator + one of the output
    assert (diff <= bound).all(), f"bias fusion: max diff/bound = {(diff / bound.clamp_min(1e-300)).max().item():.3f}"
    assert (diff > 0).any(), "expected the two rounding orders to differ somewhere (the test is not trivially true)"


# =========================================================================================
# 3. 64-row suffix, KV hoist, Euler fold, GeGLU split, dt values
# =========================================================================================


def test_pad_unpad_rows():
    x = torch.randn(1, 50, 32)
    p = fh.pad_rows(x, 64)
    assert p.shape == (1, 64, 32) and torch.equal(p[:, :50], x) and torch.count_nonzero(p[:, 50:]) == 0
    assert torch.equal(fh.unpad_rows(p, 50), x)
    assert fh.pad_rows(x, 50) is x
    assert fh.round_up(50) == 64 and fh.round_up(736) == 736 and fh.round_up(786) == 800


def test_euler_dts_equal_legacy_loop_arithmetic():
    num_steps = 10
    timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]  # the legacy loop
    legacy = [timesteps[i + 1] - timesteps[i] for i in range(num_steps)]
    assert fh.euler_dts(num_steps) == legacy  # python float equality: identical fp32 scalar attributes
    assert all(abs(dt + 0.1) < 1e-9 for dt in legacy)


def test_euler_step_dit_fold_is_exact():
    """x_t + dt * (h @ W + b) * ones  ==  x_t + dt * velocity (multiplying by exactly 1.0 is exact)."""
    h = torch.randn(1, 64, 1024, dtype=F64)
    w = torch.randn(1024, 32, dtype=F64)  # ttnn [K, N] layout (== action_out_proj.weight.T)
    b = torch.randn(1, 32, dtype=F64)
    x_t = torch.randn(1, 64, 32, dtype=F64)
    ones = torch.ones(1, 32, dtype=F64)
    for dt in fh.euler_dts(10):
        fused = fh.dit_reference(h, w, dt, x_t, ones, bias=b)
        legacy = fh.euler_step_reference(h, w, b, x_t, dt)
        assert torch.equal(fused, legacy)


def test_gated_residual_dit_math():
    """dit(act, W, 1.0, hidden, gate) == hidden + (act @ W) * gate  (the legacy mac(gate, linear, hidden))."""
    act = torch.randn(1, 64, 2048, dtype=F64)
    w = torch.randn(2048, 1024, dtype=F64)
    hidden = torch.randn(1, 64, 1024, dtype=F64)
    gate = torch.randn(1, 1, 1024, dtype=F64)
    fused = fh.dit_reference(act, w, 1.0, hidden, gate)
    legacy = gate * (act @ w) + hidden  # mac(gate, y, x) = gate * y + x
    assert torch.equal(fused, legacy)


def test_gate_up_split_equals_fused_gate_up():
    x = torch.randn(1, 64, 1024, dtype=F64)
    wg = torch.randn(4096, 1024, dtype=F64)
    wu = torch.randn(4096, 1024, dtype=F64)
    fused = F.linear(x, torch.cat([wg, wu], dim=0))
    g_f, u_f = fused[..., :4096], fused[..., 4096:]
    _assert_close(F.linear(x, wg), g_f, 1e-12, "gate split")
    _assert_close(F.linear(x, wu), u_f, 1e-12, "up split")
    # GeGLU on the split == GeGLU on the fused output (gelu is elementwise)
    _assert_close(F.gelu(F.linear(x, wg)) * F.linear(x, wu), F.gelu(g_f) * u_f, 1e-12, "GeGLU")


def test_kv_cache_plan_and_constraints():
    plan = fh.kv_cache_plan(736, 50)
    assert plan == {
        "prefix_len": 736,
        "action_horizon": 50,
        "logical_len": 786,
        "padded_len": 800,
        "suffix_rows": 64,
        "expert_update_idx": 736,
        "vlm_update_idx": 0,
    }
    # rotary_embedding_to_cache / fill_cache: update_idx % 32 == 0 and update_idx + rows <= padded rows
    assert plan["expert_update_idx"] % 32 == 0 and plan["vlm_update_idx"] % 32 == 0
    assert plan["expert_update_idx"] + plan["suffix_rows"] <= plan["padded_len"]
    assert plan["vlm_update_idx"] + plan["prefix_len"] <= plan["padded_len"]
    # served shapes: 2 x 256 + 224 (default) and 2 x 256 + 32 (README config), 1..3 cameras
    for n_img, L in ((2, 224), (2, 32), (1, 224), (3, 224)):
        fh.check_fused_shape_contract(n_img, L, 50)
    for bad in ((2, 100), (2, 0)):
        try:
            fh.check_fused_shape_contract(*bad, 50)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")
    specs = fh.persistent_input_specs(2, 224, 50, 32)
    assert specs["im2col"]["shape"] == (2, 256, 608) and specs["im2col"]["layout"] == "ROW_MAJOR"
    assert specs["tokens"]["shape"] == (1, 224) and specs["tokens"]["dtype"] == "uint32"
    assert specs["noise"]["shape"] == (1, 64, 32) and specs["noise"]["memory"] == "L1"
    assert specs["prefix_len"] == 736


def test_kv_hoist_cache_equals_concat_and_refreshes_per_call():
    """VLM writes rows 0..P-1 (rotated K, V), expert writes rows P..P+63: rows < P+50 equal the
    reference torch.cat([past_k, k]); a second call with a new prefix leaves no stale rows."""
    plan = fh.kv_cache_plan(96, 50)  # tiny: P=96, logical 146, padded 160
    pk, pv = torch.randn(1, 1, 96, 32), torch.randn(1, 1, 96, 32)
    sk, sv = torch.randn(1, 1, 64, 32), torch.randn(1, 1, 64, 32)
    ck, cv = fh.build_kv_cache_reference(pk, pv, sk, sv, plan)
    assert ck.shape == (1, 1, 160, 32)
    ref_k = torch.cat([pk, sk[:, :, :50]], dim=2)  # reference GemmaAttention: cat(past_k, k) with 50 rows
    ref_v = torch.cat([pv, sv[:, :, :50]], dim=2)
    assert torch.equal(ck[:, :, :146], ref_k) and torch.equal(cv[:, :, :146], ref_v)
    # rows 146..159 hold the padded suffix rows (masked by SDPA: beyond the logical length)
    assert torch.equal(ck[:, :, 146:160], sk[:, :, 50:64])
    # second call (new observation): the VLM rewrites rows 0..95 -> no stale prefix
    pk2, pv2 = torch.randn(1, 1, 96, 32), torch.randn(1, 1, 96, 32)
    ck2, cv2 = fh.build_kv_cache_reference(pk2, pv2, sk, sv, plan)
    assert torch.equal(ck2[:, :, :96], pk2) and torch.equal(cv2[:, :, :96], pv2)
    assert not torch.equal(ck2[:, :, :96], pk)


# ---- tiny reference expert / VLM -------------------------------------------------------------


def _tiny_expert_cfg() -> GemmaConfig:
    return GemmaConfig(width=64, depth=2, mlp_dim=128, num_heads=2, num_kv_heads=1, head_dim=32, use_adarms=True, adarms_cond_dim=64)


def _tiny_vlm_cfg() -> GemmaConfig:
    return GemmaConfig(width=64, depth=3, mlp_dim=128, num_heads=2, num_kv_heads=1, head_dim=32, use_adarms=False)


def _tiny_block_weights(cfg: GemmaConfig, g: torch.Generator) -> dict:
    w, hd, h, kv, ff = cfg.width, cfg.head_dim, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_dim
    r = lambda *s: torch.randn(*s, generator=g, dtype=F64) * 0.1  # noqa: E731
    d = {
        "self_attn.q_proj.weight": r(h * hd, w),
        "self_attn.k_proj.weight": r(kv * hd, w),
        "self_attn.v_proj.weight": r(kv * hd, w),
        "self_attn.o_proj.weight": r(w, h * hd),
        "mlp.gate_proj.weight": r(ff, w),
        "mlp.up_proj.weight": r(ff, w),
        "mlp.down_proj.weight": r(w, ff),
    }
    if cfg.use_adarms:
        d.update(
            {
                "input_layernorm.dense.weight": r(3 * w, cfg.adarms_cond_dim),
                "input_layernorm.dense.bias": r(3 * w),
                "post_attention_layernorm.dense.weight": r(3 * w, cfg.adarms_cond_dim),
                "post_attention_layernorm.dense.bias": r(3 * w),
            }
        )
    else:
        d.update({"input_layernorm.weight": r(w), "post_attention_layernorm.weight": r(w)})
    return d


def _key_mask(q_rows: int, kv_len: int, valid: int) -> torch.Tensor:
    """Additive mask that hides keys >= valid (what the SDPA kernel does with rows beyond the cache's
    logical length)."""
    m = torch.zeros(1, 1, q_rows, kv_len, dtype=F64)
    m[..., valid:] = float("-inf")
    return m


def test_64_row_suffix_matches_50_rows_on_rows_0_49():
    """Fused expert: 64-row (zero-padded) suffix, K/V rows 50..63 written but outside the cache's
    logical length (masked). Rows 0..49 of every layer output, the final adaRMS and the Euler step
    equal the 50-row legacy run (all ops are row-independent; keys beyond the logical length are
    masked exactly like the legacy 786-row buffer's padding)."""
    cfg = _tiny_expert_cfg()
    g = torch.Generator().manual_seed(2)
    blocks = [GemmaBlock(cfg, _tiny_block_weights(cfg, g), i) for i in range(cfg.depth)]
    cos, sin = precompute_freqs_cis(cfg.head_dim, 512, dtype=F64)
    P, H = 96, 50
    plan = fh.kv_cache_plan(P, H)
    prefix = [(torch.randn(1, 1, P, 32, generator=g, dtype=F64), torch.randn(1, 1, P, 32, generator=g, dtype=F64)) for _ in blocks]
    cond = torch.randn(1, cfg.adarms_cond_dim, generator=g, dtype=F64)
    x50 = torch.randn(1, H, cfg.width, generator=g, dtype=F64)
    x64 = fh.pad_rows(x50, plan["suffix_rows"])
    mask64 = _key_mask(plan["suffix_rows"], plan["padded_len"], plan["logical_len"])  # keys >= P+50 hidden

    h50, h64 = x50, x64
    for i, blk in enumerate(blocks):
        h50, _ = blk.forward(h50, cos, sin, None, None, prefix[i], False, adarms_cond=cond)
        h64, _ = blk.forward(h64, cos, sin, mask64, None, prefix[i], False, adarms_cond=cond)
        _assert_close(h64[:, :H], h50, 1e-12, f"expert layer {i} rows 0..49")

    # final adaRMS + out-proj + Euler on 64 rows, host [:50]
    dense_w = torch.randn(3 * cfg.width, cfg.adarms_cond_dim, generator=g, dtype=F64) * 0.1
    dense_b = torch.randn(3 * cfg.width, generator=g, dtype=F64) * 0.1
    n50, _ = adarms_norm(h50, dense_w, dense_b, cond)
    n64, _ = adarms_norm(h64, dense_w, dense_b, cond)
    w_out = torch.randn(cfg.width, 32, generator=g, dtype=F64) * 0.1
    b_out = torch.randn(1, 32, generator=g, dtype=F64) * 0.1
    noise = torch.randn(1, H, 32, generator=g, dtype=F64)
    dt = fh.euler_dts(10)[0]
    legacy = fh.euler_step_reference(n50, w_out, b_out, noise, dt)
    fused = fh.dit_reference(n64, w_out, dt, fh.pad_rows(noise, 64), torch.ones(1, 32, dtype=F64), bias=b_out)
    _assert_close(fh.unpad_rows(fused, H), legacy, 1e-12, "Euler on 64 rows, rows 0..49")


def test_vlm_tail_skip_leaves_the_kv_cache_unchanged():
    """sample_actions only consumes the VLM KV cache. The last layer's o_proj, both residual adds,
    its MLP and the final norm never feed a K/V -> skipping them is exact. (Sanity: perturbing the
    last layer's k_proj DOES change its cache.)"""
    cfg = _tiny_vlm_cfg()
    g = torch.Generator().manual_seed(3)
    weights = [_tiny_block_weights(cfg, g) for _ in range(cfg.depth)]
    cos, sin = precompute_freqs_cis(cfg.head_dim, 512, dtype=F64)
    x = torch.randn(1, 96, cfg.width, generator=g, dtype=F64)
    final_norm_w = torch.randn(cfg.width, generator=g, dtype=F64)

    def run(ws, final_w):
        h = x
        caches = []
        for i, w in enumerate(ws):
            h, kv = GemmaBlock(cfg, w, i).forward(h, cos, sin, None, None, None, True)
            caches.append(kv)
        return rms_norm(h, final_w), caches

    out_ref, caches_ref = run(weights, final_norm_w)
    perturbed = [dict(w) for w in weights]
    last = perturbed[-1]
    for k in ("self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "post_attention_layernorm.weight"):
        last[k] = last[k] + torch.randn_like(last[k])
    out_p, caches_p = run(perturbed, final_norm_w + 1.0)
    assert not torch.equal(out_p, out_ref)  # the (unused) VLM output did change ...
    for (k_r, v_r), (k_p, v_p) in zip(caches_ref, caches_p):
        assert torch.equal(k_r, k_p) and torch.equal(v_r, v_p)  # ... but no K/V did
    sanity = [dict(w) for w in weights]
    sanity[-1]["self_attn.k_proj.weight"] = sanity[-1]["self_attn.k_proj.weight"] + 1.0
    _, caches_s = run(sanity, final_norm_w)
    assert not torch.equal(caches_s[-1][0], caches_ref[-1][0])


def test_expert_positions_start_at_zero_for_the_padded_suffix():
    """The expert's cos/sin slice grows 50 -> 64 rows: rows 0..49 are the same positions (the reference
    uses cos[:seq_len], positions 0..seq_len-1), rows 50..63 only touch masked keys / discarded queries."""
    cos, sin = precompute_freqs_cis(256, 2048)
    assert torch.equal(cos[:64][:50], cos[:50]) and torch.equal(sin[:64][:50], sin[:50])


def test_language_embedding_scale_is_layout_independent():
    """TILE vs ROW_MAJOR is a layout; the values (bf16 gather rows x sqrt(2048)) are the same."""
    table = torch.randn(1000, 2048).to(torch.bfloat16)
    ids = torch.randint(0, 1000, (1, 224))
    emb = F.embedding(ids, table)
    scale = math.sqrt(2048)
    a = (emb.float() * scale).to(torch.bfloat16)
    b = (emb.float().reshape(-1, 2048) * scale).to(torch.bfloat16).reshape(1, 224, 2048)  # "different layout"
    assert torch.equal(a, b)


# =========================================================================================
# plain-script entry point
# =========================================================================================

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
