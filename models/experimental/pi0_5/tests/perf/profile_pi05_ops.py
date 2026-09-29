# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Op-by-op device profile of the shipped fused/traced pi0.5 path (``PI0ModelTTNN.sample_actions_fused``).

Sub-commands (one device hold each for ``run`` / ``clock``; ``summarise`` is CPU only):

``run``   -- launch UNDER the tracy op profiler (one process per shape)::

              python -m tracy -r -p -v --no-op-info-cache -o <dir> \\
                  -m models.experimental.pi0_5.tests.perf.profile_pi05_ops run --shape base --run-json <json>

            Builds the model, first call = compile + trace capture (as the server's warm-up), drains the profiler, then
            (a) one EAGER pass of the same device graph (``_fused_device_graph``) with stage signposts pushed / popped by
            wrappers around every stage function (labels every op), and (b) ``--replays`` traced replays, each
            synchronised and drained. Traced ops get the labels of the eager op at the same index (the op-code
            sequences must match 1:1; ``summarise`` refuses otherwise).  The shipped code is not modified: wrappers are
            instance attributes installed only for the eager pass.

``clock`` -- the unprofiled reference in the same process type: per-call and replay-only wall clock, and an aiclk
            snapshot (tt-smi) while replaying back to back.

``summarise`` -- ops CSV + run JSON (+ clock JSON) -> summary JSON (per-op rows, per-stage / per-layer sums, gaps,
            matmul efficiency, weight-stream bandwidth).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_WEIGHTS = os.environ.get(
    "PI05_BASE_WEIGHTS",
    "/home/deepgadget/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba",
)
TT_SMI = Path.home() / ".tenstorrent-venv" / "bin" / "tt-smi"


# ----------------------------------------------------------------------------------------------------------------
# model construction (identical to tests/perf/test_perf_pi05_fused.py and the verification scripts)
# ----------------------------------------------------------------------------------------------------------------
def open_device():
    import ttnn
    from models.experimental.pi0_5.common.fused_config import FusedConfig

    from models.experimental.pi0_5.common.device_open import device_kwargs

    fused = FusedConfig.from_env()
    dev = ttnn.open_device(**device_kwargs(fused, device_id=int(os.environ.get("PI0_DEVICE_ID", "0"))))
    dev.enable_program_cache()
    return dev, fused


def build(shape: str, device, fused):
    """-> (model, (images, tokens, noise, lang_mask), meta)."""
    import torch
    from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
    from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

    if shape == "base":
        cfg = PI0ModelConfig(action_dim=32, action_horizon=50, state_dim=32, pi05=True)
        cfg.siglip_config = SigLIPConfig(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
                                         num_attention_heads=16, image_size=224, patch_size=14)
        torch.manual_seed(42)
        model = PI0ModelTTNN(cfg, PI0WeightLoader(BASE_WEIGHTS), device, fused=fused)
        # the perf test's inputs: 2 cameras, 224 real tokens
        images = [torch.full((1, 3, 224, 224), -1.0) for _ in range(2)]
        tokens = torch.randint(1, 256000, (1, 224))
        inputs = (images, tokens, None, None)
        meta = {"weights": BASE_WEIGHTS, "token_len": 224, "n_lang": 224, "H": 50}
    elif shape == "libero":
        from models.experimental.pi0_5.tests.pcc.golden_openpi import (
            LIBERO_WEIGHTS, libero_config, load_records, record_inputs)

        torch.manual_seed(42)
        model = PI0ModelTTNN(libero_config(), PI0WeightLoader(LIBERO_WEIGHTS), device, fused=fused)
        rec = load_records()[0]
        images, tokens, mask, noise = record_inputs(rec, 32)
        inputs = (images, tokens, noise, mask)
        meta = {"weights": LIBERO_WEIGHTS, "token_len": 32, "n_lang": int(mask.sum()), "H": 10, "record": rec.get("tag")}
    else:
        raise ValueError(shape)
    return model, inputs, meta


def call(model, inputs):
    images, tokens, noise, mask = inputs
    return model.sample_actions_fused(images, tokens, noise, lang_masks=mask)


# ----------------------------------------------------------------------------------------------------------------
# stage labelling (eager pass only)
# ----------------------------------------------------------------------------------------------------------------
class Labeller:
    def __init__(self):
        self.stack: List[str] = []
        self.counts: Dict[str, int] = defaultdict(int)
        self.installed: List = []

    def _emit(self):
        import ttnn

        ttnn.tracy_message("`TT_SIGNPOST: L:" + "/".join(self.stack) + "`")

    def wrap(self, obj, attr: str, label):
        orig = getattr(obj, attr)
        lab = self

        def wrapped(*a, **k):
            name = label(a, k) if callable(label) else label
            lab.stack.append(name)
            lab._emit()
            try:
                return orig(*a, **k)
            finally:
                lab.stack.pop()
                lab._emit()

        had = attr in obj.__dict__ if hasattr(obj, "__dict__") else False
        setattr(obj, attr, wrapped)
        self.installed.append((obj, attr, orig, had))

    def uninstall(self):
        for obj, attr, orig, had in reversed(self.installed):
            if had:
                setattr(obj, attr, orig)
            else:
                delattr(obj, attr)
        self.installed = []


def install_labels(model) -> Labeller:
    lab = Labeller()
    bb = model.backbone
    vt = bb.vision_tower
    lab.wrap(model.prefix_embedding, "embed_prefix_fused", "prefix")
    lab.wrap(vt, "forward_fused", "siglip")
    lab.wrap(vt.patch_embed, "forward_fused", "patch_embed")
    for i, blk in enumerate(vt.blocks):
        lab.wrap(blk, "forward_fused", f"B{i:02d}")
        if hasattr(blk, "attention"):
            lab.wrap(blk.attention, "forward_fused", "attn")
        if hasattr(blk, "mlp"):
            lab.wrap(blk.mlp, "forward_fused", "mlp")
    lab.wrap(bb.mm_projector, "forward", "projector")
    lab.wrap(bb, "embed_language_tokens_fused", "lang_embed")
    lab.wrap(bb, "forward_vlm_fused", "vlm")
    for mk in getattr(model, "_mk", {}).values():
        lab.wrap(mk, "run", "megakernel")
    for i, blk in enumerate(bb.vlm_blocks):
        lab.wrap(blk, "forward_fused_vlm", f"L{i:02d}")
        lab.wrap(blk.attention, "forward_fused_vlm", "attn")
        lab.wrap(blk.attention, "_qkv_proj", "qkv")
        lab.wrap(blk.mlp, "forward_fused_vlm", "mlp")
    step_ctr = {"n": 0}

    def step_label(a, k):
        return f"s{k.get('step', step_ctr['n']):02d}"

    lab.wrap(model.suffix_embedding, "embed_actions_fused", "action_in")
    lab.wrap(bb, "forward_expert_fused", lambda a, k: "expert_" + step_label(a, k))
    lab.wrap(model.suffix_embedding, "euler_step_fused", "euler")
    for i, blk in enumerate(bb.expert_blocks):
        lab.wrap(blk, "forward_fused_expert", f"L{i:02d}")
        at = blk.attention
        lab.wrap(at, "_qkv_proj", "qkv")
        if at._fused_attn is not None:
            lab.wrap(at, "_fused_attn", "fused_attn")
        lab.wrap(at, "forward_fused_expert", "attn")
        lab.wrap(blk, "_dit_residual", "dit")
        if blk._row_rsqrt is not None:
            lab.wrap(blk, "_row_rsqrt", "row_rsqrt")
        if blk._geglu_rc is not None:
            lab.wrap(blk, "_geglu_rc", "geglu_rc")
        lab.wrap(blk.mlp, "up_gate_linear", "up_gate")
    return lab


# ----------------------------------------------------------------------------------------------------------------
def signpost(msg: str):
    import ttnn

    ttnn.tracy_message(f"`TT_SIGNPOST: {msg}`")


def read_profiler(device):
    import ttnn

    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        ttnn.ReadDeviceProfiler(device)


def _stats(xs):
    xs = sorted(xs)
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "median": statistics.median(xs), "min": xs[0], "max": xs[-1], "all": [round(x, 4) for x in xs]}


def device_facts(device) -> Dict[str, Any]:
    g = device.compute_with_storage_grid_size()
    return {"compute_grid": [int(g.x), int(g.y)], "num_cores": int(g.x) * int(g.y)}


def smi_snapshot(label: str) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"label": label, "time": datetime.now().isoformat(timespec="seconds")}
    if not TT_SMI.exists():
        rec["error"] = "no tt-smi"
        return rec
    try:
        res = subprocess.run([str(TT_SMI), "-s", "--snapshot_no_tty"], capture_output=True, text=True, timeout=30)
        txt = res.stdout
        d = json.loads(txt[txt.find("{"):])
        tel = d["device_info"][0].get("telemetry", {})
        rec.update({"aiclk_mhz": tel.get("aiclk"), "power_w": tel.get("power"), "asic_temp_c": tel.get("asic_temperature")})
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def cmd_run(args) -> int:
    import torch
    import ttnn

    torch.set_grad_enabled(False)
    out: Dict[str, Any] = {"schema": "pi05-op-profile-run/1", "created": datetime.now().astimezone().isoformat(timespec="seconds"),
                           "shape": args.shape, "argv": sys.argv[1:],
                           "env": {k: v for k, v in os.environ.items() if k.startswith(("TT_METAL", "TTNN", "PI05_"))}}
    device, fused = open_device()
    try:
        out["device"] = device_facts(device)
        out["fused_cfg"] = {k: str(v) for k, v in vars(fused).items()}
        model, inputs, meta = build(args.shape, device, fused)
        out["meta"] = meta
        signpost("phase:warm_capture")
        t0 = time.perf_counter()
        ref = call(model, inputs).clone()
        out["first_call_s"] = time.perf_counter() - t0
        ttnn.synchronize_device(device)
        read_profiler(device)
        assert model._fused_trace_id is not None, "trace not captured (PI05_TRACE=0?)"

        # (a) eager labelled pass of the same device graph
        lab = install_labels(model)
        signpost("phase:eager")
        t0 = time.perf_counter()
        o = model._fused_device_graph()
        ttnn.synchronize_device(device)
        out["eager_s"] = time.perf_counter() - t0
        signpost("phase:eager_end")
        eager_out = ttnn.to_torch(o).float()
        ttnn.deallocate(o)
        lab.uninstall()
        read_profiler(device)

        # (b) traced replays (inputs unchanged since the first call)
        walls = []
        outs_equal = []
        for r in range(args.replays):
            signpost(f"phase:replay:{r}")
            t0 = time.perf_counter()
            ttnn.execute_trace(device, model._fused_trace_id, cq_id=0, blocking=True)
            walls.append((time.perf_counter() - t0) * 1e3)
            y = ttnn.to_torch(model._fused_out).float()
            outs_equal.append(bool(torch.equal(y[:, : ref.shape[1]], ref)))
            read_profiler(device)
        signpost("phase:done")
        out["replay_wall_ms_profiled"] = _stats(walls)
        out["replay_bit_identical_to_first_call"] = outs_equal
        out["eager_equal_to_first_call"] = bool(torch.equal(eager_out[:, : ref.shape[1]], ref))
        model.release_trace()
    finally:
        ttnn.close_device(device)
    Path(args.run_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.run_json, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", args.run_json)
    return 0


def cmd_clock(args) -> int:
    import torch
    import ttnn

    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        raise RuntimeError("clock is the unprofiled reference")
    torch.set_grad_enabled(False)
    out: Dict[str, Any] = {"schema": "pi05-op-profile-clock/1", "created": datetime.now().astimezone().isoformat(timespec="seconds"),
                           "shape": args.shape, "argv": sys.argv[1:]}
    out["smi_idle"] = smi_snapshot("idle_before_open")
    device, fused = open_device()
    try:
        out["device"] = device_facts(device)
        model, inputs, meta = build(args.shape, device, fused)
        out["meta"] = meta
        ref = call(model, inputs).clone()
        for _ in range(3):
            call(model, inputs)
        per_call, replay = [], []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            y = call(model, inputs)
            per_call.append((time.perf_counter() - t0) * 1e3)
        for _ in range(args.reps):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, model._fused_trace_id, cq_id=0, blocking=True)
            replay.append((time.perf_counter() - t0) * 1e3)
        out["per_call_ms"] = _stats(per_call)
        out["replay_ms"] = _stats(replay)
        out["last_bit_identical"] = bool(torch.equal(y, ref))
        # sustained back-to-back replays with tt-smi snapshots from a helper thread
        snaps: List[Dict[str, Any]] = []
        stop = threading.Event()

        def loop():
            while not stop.is_set():
                snaps.append(smi_snapshot("sustain"))
                stop.wait(0.5)

        th = threading.Thread(target=loop, daemon=True)
        th.start()
        t_end = time.perf_counter() + args.seconds
        sus = []
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            ttnn.execute_trace(device, model._fused_trace_id, cq_id=0, blocking=True)
            sus.append((time.perf_counter() - t0) * 1e3)
        stop.set()
        th.join(timeout=40)
        out["sustain_replay_ms"] = {k: v for k, v in _stats(sus).items() if k != "all"}
        out["sustain_smi"] = snaps
        model.release_trace()
    finally:
        ttnn.close_device(device)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", args.out)
    return 0


# ----------------------------------------------------------------------------------------------------------------
# summarise (CPU)
# ----------------------------------------------------------------------------------------------------------------
TILE_BYTES = {"BFLOAT16": 2048, "BFLOAT8_B": 1088, "BFLOAT4_B": 576, "FLOAT32": 4096, "UINT32": 4096}
MATMUL_CODES = ("Matmul", "MinimalMatmul", "DitMinimal")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _pad(v) -> Optional[int]:
    m = re.match(r"\s*(\d+)", str(v or ""))
    return int(m.group(1)) if m else None


def tensors(row, io):
    out = []
    for i in range(8):
        k = f"{io}_{i}_X_PAD[LOGICAL]"
        if not row.get(k):
            break
        dims = [_pad(row.get(f"{io}_{i}_{d}_PAD[LOGICAL]")) or 1 for d in "WZYX"]
        out.append({"shape": dims, "dtype": row.get(f"{io}_{i}_DATATYPE"), "mem": row.get(f"{io}_{i}_MEMORY")})
    return out


def tbytes(t) -> float:
    w, z, y, x = t["shape"]
    tiles = w * z * math.ceil(y / 32) * math.ceil(x / 32)
    return tiles * TILE_BYTES.get(t["dtype"], 2048)


def fidelity(row) -> Optional[str]:
    mf = (row.get("MATH FIDELITY") or "").strip()
    if mf:
        return mf
    m = re.search(r"math_fidelity['\"]?\s*[:=]\s*['\"]?(?:MathFidelity::)?(\w+)", row.get("ATTRIBUTES") or "")
    return m.group(1) if m else None


def open_csv(p: Path):
    return gzip.open(p, "rt") if str(p).endswith(".gz") else open(p, newline="")


def classify(label: str, code: str) -> Dict[str, Any]:
    parts = label.split("/") if label else []
    d: Dict[str, Any] = {"stage": "other", "layer": None, "step": None, "sub": ""}
    if not parts:
        return d
    if parts[0] == "prefix":
        if len(parts) > 1 and parts[1] == "siglip":
            d["stage"] = "siglip"
            if len(parts) > 2 and parts[2].startswith("B"):
                d["layer"] = int(parts[2][1:])
                d["sub"] = "/".join(parts[3:])
            else:
                d["sub"] = "/".join(parts[2:]) or "tower"
        elif len(parts) > 1 and parts[1] == "projector":
            d["stage"] = "projector"
        elif len(parts) > 1 and parts[1] == "lang_embed":
            d["stage"] = "prefix_assembly"
            d["sub"] = "lang_embed"
        else:
            d["stage"] = "prefix_assembly"
    elif parts[0] == "vlm":
        d["stage"] = "vlm"
        if len(parts) > 1 and parts[1].startswith("L"):
            d["layer"] = int(parts[1][1:])
            d["sub"] = "/".join(parts[2:])
    elif parts[0].startswith("expert_s"):
        d["stage"] = "expert"
        d["step"] = int(parts[0][len("expert_s"):])
        if len(parts) > 1 and parts[1].startswith("L"):
            d["layer"] = int(parts[1][1:])
            d["sub"] = "/".join(parts[2:])
        else:
            d["sub"] = "final_norm"
    elif parts[0] in ("action_in", "euler"):
        d["stage"] = "action_io"
        d["sub"] = parts[0]
    return d


def role_names(ops: List[Dict[str, Any]]) -> None:
    """role = sub-label + op short name, numbered within (layer instance, sub-label); dits -> o_proj / down."""
    ctr: Dict[Any, int] = defaultdict(int)
    for op in ops:
        short = op["code"].replace("DeviceOperation", "").replace("Operation", "")
        key = (op["stage"], op["step"], op["layer"], op["sub"], short)
        n = ctr[key]
        ctr[key] += 1
        sub = op["sub"]
        if op["stage"] == "expert" and sub == "dit":
            sub = ("o_proj" if n == 0 else "down") + "_dit"
            n = 0
        role = (sub + ":" if sub else "") + short + (f"#{n}" if n else "")
        op["role"] = role


def matmul_dims(op) -> Optional[Dict[str, int]]:
    ins = op["inputs"]
    if len(ins) < 2 or not any(c in op["code"] for c in MATMUL_CODES):
        return None
    a, b = ins[0]["shape"], ins[1]["shape"]
    m = a[0] * a[1] * a[2]
    k = a[3]
    n = b[3]
    if b[2] != k:
        return None
    return {"M": m, "K": k, "N": n}


def summarise(args) -> int:
    run = json.load(open(args.run_json))
    rows = list(csv.DictReader(open_csv(Path(args.csv))))
    freq_mhz = args.freq_mhz
    ncores = run["device"]["num_cores"]
    # ---------- eager rows, labelled by the latest L: signpost (CSV order == host order for eager rows)
    in_eager = False
    cur = ""
    eager: List[Dict[str, Any]] = []
    traced: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        code = row.get("OP CODE", "")
        if row.get("OP TYPE") == "signpost":
            if code == "phase:eager":
                in_eager = True
            elif code == "phase:eager_end":
                in_eager = False
            elif code.startswith("L:"):
                cur = code[2:]
            continue
        tid = row.get("METAL TRACE ID")
        rec = {"code": code, "fw_ns": _f(row.get("DEVICE FW DURATION [ns]")), "kernel_ns": _f(row.get("DEVICE KERNEL DURATION [ns]")),
               "start": int(row["DEVICE FW START CYCLE"]) if row.get("DEVICE FW START CYCLE") else None,
               "end": int(row["DEVICE FW END CYCLE"]) if row.get("DEVICE FW END CYCLE") else None,
               "cores": int(float(row["CORE COUNT"])) if row.get("CORE COUNT") else None,
               "inputs": tensors(row, "INPUT"), "outputs": tensors(row, "OUTPUT"), "fidelity": fidelity(row),
               "attr": (row.get("ATTRIBUTES") or "")[:600]}
        if tid not in (None, ""):
            traced[int(row["METAL TRACE REPLAY SESSION ID"] or -1)].append(rec)
        elif in_eager:
            rec["label"] = cur
            eager.append(rec)
    if not eager:
        raise SystemExit("no eager rows (signposts missing?)")
    for op in eager:
        op.update(classify(op["label"], op["code"]))
    role_names(eager)
    sessions = sorted(traced)
    report: Dict[str, Any] = {"shape": run["shape"], "run_json": args.run_json, "csv": args.csv, "freq_mhz": freq_mhz,
                              "num_cores": ncores, "eager_op_count": len(eager), "sessions": {}}
    good_sessions = []
    for s in sessions:
        ops = sorted(traced[s], key=lambda r: r["start"] or 0)
        codes_ok = len(ops) == len(eager) and all(a["code"] == b["code"] for a, b in zip(ops, eager))
        report["sessions"][s] = {"n_ops": len(ops), "matches_eager": codes_ok}
        if codes_ok:
            good_sessions.append(ops)
    if not good_sessions:
        raise SystemExit(f"PROFILER TRUNCATED / MISMATCH: sessions {report['sessions']} vs eager {len(eager)} ops")
    # ---------- per-op medians across good replays
    n = len(eager)
    per_op = []
    for i in range(n):
        # FW START is taken early on cores the previous op does not use, so FW durations overlap; charge every op its
        # end-to-end increment (sums exactly to the replay span) = kernel duration + launch gap.
        fw = [(ss[i]["end"] - ss[i - 1]["end"]) * 1e3 / freq_mhz if i else ss[i]["fw_ns"] for ss in good_sessions]
        kn = [ss[i]["kernel_ns"] for ss in good_sessions]
        gaps = [f - k for f, k in zip(fw, kn)]
        e = eager[i]
        r = {k: e[k] for k in ("code", "label", "stage", "layer", "step", "sub", "role", "cores", "fidelity")}
        r["incr_us"] = statistics.median(fw) / 1e3
        r["fw_us"] = statistics.median(kn) / 1e3  # kernel time; totals add gap_after_us (= launch gap before it)
        r["kernel_us"] = statistics.median(kn) / 1e3
        r["gap_after_us"] = statistics.median(gaps) / 1e3
        r["raw_fw_us"] = statistics.median(ss[i]["fw_ns"] for ss in good_sessions) / 1e3
        r["eager_fw_us"] = e["fw_ns"] / 1e3
        r["in"] = [f"{'x'.join(str(d) for d in t['shape'])}:{t['dtype']}:{(t['mem'] or '')[:12]}" for t in e["inputs"]]
        r["out"] = [f"{'x'.join(str(d) for d in t['shape'])}:{t['dtype']}" for t in e["outputs"]]
        mm = matmul_dims(e)
        if mm:
            r.update(mm)
            flops = 2.0 * mm["M"] * mm["K"] * mm["N"]
            wbytes = tbytes(e["inputs"][1])
            r["flops"] = flops
            r["w_bytes"] = wbytes
            r["w_gbps"] = wbytes / (r["kernel_us"] * 1e3) if r["kernel_us"] > 0 else None
            fid = (r["fidelity"] or "HiFi2")
            div = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}.get(fid, 2)
            peak_full = ncores * 4096.0 / div * freq_mhz * 1e6
            r["tflops"] = flops / (r["kernel_us"] * 1e-6) / 1e12 if r["kernel_us"] > 0 else None
            r["pct_peak_full_grid"] = 100.0 * flops / (r["kernel_us"] * 1e-6) / peak_full if r["kernel_us"] > 0 else None
            r["pct_peak_op_cores"] = (100.0 * flops / (r["kernel_us"] * 1e-6) / (r["cores"] * 4096.0 / div * freq_mhz * 1e6)
                                      if r["kernel_us"] > 0 and r["cores"] else None)
        per_op.append(r)
    spans = [(ss[-1]["end"] - ss[0]["start"]) / freq_mhz / 1e3 for ss in good_sessions]  # ms
    report["replay_device_span_ms"] = _stats(spans)
    report["replay_sum_fw_ms"] = sum(r["fw_us"] for r in per_op) / 1e3
    report["replay_sum_gap_ms"] = sum(r["gap_after_us"] for r in per_op) / 1e3
    report["n_good_sessions"] = len(good_sessions)
    # ---------- per stage
    def agg(sel):
        xs = [r for r in per_op if sel(r)]
        return {"n_ops": len(xs), "fw_ms": sum(r["fw_us"] for r in xs) / 1e3, "gap_ms": sum(r["gap_after_us"] for r in xs) / 1e3}

    stages = OrderedDict()
    for st in ("siglip", "projector", "prefix_assembly", "vlm", "action_io", "expert", "other"):
        stages[st] = agg(lambda r, st=st: r["stage"] == st)
    report["stages"] = stages
    # ---------- per-layer role tables (median over layers / steps)
    def role_table(stage):
        by = defaultdict(list)
        order = []
        for r in per_op:
            if r["stage"] != stage or r["layer"] is None:
                continue
            if r["role"] not in by:
                order.append(r["role"])
            by[r["role"]].append(r)
        tab = []
        for role in order:
            xs = by[role]
            ex = xs[0]
            t = {"role": role, "code": ex["code"], "count": len(xs), "median_fw_us": statistics.median(x["fw_us"] for x in xs),
                 "sum_fw_ms": sum(x["fw_us"] for x in xs) / 1e3, "median_gap_us": statistics.median(x["gap_after_us"] for x in xs),
                 "cores": ex["cores"], "in": ex["in"], "fidelity": ex["fidelity"]}
            for k in ("M", "K", "N", "w_bytes"):
                if k in ex:
                    t[k] = ex[k]
            if "flops" in ex:
                t["median_tflops"] = statistics.median(x["tflops"] for x in xs)
                t["median_pct_peak_full_grid"] = statistics.median(x["pct_peak_full_grid"] for x in xs)
                t["median_w_gbps"] = statistics.median(x["w_gbps"] for x in xs)
            tab.append(t)
        return tab

    report["siglip_layer_roles"] = role_table("siglip")
    report["vlm_layer_roles"] = role_table("vlm")
    report["expert_layer_roles"] = role_table("expert")
    # per-layer totals
    for stage in ("siglip", "vlm", "expert"):
        tot = defaultdict(float)
        for r in per_op:
            if r["stage"] == stage and r["layer"] is not None:
                tot[(r["step"], r["layer"])] += r["fw_us"] + r["gap_after_us"]
        vals = list(tot.values())
        report[f"{stage}_per_layer_us"] = _stats(vals) if vals else None
    # expert per step
    # one denoise step = action_in + 18 expert layers + final norm + euler (action_in opens a step)
    st_tot = defaultdict(float)
    k_step = -1
    prev_sub = None
    for r in per_op:
        is_in = r["stage"] == "action_io" and r["sub"] == "action_in"
        if is_in and prev_sub != "action_in":
            k_step += 1
        if r["stage"] in ("expert", "action_io"):
            st_tot[k_step] += r["fw_us"] + r["gap_after_us"]
        prev_sub = "action_in" if is_in else r["stage"]
    report["step_ms"] = {str(k): v / 1e3 for k, v in sorted(st_tot.items())}
    # ---------- expert weight stream
    exp_mm = [r for r in per_op if r["stage"] == "expert" and "w_bytes" in r]
    wb = sum(r["w_bytes"] for r in exp_mm)
    kt = sum(r["kernel_us"] for r in exp_mm)
    exp_all = [r for r in per_op if r["stage"] == "expert"]
    report["expert_weight_stream"] = {
        "matmul_ops": len(exp_mm), "weight_bytes_total": wb, "weight_bytes_per_step": wb / 10,
        "matmul_kernel_ms": kt / 1e3, "gbps_over_matmul_time": wb / (kt * 1e3) if kt else None,
        "expert_stage_ms_incl_gaps": sum(r["fw_us"] + r["gap_after_us"] for r in exp_all) / 1e3,
        "gbps_over_expert_stage_time": wb / (sum(r["fw_us"] + r["gap_after_us"] for r in exp_all) * 1e3),
    }
    # ---------- matmul class table (by stage + role)
    cls = defaultdict(list)
    for r in per_op:
        if "flops" in r:
            cls[(r["stage"], r["role"])].append(r)
    report["matmul_classes"] = [
        {"stage": st, "role": ro, "count": len(xs), "M": xs[0]["M"], "K": xs[0]["K"], "N": xs[0]["N"], "fidelity": xs[0]["fidelity"],
         "cores": xs[0]["cores"], "w_dtype": xs[0]["in"][1].split(":")[1] if len(xs[0]["in"]) > 1 else None,
         "median_kernel_us": statistics.median(x["kernel_us"] for x in xs), "sum_ms": sum(x["fw_us"] for x in xs) / 1e3,
         "median_tflops": statistics.median(x["tflops"] for x in xs),
         "median_pct_peak_full_grid": statistics.median(x["pct_peak_full_grid"] for x in xs),
         "median_pct_peak_op_cores": statistics.median(x["pct_peak_op_cores"] for x in xs if x["pct_peak_op_cores"] is not None),
         "median_w_gbps": statistics.median(x["w_gbps"] for x in xs)}
        for (st, ro), xs in cls.items()]
    # ---------- op-family totals (whole replay)
    fam = defaultdict(lambda: [0, 0.0])
    for r in per_op:
        fam[r["code"]][0] += 1
        fam[r["code"]][1] += r["fw_us"]
    report["op_code_totals"] = sorted(([k, v[0], v[1] / 1e3] for k, v in fam.items()), key=lambda x: -x[2])
    if args.clock_json and Path(args.clock_json).exists():
        report["clock"] = {k: v for k, v in json.load(open(args.clock_json)).items() if k not in ("sustain_smi",)}
    report["per_op"] = per_op
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print("wrote", args.out)
    print(json.dumps({k: report[k] for k in ("eager_op_count", "sessions", "replay_device_span_ms", "replay_sum_fw_ms",
                                              "replay_sum_gap_ms", "stages", "expert_weight_stream")}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------------------------------------------
# report (CPU): summary JSONs -> markdown tables (PROFILE.md body); every number printed comes from the summaries
# ----------------------------------------------------------------------------------------------------------------
PEAK_TF = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}


def _md_stage_table(d) -> List[str]:
    span = d["replay_device_span_ms"]["median"]
    out = ["| stage | ops | kernel ms | launch gaps ms | total ms | % of span |", "|---|---:|---:|---:|---:|---:|"]
    names = {"siglip": "SigLIP x2 cameras (27 layers, batched)", "projector": "multimodal projector",
             "prefix_assembly": "prefix assembly (lang embed, scale, concat)", "vlm": "VLM prefill (18 layers)",
             "action_io": "action in-proj + Euler (10 steps)", "expert": "expert (10 steps x 18 layers + final norm)"}
    for k, v in d["stages"].items():
        if v["n_ops"] == 0:
            continue
        tot = v["fw_ms"] + v["gap_ms"]
        out.append(f"| {names.get(k, k)} | {v['n_ops']} | {v['fw_ms']:.3f} | {v['gap_ms']:.3f} | {tot:.3f} | {100 * tot / span:.1f} |")
    out.append(f"| **replay (device span)** | {d['eager_op_count']} | {d['replay_sum_fw_ms']:.3f} | {d['replay_sum_gap_ms']:.3f} | "
               f"**{span:.3f}** | 100 |")
    return out


def _md_role_table(d, key, per) -> List[str]:
    out = [f"| op ({per}) | code | cores | M x K x N | fidelity | kernel us (median) | gap us | TFLOP/s | % peak (110 cores) | weight GB/s |",
           "|---|---|---:|---|---|---:|---:|---:|---:|---:|"]
    tot_k = tot_g = 0.0
    for t in d[key]:
        mkn = f"{t['M']}x{t['K']}x{t['N']}" if "M" in t else ""
        tf = f"{t['median_tflops']:.1f}" if "median_tflops" in t else ""
        pk = f"{t['median_pct_peak_full_grid']:.1f}" if "median_pct_peak_full_grid" in t else ""
        bw = f"{t['median_w_gbps']:.0f}" if "median_w_gbps" in t else ""
        code = t["code"].replace("DeviceOperation", "").replace("Operation", "")
        out.append(f"| {t['role']} | {code} | {t['cores']} | {mkn} | {t['fidelity'] or ''} | {t['median_fw_us']:.2f} | "
                   f"{t['median_gap_us']:.2f} | {tf} | {pk} | {bw} |")
        tot_k += t["median_fw_us"] * t["count"]
        tot_g += t["median_gap_us"] * t["count"]
    return out


def cmd_report(args) -> int:
    lines: List[str] = []
    for path in args.summary:
        d = json.load(open(path))
        lines += [f"### shape `{d['shape']}` -- `{path}`", ""]
        lines += [f"Traced replay sessions matching the eager op sequence 1:1: {d['n_good_sessions']} "
                  f"({d['eager_op_count']} ops each). Device span median {d['replay_device_span_ms']['median']:.3f} ms "
                  f"(min {d['replay_device_span_ms']['min']:.3f}, max {d['replay_device_span_ms']['max']:.3f}).", ""]
        if "clock" in d:
            c = d["clock"]
            lines += [f"Unprofiled (clock JSON): per call {c['per_call_ms']['median']:.2f} ms median, replay only "
                      f"{c['replay_ms']['median']:.2f} ms median, sustained replay {c['sustain_replay_ms']['median']:.2f} ms.", ""]
        lines += _md_stage_table(d) + [""]
        for k in ("siglip", "vlm", "expert"):
            pl = d.get(f"{k}_per_layer_us")
            if pl:
                lines.append(f"- {k} per layer (kernel + gaps): median {pl['median']:.1f} us, min {pl['min']:.1f}, max {pl['max']:.1f} (n = {pl['n']})")
        sm = d.get("step_ms", {})
        if sm:
            v = list(sm.values())
            lines.append(f"- denoise step (action in + 18 layers + final norm + Euler): median {statistics.median(v):.3f} ms, "
                         f"min {min(v):.3f}, max {max(v):.3f}")
        w = d["expert_weight_stream"]
        lines += ["", f"Expert weight stream: {w['weight_bytes_total'] / 1e9:.3f} GB per request "
                  f"({w['weight_bytes_per_step'] / 1e6:.1f} MB per step) over {w['matmul_ops']} matmuls; "
                  f"{w['gbps_over_matmul_time']:.0f} GB/s over the matmuls' kernel time ({w['matmul_kernel_ms']:.2f} ms), "
                  f"{w['gbps_over_expert_stage_time']:.0f} GB/s over the whole expert stage ({w['expert_stage_ms_incl_gaps']:.2f} ms).", ""]
        for key, per in (("siglip_layer_roles", "per SigLIP layer"), ("vlm_layer_roles", "per VLM layer"),
                         ("expert_layer_roles", "per expert layer")):
            lines += [f"#### {per}", ""] + _md_role_table(d, key, per) + [""]
        lines += ["#### ops outside the layers (one instance each unless stated)", "",
                  "| stage | op | cores | kernel us | gap us | inputs |", "|---|---|---:|---:|---:|---|"]
        seen = defaultdict(list)
        for r in d["per_op"]:
            if r["layer"] is None:
                base = re.sub(r"#\d+$", "", r["role"])
                seen[(r["stage"], base)].append(r)
        for (st, role), xs in seen.items():
            k = statistics.median(x["fw_us"] for x in xs)
            g = statistics.median(x["gap_after_us"] for x in xs)
            lines.append(f"| {st} | {role}{' x' + str(len(xs)) if len(xs) > 1 else ''} | {xs[0]['cores']} | {k:.2f} | {g:.2f} | "
                         f"{', '.join(xs[0]['in'][:2])} |")
        lines += ["", "#### matmul classes (whole replay)", "",
                  "| stage | op | count | M x K x N | w dtype | fidelity | cores | kernel us | sum ms | TFLOP/s | % peak 110 cores | % peak op cores | weight GB/s |",
                  "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m in sorted(d["matmul_classes"], key=lambda m: -m["sum_ms"]):
            lines.append(f"| {m['stage']} | {m['role']} | {m['count']} | {m['M']}x{m['K']}x{m['N']} | {m['w_dtype']} | {m['fidelity']} | "
                         f"{m['cores']} | {m['median_kernel_us']:.2f} | {m['sum_ms']:.3f} | {m['median_tflops']:.1f} | "
                         f"{m['median_pct_peak_full_grid']:.1f} | {m['median_pct_peak_op_cores']:.1f} | {m['median_w_gbps']:.0f} |")
        lines += ["", "#### op-code totals (kernel ms, whole replay)", "", "| op code | count | kernel ms |", "|---|---:|---:|"]
        for code, n, ms in d["op_code_totals"]:
            lines.append(f"| {code} | {n} | {ms:.3f} |")
        lines.append("")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("wrote", args.out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--shape", choices=("base", "libero"), required=True)
    a.add_argument("--replays", type=int, default=3)
    a.add_argument("--run-json", required=True)
    c = sub.add_parser("clock")
    c.add_argument("--shape", choices=("base", "libero"), required=True)
    c.add_argument("--reps", type=int, default=20)
    c.add_argument("--seconds", type=float, default=10.0)
    c.add_argument("--out", required=True)
    s = sub.add_parser("summarise")
    s.add_argument("--csv", required=True)
    s.add_argument("--run-json", required=True)
    s.add_argument("--clock-json")
    s.add_argument("--freq-mhz", type=float, default=1350.0)
    s.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--summary", nargs="+", required=True)
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    return {"run": cmd_run, "clock": cmd_clock, "summarise": summarise, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
