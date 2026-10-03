"""pi0.5 LIBERO policy server on one Tenstorrent Blackhole p150a (openpi websocket protocol).

Serves `lerobot/pi05_libero` through the pi0.5 megakernel (`PI05MegakernelTTNN`: SigLIP | VLM prefill | action expert
as three ttnn.generic_op programs, replayed from one Metal trace per prompt bucket) with openpi's `pi05_libero` host
conventions, so openpi's own LIBERO client (`examples/libero/main.py`) talks to it unchanged:

  request  msgpack_numpy({"observation/image": uint8 (224,224,3), "observation/wrist_image": uint8 (224,224,3),
                          "observation/state": (8,), "prompt": str [, "__noise_seed__": int]})
  reply    msgpack_numpy({"actions": float32 (10, 7), "actions_norm": (10, 32), "policy_timing": {"infer_ms": ...}})

Host conventions (bit-exact to openpi's PyTorch policy inputs, checked against openpi GPU goldens):
  * images: x.float32 * float32(1/255) * 2 - 1, NCHW. That is what openpi's `x / 255.0 * 2.0 - 1.0` computes on CUDA
    (a scalar division runs as a multiply by the float32 reciprocal); a true division differs by 1 ulp.
    --cameras 2: [base (agentview), wrist]; --cameras 1: [base] only, on a 1-camera model (the wrist image is ignored).
    openpi's third, masked camera slot is left out: the megakernel takes only real cameras.
  * prompt: BOS + sentencepiece(task.strip() with "_" and "\n" -> " ") + sentencepiece("\n"), right-padded with 0
    (openpi pi05_libero, discrete_state_input=False: the state is not used); the smallest prompt bucket that holds it.
  * actions: the first 7 of 32 dims, unnormalised with openpi pi05_libero quantiles: (y+1)/2*(q99-q01+1e-6)+q01.
  * noise: `__noise_seed__` -> np.random.default_rng(seed).standard_normal((1, 10, 32)) (float32); otherwise
    np.random.standard_normal. Action horizon 10, --num-steps flow-matching steps (1..10, default 10).

Requirements: tt-metal with ttnn (Blackhole), this repo's `models/` on PYTHONPATH, torch, numpy, sentencepiece,
websockets >= 13, openpi-client (msgpack_numpy), the checkpoint (lerobot/pi05_libero), openpi's pi05_libero
norm_stats.json and the PaliGemma tokenizer (gs://big_vision/paligemma_tokenizer.model).

Example:
  python serve_pi05_libero.py --ckpt lerobot/pi05_libero --norm-stats norm_stats.json \
      --tokenizer paligemma_tokenizer.model --cameras 2 --num-steps 10 --port 8000
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
import traceback

import numpy as np
import torch

H = 10  # openpi pi05_libero action horizon
TOKEN_LEN = 200  # openpi pi05_libero max_token_len (the model trims to its prompt bucket)


def tokenize_openpi(sp, prompt, lang_len=TOKEN_LEN):
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    ids = sp.encode(cleaned, add_bos=True) + sp.encode("\n")
    if len(ids) > lang_len:
        raise ValueError(f"prompt has {len(ids)} tokens > {lang_len}: {prompt!r}")
    toks = np.zeros(lang_len, np.int64)
    toks[: len(ids)] = ids
    mask = np.zeros(lang_len, bool)
    mask[: len(ids)] = True
    return toks, mask, len(ids)


def image_to_model(img_u8):
    a = np.asarray(img_u8)
    if a.dtype != np.uint8 or a.shape != (224, 224, 3):
        raise ValueError(f"expected a uint8 (224, 224, 3) image, got {a.dtype} {a.shape}")
    x = torch.from_numpy(a).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    return (x * torch.tensor(1.0 / 255.0, dtype=torch.float32) * 2.0 - 1.0).contiguous()


class Pi05LiberoPolicy:
    """openpi pi05_libero pre/post-processing around PI05MegakernelTTNN; owns the device. close() releases both."""

    def __init__(self, ckpt, norm_stats, tokenizer, cameras=2, num_steps=10, device_id=0):
        import sentencepiece
        import ttnn

        from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
        from models.experimental.pi0.common.weight_loader import PI0WeightLoader
        from models.experimental.pi0.tt.ttnn_pi05_model import PI05_DEVICE_PARAMS, PI05MegakernelTTNN

        self.ttnn = ttnn
        self.H, self.cameras, self.num_steps = H, int(cameras), int(num_steps)
        ns = json.load(open(norm_stats))["norm_stats"]
        self.aq01 = np.asarray(ns["actions"]["q01"], np.float32)[:7]
        self.aq99 = np.asarray(ns["actions"]["q99"], np.float32)[:7]
        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer))
        cfg = PI0ModelConfig(action_dim=32, action_horizon=H, state_dim=32, pi05=True)
        cfg.num_denoising_steps = self.num_steps
        cfg.num_cameras = self.cameras
        cfg.siglip_config = SigLIPConfig(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
                                         num_attention_heads=16, image_size=224, patch_size=14)
        self.device = ttnn.open_device(device_id=device_id, **PI05_DEVICE_PARAMS)
        self.model = None
        try:
            t0 = time.time()
            torch.manual_seed(42)
            self.model = PI05MegakernelTTNN(cfg, PI0WeightLoader(ckpt), self.device)
            self.load_s = time.time() - t0
            t0 = time.time()
            self.model.warmup()  # compile + capture every prompt bucket before serving
            self.warmup_s = time.time() - t0
        except BaseException:
            self.close()
            raise
        self.n_calls = 0
        self.last_inputs = None

    @property
    def backend_stamp(self):
        m = self.model
        return (f"tt-p150a:{type(m).__module__}.{type(m).__name__}[cameras={m.cameras},H={m.horizon},"
                f"num_steps={m.config.num_denoising_steps},kv_dram={m.kv_dram},digest={str(m.kernel_digest)[:16]}]")

    def infer(self, obs):
        seed = obs.get("__noise_seed__")
        if seed is not None:
            noise = np.random.default_rng(int(seed)).standard_normal((1, self.H, 32)).astype(np.float32)
        else:
            noise = np.random.standard_normal((1, self.H, 32)).astype(np.float32)
        t0 = time.monotonic()
        toks, lmask, n_lang = tokenize_openpi(self.sp, obs["prompt"])
        images = [image_to_model(obs["observation/image"])]
        if self.cameras == 2:
            images.append(image_to_model(obs["observation/wrist_image"]))
        tokens, lang_mask = torch.from_numpy(toks)[None], torch.from_numpy(lmask)[None]
        noise_t = torch.from_numpy(noise)
        with torch.no_grad():
            out = self.model.sample_actions(images, [torch.ones(1, dtype=torch.bool)] * self.cameras, tokens,
                                            lang_masks=lang_mask, noise=noise_t)
        a_norm = out.float().numpy().reshape(self.H, -1)
        actions = (a_norm[:, :7] + 1.0) / 2.0 * (self.aq99 - self.aq01 + 1e-6) + self.aq01
        ms = (time.monotonic() - t0) * 1000.0
        self.n_calls += 1
        self.last_inputs = dict(images=images, tokens=tokens, lang_mask=lang_mask, noise=noise_t, n_lang=n_lang,
                                actions_model_norm=torch.from_numpy(a_norm.copy()).reshape(1, self.H, -1),
                                preset=self.model.preset_for(n_lang).key)
        return {"actions": actions.astype(np.float32), "actions_norm": a_norm.astype(np.float32),
                "policy_timing": {"infer_ms": ms}}

    def close(self):
        try:
            if self.model is not None:
                self.model.close()
        finally:
            self.ttnn.close_device(self.device)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default="lerobot/pi05_libero", help="checkpoint directory or HuggingFace id")
    ap.add_argument("--norm-stats", required=True, help="openpi pi05_libero norm_stats.json")
    ap.add_argument("--tokenizer", required=True, help="PaliGemma sentencepiece model (paligemma_tokenizer.model)")
    ap.add_argument("--cameras", type=int, choices=(1, 2), default=2)
    ap.add_argument("--num-steps", type=int, default=10, help="flow-matching steps, 1..10")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--warmup", type=int, default=2, help="warm-up calls before serving")
    ap.add_argument("--max-idle-s", type=float, default=0.0, help="exit after this long without a request (0 = never)")
    ap.add_argument("--max-call-s", type=float, default=120.0, help="exit if one policy call exceeds this (hang)")
    ap.add_argument("--ready-file", default="", help="write this file once serving")
    ap.add_argument("--call-log", default="", help="append one JSON line per policy call")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    from openpi_client import msgpack_numpy
    import websockets
    from websockets.sync.server import serve

    pol = Pi05LiberoPolicy(a.ckpt, a.norm_stats, a.tokenizer, a.cameras, a.num_steps, a.device_id)
    logging.info("policy loaded in %.1fs (+ compile/capture %.1fs): %s", pol.load_s, pol.warmup_s, pol.backend_stamp)
    rng = np.random.default_rng(0)
    for i in range(a.warmup):
        t0 = time.time()
        pol.infer({"observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                   "observation/wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                   "observation/state": np.zeros(8, np.float32), "prompt": "warm up the kernels", "__noise_seed__": i})
        logging.info("warm-up call %d: %.1f ms", i, (time.time() - t0) * 1000)
    pol.n_calls = 0
    metadata = dict(backend=pol.backend_stamp, action_horizon=pol.H, num_steps=pol.num_steps, cameras=pol.cameras,
                    ckpt=str(a.ckpt), norm_stats=str(a.norm_stats), config="pi05_libero (openpi conventions)")
    calls = open(a.call_log, "a") if a.call_log else None
    state = {"last": time.time(), "in_call": None}
    lock = threading.Lock()

    def handler(ws):
        logging.info("connection from %s", ws.remote_address)
        packer = msgpack_numpy.Packer()
        ws.send(packer.pack(metadata))
        while True:
            try:
                msg = ws.recv()
            except websockets.ConnectionClosed:
                logging.info("connection closed")
                return
            try:
                obs = msgpack_numpy.unpackb(msg)
                for k in ("__golden_tag__", "__dump_golden__"):  # GPU-reference client keys: accepted, unused
                    obs.pop(k, None)
                with lock:
                    state["in_call"] = time.time()
                    out = pol.infer(obs)
                    state["in_call"] = None
                if calls is not None:
                    calls.write(json.dumps(dict(call=pol.n_calls, noise_seed=obs.get("__noise_seed__"),
                                                infer_ms=out["policy_timing"]["infer_ms"], t=time.time())) + "\n")
                    calls.flush()
                ws.send(packer.pack(out))
                state["last"] = time.time()
            except Exception:
                tb = traceback.format_exc()
                logging.error("infer failed:\n%s", tb)
                ws.send(tb)
                ws.close()
                return

    with serve(handler, a.host, a.port, compression=None, max_size=None, ping_interval=None) as server:
        def watchdog():
            while True:
                time.sleep(2)
                t = time.time()
                if state["in_call"] is not None and t - state["in_call"] > a.max_call_s:
                    logging.error("policy call running %.0f s > --max-call-s: device hang? exiting", t - state["in_call"])
                    logging.shutdown()
                    os._exit(3)
                if state["in_call"] is None and a.max_idle_s and t - state["last"] > a.max_idle_s:
                    logging.info("idle %.0f s -> shutting down", t - state["last"])
                    server.shutdown()
                    return

        threading.Thread(target=watchdog, daemon=True).start()
        logging.info("serving on %s:%d (%s)", a.host, a.port, pol.backend_stamp)
        if a.ready_file:
            open(a.ready_file, "w").write(f"{time.strftime('%F %T')} {pol.backend_stamp}\n")
        server.serve_forever()
    logging.info("served %d policy calls; closing model and device", pol.n_calls)
    pol.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
