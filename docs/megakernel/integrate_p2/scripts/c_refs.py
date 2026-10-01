"""CPU fp32 torch reference of the WHOLE model per input, keeping its VLM K/V cache (the P2-3 oracle) as well.
My pipeline = forward_inference's steps written out (embed_prefix -> prefix_attention_inputs -> forward_vlm(cache) ->
denoising.sample_actions with the fixed noise). Positive control: equals ref.sample_actions bit-exactly (2 inputs/shape)."""
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vc2 import BASE_WEIGHTS, KV_SEEDS, OUT, SPEC, base_config, now, obs_for, pcc  # noqa: E402

from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model, prefix_attention_inputs
from models.experimental.pi0_5.tests.pcc.golden_openpi import LIBERO_WEIGHTS, libero_config, load_records, record_inputs

torch.set_grad_enabled(False)
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 16)
which = sys.argv[1]  # base | libero
IM = [torch.ones(1, dtype=torch.bool)] * 2


def run(ref, images, tokens, noise, mask, H):
    ref.denoising.sample_noise = lambda *a, _n=noise, **k: _n.clone()
    pe, ppm, _ = ref.embed_prefix(images, IM, tokens, mask)
    vm, vp, em, ep = prefix_attention_inputs(ppm.reshape(1, -1), H)
    _, cache = ref.backbone.forward_vlm(pe, attention_mask=vm.to(pe.dtype), position_ids=vp, use_cache=True)
    out = ref.denoising.sample_actions(1, prefix_kv_cache=cache, device=pe.device, state=torch.zeros(1, 32),
                                       attention_mask=em.to(pe.dtype), position_ids=ep).float()
    return out, cache, ppm.reshape(1, -1).bool()


res = {"refs": {}, "valid": {}, "kv": {}, "ctrl": {}}
if which == "base":
    ref = PI0Model(base_config(), PI0WeightLoader(BASE_WEIGHTS))
    items = [(s, obs_for(s, n)) for s, n in SPEC]
    H, keep = 50, set(KV_SEEDS)
else:
    ref = PI0Model(libero_config(), PI0WeightLoader(LIBERO_WEIGHTS))
    recs = load_records()
    items = []
    for r in recs:
        im, tk, mk, nz = record_inputs(r, 32)
        items.append((r["tag"], (im, tk, nz, mk)))
    H, keep = 10, {t for t, _ in items}
for i, (tag, (images, tokens, noise, mask)) in enumerate(items):
    out, cache, valid = run(ref, images, tokens, noise, mask, H)
    res["refs"][tag], res["valid"][tag] = out, valid
    if tag in keep:
        res["kv"][tag] = [(k.float().clone(), v.float().clone()) for k, v in cache]
    if i < 2:
        ref.denoising.sample_noise = lambda *a, _n=noise, **k: _n.clone()
        o2 = ref.sample_actions(images, IM, tokens, mask, torch.zeros(1, 32)).float()
        res["ctrl"][tag] = {"bit_equal_sample_actions": bool(torch.equal(o2, out)), "pcc": pcc(o2, out),
                            "cache_shape": list(cache[0][0].shape), "n_valid": int(valid.sum())}
        print(now(), "ctrl", tag, res["ctrl"][tag], flush=True)
    print(now(), which, "ref", tag, flush=True)
torch.save(res, f"{OUT}/ref_{which}.pt")
print(now(), "DONE", json.dumps(res["ctrl"]))
