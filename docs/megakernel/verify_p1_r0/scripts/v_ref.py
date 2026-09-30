"""CPU: full fp32 torch reference per SPEC seed + the oracle positive control (oracle fed with the reference's OWN
VLM cache must reproduce the reference output). Saves out/ref_base.pt."""
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vcommon import BASE_WEIGHTS, OUT, SPEC, base_config, now, obs_for, oracle, pcc  # noqa: E402

from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model, prefix_attention_inputs

torch.set_grad_enabled(False)
ref = PI0Model(base_config(), PI0WeightLoader(BASE_WEIGHTS))
res = {"refs": {}, "valid": {}, "ctrl": {}}
for i, (seed, n) in enumerate(SPEC):
    images, tokens, noise, mask = obs_for(seed, n)
    ref.denoising.sample_noise = lambda *a, _n=noise, **k: _n.clone()
    out = ref.sample_actions(images, [torch.ones(1, dtype=torch.bool)] * 2, tokens, mask, torch.zeros(1, 32)).float()
    res["refs"][seed] = out
    valid = torch.cat([torch.ones(1, 512, dtype=torch.bool), mask], dim=1)
    res["valid"][seed] = valid
    if i < 2:  # positive control of the oracle + prefix-valid construction
        pe, ppm, _ = ref.embed_prefix(images, [torch.ones(1, dtype=torch.bool)] * 2, tokens, mask)
        assert torch.equal(ppm.reshape(1, -1).bool(), valid), "prefix validity construction differs from embed_prefix"
        vm, vp, _, _ = prefix_attention_inputs(ppm.reshape(1, -1), 50)
        _, cache = ref.backbone.forward_vlm(pe, attention_mask=vm.to(pe.dtype), position_ids=vp, use_cache=True)
        o2 = oracle(ref, cache, valid, noise, 50)
        res["ctrl"][seed] = {"bit_equal": bool(torch.equal(o2, out)), "maxabs": float((o2 - out).abs().max()),
                             "cache_shape": list(cache[0][0].shape)}
        print(now(), "ctrl", seed, res["ctrl"][seed], flush=True)
    print(now(), "ref", seed, n, flush=True)
torch.save(res, f"{OUT}/ref_base.pt")
print(now(), "DONE", json.dumps(res["ctrl"]))
