"""CPU: full fp32 torch reference per SPEC seed (whole model) + positive controls of the oracle:
(1) my expert_loop fed the reference's OWN VLM cache reproduces ref.sample_actions (tolerance: dt rounding only);
(2) my mask / positions equal prefix_attention_inputs'."""
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from oracle import expert_inputs, expert_loop  # noqa: E402
from vc import BASE_WEIGHTS, OUT, SPEC, base_config, now, obs_for, pcc  # noqa: E402

from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model, prefix_attention_inputs

torch.set_grad_enabled(False)
ref = PI0Model(base_config(), PI0WeightLoader(BASE_WEIGHTS))
res = {"refs": {}, "valid": {}, "ctrl": {}}
for i, (seed, n) in enumerate(SPEC):
    images, tokens, noise, mask = obs_for(seed, n)
    ref.denoising.sample_noise = lambda *a, _n=noise, **k: _n.clone()
    out = ref.sample_actions(images, [torch.ones(1, dtype=torch.bool)] * 2, tokens, mask, torch.zeros(1, 32)).float()
    valid = torch.cat([torch.ones(1, 512, dtype=torch.bool), mask], dim=1)
    res["refs"][seed], res["valid"][seed] = out, valid
    if i in (0, 1, 6, 11):
        pe, ppm, _ = ref.embed_prefix(images, [torch.ones(1, dtype=torch.bool)] * 2, tokens, mask)
        c = {"valid_eq_embed_prefix": bool(torch.equal(ppm.reshape(1, -1).bool(), valid))}
        vm, vp, em, ep = prefix_attention_inputs(ppm.reshape(1, -1), 50)
        mm, mp = expert_inputs(valid, 50)
        c["mask_eq"] = bool(torch.equal(mm, em.float())), 
        c["pos_eq"] = bool(torch.equal(mp, ep))
        _, cache = ref.backbone.forward_vlm(pe, attention_mask=vm.to(pe.dtype), position_ids=vp, use_cache=True)
        o2 = expert_loop(ref, cache, valid, noise, 50)
        c.update(pcc=pcc(o2, out), maxabs=float((o2 - out).abs().max()), cache_shape=list(cache[0][0].shape))
        res["ctrl"][seed] = c
        print(now(), "ctrl", seed, c, flush=True)
    print(now(), "ref", seed, n, flush=True)
torch.save(res, f"{OUT}/ref_base.pt")
print(now(), "DONE", json.dumps(res["ctrl"]))
