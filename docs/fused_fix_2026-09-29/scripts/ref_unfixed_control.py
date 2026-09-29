"""Control: the reference WITHOUT the fix (no masks, expert positions [0,H)), same records -> PCC vs openpi."""
import json, torch
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model
from models.experimental.pi0_5.tests.pcc.golden_openpi import LIBERO_WEIGHTS, libero_config, load_records, pcc7, record_inputs
m = PI0Model(libero_config(), PI0WeightLoader(LIBERO_WEIGHTS))
def unfixed(images, tokens, lang_mask, noise):
    pe, _, _ = m.embed_prefix(images, [torch.ones(1, dtype=torch.bool)] * 2, tokens, lang_mask)
    _, kv = m.backbone.forward_vlm(pe, use_cache=True)
    m.denoising.sample_noise = lambda *a, **k: noise.clone()
    return m.denoising.sample_actions(1, prefix_kv_cache=kv, state=torch.zeros(1, 32))
out = {}
with torch.no_grad():
    for L in (32, 224):
        v = [pcc7(unfixed(*record_inputs(r, L)), r["actions_model_norm"]) for r in load_records()]
        out[L] = {"mean": sum(v) / len(v), "min": min(v), "all": v}
        print(L, json.dumps(out[L]), flush=True)
json.dump(out, open(__file__.replace(".py", ".json").replace("fusedfix/", "fusedfix/results/"), "w"), indent=1)
