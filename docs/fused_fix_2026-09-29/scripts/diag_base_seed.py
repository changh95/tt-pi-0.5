"""Is the 0.938 of (seed 3, 12 real tokens) the seed or the padding? Same seeds at other prompt lengths."""
import json, torch, ttnn
import models.experimental.pi0_5.tests.pcc.test_pcc_pi05_fused as t
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model as R
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN
cases = [(3, 12), (3, 224), (3, 64), (7, 12), (8, 12), (9, 12), (7, 224), (8, 224), (9, 224)]
loader = PI0WeightLoader(t.BASE_WEIGHTS)
obs = [t.padded_prompt(s, n) for s, n in cases]
ref = R(t.base_config(), loader); refs = []
for im, tok, nz, m in obs:
    ref.denoising.sample_noise = lambda *a, _n=nz, **k: _n.clone()
    with torch.no_grad():
        refs.append(ref.sample_actions(im, [torch.ones(1, dtype=torch.bool)] * 2, tok, m, torch.zeros(1, 32)).float())
del ref
dev = t.open_device()
try:
    torch.manual_seed(42)
    model = PI0ModelTTNN(t.base_config(), loader, dev, fused=FusedConfig.from_env())
    for (s, n), (im, tok, nz, m), r in zip(cases, obs, refs):
        print("RESULT", json.dumps({"seed": s, "n_real": n, "pcc": t.pcc(model.sample_actions_fused(im, tok, nz, lang_masks=m), r)}), flush=True)
    model.release_trace()
finally:
    ttnn.close_device(dev)
