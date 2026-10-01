"""ONE create_commit to changh95/pi05-base-p150 with parent_commit pinned to 2cac5ec1. --dry-run lists the operations."""
import json, os, sys
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
REPO = "changh95/pi05-base-p150"; PARENT = "2900530f342e9ce12ae809c80389f47e73200a99"
S = "/home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused"; B = "/home/deepgadget/experiments/tt-models/build/pi05-base-p150"
api = HfApi()
remote = {s.rfilename for s in api.model_info(REPO, revision=PARENT).siblings}
local = {}
for f in ["README.md", "SERVING.md", "GPU_COMPARISON.md", "tt-model.yaml", "requirements.lock",
          "demo/README.md", "demo/libero_eval.json", "demo/pi05_libero_spatial.mp4", "demo/pi05_libero_spatial_poster.png",
          "demo/pi05_tt_libero_spatial_t0_i0.mp4", "demo/pi05_tt_libero_spatial_t3_i0.mp4",
          "demo/pi05_tt_libero_spatial_t5_i0.mp4", "demo/pi05_tt_libero_spatial_t7_i0.mp4"]:
    local[f] = f"{S}/{f}"
local["tt_kernel_manifest.json"] = f"{B}/tt_kernel_manifest.json"
for top in ("code", "image"):
    for d, _, fs in os.walk(f"{B}/{top}"):
        for x in fs:
            p = os.path.join(d, x); local[os.path.relpath(p, B)] = p
assert not any("__pycache__" in k or k.endswith(".pyc") for k in local)
keep = {".gitattributes", "media/sample_base.png", "media/sample_wrist.png"}
deletes = sorted(r for r in remote if r not in local and r not in keep)
unexpected = [r for r in deletes if not r.startswith("image/blobs/")]
assert not unexpected, unexpected
ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in sorted(local.items())] + [CommitOperationDelete(path_in_repo=d) for d in deletes]
print(f"adds {len(local)} (new paths {len(set(local) - remote)}), deletes {len(deletes)}, kept {sorted(keep & remote)}")
print("deletes:", *[d for d in deletes if not d.startswith("image/")], f"+ {sum(d.startswith('image/') for d in deletes)} image/ blobs", sep="\n  ")
json.dump({"adds": {k: v for k, v in sorted(local.items())}, "deletes": deletes}, open(os.path.dirname(__file__) + "/hf_ops.json", "w"), indent=1)
if "--dry-run" in sys.argv:
    sys.exit(0)
msg = "pi0.5 phase-1 expert megakernel (one persistent generic_op for the 10-step expert loop): 70.8 ms served; LIBERO demo 99/100"
desc = ("Image tt-model/pi05-base-p150:fe0d2e3d68a7 (sha256:fe0d2e3d68a752709a443cbe6b8e5aa5f9d914ab21f03631959cb81623aeeafe) built from tt-metal 668c2907575 (clean) + "
        "changh95/tt-pi-0.5 @ f7f173bd (main, merge of PR #2). code/ = that commit's models/ (+ tt-metal models/common/lightweightmodule.py). "
        "README / SERVING / GPU_COMPARISON / tt-model.yaml updated from measured files; demo/ replaced with the megakernel LIBERO run (same file names, new content).")
info = api.create_commit(repo_id=REPO, operations=ops, commit_message=msg, commit_description=desc, parent_commit=PARENT)
print("COMMIT", info.oid, info.commit_url)
