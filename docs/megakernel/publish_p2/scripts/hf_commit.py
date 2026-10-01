"""ONE create_commit to changh95/pi05-base-p150 with parent_commit pinned to cf08fb95. --dry-run lists the operations."""
import json, os, sys
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
REPO = "changh95/pi05-base-p150"; PARENT = "cf08fb955447b6ed877818bf43a554a0879a224a"
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
IMG = json.load(open(f"{B}/tt_kernel_manifest.json"))["container"]["image"]
msg = "pi0.5 phase-2 whole-model megakernel (the whole model = ONE persistent generic_op per call): %s ms served; LIBERO demo 99/100" % sys.argv[1]
desc = ("Image %s (%s) built from tt-metal 668c2907575 (clean) + changh95/tt-pi-0.5 @ 821e8c5 (main, merge of PR #3). "
        "code/ = that commit's models/ (+ tt-metal models/common/lightweightmodule.py). README / SERVING / GPU_COMPARISON / tt-model.yaml "
        "rewritten from measured files (label fixes of cf08fb95 kept); demo/ replaced with the whole-model LIBERO run (same file names, new content)." % (IMG["tag"], IMG["digest"]))
print(msg); print(desc)
if "--print" in sys.argv:
    sys.exit(0)
info = api.create_commit(repo_id=REPO, operations=ops, commit_message=msg, commit_description=desc, parent_commit=PARENT)
print("COMMIT", info.oid, info.commit_url)
