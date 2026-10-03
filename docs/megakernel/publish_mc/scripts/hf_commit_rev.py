"""ONE create_commit to changh95/pi05-base-p150 with parent_commit pinned to 990e22b5 (the HF head on 2026-10-03).
--dry-run: list the operations and write hf_ops_mc.json + a diff summary (no network writes). Without --dry-run it
commits -- do NOT run without the lead's go."""
import hashlib, json, os, sys
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
REPO = "changh95/pi05-base-p150"; PARENT = "ab447b79a73f7bf2d0b372beba4c8df7cc640076"
S = "/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc"; B = "/home/deepgadget/experiments/tt-models/build/pi05-base-p150"
SP = os.path.dirname(os.path.abspath(__file__))
api = HfApi()
info = api.model_info(REPO, revision=PARENT, files_metadata=True)
remote = {s.rfilename: s for s in info.siblings}
local = {}
for f in ["README.md", "SERVING.md", "GPU_COMPARISON.md", "PERF_PRESETS.md", "tt-model.yaml", "requirements.lock",
          "demo/README.md", "demo/manifest.json", "demo/pi05_libero_spatial.mp4", "demo/pi05_libero_spatial_poster.png",
          "demo/pi05_tt_libero_spatial_t0_i0.mp4", "demo/pi05_tt_libero_spatial_t3_i0.mp4",
          "demo/pi05_tt_libero_spatial_t5_i0.mp4", "demo/pi05_tt_libero_spatial_t7_i0.mp4"]:
    local[f] = f"{S}/{f}"
local["tt_kernel_manifest.json"] = f"{B}/tt_kernel_manifest.json"
for top in ("code", "image"):
    for d, _, fs in os.walk(f"{B}/{top}"):
        for x in fs:
            p = os.path.join(d, x); local[os.path.relpath(p, B)] = p
assert not any("__pycache__" in k or k.endswith(".pyc") for k in local)
assert all(os.path.isfile(v) for v in local.values())
keep = {".gitattributes", "media/sample_base.png", "media/sample_wrist.png"}
deletes = sorted(r for r in remote if r not in local and r not in keep)
unexpected = [r for r in deletes if not r.startswith("image/blobs/")]
assert not unexpected, unexpected

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

def git_blob_sha1(p):
    data = open(p, "rb").read(); return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()

new, changed, same = [], [], []
for k, v in sorted(local.items()):
    if k not in remote:
        new.append(k); continue
    r = remote[k]
    if r.lfs:
        (same if r.lfs.sha256 == sha(v) else changed).append(k)
    else:
        (same if r.blob_id == git_blob_sha1(v) else changed).append(k)
ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in sorted(local.items()) if k not in same] + \
      [CommitOperationDelete(path_in_repo=d) for d in deletes]
summary = {"repo": REPO, "parent": PARENT, "remote_files": len(remote), "local_files": len(local),
           "new": new, "changed": changed, "unchanged": len(same), "deleted": deletes,
           "new_count": len(new), "changed_count": len(changed), "deleted_count": len(deletes),
           "add_ops": len(local) - len(same), "image": json.load(open(f"{B}/tt_kernel_manifest.json"))["container"]["image"]}
json.dump({"adds": {k: v for k, v in sorted(local.items()) if k not in same}, "deletes": deletes, "unchanged": same},
          open(f"{SP}/hf_ops_rev.json", "w"), indent=1)
json.dump(summary, open(f"{SP}/hf_diff_rev.json", "w"), indent=1)
print(f"remote {len(remote)} files @ {PARENT[:8]}; local {len(local)}: new {len(new)}, changed {len(changed)}, unchanged {len(same)}, deleted {len(deletes)}")
print("changed (non-image):", [c for c in changed if not c.startswith("image/")])
print("new (non-image):", len([n for n in new if not n.startswith("image/")]), "; new image blobs:", len([n for n in new if n.startswith("image/")]))
print("deleted:", len(deletes), "image blobs")
if "--dry-run" in sys.argv:
    sys.exit(0)
IMG = summary["image"]
msg = "STE (ASD-STE100) docs and one serve profile p150 (configuration through PI05_NUM_IMAGES / PI05_ACTION_HORIZON / PI05_NUM_STEPS)"
desc = ("Image %s (%s): the repackage of 36f651704bf1 with one serve profile p150 (same packages, libraries and model code; "
        "only the baked serve-default.sh differs; re-validated: golden PCC7 0.999981 / 0.999958, c1-c4 bit-identical, served 56.40 ms). "
        "README / SERVING / GPU_COMPARISON / PERF_PRESETS / demo README rewritten in STE; code/ unchanged (changh95/tt-pi-0.5 @ 5edf139, docs merge 536c507)." % (IMG["tag"], IMG["digest"]))
print(msg); print(desc)
info = api.create_commit(repo_id=REPO, operations=ops, commit_message=msg, commit_description=desc, parent_commit=PARENT)
print("COMMIT", info.oid, info.commit_url)
