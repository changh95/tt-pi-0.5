"""Verify the HF tree at REV against the staging manifest: file set, sha256 (LFS: Hub lfs.sha256; non-LFS: git blob
sha1 of the local file == Hub blob_id), deletes gone, parent."""
import hashlib, json, sys
from huggingface_hub import HfApi
REPO = "changh95/pi05-base-p150"; REV = sys.argv[1]
M = json.load(open("/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc.STAGING_MANIFEST.json"))
api = HfApi(); info = api.model_info(REPO, revision=REV, files_metadata=True)
remote = {s.rfilename: s for s in info.siblings}
expected = set(M["adds_or_replaces"]) | set(M["unchanged_on_hub"]) | set(M["kept"])
print("rev", info.sha, "| files", len(remote), "| expected", len(expected))
print("missing", sorted(expected - set(remote)), "| unexpected", sorted(set(remote) - expected))
print("deleted still present", [d for d in M["deletes"] if d in remote])
def blob(p):
    d = open(p, "rb").read(); return hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest()
bad = []; n_lfs = n_git = 0
for k, v in M["adds_or_replaces"].items():
    r = remote[k]
    if r.lfs:
        n_lfs += 1; ok = r.lfs.sha256 == v["sha256"]
    else:
        n_git += 1; ok = r.blob_id == blob(v["src"]) and hashlib.sha256(open(v["src"], "rb").read()).hexdigest() == v["sha256"]
    if not ok: bad.append(k)
print(f"adds/replaces checked: {n_lfs} LFS by sha256, {n_git} git blobs by sha1; mismatches {bad}")
cm = api.list_repo_commits(REPO, revision=REV)[:2]
print("commits:", [(c.commit_id[:8], c.title[:80]) for c in cm])
