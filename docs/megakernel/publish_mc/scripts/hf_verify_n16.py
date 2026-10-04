"""Verify HF changh95/pi05-base-p150 @ REV against the commit's op list (n16/hf_ops_mc.json): every add by content
(LFS sha256 / git blob sha1 of the local file), unchanged files present, deletes gone, no extra files, parent."""
import hashlib, json, os, sys
from huggingface_hub import HfApi
REPO, REV, PARENT = "changh95/pi05-base-p150", sys.argv[1], "061677cfd1d1458b5a9f22e125e0d2a5ee6a32e5"
O = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_ops_mc.json")))
api = HfApi(); info = api.model_info(REPO, revision=REV, files_metadata=True); remote = {s.rfilename: s for s in info.siblings}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def blob(p):
    d = open(p, "rb").read(); return hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest()
keep = {".gitattributes", "media/sample_base.png", "media/sample_wrist.png"}
expected = set(O["adds"]) | set(O["unchanged"]) | keep
bad = [k for k, v in O["adds"].items() if k not in remote or (remote[k].lfs.sha256 != sha(v) if remote[k].lfs else remote[k].blob_id != blob(v))]
print("rev", info.sha, "| files", len(remote), "| expected", len(expected))
print("missing", sorted(expected - set(remote)), "| unexpected", sorted(set(remote) - expected))
print("adds checked by content:", len(O["adds"]), "mismatches", bad, "| deletes still present:", [d for d in O["deletes"] if d in remote])
cm = api.list_repo_commits(REPO, revision=REV)[:2]
print("commits:", [(c.commit_id[:8], c.title[:70]) for c in cm], "| parent ok:", cm[1].commit_id == PARENT)
ok = info.sha == REV and expected == set(remote) and not bad and not [d for d in O["deletes"] if d in remote] and cm[1].commit_id == PARENT
print("TREE", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
