import hashlib, json, os, sys, filecmp, urllib.request
from huggingface_hub import HfApi, hf_hub_download
REPO = "changh95/pi05-base-p150"; REV = sys.argv[1]
SP = os.path.dirname(os.path.abspath(__file__)); ops = json.load(open(f"{SP}/hf_ops.json"))
api = HfApi(); info = api.model_info(REPO, revision=REV, files_metadata=True)
print("repo sha", info.sha)
remote = {s.rfilename: s for s in info.siblings}
expected = set(ops["adds"]) | {".gitattributes", "media/sample_base.png", "media/sample_wrist.png"}
print("tree:", len(remote), "files; missing:", sorted(expected - set(remote)), "unexpected:", sorted(set(remote) - expected))
print("deleted still present:", [d for d in ops["deletes"] if d in remote])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
dl = f"{SP}/hf_dl"; bad = []; n_dl = n_meta = 0
for path, lp in sorted(ops["adds"].items()):
    ls = sha(lp)
    if path.startswith("image/blobs/"):
        s = remote[path]; rs = s.lfs.sha256 if s.lfs else None
        if rs is None:  # small non-LFS blob: download it
            rs = sha(hf_hub_download(REPO, path, revision=REV, local_dir=dl)); n_dl += 1
        else:
            n_meta += 1
        ok = rs == ls == path.rsplit("/", 1)[1]
    else:
        rs = sha(hf_hub_download(REPO, path, revision=REV, local_dir=dl)); n_dl += 1; ok = rs == ls
    if not ok: bad.append((path, ls, rs))
print(f"sha256: {n_dl} files downloaded and hashed, {n_meta} LFS image blobs checked via Hub lfs.sha256 (== local sha == blob name); mismatches: {bad}")
# code/ vs GitHub main clone
gh = f"{SP}/ghmain/repo/models"; hc = f"{dl}/code/models"
def walk(r): return sorted(os.path.relpath(os.path.join(d, f), r) for d, _, fs in os.walk(r) for f in fs)
g, h = walk(gh), walk(hc)
diff = [p for p in g if p not in h or not filecmp.cmp(f"{gh}/{p}", f"{hc}/{p}", shallow=False)]
print(f"code/: {len(g)} GitHub main files, all byte-identical in HF code/: {not diff} {diff}; HF-only: {[p for p in h if p not in g]}")
for u in ["https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial.mp4",
          "https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial_poster.png",
          "https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_tt_libero_spatial_t3_i0.mp4",
          "https://huggingface.co/changh95/pi05-base-p150/resolve/main/media/sample_base.png",
          "https://huggingface.co/changh95/pi05-base-p150"]:
    r = urllib.request.urlopen(urllib.request.Request(u, method="HEAD"), timeout=60)
    print("HEAD", r.status, r.headers.get("Content-Type"), r.headers.get("Content-Length"), u)
