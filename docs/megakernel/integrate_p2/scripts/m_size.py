"""verify-p2-r0 offline size gate: mock-cluster compile (no card) of the WHOLE program into an EMPTY private cache,
then my own accounting: every ELF the compile produced (must be exactly one per RISC for whole_*), text+data measured
with readelf (PT_LOAD filesz, and SHF_ALLOC PROGBITS sections), runtime args counted from the ProgramDescriptor the
model builds (max per-core unique args per kernel + common args), 16 B per CB id, semaphores.
  python m_size.py SHAPE OUT.json   (TT_METAL_CACHE must be an empty directory)"""
import glob, json, os, subprocess, sys, time
os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = os.path.join(
    os.environ["TT_METAL_HOME"], "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml")
shape, outp = sys.argv[1], sys.argv[2]
cache = os.environ["TT_METAL_CACHE"]
pre = glob.glob(cache + "/**/*.elf", recursive=True)
assert not pre, f"cache not empty: {len(pre)} ELFs"
import torch, ttnn  # noqa: E402
from models.experimental.pi0_5.tt.megakernel.pe_size_check import build_dummy  # noqa: E402
from models.experimental.pi0_5.tt.megakernel import pe_geometry as P  # noqa: E402
from models.experimental.pi0_5.tt.megakernel.pe_program import kernel_digest2  # noqa: E402


def elf_sizes(p):
    lo = subprocess.run(["readelf", "-lW", p], capture_output=True, text=True).stdout
    load = sum(int(l.split()[4], 16) for l in lo.splitlines() if l.strip().startswith("LOAD"))
    se = subprocess.run(["readelf", "-SW", p], capture_output=True, text=True).stdout
    alloc = 0
    for l in se.splitlines():
        if "]" in l and "PROGBITS" in l:
            f = l.split("]", 1)[1].split()
            # name type addr off size es flg ...
            if "A" in f[6]:
                alloc += int(f[4], 16)
    return {"load_filesz": load, "alloc_progbits": alloc}


res = {"shape": shape, "t0": time.strftime("%F %T"), "kernel_digest": kernel_digest2()}
dev = ttnn.open_device(device_id=0, l1_small_size=24576, worker_l1_size=1_395_712)
try:
    wm, kv, mask, tables, noise = build_dummy(dev, shape)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, wm.shape.suffix_rows, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                         dev, ttnn.L1_MEMORY_CONFIG)
    prog = wm.program(kv, mask, tables, noise, out, whole=True)
    from models.experimental.pi0_5.tt.megakernel import geometry as G
    lens = [len(list(wm.core_args((x, y)))) for x in range(G.GRID[0]) for y in range(G.GRID[1])]
    rt_len = {"whole_ncrisc.cpp": max(lens), "whole_brisc.cpp": max(lens), "whole_trisc.cpp": min(max(lens), G.N_RT_ARGS)}
    args = []
    for kd in prog.kernels:
        src = os.path.basename(str(kd.kernel_source))
        args.append({"src": src, "rt_max": rt_len[src], "rt_min_core": min(lens), "common": len(kd.common_runtime_args),
                     "n_cores": len(lens), "n_ct": len(kd.compile_time_args)})
    res["args"] = args
    res["n_cbs_ids"] = sum(len(c.format_descriptors) for c in prog.cbs)
    res["cb_ids"] = sorted(f.buffer_index for c in prog.cbs for f in c.format_descriptors)
    t0 = time.time()
    ttnn.generic_op(wm.io_tensors(kv, mask, tables, noise, out), prog)
    ttnn.synchronize_device(dev)
    res["compile_s"] = round(time.time() - t0, 1)
finally:
    ttnn.close_device(dev)
elfs = sorted(glob.glob(cache + "/**/*.elf", recursive=True))
res["elfs"] = {}
for p in elfs:
    res["elfs"][p.split("/kernels/")[1] if "/kernels/" in p else p] = elf_sizes(p)
whole = {k: v for k, v in res["elfs"].items() if k.startswith("whole_") and not k.endswith(".xip.elf")}
res["n_whole_elfs"] = len(whole)
binsum = sum((v["load_filesz"] + 15) // 16 * 16 for v in whole.values())
binsum_alloc = sum((v["alloc_progbits"] + 15) // 16 * 16 for v in whole.values())
argb = 4 * sum(a["rt_max"] + a["common"] for a in args if isinstance(a["rt_max"], int))
cbb = 16 * (max(res["cb_ids"]) + 1)
sem = 16 * 4
res["footprint"] = {"binaries_load": binsum, "binaries_alloc": binsum_alloc, "args_bytes": argb, "cb_bytes": cbb,
                    "sem_bytes": sem, "total_load": binsum + argb + cbb + sem,
                    "total_alloc": binsum_alloc + argb + cbb + sem, "gate": 131072, "ring": 136192}
res["footprint"]["pass"] = max(res["footprint"]["total_load"], res["footprint"]["total_alloc"]) <= 131072
json.dump(res, open(outp, "w"), indent=1)
print("RESULT", json.dumps({k: v for k, v in res.items() if k != "elfs"}), flush=True)
