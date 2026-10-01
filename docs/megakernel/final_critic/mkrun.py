"""Build a `docker run` for the profiled container from the inspect of the tt-model-served one (same image, devices,
binds, env), adding the device profiler env and a private cache + profiler dir."""
import json, shlex, sys
d = json.load(open(sys.argv[1]))[0]
fc = sys.argv[2]
hc, cf = d["HostConfig"], d["Config"]
a = ["docker", "run", "-d", "--name", "pi05-mkprof", "--user", cf.get("User") or "1000"]
for dv in hc.get("Devices") or []:
    a += ["--device", f'{dv["PathOnHost"]}:{dv["PathInContainer"]}']
for b in hc.get("Binds") or []:
    a += ["-v", b]
for m in d.get("Mounts") or []:
    if m["Type"] == "bind" and not any(b.split(":")[0] == m["Source"] for b in hc.get("Binds") or []):
        a += ["-v", f'{m["Source"]}:{m["Destination"]}' + ("" if m.get("RW", True) else ":ro")]
if hc.get("IpcMode"): a += ["--ipc", hc["IpcMode"]]
if hc.get("ShmSize"): a += ["--shm-size", str(hc["ShmSize"])]
for u in hc.get("Ulimits") or []:
    a += ["--ulimit", f'{u["Name"]}={u["Soft"]}:{u["Hard"]}']
a += ["--network", "host"] if hc.get("NetworkMode") == "host" else ["-p", "20001:20000"]
env = [e for e in cf["Env"] if not e.startswith(("TT_METAL_CACHE=",))]
env += ["TT_METAL_CACHE=/fc/ttcache", "TT_METAL_DEVICE_PROFILER=1", "TT_METAL_PROFILER_TRACE_TRACKING=1",
        "TT_METAL_PROFILER_CPP_POST_PROCESS=1", "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=16000",
        "TT_METAL_PROFILER_DIR=/fc/prof"]
for e in env:
    a += ["-e", e]
a += ["-v", f"{fc}:/fc"]
if cf.get("Entrypoint"): a += ["--entrypoint", cf["Entrypoint"][0]]
a += [d["Image"]] + (cf["Entrypoint"][1:] if cf.get("Entrypoint") else []) + (cf.get("Cmd") or [])
print(shlex.join(a))
