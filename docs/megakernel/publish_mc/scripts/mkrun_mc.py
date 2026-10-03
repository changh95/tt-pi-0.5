"""`docker run --rm` of the SAME image, devices, binds, env and entrypoint as a tt-model-served container (from its
docker inspect), running a custom command instead of the server. Optional extra env (e.g. the device profiler).
usage: mkrun_mc.py INSPECT_JSON NAME "EXTRA_DOCKER_ARGS" "COMMAND" [ENV=VAL ...]"""
import json, shlex, sys

d = json.load(open(sys.argv[1]))[0]
name, extra, cmd = sys.argv[2], sys.argv[3], sys.argv[4]
hc, cf = d["HostConfig"], d["Config"]
a = ["docker", "run", "--rm", "--name", name, "--user", cf.get("User") or "1000"]
for dv in hc.get("Devices") or []:
    a += ["--device", f'{dv["PathOnHost"]}:{dv["PathInContainer"]}']
for b in hc.get("Binds") or []:
    a += ["-v", b]
for m in d.get("Mounts") or []:
    if m["Type"] == "bind" and not any(b.split(":")[0] == m["Source"] for b in hc.get("Binds") or []):
        a += ["-v", f'{m["Source"]}:{m["Destination"]}' + ("" if m.get("RW", True) else ":ro")]
if hc.get("IpcMode"):
    a += ["--ipc", hc["IpcMode"]]
if hc.get("ShmSize"):
    a += ["--shm-size", str(hc["ShmSize"])]
for u in hc.get("Ulimits") or []:
    a += ["--ulimit", f'{u["Name"]}={u["Soft"]}:{u["Hard"]}']
for e in cf["Env"] + sys.argv[5:]:
    a += ["-e", e]
a += shlex.split(extra)
if cf.get("Entrypoint"):
    a += ["--entrypoint", cf["Entrypoint"][0]]
a += [d["Image"]] + (cf["Entrypoint"][1:] if cf.get("Entrypoint") else []) + shlex.split(cmd)
print(shlex.join(a))
