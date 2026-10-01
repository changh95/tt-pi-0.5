import sys, yaml, subprocess, os
fo = sys.argv[1]; d = yaml.safe_load(open(sys.argv[2]))
only = sys.argv[3:]  # substrings to select
env = dict(os.environ, PYTHONPATH=fo, PYTHONDONTWRITEBYTECODE="1")
for i, v in enumerate(d["verify"]):
    if only and not any(o in v for o in only): continue
    code = v.replace("/opt/tt-metal", fo)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=fo)
    print(i, "OK" if r.returncode == 0 else "FAIL", v[:110].replace("\n", " "), "" if r.returncode == 0 else r.stderr.strip().splitlines()[-1][:300])
