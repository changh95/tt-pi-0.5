#!/usr/bin/env bash
# Build the N16 / 2-profile image once the LIBERO server's golden control passed. Usage: build_n16.sh <expected sha256 of serve_pi05_libero.py>
set -eu -o pipefail
EXP=$1
F=/home/deepgadget/experiments/gr00t/libero_eval/pi05/tt_dispatch_matrix/package/serve_pi05_libero.py
P=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
S=/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc
SP=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc/n16
GOT=$(sha256sum $F | cut -d' ' -f1); [ "$GOT" = "$EXP" ] || { echo "serve_pi05_libero.py sha $GOT != $EXP"; exit 1; }
cp $F $P/models/experimental/pi0_5/server/serve_pi05_libero.py
cd $P && git add models && git commit -q -F $SP/code_commit_msg.txt && SRC=$(git rev-parse HEAD) && echo "code commit $SRC"
git diff --name-only HEAD~1 HEAD | grep -E "generated/|\.omc/|__pycache__" && { echo "forbidden paths"; exit 1; } || true
rm -rf $S/code && mkdir $S/code && git archive $SRC models | tar -x -C $S/code
find $S/code -name __pycache__ | grep -q . && { echo "pycache in code/"; exit 1; } || true
python3 - "$S/tt-model.yaml" "$SRC" <<'PY'
import sys, re
p, src = sys.argv[1:3]; s = open(p).read()
s2 = re.sub(r'PI05_SOURCE_COMMIT: "[0-9a-f]{40}"', f'PI05_SOURCE_COMMIT: "{src}"', s); assert s2 != s or src in s
open(p, "w").write(s2)
PY
grep -n "PI05_SOURCE_COMMIT:" $S/tt-model.yaml
df -h / | tail -1
cd $S && source /home/deepgadget/experiments/tt-models/bin/docker-env.sh
echo "[$(date)] package start" > $SP/pkg.log
PYTHONDONTWRITEBYTECODE=1 /home/deepgadget/experiments/tt-models/.venv/bin/tt-model package --container tt-model.yaml --out /home/deepgadget/experiments/tt-models/build >> $SP/pkg.log 2>&1; RC=$?
echo "[$(date)] package rc=$RC" >> $SP/pkg.log; cp ~/.cache/tt-model/build/pi05-base-p150.log $SP/pkg-buildkit.log
df -h / | tail -1; exit $RC
