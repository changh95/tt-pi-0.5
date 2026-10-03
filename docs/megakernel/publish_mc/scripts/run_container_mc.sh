#!/usr/bin/env bash
# Hardware validation of the multi-config pi05-base-p150 IMAGE inside ONE device-lock hold:
#   DEVICE_LOCK_TIMEOUT=14400 with-device.sh timeout --signal=TERM --kill-after=120 5400 run_container_mc.sh
# 1. default profile c2-h50-n10: cycle 1 cold (this package's ~/.cache/tt-model/pi05-base-p150 removed), cycle 2 warm;
#    each: serve (timed) -> /info (backend mc, digest, source commit) -> smoke_test (staged code/) -> 100-request bench.
# 2. every other serve profile once: serve --profile -> /info -> smoke -> 30-request bench.
# 3. img_check.py inside the image (golden PCC7, c1-c4 bit-identity outputs, refusals), container spec of tt-model's own.
set -u -o pipefail
ROOT=/home/deepgadget/experiments/tt-models
S=$ROOT/models/pi05-base-p150-mc
PKG=pi05-base-p150
MAN=$ROOT/build/$PKG/tt_kernel_manifest.json
SP=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc
LOG=$SP/logs/img; mkdir -p $LOG
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
DIGEST=1429d5bea05c31ad
SRC=$(python3 -c "import yaml;print(yaml.safe_load(open('$S/tt-model.yaml'))['serve']['env']['PI05_SOURCE_COMMIT'])")
source $ROOT/bin/docker-env.sh
TTM=$ROOT/.venv/bin/tt-model
PROFILES=${PROFILES:-"c1-h50-n10 c3-h50-n10 c4-h50-n10 c2-h10-n10 c2-h10-n5"}
echo "[$(date +%T)] start; manifest $MAN; image $(python3 -c "import json;print(json.load(open('$MAN'))['container']['image'])"); source $SRC"
RC_ALL=0
cleanup() {
  echo "[trap] stop"; $TTM stop "$MAN" 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | tail -2 || true
  docker rm -f pi05-mccheck >/dev/null 2>&1
  docker rm -f $(docker ps -aq --filter label=org.tenstorrent.tt-model=$PKG) >/dev/null 2>&1 || true
  echo "[trap] device users after stop: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"
}
trap cleanup EXIT

serve_check() {  # $1 tag, $2 profile (""=default), $3 bench n
  local TAG=$1 PROF=$2 N=$3
  echo "=== $TAG profile=${PROF:-default} [$(date +%T)] ==="
  docker rm -f $(docker ps -aq --filter label=org.tenstorrent.tt-model=$PKG) >/dev/null 2>&1 || true
  local T0=$(date +%s.%N)
  if [ -n "$PROF" ]; then $TTM serve "$MAN" --profile "$PROF" > "$LOG/serve-$TAG-$STAMP.log" 2>&1; else $TTM serve "$MAN" > "$LOG/serve-$TAG-$STAMP.log" 2>&1; fi
  local SRC_RC=$? T1=$(date +%s.%N)
  echo "[$(date +%T)] serve exit=$SRC_RC wall=$(python3 -c "print(round($T1-$T0,1))")s"
  local CID=$(docker ps -q --filter label=org.tenstorrent.tt-model=$PKG | head -1)
  if [ $SRC_RC -ne 0 ] || [ -z "$CID" ]; then
    echo "serve FAILED"; sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOG/serve-$TAG-$STAMP.log" | tail -20
    local CIDA=$(docker ps -aq --filter label=org.tenstorrent.tt-model=$PKG | head -1); [ -n "$CIDA" ] && docker logs -t "$CIDA" 2>&1 | tail -80 | tee "$LOG/container-$TAG-$STAMP.log"
    RC_ALL=1; return 1
  fi
  local PORT=$(docker port "$CID" | grep -o '0.0.0.0:[0-9]*' | head -1 | cut -d: -f2); local URL=http://127.0.0.1:${PORT:-20000}
  docker logs -t "$CID" > "$LOG/container-$TAG-boot-$STAMP.log" 2>&1
  grep -E "Opening|Model built|Warm|warm|refused" "$LOG/container-$TAG-boot-$STAMP.log" | cut -c1-220 | sed 's/^/    /' | tail -8
  curl -s "$URL/info" > "$LOG/info-$TAG-$STAMP.json"
  python3 - "$LOG/info-$TAG-$STAMP.json" "$DIGEST" "$SRC" <<'PY' || { echo "INFO CHECK FAILED"; RC_ALL=1; }
import json, sys
d = json.load(open(sys.argv[1])); mk = d.get("megakernel") or {}; pr = mk.get("program") or {}
print("info:", mk.get("backend"), json.dumps(pr), "source", d["source"]["commit"], "inputs.num_images", d["inputs"]["num_images"], "outputs", d["outputs"]["actions"], "warmup", d.get("warmup_latency_ms"))
sys.exit(0 if mk.get("backend") == "mc" and pr.get("kernel_digest") == sys.argv[2] and d["source"]["commit"] == sys.argv[3] else 1)
PY
  python3 $S/code/models/experimental/pi0_5/server/smoke_test.py --url "$URL" --timeout 300 2>&1 | tail -1 | tee "$LOG/smoke-$TAG-$STAMP.log"; [ ${PIPESTATUS[0]} -ne 0 ] && RC_ALL=1
  python3 $SP/bench_http_pi05.py --url "$URL" --media $S/media --n $N --warmup 5 --out "$LOG/bench-$TAG-$STAMP.json" 2>&1 | tail -1 | tee "$LOG/bench-$TAG-$STAMP.log"; [ ${PIPESTATUS[0]} -ne 0 ] && RC_ALL=1
  docker logs -t "$CID" > "$LOG/container-$TAG-$STAMP.log" 2>&1
  if [ "$TAG" = "c2-default-c2" ]; then docker inspect "$CID" > "$LOG/inspect-$STAMP.json"; fi
  $TTM stop "$MAN" 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | tail -1
  echo "device users after stop: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"
}

if [ "${SKIP_DEFAULT:-0}" != "1" ]; then
  echo "removing ~/.cache/tt-model/$PKG/cache ($(du -sh ~/.cache/tt-model/$PKG/cache 2>/dev/null | cut -f1)) for a cold boot"; rm -rf ~/.cache/tt-model/$PKG/cache
  serve_check c2-default-c1 "" 100
  serve_check c2-default-c2 "" 100
fi
for P in $PROFILES; do serve_check "$P" "$P" 30; done

if [ "${SKIP_IMGCHECK:-0}" != "1" ]; then
  echo "=== img_check in the image [$(date +%T)] ==="
  INS=$(ls -t $LOG/inspect-*.json | head -1)
  mkdir -p $SP/check_img; chmod 777 $SP/check_img
  CMD=$(python3 $SP/mkrun_mc.py "$INS" pi05-mccheck "-v $SP/chk:/chk:ro -v $SP:/sp" "python /sp/img_check.py --out /sp/check_img" PYTHONDONTWRITEBYTECODE=1)
  echo "RUN $CMD" > $LOG/imgcheck_cmd-$STAMP.txt
  eval "$CMD" > $LOG/imgcheck-$STAMP.log 2>&1; R=$?
  grep -E "GOLDEN|BITID|REFUSE|DONE|Traceback|Error" $LOG/imgcheck-$STAMP.log | cut -c1-300
  echo "[$(date +%T)] img_check rc=$R"; [ $R -ne 0 ] && RC_ALL=1
fi
if [ "${SKIP_PROF:-0}" != "1" ]; then
  echo "=== device profiler: programs per replay [$(date +%T)] ==="
  INS=$(ls -t $LOG/inspect-*.json | head -1)
  rm -rf $SP/prof $SP/ttcache_prof; mkdir -p $SP/prof $SP/ttcache_prof; chmod 777 $SP/prof $SP/ttcache_prof
  CMD=$(python3 $SP/mkrun_mc.py "$INS" pi05-mcprof "-v $SP:/sp" "python /sp/prof_ops.py" PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=/sp/ttcache_prof TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=16000 TT_METAL_PROFILER_DIR=/sp/prof)
  echo "RUN $CMD" > $LOG/prof_cmd-$STAMP.txt
  eval "$CMD" > $LOG/prof-$STAMP.log 2>&1; R=$?
  grep -E "call|PROF DONE|Traceback|Error|ReadDevice" $LOG/prof-$STAMP.log | cut -c1-200 | tail -16
  echo "[$(date +%T)] profiler run rc=$R"; [ $R -ne 0 ] && RC_ALL=1
  CSV=$(find $SP/prof -name cpp_device_perf_report.csv | head -1); echo "csv: $CSV"
  [ -n "$CSV" ] && python3 $SP/prof_analyze.py "$CSV" $LOG/prof_ops-$STAMP.json | head -60
fi
echo "[$(date +%T)] done rc=$RC_ALL"; exit $RC_ALL
