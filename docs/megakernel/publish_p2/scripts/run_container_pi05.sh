#!/usr/bin/env bash
# Hardware validation of the pi05-base-p150 IMAGE inside ONE device-lock hold:
#   DEVICE_LOCK_TIMEOUT=14400 with-device.sh timeout --signal=TERM --kill-after=120 2280 run_container_pi05.sh [cycles]
# cycle 1 cold (this package's ~/.cache/tt-model/pi05-base-p150 removed first), cycle 2 warm. Each cycle: serve (timed) ->
# boot landmarks -> /info -> smoke_test.py (unchanged, from the staged code/) -> bench 100 warm requests -> stop by manifest path.
set -u -o pipefail
CYCLES=${1:-2}; COLD=${COLD:-1}
ROOT=/home/deepgadget/experiments/tt-models
S=$ROOT/models/pi05-base-p150-fused
PKG=pi05-base-p150
MAN=$ROOT/build/$PKG/tt_kernel_manifest.json
SP=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pub2
LOG=$SP/logs; mkdir -p $LOG
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
source $ROOT/bin/docker-env.sh
TTM=$ROOT/.venv/bin/tt-model
echo "[$(date +%T)] start; manifest $MAN; image $(python3 -c "import json;print(json.load(open('$MAN'))['container']['image'])" 2>/dev/null)"
if [ "$COLD" = "1" ]; then echo "removing ~/.cache/tt-model/$PKG ($(du -sh ~/.cache/tt-model/$PKG 2>/dev/null | cut -f1)) for a cold boot"; rm -rf ~/.cache/tt-model/$PKG; fi
echo "device users before: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"
RC_ALL=0
cleanup() {
  echo "[trap] stop"; $TTM stop "$MAN" 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | tail -3 || true
  docker rm -f $(docker ps -aq --filter label=org.tenstorrent.tt-model=$PKG) >/dev/null 2>&1 || true
  echo "[trap] docker ps -a (tt-model): [$(docker ps -a --filter label=org.tenstorrent.tt-model --format '{{.Names}} {{.Status}}' | tr '\n' ';')]"
  echo "[trap] device users after stop: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"
}
trap cleanup EXIT
for C in $(seq 1 "$CYCLES"); do
  echo "=== cycle $C/$CYCLES [$(date +%T)] ==="
  docker rm -f $(docker ps -aq --filter label=org.tenstorrent.tt-model=$PKG) >/dev/null 2>&1 || true
  T0=$(date +%s.%N); $TTM serve "$MAN" > "$LOG/serve-c$C-$STAMP.log" 2>&1; SRC=$?; T1=$(date +%s.%N)
  echo "[$(date +%T)] serve exit=$SRC wall=$(python3 -c "print(round($T1-$T0,1))")s"
  sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$LOG/serve-c$C-$STAMP.log" | grep -v '^\s*$' | tail -25
  CID=$(docker ps -q --filter label=org.tenstorrent.tt-model=$PKG | head -1)
  if [ $SRC -ne 0 ] || [ -z "$CID" ]; then
    echo "serve FAILED"; CIDA=$(docker ps -aq --filter label=org.tenstorrent.tt-model=$PKG | head -1); [ -n "$CIDA" ] && docker logs -t "$CIDA" 2>&1 | tail -150 | tee "$LOG/container-c$C-$STAMP.log"
    RC_ALL=1; break
  fi
  PORT=$(docker port "$CID" | grep -o '0.0.0.0:[0-9]*' | head -1 | cut -d: -f2); URL=http://127.0.0.1:${PORT:-20000}; echo "endpoint $URL"
  docker logs -t "$CID" > "$LOG/container-c$C-boot-$STAMP.log" 2>&1
  grep -E "Loading|Opening|Warm|warm|startup|JIT|trace|Trace|built|tokenizer" "$LOG/container-c$C-boot-$STAMP.log" | cut -c1-300 | sed 's/^/    /' | tail -30
  curl -s "$URL/info" > "$LOG/info-c$C-$STAMP.json"; python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('info:', json.dumps({k: d.get(k) for k in ('name','hardware','megakernel','weights','source','warmup_latency_ms')})[:1500])" "$LOG/info-c$C-$STAMP.json"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); mk=(d.get('megakernel') or {}).get('backend'); print('info megakernel backend:', mk, 'source commit:', d['source']['commit']); dg=((d.get('megakernel') or {}).get('program') or {}).get('kernel_digest'); print('kernel_digest', dg); sys.exit(0 if mk == 'whole' and dg == '4aa02cdf21ed0c94' else 1)" "$LOG/info-c$C-$STAMP.json" || { echo "megakernel backend / digest is NOT whole / 4aa02cdf21ed0c94"; RC_ALL=1; }
  python3 $S/code/models/experimental/pi0_5/server/smoke_test.py --url "$URL" --timeout 300 2>&1 | tee "$LOG/smoke-c$C-$STAMP.log"; RC=${PIPESTATUS[0]}; echo "[$(date +%T)] smoke rc=$RC"; [ $RC -ne 0 ] && RC_ALL=1
  python3 $SP/bench_http_pi05.py --url "$URL" --media $S/media --n 100 --warmup 5 --out "$LOG/bench-c$C-$STAMP.json" 2>&1 | tail -3 | tee "$LOG/bench-c$C-$STAMP.log"; [ ${PIPESTATUS[0]} -ne 0 ] && RC_ALL=1
  docker logs -t "$CID" > "$LOG/container-c$C-$STAMP.log" 2>&1
  T2=$(date +%s.%N); $TTM stop "$MAN" 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | tail -3; T3=$(date +%s.%N)
  echo "[$(date +%T)] stop wall=$(python3 -c "print(round($T3-$T2,1))")s; caches:"; du -sh ~/.cache/tt-model/$PKG/* 2>/dev/null
  echo "device users after stop: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"
done
echo "[$(date +%T)] done rc=$RC_ALL"; exit $RC_ALL
