#!/usr/bin/env bash
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
# EW-v2 live trial, one scheduled issue day (amendment §5). Schedule (project owner's crontab, UTC):
#     CRON_TZ=UTC
#     0 12 * * * /home/bcheng/STOCKHOLM/SENTINEL-STOCKHOLM/scripts/live/run_daily.sh live
# (without CRON_TZ=UTC, "0 12" would run at 12:00 LOCAL time). On-time = issued <= D 23:59 UTC; inputs are capped at
# d = D - 1, so a later run on the same day gains no information.
# EVERY scheduled day ends with a chained hash-log row: "issued" (written by issue.py) or "failed" (written here on
# EXIT by log_status.py with the failing stage, exit code and exception class). Then all rows without evidence are
# timestamped (stamp.py; hash-only, only backends approved in FREEZE_ew_live.json) and the log is verified.
# A failed gridMET fetch does not stop the day: the issue goes ahead with NaN weather (H2/H4; H1 uses no weather).
# Mode "live" is refused by issue.py until docs/preregistration/FREEZE_ew_live.json exists (lead).
# CPU only, <= 6 threads, niced. The lead commits + pushes results/live/ (log and timestamp evidence).
set -uo pipefail
MODE="${1:-live}"
DATE="${2:-$(date -u +%F)}"
PY=/home/bcheng/miniconda3/envs/sentinel2-live/bin/python   # dedicated env (review C-D2); no other workstream uses it
HERE="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
ROOT=/home/bcheng/STOCKHOLM/sentinel_runs/live/ew_v2
[ "$MODE" = "dryrun" ] && ROOT="$ROOT/dryrun"
LOGDIR="$ROOT/daily_logs/$DATE"
mkdir -p "$LOGDIR"
LOCK=/home/bcheng/STOCKHOLM/sentinel_runs/live/ew_v2/.run_daily.lock
exec 9>"$LOCK"
flock -n 9 || { echo "another run_daily is active"; exit 1; }
cd "$HERE"
STAGE=start
RC=0
finish() {
  RC=$?
  "$PY" -B log_status.py --mode "$MODE" --date "$DATE" --stage "$STAGE" --rc "$RC" --stage-log "$LOGDIR/$STAGE.log"
  "$PY" -B stamp.py --mode "$MODE" > "$LOGDIR/stamp.log" 2>&1
  "$PY" -B push_evidence.py --mode "$MODE" > "$LOGDIR/push.log" 2>&1   # dormant until enabled in the freeze record
  if [ "$MODE" = "live" ]; then "$PY" -B verify_hashlog.py --no-ots > "$LOGDIR/verify.log" 2>&1
  else "$PY" -B verify_hashlog.py --dryrun --no-ots > "$LOGDIR/verify.log" 2>&1; fi
}
trap finish EXIT
STAGE=fetch_usgs
nice "$PY" -B fetch_usgs.py --mode "$MODE" --date "$DATE" > "$LOGDIR/$STAGE.log" 2>&1 || exit $?
STAGE=fetch_gridmet
nice "$PY" -B fetch_gridmet.py --mode "$MODE" --date "$DATE" > "$LOGDIR/$STAGE.log" 2>&1 || echo "gridMET fetch failed: NaN weather today"
STAGE=issue
nice "$PY" -B issue.py --mode "$MODE" --date "$DATE" > "$LOGDIR/$STAGE.log" 2>&1 || exit $?
STAGE=done
