# Audit of the five completed downstream test looks (2026-11-09)

**Why:** the D4 pre-test review showed that the shared downstream guard (`sentinel/models/downstream/testlook.py`)
would allow a second look if its marker files were deleted. This audit checks that each completed look happened
exactly once. Read-only; no hashed file touched.

| App (look order) | Runner log (local) | Claim → saved → scored (marker contents = mtimes = ctimes) | Registry test rows | Duplicate (exp, model, metric, split) keys | Rescore rows | Hosts | Registry window (UTC) | Prereg snapshots |
|---|---|---|---|---|---|---|---|---|
| D2 toxinwatch | 20:22:23–20:25:25 | 20:22:52 → 20:23:41 → 20:25:25 (pid 1088222) | 86 | 0 | 0 | comet | 01:23:44–01:25:25 | 1 |
| D5 biointegrity | 20:25:25–20:32:30 | 20:25:31 → 20:28:48 → 20:32:29 (pid 1089855) | 168 | 0 | 0 | comet | 01:28:50–01:32:29 | 1 |
| D1 safeswim | 20:32:30–20:33:42 | 20:32:35 → 20:32:39 → 20:33:41 (pid 1091854) | 36 | 0 | 0 | comet | 01:32:42–01:33:41 | 1 |
| D1n d1n_beaches | 20:33:42–20:40:14 | 20:33:48 → 20:34:31 → 20:40:13 (pid 1092754) | 52 | 0 | 0 | comet | 01:34:44–01:40:13 | 1 |
| D6 musselguard | 20:40:14–22:12:37 | 20:40:17 → 20:48:30 → 22:12:36 (pid 1094997) | 234 | 0 | 0 | comet | 01:48:59–03:12:36 | 1 |

Checks: (i) every registry test row of each app lies inside its runner window (UTC = EST + 5 h) and was written by
the single host; (ii) no (experiment, model, metric, split) key appears twice — a second scoring pass would duplicate
every key; (iii) marker JSON timestamps equal the files' mtimes and inode change times (no delete/re-create);
(iv) each `test_look/` directory holds exactly the five expected files; the sealed `predictions.parquet` matches the
sha256 in `PREDICTIONS_SAVED.json`; (v) no `downstream` registry row outside these windows references a test split or
a `test_look` experiment; (vi) exactly one runner log per app (`$SENTINEL_RUNS/downstream/test_look_logs/`) and one
prereg snapshot row per app. **Conclusion: exactly one look per app; no evidence of a second look or rescore.**
The guard weakness is real for future looks; D4/D7 use dedicated hardened modules (registry-anchored claim/seal).
