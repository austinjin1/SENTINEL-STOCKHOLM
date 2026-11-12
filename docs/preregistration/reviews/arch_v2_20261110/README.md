# Architecture round v2: independent pre-stamp reviews (verbatim)

Saved by the lead (AI-assisted). Index last written 2026-11-11 13:15 EST (first written 2026-11-10 14:53 EST; the M4 line of the file
list was wrong until 2026-11-11 and is corrected below).

Each JSON file is the unedited structured output of one independent review agent that did not write the
file it reviewed. `review` = first-stage reviewer (one lens); `verify` = second-stage adversarial verifier,
which re-checked every blocking/should-fix finding against the repository. A reviewer finding that has no
verifier verdict is **unverified** and is not a lead ruling. Lead rulings live in the `*_lead_rulings.md` files.

| Model | Stage | Lens | File | Workflow run | sha256 the agent read | Items | By severity | Recommendation |
|---|---|---|---|---|---|---|---|---|
| M1dev | review |  | `M1dev_review.json` | wf_349862cd-ad0 | `031ddddc8bb8fecb4788ae639bc358f9c28ad042b08cb4ea3d8fd9b1ab2fcd92` | 16 | {'blocking': 4, 'note': 2, 'should_fix': 10} |  |
| M1p2 | review | code | `M1p2_review_code.json` | wf_c4ead573-356 | `b17f3ec67af92809f2c2c2b31dca210275d7712c45d9a7a3d8e9930e788a63f4` | 17 | {'blocking': 3, 'note': 6, 'should_fix': 8} | REVISE |
| M1p2 | review | integrity | `M1p2_review_integrity.json` | wf_c4ead573-356 | `b17f3ec67af92809f2c2c2b31dca210275d7712c45d9a7a3d8e9930e788a63f4` | 14 | {'blocking': 4, 'note': 2, 'should_fix': 8} | REVISE |
| M1p2 | verify |  | `M1p2_verify.json` | wf_c4ead573-356 | `b17f3ec67af92809f2c2c2b31dca210275d7712c45d9a7a3d8e9930e788a63f4` | 31 | {'?': 31} | REVISE |
| M2 | review | consistency | `M2_review_consistency.json` | wf_349862cd-ad0 | `5001373da42652926c6ea4a84028ac46628aa0e6d8b6e99380244417aeb8df8a` | 19 | {'note': 5, 'should_fix': 14} |  |
| M2 | review | leakage | `M2_review_leakage.json` | wf_349862cd-ad0 | `5001373da42652926c6ea4a84028ac46628aa0e6d8b6e99380244417aeb8df8a` | 16 | {'note': 6, 'should_fix': 10} |  |
| M2 | review | stats | `M2_review_stats.json` | wf_349862cd-ad0 | `5001373da42652926c6ea4a84028ac46628aa0e6d8b6e99380244417aeb8df8a` | 17 | {'note': 2, 'should_fix': 15} |  |
| M2 | verify |  | `M2_verify.json` | wf_349862cd-ad0 | `` | 28 | {'should_fix': 28} | stamp_with_rulings |
| M3 | review | design | `M3_review_design.json` | wf_5a96bec9-d7b | `022841f3e94498efd94b4258449730472f790c5c2ef800556350cfe55d82eb6a` | 20 | {'note': 5, 'should_fix': 15} |  |
| M3 | review | integrity | `M3_review_integrity.json` | wf_5a96bec9-d7b | `022841f3e94498efd94b4258449730472f790c5c2ef800556350cfe55d82eb6a` | 14 | {'note': 3, 'should_fix': 11} |  |
| M3 | verify |  | `M3_verify.json` | wf_5a96bec9-d7b | `022841f3e94498efd94b4258449730472f790c5c2ef800556350cfe55d82eb6a` | 23 | {'should_fix': 23} | stamp_with_rulings |
| M4 | review | consistency | `M4_review_consistency.json` | wf_349862cd-ad0 | `b9326d49aca711f0ceeec1362c21883030a94f83dd6971a2db15cba1babc9a9e` | 15 | {'blocking': 4, 'note': 1, 'should_fix': 10} |  |
| M4 | review | leakage | `M4_review_leakage.json` | wf_349862cd-ad0 | `b9326d49aca711f0ceeec1362c21883030a94f83dd6971a2db15cba1babc9a9e` | 14 | {'blocking': 5, 'note': 2, 'should_fix': 7} |  |
| M4 | review | stats | `M4_review_stats.json` | wf_349862cd-ad0 | `b9326d49aca711f0ceeec1362c21883030a94f83dd6971a2db15cba1babc9a9e` | 16 | {'blocking': 3, 'note': 4, 'should_fix': 9} |  |
| M4 | verify |  | `M4_verify.json` | wf_349862cd-ad0 | `` | 27 | {'blocking': 5, 'note': 3, 'should_fix': 19} | revise_then_stamp |
| M6 | review | design | `M6_review_design.json` | wf_5a96bec9-d7b | `a65e6eb578f5bebe001dada100cd8ea49f2383b80987cd3927014f759696e656` | 29 | {'note': 6, 'should_fix': 23} |  |
| M6 | review | integrity | `M6_review_integrity.json` | wf_5a96bec9-d7b | `a65e6eb578f5bebe001dada100cd8ea49f2383b80987cd3927014f759696e656` | 17 | {'note': 5, 'should_fix': 12} |  |
| M6 | verify |  | `M6_verify.json` | wf_5a96bec9-d7b | `a65e6eb578f5bebe001dada100cd8ea49f2383b80987cd3927014f759696e656` | 27 | {'should_fix': 27} | stamp_with_rulings |
| M7 | review | design | `M7_review_design.json` | wf_5a96bec9-d7b | `2b369018d2fc2a340594790756b3edbef3b1de28a38cf6073eb5908111908a56` | 14 | {'blocking': 2, 'note': 1, 'should_fix': 11} |  |
| M7 | review | integrity | `M7_review_integrity.json` | wf_5a96bec9-d7b | `2b369018d2fc2a340594790756b3edbef3b1de28a38cf6073eb5908111908a56` | 13 | {'blocking': 2, 'note': 2, 'should_fix': 9} |  |
| M7 | verify |  | `M7_verify.json` | wf_5a96bec9-d7b | `2b369018d2fc2a340594790756b3edbef3b1de28a38cf6073eb5908111908a56` | 17 | {'blocking': 1, 'note': 1, 'should_fix': 15} | revise_then_stamp |
| M4 | verify | arch v2 | `M4v2_verify.json` | (single reviewer agent) | `fb52cd4b072fa19c8c1e19b9fe1374a0e8abb71d9492eff764f2034cb26d293b` |  |  | revise_then_stamp |
| M4 | review | guard code | `M4v2_guard_review.json` + `.md` | (single reviewer agent) | guard code state of 2026-11-10 |  |  | fix_then_rereview |
| M6 | review | entry 1 | `M6e1_review.json` + `.md` | (single reviewer agent) | `a65e6eb578f5bebe001dada100cd8ea49f2383b80987cd3927014f759696e656` |  |  | see file |
| M7 | review | stamp package | `M7v2_stamp_review.json` + `.md` | (single reviewer agent) | `2b369018d2fc2a340594790756b3edbef3b1de28a38cf6073eb5908111908a56` | 15 findings + 45 verifier items + 5 open points | {'see review.json'} | stamp_with_lead_rulings |

Files reviewed and their sha256 when this index was written:

- M1dev: `docs/preregistration/sonde_arch_v2_deviations.md` `031ddddc8bb8fecb4788ae639bc358f9c28ad042b08cb4ea3d8fd9b1ab2fcd92`
- M1p2: `docs/preregistration/sonde_arch_v2_deviations_part2.md` `b17f3ec67af92809f2c2c2b31dca210275d7712c45d9a7a3d8e9930e788a63f4`
- M2: `docs/preregistration/orbit_arch_v2_amendment.md` `5001373da42652926c6ea4a84028ac46628aa0e6d8b6e99380244417aeb8df8a`
- M3: `docs/preregistration/microbe_arch_v2_amendment.md` `022841f3e94498efd94b4258449730472f790c5c2ef800556350cfe55d82eb6a`
- M4: the three `M4_review_*.json` files and `M4_verify.json` read the v1 amendment, which is now
  `docs/preregistration/chemtox_arch_v2_amendment_v1_unstamped.md`
  `b9326d49aca711f0ceeec1362c21883030a94f83dd6971a2db15cba1babc9a9e`. The stamped v2 amendment is
  `docs/preregistration/chemtox_arch_v2_amendment.md`
  `fb52cd4b072fa19c8c1e19b9fe1374a0e8abb71d9492eff764f2034cb26d293b`, which `M4v2_verify.json` read.
  (The hash `4c4e9ba9...` printed here until 2026-11-11 matched no file; it was an index error, and no
  ruling ever rested on it.)
- M6: `docs/preregistration/hydromet_arch_v2_amendment.md` `a65e6eb578f5bebe001dada100cd8ea49f2383b80987cd3927014f759696e656`
- M7: `docs/preregistration/lite_arch_v2_amendment.md` `2b369018d2fc2a340594790756b3edbef3b1de28a38cf6073eb5908111908a56`
