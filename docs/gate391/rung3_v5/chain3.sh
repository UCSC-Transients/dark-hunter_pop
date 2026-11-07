#!/bin/zsh
# Coordinator order (2026-11-07), continuing chain.sh after gen 1008. One heavy job at a time via run_heavy.sh.
# plots_v5 (already running) -> rank/compare -> suite #623 -> [#623 merged] #621 M1 table -> [#622 merged] #620 real
# M2_app -> R5-T1 bin freeze --dry-run on CO 1006-1008 (the real freeze is decided from its report) -> Student-t nu
# re-calibration on the held-out partition of 1006-1008 -> dark/WD selection-function table + gate 4b on 1006-1008
# -> suites #625, #627 -> gate5a -> suites #629, #631 (no auto-merge) -> own suites #609/#616/#615.
D=/Users/rfoley/darkhunter/pop/dark-hunter_pop; W=/Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees
V5=$W/rung3-v5; cd $V5
V=$D/output/proposal_set/v21tr1_e2st1; Q=$V/queue.log; G=docs/gate391/rung3_v5; H=$D/scripts/jobs/run_heavy.sh; O=output/v5
NC=docs/gate391/noise_check_v21tr1; V4D=$W/rung3-v4/output/v4/fit; V3D=$W/rung3-v3/output/v3/fit
PAR=$D/data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj; F=$W/topup-607/config/population
ARTS=($V/co_gen1006.h5 $V/co_gen1007.h5 $V/co_gen1008.h5)
FRAGS=($F/proposal_set_co_v1_wd1006.yaml $F/proposal_set_co_v1_wd1007.yaml $F/proposal_set_co_v1_dark1008.yaml)
log() { echo "$1 $(date '+%F %T')" >> $Q; }
fail() { log "FAIL $1"; log "QUEUE STOPPED at step $1 (#615 chain3)"; exit 1; }
waitmerged() { for i in {1..90}; do [ "$(gh pr view $1 --json state -q .state 2>/dev/null)" = MERGED ] && return 0; sleep 120; done; return 1; }
suite() {  # suite <pr> <worktree> [nomerge]
  REPO=$2 LOG_DIR=$D/output/suite_prs $H suite_$1 -m pytest -m "unit or physics or api" -q -p no:cacheprovider
  local rc=$? res=$(grep -E "passed|failed" $D/output/suite_prs/suite_$1.log | tail -1)
  if [ $rc -eq 0 ] && [ -z "$3" ]; then gh pr merge $1 --auto --merge >/dev/null 2>&1; log "SUITE PR $1 PASS ($res); auto-merge on"
  elif [ $rc -eq 0 ]; then log "SUITE PR $1 PASS ($res); auto-merge NOT enabled (awaiting Ryan)"
  else log "SUITE PR $1 FAIL rc=$rc ($res); left open"; fi
}
while pgrep -f "run_heavy.sh plots_v5" >/dev/null; do sleep 30; done
grep -q "^exit 0" $O/plots_v5.log || fail plots_v5
export REPO=$V5 LOG_DIR=$O
[ -f $G/v5_rank_v3edges/ranking.json ] || $H rank_v5 scripts/rung3_plots_391.py --dump $O/fit/plot_dump.npz --summary $O/fit/summary.json --out-dir $G/v5_rank_v3edges --edges-in $NC/v3_rank/edges.json --rank-only --boot-2d || fail rank_v5
$H compare_v5 scripts/rung3_v4_compare_603.py --edges $NC/v3_rank/edges.json \
  --run "v3=$V3D/plot_dump.npz:$NC/v3_rank:docs/gate391/rung3_v3/fit/summary.json" \
  --run "v4=$V4D/plot_dump.npz:docs/gate391/rung3_v4/v4_rank_v3edges:docs/gate391/rung3_v4/fit/summary.json" \
  --run "v5=$O/fit/plot_dump.npz:$G/v5_rank_v3edges:$O/fit/summary.json" --out-dir $G/compare || fail compare_v5
log "STEP v5 plots OK"
git -C $D/.worktrees/rung5-t2b fetch -q origin && git -C $D/.worktrees/rung5-t2b merge -q --ff-only origin/feat/rung5-t2b-host-m1 2>/dev/null
log "suite 623 on head $(git -C $D/.worktrees/rung5-t2b rev-parse --short HEAD)"
suite 623 $D/.worktrees/rung5-t2b
if waitmerged 623; then
  git -C $D pull -q --ff-only origin main || fail pull_main_623
  REPO=$D LOG_DIR=$D/output/rung5 $H nss_host_m1_621 scripts/build_nss_host_m1_table_621.py --published-orbits output/gate390/published_orbits.h5 \
    --input-columns data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns \
    --bj-snapshot data/dr3/external_catalogs/20261107T052615Z_bj_distances_nss_hosts --out output/rung5/nss_host_m1_dr3.h5 || fail nss_host_m1_621
  log "STEP nss_host_m1_621 OK"
  if waitmerged 622; then
    git -C $D pull -q --ff-only origin main || fail pull_main_622
    REPO=$D LOG_DIR=$D/output/rung5 $H real_m2app_620 scripts/rung5_real_m2app.py --published-orbits output/gate390/published_orbits.h5 \
      --m1-table output/rung5/nss_host_m1_dr3.h5 --out output/rung5/real_m2app_dr3.h5 || fail real_m2app_620
    log "STEP real_m2app_620 OK"
  else log "real_m2app_620 NOT run: #622 not merged within 3 h"; fi
else log "nss_host_m1_621 / real_m2app_620 NOT run: #623 not merged within 3 h"; fi
git -C $D pull -q --ff-only origin main
REPO=$D LOG_DIR=$D/output/rung5 $H bins_freeze_dry scripts/rung5_freeze_bins.py --dry-run --parent-dir $PAR --artifact $ARTS --fragment $FRAGS \
  --out-dir $D/output/rung5/bins_dr3_1006_1008 && log "STEP bins_freeze_dry OK" || log "FAIL bins_freeze_dry (continuing)"
REPO=$D LOG_DIR=$D/output/rung5 $H student_t_1006_1008 scripts/calibrate_student_t_564.py --artifact $ARTS --fragment $FRAGS \
  --snr-edges 0 20 50 100 1e9 --out $D/output/rung5/student_t_1006_1008.json && log "STEP student_t_1006_1008 OK" || log "FAIL student_t_1006_1008 (continuing)"
REPO=$D LOG_DIR=$D/output/rung5 $H sf_table_1006_1008 scripts/build_co_selection_table.py --parent-dir $PAR --artifact $ARTS --fragment $FRAGS \
  --out-dir $D/output/rung5/sf_1006_1008 && log "STEP sf_table_1006_1008 OK" || log "FAIL sf_table_1006_1008 (continuing)"
suite 625 $D/.worktrees/rung5-t7
suite 627 $D/.worktrees/fl-t3
REPO=$D/.worktrees/rung5-t7 LOG_DIR=$D/output/rung5 $H gate5a scripts/rung5_known_objects_5a.py inject --out $D/output/rung5/gate5a --workers 4 || fail gate5a
REPO=$D/.worktrees/rung5-t7 LOG_DIR=$D/output/rung5 $H gate5a_report scripts/rung5_known_objects_5a.py report --out $D/output/rung5/gate5a || fail gate5a_report
log "STEP gate5a OK"
suite 629 $D/.worktrees/fetch-hosts
suite 631 $D/.worktrees/rung5-nuis nomerge
log "PAUSE after chain3 (#615): own suites #609/#616/#615 run after the close-out commit"
