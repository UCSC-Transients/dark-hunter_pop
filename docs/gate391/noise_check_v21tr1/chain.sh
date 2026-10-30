#!/bin/zsh
# #587 noise-model go/no-go chain (after gen 50 is assembled). Strictly sequential via run_heavy.sh (nice, memory guard);
# stops at the first failure. (a) v3 θ on gens 47-50 + the v21tr1 spurious template, WD off; (b) luminous + spurious refit.
cd /Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/noise-check-587
export REPO=$PWD LOG_DIR=output/nc587
D=/Users/rfoley/darkhunter/pop/dark-hunter_pop; V=$D/output/proposal_set/v21tr1; C=config/population; G=docs/gate391/noise_check_v21tr1
V3=/Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/rung3-v3/output/v3/fit; S3=docs/gate391/rung3_v3/fit/summary.json
COMMON=(--fit-config $C/rung3_noise_check_587.yaml --cache $V/weights_gen47_50.npz --artifact-dir $V
 --artifacts single_gen47.h5 single_gen48.h5 single_gen49.h5 single_gen50.h5 --no-co
 --calibration $V/spur/spurious_calibration --fragment $C/proposal_set_single50.yaml --gate390 $D/output/gate390
 --parent-dir $D/data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj
 --real-snapshot $D/data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/query.ecsv
 --real-input-columns $D/data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns
 --accel-snapshot $D/data/dr3/gaia_snapshots/20261009T205231Z_nss_acceleration_astro --init-from $S3 --fix ln_R_wd=-50 ln_R_dark=-50)
H=scripts/jobs/run_heavy.sh; O=output/nc587
PL=(--edges-in $G/v3_rank/edges.json --boot-2d)
[ -f $V/weights_gen47_50.npz ] || { $H cache47_50 $G/build_cache_47_50.py || exit 1; }
$H fit_a scripts/rung3_joint_391.py $COMMON --eval-only --out-dir $O/a_v3theta || exit 1
$H plots_a scripts/rung3_plots_391.py --dump $O/a_v3theta/plot_dump.npz --summary $O/a_v3theta/summary.json --out-dir $G/a_v3theta $PL \
  --title "#587 (a): v3 θ on gens 47-50 (colour_v2.1+tr1), luminous + spurious (WD off)" \
  --intro "Luminous gens 47-50 (colour_v2.1+tr1) weighted at the rung-3 v3 best-fit θ (N_s as v3), refit §12.16 spurious template (v21tr1), WD off. " || exit 1
$H fit_b scripts/rung3_joint_391.py $COMMON --n-starts 8 --out-dir $O/b_refit || exit 1
$H plots_b scripts/rung3_plots_391.py --dump $O/b_refit/plot_dump.npz --summary $O/b_refit/summary.json --out-dir $G/b_refit $PL \
  --title "#587 (b): luminous + spurious refit on gens 47-50 (colour_v2.1+tr1), WD off" \
  --intro "Luminous gens 47-50 (colour_v2.1+tr1) + refit §12.16 spurious template (v21tr1), refit (8 starts), WD off, R_dark = 0. " || exit 1
$H compare scripts/noise_check_compare_587.py --edges $G/v3_rank/edges.json \
  --run "v3=$V3/plot_dump.npz:$G/v3_rank" --run "v3 L+S=$V3/plot_dump.npz:$G/v3ls_rank:wd" \
  --run "(a) v3θ new=$O/a_v3theta/plot_dump.npz:$G/a_v3theta" --run "(b) refit new=$O/b_refit/plot_dump.npz:$G/b_refit" --out-dir $G/compare || exit 1
echo DONE > $O/chain.done
