#!/bin/zsh
# #597 validation runs (e2 + st1 on), strictly sequential via run_heavy.sh; stops at the first failure.
cd /Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/diag-v21tr1
export REPO=$PWD LOG_DIR=output/st1
D=/Users/rfoley/darkhunter/pop/dark-hunter_pop; H=scripts/jobs/run_heavy.sh
$H reinject_st1 scripts/reinject_noise_colour_572.py --out output/st1/reinject_st1 --enable-type-model --n-realizations 2 --workers 6 || exit 1
$H singles_st1 scripts/single_star_noise_control.py simulate --parent-dir $D/data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj \
  --out output/st1/singles_st1_r16 --g-max 13.5 --n-per-bin 6000 --realizations 16 --seed 572 --enable-type-model --workers 6 || exit 1
echo DONE > output/st1/chain.done
