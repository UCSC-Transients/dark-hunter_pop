#!/bin/zsh
# #595 ss1 validation runs, strictly sequential via run_heavy.sh (nice, memory guard); stops at the first failure.
cd /Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/diag-v21tr1
export REPO=$PWD LOG_DIR=output/ss1
D=/Users/rfoley/darkhunter/pop/dark-hunter_pop; H=scripts/jobs/run_heavy.sh
$H reinject_ss1 scripts/reinject_noise_colour_572.py --out output/ss1/reinject_ss1 --enable-source-scale --n-realizations 2 --workers 6 || exit 1
$H singles_ss1 scripts/single_star_noise_control.py simulate --parent-dir $D/data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj \
  --out output/ss1/singles_ss1_r16 --g-max 13.5 --n-per-bin 6000 --realizations 16 --seed 572 --enable-source-scale --workers 6 || exit 1
echo DONE > output/ss1/chain.done
