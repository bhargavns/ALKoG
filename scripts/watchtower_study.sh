#!/usr/bin/env bash
# The watchtower study: 3 stages x 3 seeds, each with a silent control, 2,500 updates each, all in parallel.
#   bash scripts/watchtower_study.sh [python]
PY=${1:-python}
OUT=results/watchtower
mkdir -p $OUT
for stage in 1 2 3; do
  for seed in 0 1 2; do
    $PY -m foraging.watchtower.train --stage $stage --seed $seed --updates 2500 --threads 1 --output $OUT/s${stage}_${seed} > $OUT/s${stage}_${seed}.log 2>&1 &
    $PY -m foraging.watchtower.train --stage $stage --seed $seed --updates 2500 --threads 1 --silent --output $OUT/s${stage}_silent_${seed} > $OUT/s${stage}_silent_${seed}.log 2>&1 &
  done
done
wait
$PY -m foraging.watchtower.analysis --summarise $OUT > $OUT/summary.txt 2>&1
cat $OUT/summary.txt
