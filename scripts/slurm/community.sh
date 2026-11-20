#!/bin/bash
# Community study (foraging/community/README.md): six runs at a time on one GPU, then the analysis.
#   BATCH=A  speak_{0,1,2}, silent_{0,1,2}              (step 1)
#   BATCH=B  turnover_{0,1,2}, generations_{0,1,2}      (step 2; needs batch A's silent runs for the analysis)
#   BATCH=C  nobottleneck_{0,1,2}                        (step 2 control)
#
# On a Blanca LOGIN node, from the repository root:
#   module load slurm/blanca
#   git checkout errand_setup && git pull
#   mkdir -p logs_errand
#   sbatch --export=ALL,BATCH=A scripts/slurm/community.sh
#   sbatch --export=ALL,BATCH=B --dependency=afterok:<job id of A> scripts/slurm/community.sh
#   sbatch --export=ALL,BATCH=C --dependency=afterok:<job id of B> scripts/slurm/community.sh
#   short test first:  sbatch --time=00:40:00 --export=ALL,BATCH=A,UPDATES=30,TAG=test scripts/slurm/community.sh
#
# Training output goes to /scratch/alpine (CURC: /projects is "Not for job output"); the logs, curves,
# final checkpoints and summary are copied back to results/community in the repository at the end.
#
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --account=blanca-clearlab1
#SBATCH --qos=blanca-clearlab1
#SBATCH --partition=blanca-clearlab1
#SBATCH --nodelist=bgpu-g4-u30
#SBATCH --gres=gpu:1
#SBATCH --job-name=community
#SBATCH --output=/projects/%u/ALKoG/logs_errand/community-%j.out
#SBATCH --error=/projects/%u/ALKoG/logs_errand/community-%j.err

set -u
BATCH=${BATCH:-A}
UPDATES=${UPDATES:-2000}
TAG=${TAG:-main}
REPO=/projects/$USER/ALKoG
WORK=/scratch/alpine/$USER/community_$TAG
DEST=$REPO/results/community_$TAG

stamp() { echo "[$(date '+%F %T')] $*"; }
stamp "job $SLURM_JOB_ID on $(hostname), CPUs: $(nproc), batch: $BATCH, updates: $UPDATES"
nvidia-smi

cd "$REPO" || exit 1
module purge
module load anaconda
conda activate morphorag
git log -1 --oneline
mkdir -p "$WORK" "$DEST" || exit 1

case $BATCH in
  A) RUNS="speak:  silent:--silent" ;;
  B) RUNS="turnover:--turnover=100 generations:--turnover=100,--generation=500,--bottleneck=512" ;;
  C) RUNS="nobottleneck:--turnover=100,--generation=500,--bottleneck=0" ;;
  *) stamp "unknown BATCH $BATCH"; exit 1 ;;
esac

export OMP_NUM_THREADS=1
pids=(); names=()
for s in 0 1 2; do
  for spec in $RUNS; do
    kind=${spec%%:*}; flags=${spec#*:}; flags=${flags//,/ }
    python -m foraging.community.train --seed $s --updates $UPDATES --log 25 $flags \
        --mem-fraction 0.15 --threads 1 --output $WORK/${kind}_$s > $WORK/${kind}_$s.log 2>&1 &
    pids+=($!); names+=(${kind}_$s); stamp "started ${kind}_$s ($flags) pid $!"
  done
done

# Progress every 10 minutes: latest update and return of each run.
(
  while sleep 600; do
    line="[$(date '+%F %T')]"
    for n in "${names[@]}"; do
      last=$(grep '"update"' $WORK/$n.log 2>/dev/null | grep '"ret"' | tail -1 | sed -E 's/.*"update": ([0-9]+), "ret": (-?[0-9.]+).*/\1 \2/')
      line="$line  $n: ${last:-starting}"
    done
    echo "$line"
  done
) &
progress=$!

failed=0
for i in "${!pids[@]}"; do
  if wait ${pids[$i]}; then stamp "finished ${names[$i]}"; else stamp "FAILED ${names[$i]} (see its log)"; failed=1; fi
done
kill $progress 2>/dev/null
nvidia-smi
[ $failed -ne 0 ] && grep -l Traceback $WORK/*.log

# Analyses every finished run in the folder (pairs each run with silent_<seed>).
stamp "analysis started"
python -m foraging.community.analysis --summarise $WORK > $WORK/summary.txt 2>&1; cat $WORK/summary.txt

(cd $WORK && find . \( -name '*.log' -o -name '*.json' -o -name '*.txt' -o -name checkpoint.pt \) -print0 \
   | xargs -0 cp --parents -t "$DEST")
stamp "copied results to $DEST; full output in $WORK"
