#!/bin/bash
# Errand study, private rules: 3 seeds x (speaking, silent), 2,000 updates each, then the analysis.
# Six training processes share one GPU; each keeps about one CPU core busy.
#
# Before submitting, on a Blanca LOGIN node, from the repository root:
#   module load slurm/blanca
#   git checkout errand_setup && git pull
#   mkdir -p logs_errand
#   sbatch scripts/slurm/errand_private2k.sh                 # full study
#   sbatch --time=00:30:00 --export=ALL,UPDATES=50 scripts/slurm/errand_private2k.sh   # short test first
#
# Training output goes to /scratch/alpine (CURC: /projects is "Not for job output"); the logs,
# curves, final checkpoints and summary are copied back to results/ in the repository at the end.
#
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --account=blanca-clearlab1
#SBATCH --qos=blanca-clearlab1
#SBATCH --partition=blanca-clearlab1
#SBATCH --nodelist=bgpu-g4-u30
#SBATCH --gres=gpu:1
#SBATCH --job-name=errand-private2k
#SBATCH --output=/projects/%u/ALKoG/logs_errand/private2k-%j.out
#SBATCH --error=/projects/%u/ALKoG/logs_errand/private2k-%j.err

set -u
UPDATES=${UPDATES:-2000}
USE_MPS=${USE_MPS:-0}          # NVIDIA MPS is not documented by CURC; off unless asked for
REPO=/projects/$USER/ALKoG
WORK=/scratch/alpine/$USER/errand_private2k_$SLURM_JOB_ID
DEST=$REPO/results/errand/private2k_h100_$SLURM_JOB_ID

stamp() { echo "[$(date '+%F %T')] $*"; }
stamp "job $SLURM_JOB_ID on $(hostname), CPUs: $(nproc), updates: $UPDATES, MPS: $USE_MPS"
nvidia-smi

cd "$REPO" || exit 1
module purge
module load anaconda
conda activate morphorag
git log -1 --oneline
mkdir -p "$WORK" "$DEST" || exit 1

MPS=0
cleanup() { [ $MPS = 1 ] && echo quit | nvidia-cuda-mps-control; }
trap cleanup EXIT
if [ "$USE_MPS" = 1 ]; then
  export CUDA_MPS_PIPE_DIRECTORY=$SLURM_SCRATCH/mps_pipe CUDA_MPS_LOG_DIRECTORY=$SLURM_SCRATCH/mps_log
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  if nvidia-cuda-mps-control -d; then MPS=1; stamp "MPS started"; else stamp "MPS could not start; continuing without it"; fi
fi

export OMP_NUM_THREADS=1
# 6 processes x 0.15 of the GPU's memory (6 GB each on the 40 GB MIG slice bgpu-g4-u30 gives).
pids=(); names=()
for s in 0 1 2; do
  for kind in private private_silent; do
    flag=""; [ $kind = private_silent ] && flag="--silent"
    python -m foraging.errand.train --rules private $flag --seed $s --updates $UPDATES --log 25 \
        --mem-fraction 0.15 --threads 1 --output $WORK/${kind}_$s > $WORK/${kind}_$s.log 2>&1 &
    pids+=($!); names+=(${kind}_$s); stamp "started ${kind}_$s (pid $!)"
  done
done

# Progress every 10 minutes: latest update and return of each run.
(
  while sleep 600; do
    line="[$(date '+%F %T')]"
    for n in "${names[@]}"; do
      last=$(grep '"update"' $WORK/$n.log 2>/dev/null | tail -1 | sed -E 's/.*"update": ([0-9]+), "ret": (-?[0-9.]+).*/\1 \2/')
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

# Analyses whatever runs finished (pairs each speaking run with its silent run).
stamp "analysis started"
python -m foraging.errand.analysis --summarise $WORK > $WORK/summary.txt 2>&1; cat $WORK/summary.txt

# Copy the small files and the final checkpoints back; the scratch copy stays (purged after 90 days).
(cd $WORK && find . \( -name '*.log' -o -name '*.json' -o -name '*.txt' -o -name checkpoint.pt \) -print0 \
   | xargs -0 cp --parents -t "$DEST")
stamp "copied results to $DEST; full output in $WORK"
