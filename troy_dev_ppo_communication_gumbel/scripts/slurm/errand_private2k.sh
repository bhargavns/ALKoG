#!/bin/bash
# Errand study, private rules: 3 seeds x (speaking, silent), 2,000 updates each, then the analysis.
# One H100 holds all six runs at once: each needs about 8 GB, and training is limited by the CPU
# launching small GPU kernels, so six separate processes keep the GPU far busier than one.
#
#   mkdir -p logs_errand && sbatch scripts/slurm/errand_private2k.sh   (from troy_dev_ppo_communication_gumbel/)
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --account=blanca-clearlab1
#SBATCH --qos=blanca-clearlab1
#SBATCH --partition=blanca-clearlab1
#SBATCH --gres=gpu:1
#SBATCH --job-name=errand-private2k
#SBATCH --output=logs_errand/private2k-%j.out
#SBATCH --error=logs_errand/private2k-%j.err

set -u
nvidia-smi

# Change to wherever the repository is cloned on the cluster.
cd /projects/bhsh5847/ALKoG/troy_dev_ppo_communication_gumbel
module load anaconda
conda activate alkog

git status
git log -1 --oneline

O=results/errand/private2k_h100
mkdir -p "$O"
export OMP_NUM_THREADS=2

# 6 processes x 0.15 of 80 GB = 72 GB, leaving headroom.
pids=()
for s in 0 1 2; do
  python -m experiments.errand.train --rules private --seed $s --updates 2000 --log 25 \
      --mem-fraction 0.15 --threads 2 --output $O/private_$s > $O/private_$s.log 2>&1 &
  pids+=($!)
  python -m experiments.errand.train --rules private --silent --seed $s --updates 2000 --log 25 \
      --mem-fraction 0.15 --threads 2 --output $O/private_silent_$s > $O/private_silent_$s.log 2>&1 &
  pids+=($!)
done

failed=0
for p in "${pids[@]}"; do wait $p || failed=1; done
nvidia-smi
if [ $failed -ne 0 ]; then
  echo "at least one run failed; see $O/*.log"
  grep -l Traceback $O/*.log
fi

# Analyses whatever runs finished (pairs each speaking run with its silent run).
python -m experiments.errand.analysis --summarise $O | tee $O/summary.txt
