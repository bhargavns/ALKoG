# Foraging communication

Can environmental pressure make two learning agents develop communication that
looks like language, with words that combine?  A **scout** in a tower sees the
whole map; a **forager** on the ground sees only the cells around it.  They learn
together with recurrent PPO from a shared team reward, and the scout's only way
to help is to talk.

| study | question | status |
|---|---|---|
| [watchtower](foraging/watchtower/README.md) | Does a scout that sees everything learn to describe the scene? | done: the scout learned only "go to glade X" |
| [errand](foraging/errand/README.md) | Does splitting knowledge (the forager has a goal and knows which animals are fed, the scout knows where everything is) force descriptive messages? | pilots done; three-seed study running |

## Layout

```
foraging/
  metrics.py          rank correlation and edit distance for topographic similarity
  watchtower/         env.py, agents.py, train.py, analysis.py, README.md
  errand/             same files; reuses the watchtower message encoder/decoder
tests/                rule checks for both worlds
scripts/
  watchtower_study.sh       the watchtower study, all runs in parallel
  slurm/errand_private2k.sh the errand private-rules study on one Slurm GPU node
results/              summaries, logs and training curves (checkpoints stay local)
```

## Setup

Python 3.11, PyTorch and NumPy.  The worlds run as batches of tensors on the GPU.

```bash
conda create -n foraging python=3.11 -y && conda activate foraging
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m pytest -q
```

## Running

From the repository root:

```bash
python -m foraging.errand.train --rules private --seed 0 --updates 2000 --output results/errand/X/private_0
python -m foraging.errand.train --rules private --silent --seed 0 --updates 2000 --output results/errand/X/private_silent_0
python -m foraging.errand.analysis --summarise results/errand/X
```

On a Slurm cluster: edit the `cd` line in `scripts/slurm/errand_private2k.sh`, then
`mkdir -p logs_errand && sbatch scripts/slurm/errand_private2k.sh`.
