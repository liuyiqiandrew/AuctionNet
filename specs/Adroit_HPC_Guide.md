# Adroit HPC Guide

Quick reference for running jobs on Princeton's Adroit cluster.

---

## 1. Where Am I?

You're either on a **login node** or a **compute node**. Know the difference.

```bash
hostname            # e.g. adroit5 = login node, adroit-11 = compute node
```

- **Login nodes** (adroit1-adroit5): SSH entry point. Light work only — editing, submitting jobs, checking status. Do NOT run heavy computation here.
- **Compute nodes** (adroit-08 through adroit-16, adroit-h11n*): Where SLURM jobs run. You get here by submitting a job or requesting an interactive session.

### Interactive session (for debugging / testing)

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=01:00:00 --partition=class
```

This drops you onto a compute node interactively. Exit with `exit` when done.

---

## 2. Environment Setup

Run every time you start a new session (or add to `~/.bashrc` for auto-loading):

```bash
module purge
module load anaconda3/2025.12
conda activate myenv
```

To verify:
```bash
which python          # should point to your conda env
conda list            # check installed packages
```

### If conda env is missing or broken

```bash
conda create -n myenv python=3.11 -y
conda activate myenv
pip install <packages>
```

---

## 3. SLURM Scripts

### Template (CPU only)

Create one `.slurm` file per project in the project directory:

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --partition=class
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=bl0919@princeton.edu

module purge
module load anaconda3/2025.12
conda activate myenv

cd /scratch/network/bl0919/YOUR_PROJECT

# your commands here
```

### Key parameters

| Parameter | What it does | Guidance |
|---|---|---|
| `--cpus-per-task` | CPU cores | 4 is a good default, up to 32 per node |
| `--mem-per-cpu` | Memory per core | 4G default. Total = cpus x mem-per-cpu |
| `--time` | Wall clock limit | Job killed if exceeded. Overestimate slightly |
| `--partition` | Queue | Use `class` for coursework |
| `--mail-type` | Email notifications | `begin,end,fail` covers all cases |

### Running a Jupyter notebook via SLURM

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1200 \
  --ExecutePreprocessor.kernel_name=python3 \
  --output OUTPUT_executed.ipynb \
  INPUT.ipynb
```

### Converting to PDF

```bash
jupyter nbconvert --to pdf OUTPUT_executed.ipynb
```

If LaTeX errors occur, use tectonic with error tolerance:
```bash
jupyter nbconvert --to pdf \
  --PDFExporter.latex_command='["tectonic", "{filename}", "--print", "-Z", "continue-on-errors"]' \
  OUTPUT_executed.ipynb
```

---

## 4. Submitting and Managing Jobs

```bash
sbatch run_notebook.slurm           # submit job
squeue -u bl0919                    # check running/queued jobs
scancel <JOBID>                     # cancel a job
scancel -u bl0919                   # cancel ALL your jobs
```

---

## 5. Checking Job History and Status

### Current jobs

```bash
squeue -u bl0919
```

### Past jobs (history)

```bash
# Summary of all past jobs
sacct -u bl0919 -X --format=JobID,JobName,Partition,State,Start,End,Elapsed,ExitCode

# Detailed stats for a specific job
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS,NCPUS,TotalCPU

# Filter by date range
sacct -u bl0919 -X --starttime=2026-04-01 --endtime=2026-04-18
```

### Job states

| State | Meaning |
|---|---|
| COMPLETED | Finished successfully (check exit code 0:0) |
| FAILED | Crashed (check slurm-JOBID.out for errors) |
| CANCELLED | You or an admin cancelled it |
| TIMEOUT | Exceeded `--time` limit |
| PENDING | Waiting for resources |
| RUNNING | Currently executing |

### Reading job output

SLURM writes stdout/stderr to `slurm-<JOBID>.out` in the directory you submitted from:

```bash
cat slurm-<JOBID>.out
```

---

## 6. Checking Cluster and Node Info

```bash
sinfo -s                        # partition summary
sinfo -p class -N -l            # detailed node list for class partition
sinfo -p class -t idle          # only show idle (available) nodes
```

### Available partitions on Adroit

| Partition | Time Limit | Use Case |
|---|---|---|
| `class` | 4 days | Coursework (CPU) |
| `all` | 7 days | General (CPU) |
| `gpu` | 7 days | GPU workloads (need `--gres=gpu:1`) |

---

## 7. Common Issues and Fixes

### Kernel crashes / out of memory

**Symptom:** Job state is FAILED, or notebook output is incomplete.

**Diagnose:**
```bash
sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS
cat slurm-<JOBID>.out
```

**Fix:** Increase memory allocation:
```bash
#SBATCH --mem-per-cpu=8G      # double it
# or set total memory directly:
#SBATCH --mem=32G
```

### Job killed for exceeding time limit

**Symptom:** State = TIMEOUT.

**Fix:** Increase `--time`:
```bash
#SBATCH --time=02:00:00       # 2 hours instead of 30 min
```

### Module or package not found

**Symptom:** `ModuleNotFoundError` in slurm output.

**Fix:** Make sure the slurm script loads the right environment:
```bash
module purge
module load anaconda3/2025.12
conda activate myenv
```

Test interactively first:
```bash
salloc --partition=class --time=00:10:00
module purge && module load anaconda3/2025.12 && conda activate myenv
python -c "import pandas; print('ok')"
exit
```

### Job stuck in PENDING

**Diagnose:**
```bash
squeue -u bl0919 -o "%.10i %.12P %.8T %.10r"     # shows REASON column
```

Common reasons:
- `Priority` — other jobs ahead of you. Wait.
- `Resources` — cluster is full. Wait or request fewer resources.
- `QOSMaxJobsPerUserLimit` — too many jobs queued. Cancel some.

### Wrong node / wrong partition

**Check:** If your script says `--partition=gpu` but you don't use `--gres=gpu`, you're wasting GPU nodes. Always use `--partition=class` for CPU-only work.

---

## 8. Workflow Checklist

1. SSH into Adroit
2. `cd /scratch/network/bl0919/YOUR_PROJECT`
3. Edit your `.slurm` script if needed
4. `sbatch your_script.slurm`
5. `squeue -u bl0919` to confirm it's queued/running
6. Wait for email notification (or poll with squeue)
7. `cat slurm-<JOBID>.out` to check output
8. `sacct -j <JOBID>` to verify completion status
