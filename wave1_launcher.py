"""
wave1_launcher.py

Generates all Wave 1 SW permutations.
Learning curve handled by wave3_launcher.py after Wave 2 completes.

Usage (on Athena, with venv active):
    python wave1_launcher.py --config analysis_config.json [--dry-run]

--dry-run: generate job configs and print sbatch command without submitting.

Job configs written to:
    {jobs_dir}/sw/job_{N:04d}.json

Results written per job by train.py:
    {results_dir}/sw/{arch}__{encoder}__{weights}__{split}__{seed}/result.json

Cluster: VCU Athena H100 partition
    athena531: 8 GPUs
    athena532: 4 GPUs
    athena533: 4 GPUs
    athena534: 4 GPUs (draining — may be unavailable)
    Total stable: 16-20 GPUs
    Concurrency limit: %60

Wave 1 SW job count:
    24 arch/encoder × 16 weights × 3 splits × 5 seeds = 5,760 jobs
    ~34 hours wall time at 60 concurrent on 20 H100s
"""

from __future__ import annotations

import json
import subprocess
import sys
from itertools import product
from pathlib import Path
import argparse


# ─── Config loader ────────────────────────────────────────────────────────────

def load_analysis_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"analysis_config not found: {p}")
    with open(p) as f:
        return json.load(f)


# ─── Permutation builder ──────────────────────────────────────────────────────

def build_sweep_jobs(cfg: dict) -> list[dict]:
    """
    Stage SW — full arch/encoder/weight sweep at n=30.
    5,760 jobs: 24 arch/encoder × 16 weights × 3 splits × 5 seeds
    """
    sw     = cfg['sweep']
    splits = cfg['splits']
    train  = cfg['training']
    output = cfg['output']

    jobs = []
    for arch_cfg, weights, (train_pct, val_pct), seed in product(
        sw['architectures'],
        sw['class_weights'],
        [tuple(r) for r in splits['ratios']],
        splits['seeds'],
    ):
        w_str  = '_'.join(str(int(w)) if w == int(w) else str(w) for w in weights)
        run_id = (
            f"sw__{arch_cfg['arch']}__{arch_cfg['encoder']}__"
            f"cw{w_str}__{train_pct}_{val_pct}__seed{seed}"
        )
        jobs.append({
            'run_id':        run_id,
            'stage':         'sweep',
            'wave':          1,
            'images_dir':    cfg['dataset']['images_dir'],
            'mag':           cfg['dataset']['mag'],
            'ctrl_prefix':   cfg['dataset']['ctrl_prefix'],
            'regen_prefix':  cfg['dataset']['regen_prefix'],
            'n_images':      sw['dataset_size'],
            'val_fraction':  round(val_pct / 100, 2),
            'train_pct':     train_pct,
            'val_pct':       val_pct,
            'seed':          seed,
            'arch':          arch_cfg['arch'],
            'encoder':       arch_cfg['encoder'],
            'class_weights': weights,
            'epochs':        train['epochs'],
            'batch_size':    train['batch_size'],
            'augmentation':  False,
            'learning_rate': train['learning_rate'],
            'weight_decay':  train['weight_decay'],
            'output': {
                'results_dir': str(Path(output['results_dir']) / 'sw'),
                'models_dir':  str(Path(output['models_dir'])  / 'sw'),
                'logs_dir':    str(Path(output['logs_dir'])    / 'sw'),
            }
        })
    return jobs


# ─── Job config writer ────────────────────────────────────────────────────────

def write_job_configs(jobs: list[dict], jobs_dir: Path) -> list[Path]:
    stage_dir = jobs_dir / 'sw'
    stage_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, job in enumerate(jobs):
        path = stage_dir / f"job_{i:04d}.json"
        with open(path, 'w') as f:
            json.dump(job, f, indent=2)
        written.append(path)
    return written


# ─── SLURM script writer ──────────────────────────────────────────────────────

def write_sbatch(
    n_jobs:   int,
    jobs_dir: Path,
    cfg:      dict,
    out_path: Path,
) -> Path:
    slurm    = cfg['slurm']
    venv     = slurm['venv']
    repo     = str(Path(venv).parent)
    logs_dir = cfg['output']['logs_dir']

    script = f"""#!/bin/bash
#SBATCH --job-name=deepaxon_wave1_sw
#SBATCH --partition={slurm['partition']}
#SBATCH --gres={slurm['gres']}
#SBATCH --nodes={slurm['nodes']}
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --time={slurm['time']}
#SBATCH --mem={slurm['mem']}
#SBATCH --array=0-{n_jobs - 1}%{slurm['max_concurrent']}
#SBATCH --output={logs_dir}/sw/%A_%a.out
#SBATCH --error={logs_dir}/sw/%A_%a.err
#SBATCH --mail-type={slurm['mail_type']}
#SBATCH --mail-user={slurm['mail_user']}

source {venv}/bin/activate
cd {repo}

JOB_CONFIG={jobs_dir}/sw/job_$(printf '%04d' $SLURM_ARRAY_TASK_ID).json

echo "Wave 1 [SW] job $SLURM_ARRAY_TASK_ID — config: $JOB_CONFIG"
python -m train --config "$JOB_CONFIG"
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script)
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepAxon Wave 1 launcher")
    parser.add_argument('--config',  default='analysis_config.json')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate configs and print command without submitting')
    args = parser.parse_args()

    cfg      = load_analysis_config(args.config)
    jobs_dir = Path(cfg['output']['jobs_dir'])
    out_dir  = jobs_dir.parent

    sw_jobs = build_sweep_jobs(cfg)
    print(f"Wave 1 SW: {len(sw_jobs)} jobs")

    write_job_configs(sw_jobs, jobs_dir)
    sbatch_sw = out_dir / 'wave1_sw.sbatch'
    write_sbatch(len(sw_jobs), jobs_dir, cfg, sbatch_sw)
    print(f"SW sbatch written → {sbatch_sw}")

    if args.dry_run:
        print("--dry-run: not submitting.")
        return

    r = subprocess.run(['sbatch', str(sbatch_sw)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"sbatch FAILED:\n{r.stderr}", file=sys.stderr)
        sys.exit(1)

    sw_job_id = r.stdout.strip().split()[-1]
    id_file   = out_dir / 'wave1_job_ids.json'
    with open(id_file, 'w') as f:
        json.dump({'sw': sw_job_id}, f, indent=2)

    print(f"Wave 1 SW submitted — SLURM array ID: {sw_job_id}")
    print(f"Job ID saved → {id_file}")
    print(f"You will be emailed at {cfg['slurm']['mail_user']} on completion/failure.")


if __name__ == '__main__':
    main()
