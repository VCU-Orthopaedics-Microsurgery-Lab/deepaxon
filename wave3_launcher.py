"""
wave3_launcher.py

Learning curve on fully optimized model from Waves 1 and 2.
Must be run after:
    1. Wave 1 SW completes + manual review → winner.json
    2. Wave 2a completes + manual review → winner_aug.json
    3. Wave 2b completes

Purpose:
    Demonstrate performance plateau at n=30, establishing dataset sufficiency.
    Uses fully optimized model — best arch, encoder, class weights, aug params.
    Produces Table 1.

Design:
    Arch/encoder/weights: from winner.json
    Aug params:           from winner_aug.json (optimized from Wave 2a)
    Aug ON:               True — fully optimized model
    Dataset sizes:        10, 20, 30
    Splits:               all 3 from analysis_config.json (70/30, 80/20, 93/7)
    Seeds:                1–5
    Total jobs:           3 sizes × 3 splits × 5 seeds = 45

Usage:
    python wave3_launcher.py --config analysis_config.json [--dry-run]

Output:
    results/lc/  — same directory as Wave 1 LC (different run_ids — no conflict)
    Table 1: learning curve — mean Dice ± SD vs dataset size

Note:
    No .pt files are saved during Wave 3 (save_checkpoint=False).
    After selecting the final winner, retrain once interactively:
        python -m train
    This produces the production .pt file (rb40x_v2) with full metadata.
"""

from __future__ import annotations

import json
import subprocess
import sys
from itertools import product
from pathlib import Path
import argparse


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as f:
        return json.load(f)


def load_winner(winner_path: Path) -> dict:
    if not winner_path.exists():
        raise FileNotFoundError(
            f"winner.json not found: {winner_path}\n"
            f"Run: python aggregator.py --config analysis_config.json --select ..."
        )
    with open(winner_path) as f:
        return json.load(f)


def load_winner_aug(winner_aug_path: Path) -> dict:
    if not winner_aug_path.exists():
        raise FileNotFoundError(
            f"winner_aug.json not found: {winner_aug_path}\n"
            f"Review Wave 2a results and write winner_aug.json manually."
        )
    with open(winner_aug_path) as f:
        return json.load(f)


# ─── Permutation builder ──────────────────────────────────────────────────────

def build_lc_jobs(cfg: dict, winner: dict, winner_aug: dict) -> list[dict]:
    """
    Build Wave 3 learning curve jobs.
    45 jobs: 3 sizes × 3 splits × 5 seeds
    Fully optimized model — aug ON with winner_aug params.
    """
    lc_cfg        = cfg['learning_curve']
    splits        = cfg['splits']
    train         = cfg['training']
    output        = cfg['output']
    ds            = cfg['dataset']
    optimized_aug = winner_aug.get('optimized_params', {})

    jobs = []

    for (train_pct, val_pct), seed, n_images in product(
        [tuple(r) for r in splits['ratios']],
        splits['seeds'],
        lc_cfg['dataset_sizes'],
    ):
        run_id = (
            f"w3lc__{winner['arch']}__{winner['encoder']}__"
            f"{train_pct}_{val_pct}__seed{seed}__n{n_images}"
        )
        jobs.append({
            'run_id':        run_id,
            'stage':         'learning_curve',
            'wave':          3,
            'images_dir':    ds['images_dir'],
            'mag':           ds['mag'],
            'ctrl_prefix':   ds['ctrl_prefix'],
            'regen_prefix':  ds['regen_prefix'],
            'n_images':      n_images,
            'val_fraction':  round(val_pct / 100, 2),
            'train_pct':     train_pct,
            'val_pct':       val_pct,
            'seed':          seed,
            'arch':          winner['arch'],
            'encoder':       winner['encoder'],
            'class_weights': winner['class_weights'],
            'epochs':        train['epochs'],
            'batch_size':    train['batch_size'],
            'augmentation':  True,
            'aug_params':    optimized_aug,
            'save_checkpoint': False,
            'learning_rate': train['learning_rate'],
            'weight_decay':  train['weight_decay'],
            'output': {
                'results_dir': str(Path(output['results_dir']) / 'lc'),
                'models_dir':  str(Path(output['models_dir'])  / 'lc'),
                'logs_dir':    str(Path(output['logs_dir'])    / 'lc'),
            }
        })

    return jobs


# ─── Job config writer ────────────────────────────────────────────────────────

def write_job_configs(jobs: list[dict], jobs_dir: Path) -> list[Path]:
    stage_dir = jobs_dir / 'wave3_lc'
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
#SBATCH --job-name=deepaxon_wave3_lc
#SBATCH --partition={slurm['partition']}
#SBATCH --gres={slurm['gres']}
#SBATCH --nodes={slurm['nodes']}
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --time={slurm['time']}
#SBATCH --mem={slurm['mem']}
#SBATCH --array=0-{n_jobs - 1}%{slurm['max_concurrent']}
#SBATCH --output={logs_dir}/wave3_lc/%A_%a.out
#SBATCH --error={logs_dir}/wave3_lc/%A_%a.err
#SBATCH --mail-type={slurm['mail_type']}
#SBATCH --mail-user={slurm['mail_user']}

source {venv}/bin/activate
cd {repo}

JOB_CONFIG={jobs_dir}/wave3_lc/job_$(printf '%04d' $SLURM_ARRAY_TASK_ID).json

echo "Wave 3 [LC] job $SLURM_ARRAY_TASK_ID — config: $JOB_CONFIG"
python -m train --config "$JOB_CONFIG"
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script)
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepAxon Wave 3 learning curve launcher")
    parser.add_argument('--config',  default='analysis_config.json')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate configs and print command without submitting')
    args = parser.parse_args()

    cfg      = load_config(args.config)
    jobs_dir = Path(cfg['output']['jobs_dir'])
    out_dir  = jobs_dir.parent

    winner_path     = out_dir / 'aggregated' / 'winner.json'
    winner_aug_path = out_dir / 'aggregated' / 'winner_aug.json'

    winner     = load_winner(winner_path)
    winner_aug = load_winner_aug(winner_aug_path)

    print(f"Winner:      {winner['arch']} + {winner['encoder']} weights={winner['class_weights']}")
    print(f"Best split:  {winner['best_split']}")
    print(f"Aug params:  {list(winner_aug.get('optimized_params', {}).keys())}")

    jobs = build_lc_jobs(cfg, winner, winner_aug)
    print(f"Wave 3 LC:   {len(jobs)} jobs")

    write_job_configs(jobs, jobs_dir)
    sbatch_path = out_dir / 'wave3_lc.sbatch'
    write_sbatch(len(jobs), jobs_dir, cfg, sbatch_path)
    print(f"sbatch written → {sbatch_path}")

    if args.dry_run:
        print("--dry-run: not submitting.")
        return

    r = subprocess.run(['sbatch', str(sbatch_path)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"sbatch FAILED:\n{r.stderr}", file=sys.stderr)
        sys.exit(1)

    job_id  = r.stdout.strip().split()[-1]
    id_file = out_dir / 'wave3_job_ids.json'
    with open(id_file, 'w') as f:
        json.dump({'lc': job_id}, f, indent=2)

    print(f"Wave 3 LC submitted — SLURM array ID: {job_id}")
    print(f"Job ID saved → {id_file}")
    print(f"You will be emailed at {cfg['slurm']['mail_user']} on completion/failure.")


if __name__ == '__main__':
    main()
