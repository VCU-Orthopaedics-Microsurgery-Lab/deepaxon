"""
wave2_launcher.py

Augmentation parameter sweep on winning model from Wave 1.
Must be run after:
    1. Wave 1 SW completes
    2. Manual review of candidates.json
    3. winner.json written via: python aggregator.py --select ...

Two steps:

    Step 2a — OAT + matrix aug parameter sweep (2,065 jobs)
        One aug type varied at a time, all others fixed at production defaults.
        Identifies optimal parameters per aug type.

    Step 2b — Aug ON vs OFF validation (20 jobs)
        Optimized aug params vs aug OFF, 5 seeds, ctrl vs regen phenotype.
        Requires step2a to complete first — winner_aug.json written manually.

Usage:
    # Step 2a
    python wave2_launcher.py --config analysis_config.json --step 2a [--dry-run]

    # After reviewing 2a results and writing winner_aug.json:
    python wave2_launcher.py --config analysis_config.json --step 2b [--dry-run]

Job counts:
    Step 2a:
        H-flip prob          :   4 × 5 seeds =   20
        V-flip prob          :   4 × 5 seeds =   20
        Rotation prob        :   4 × 5 seeds =   20
        Rotation intensity   :   5 × 5 seeds =   25
        Brightness matrix    :  28 × 5 seeds =  140
        Gamma matrix         :  28 × 5 seeds =  140
        Noise matrix         :  24 × 5 seeds =  120
        Gaussian blur matrix :  20 × 5 seeds =  100
        Elastic 3D matrix    : 100 × 5 seeds =  500
        CLAHE 3D matrix      : 196 × 5 seeds =  980
        Total 2a             :                2,065
    Step 2b: aug ON only × 5 seeds = 5
        Aug OFF baseline pulled from Wave 1 SW results by matching
        arch/encoder/weights/split/seed — no redundant aug_off jobs.
    Wave 2 total: 2,070 jobs (~4.3 hours wall time)
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
            f"Run: python aggregator.py --config analysis_config.json --select "
            f"--arch <arch> --encoder <enc> --weights <w1,w2,w3>"
        )
    with open(winner_path) as f:
        return json.load(f)


def load_winner_aug(winner_aug_path: Path) -> dict:
    if not winner_aug_path.exists():
        raise FileNotFoundError(
            f"winner_aug.json not found: {winner_aug_path}\n"
            f"Review Step 2a results and write winner_aug.json manually."
        )
    with open(winner_aug_path) as f:
        return json.load(f)


# ─── Production defaults ──────────────────────────────────────────────────────

# All aug params fixed at these values when not being swept
PRODUCTION_AUG = {
    'hflip_prob':         0.0,    # all other aug OFF during OAT sweep
    'vflip_prob':         0.0,
    'rotation_prob':      0.0,
    'rotation_deg':       15,
    'brightness_prob':    0.0,
    'brightness_scale':   [0.80, 1.20],
    'brightness_offset':  [-0.10, 0.10],
    'gamma_prob':         0.0,
    'gamma_range':        [0.70, 1.40],
    'noise_prob':         0.0,
    'noise_sigma':        0.02,
    'blur_prob':          0.0,
    'blur_sigma':         1.0,
    'elastic_prob':       0.0,
    'elastic_alpha':      20,
    'elastic_sigma':      8,
    'clahe_prob':         0.0,
    'clahe_clip':         1.5,
    'clahe_tile':         16,
}


def base_job(cfg: dict, winner: dict, run_id: str, aug_params: dict,
             stage: str, wave: int = 2) -> dict:
    """Build a base job config dict."""
    output    = cfg['output']
    train     = cfg['training']
    ds        = cfg['dataset']
    train_pct = winner['best_split'][0]
    val_pct   = winner['best_split'][1]

    return {
        'run_id':        run_id,
        'stage':         stage,
        'wave':          wave,
        'images_dir':    ds['images_dir'],
        'mag':           ds['mag'],
        'ctrl_prefix':   ds['ctrl_prefix'],
        'regen_prefix':  ds['regen_prefix'],
        'n_images':      30,
        'val_fraction':  winner['best_val_fraction'],
        'train_pct':     train_pct,
        'val_pct':       val_pct,
        'arch':          winner['arch'],
        'encoder':       winner['encoder'],
        'class_weights': winner['class_weights'],
        'epochs':        train['epochs'],
        'batch_size':    train['batch_size'],
        'augmentation':  True,
        'aug_params':    aug_params,
        'learning_rate': train['learning_rate'],
        'weight_decay':  train['weight_decay'],
        'output': {
            'results_dir': str(Path(output['results_dir']) / 'wave2' / stage),
            'models_dir':  str(Path(output['models_dir'])  / 'wave2' / stage),
            'logs_dir':    str(Path(output['logs_dir'])    / 'wave2' / stage),
        }
    }


# ─── Step 2a permutation builders ────────────────────────────────────────────

def build_2a_jobs(cfg: dict, winner: dict) -> list[dict]:
    """Build all Step 2a aug sweep jobs."""
    w2     = cfg['wave2']
    seeds  = cfg['splits']['seeds']
    jobs   = []

    def add_jobs(sweep_name: str, combos: list[dict]):
        for combo, seed in product(combos, seeds):
            aug = {**PRODUCTION_AUG, **combo}
            w_str = '_'.join(
                f"{k}{str(v).replace(' ', '').replace(',', '-').replace('[', '').replace(']', '')}"
                for k, v in combo.items()
            )
            run_id = f"w2a__{sweep_name}__{w_str}__seed{seed}"
            jobs.append(base_job(cfg, winner, run_id, aug, f'2a_{sweep_name}'))

    # ── OAT sweeps ────────────────────────────────────────────────────────────

    # H-flip probability
    add_jobs('hflip_prob', [
        {'hflip_prob': p} for p in w2['hflip']['prob_levels']
    ])

    # V-flip probability
    add_jobs('vflip_prob', [
        {'vflip_prob': p} for p in w2['vflip']['prob_levels']
    ])

    # Rotation probability
    add_jobs('rotation_prob', [
        {'rotation_prob': p} for p in w2['rotation_prob']['prob_levels']
    ])

    # Rotation intensity
    add_jobs('rotation_deg', [
        {'rotation_deg': d} for d in w2['rotation_intensity']['intensity_levels_deg']
    ])

    # ── Matrix sweeps ─────────────────────────────────────────────────────────

    # Brightness — 4 prob × 7 intensity
    bright_combos = []
    for prob in w2['brightness']['prob_levels']:
        for lvl in w2['brightness']['intensity_levels']:
            bright_combos.append({
                'brightness_prob':   prob,
                'brightness_scale':  lvl['scale'],
                'brightness_offset': lvl['offset'],
            })
    add_jobs('brightness', bright_combos)

    # Gamma — 4 prob × 7 intensity
    gamma_combos = []
    for prob in w2['gamma']['prob_levels']:
        for lvl in w2['gamma']['intensity_levels']:
            gamma_combos.append({
                'gamma_prob':  prob,
                'gamma_range': lvl,
            })
    add_jobs('gamma', gamma_combos)

    # Noise — 4 prob × 6 sigma
    noise_combos = []
    for prob in w2['noise']['prob_levels']:
        for sigma in w2['noise']['sigma_levels']:
            noise_combos.append({
                'noise_prob':  prob,
                'noise_sigma': sigma,
            })
    add_jobs('noise', noise_combos)

    # Gaussian blur — 4 prob × 5 sigma
    blur_combos = []
    for prob in w2['gaussian_blur']['prob_levels']:
        for sigma in w2['gaussian_blur']['sigma_levels']:
            blur_combos.append({
                'blur_prob':  prob,
                'blur_sigma': sigma,
            })
    add_jobs('gaussian_blur', blur_combos)

    # Elastic deformation — 4 prob × 5 alpha × 5 sigma
    elastic_combos = []
    for prob in w2['elastic']['prob_levels']:
        for alpha in w2['elastic']['alpha_levels']:
            for sigma in w2['elastic']['sigma_levels']:
                elastic_combos.append({
                    'elastic_prob':  prob,
                    'elastic_alpha': alpha,
                    'elastic_sigma': sigma,
                })
    add_jobs('elastic', elastic_combos)

    # CLAHE — 4 prob × 7 clip × 7 tile
    clahe_combos = []
    for prob in w2['clahe']['prob_levels']:
        for clip in w2['clahe']['clip_levels']:
            for tile in w2['clahe']['tile_levels']:
                clahe_combos.append({
                    'clahe_prob': prob,
                    'clahe_clip': clip,
                    'clahe_tile': tile,
                })
    add_jobs('clahe', clahe_combos)

    return jobs


# ─── Step 2b permutation builders ────────────────────────────────────────────

def build_2b_jobs(cfg: dict, winner: dict, winner_aug: dict) -> list[dict]:
    """
    Step 2b — Aug ON (optimized params) only.
    5 seeds × best_split from winner.json.
    Aug OFF baseline pulled from Wave 1 SW by matching
    arch/encoder/weights/split/seed — no redundant aug_off jobs.
    """
    seeds         = cfg['splits']['seeds']
    jobs          = []
    optimized_aug = winner_aug.get('optimized_params', PRODUCTION_AUG)

    for seed in seeds:
        run_id = f"w2b__aug_on__seed{seed}"
        job    = base_job(cfg, winner, run_id, optimized_aug, '2b_aug_comparison')
        job['augmentation'] = True
        jobs.append(job)

    return jobs


# ─── Job config writer ────────────────────────────────────────────────────────

def write_job_configs(jobs: list[dict], jobs_dir: Path, stage: str) -> list[Path]:
    stage_dir = jobs_dir / f'wave2_{stage}'
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
    step:     str,
    jobs_dir: Path,
    cfg:      dict,
    out_path: Path,
) -> Path:
    slurm    = cfg['slurm']
    venv     = slurm['venv']
    repo     = str(Path(venv).parent)
    logs_dir = cfg['output']['logs_dir']

    script = f"""#!/bin/bash
#SBATCH --job-name=deepaxon_wave2_{step}
#SBATCH --partition={slurm['partition']}
#SBATCH --gres={slurm['gres']}
#SBATCH --nodes={slurm['nodes']}
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --time={slurm['time']}
#SBATCH --mem={slurm['mem']}
#SBATCH --array=0-{n_jobs - 1}%{slurm['max_concurrent']}
#SBATCH --output={logs_dir}/wave2_{step}/%A_%a.out
#SBATCH --error={logs_dir}/wave2_{step}/%A_%a.err
#SBATCH --mail-type={slurm['mail_type']}
#SBATCH --mail-user={slurm['mail_user']}

source {venv}/bin/activate
cd {repo}

JOB_CONFIG={jobs_dir}/wave2_{step}/job_$(printf '%04d' $SLURM_ARRAY_TASK_ID).json

echo "Wave 2 [{step}] job $SLURM_ARRAY_TASK_ID — config: $JOB_CONFIG"
python -m train --config "$JOB_CONFIG"
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script)
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepAxon Wave 2 launcher")
    parser.add_argument('--config',  default='analysis_config.json')
    parser.add_argument('--step',    choices=['2a', '2b', 'both'], default='2a',
                        help='Which step to launch')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate configs and print commands without submitting')
    args = parser.parse_args()

    cfg      = load_config(args.config)
    jobs_dir = Path(cfg['output']['jobs_dir'])
    out_dir  = jobs_dir.parent

    winner_path     = out_dir / 'aggregated' / 'winner.json'
    winner_aug_path = out_dir / 'aggregated' / 'winner_aug.json'

    winner = load_winner(winner_path)
    print(f"Winner: {winner['arch']} + {winner['encoder']} weights={winner['class_weights']}")

    submitted_ids = {}

    # ── Step 2a ───────────────────────────────────────────────────────────────
    if args.step in ('2a', 'both'):
        jobs_2a = build_2a_jobs(cfg, winner)
        print(f"Step 2a: {len(jobs_2a)} jobs")
        write_job_configs(jobs_2a, jobs_dir, '2a')
        sbatch_2a = out_dir / 'wave2_2a.sbatch'
        write_sbatch(len(jobs_2a), '2a', jobs_dir, cfg, sbatch_2a)
        print(f"2a sbatch written → {sbatch_2a}")

        if not args.dry_run:
            r = subprocess.run(['sbatch', str(sbatch_2a)], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"sbatch FAILED [2a]:\n{r.stderr}", file=sys.stderr)
                sys.exit(1)
            job_id = r.stdout.strip().split()[-1]
            submitted_ids['2a'] = job_id
            print(f"2a submitted — SLURM array ID: {job_id}")
        else:
            print(f"--dry-run: would submit {sbatch_2a}")

        print("\nAfter 2a completes:")
        print("  1. Review results in analysis/results/wave2/2a_*/")
        print("  2. Write analysis/aggregated/winner_aug.json with optimized params")
        print("  3. Run: python wave2_launcher.py --config analysis_config.json --step 2b")

    # ── Step 2b ───────────────────────────────────────────────────────────────
    if args.step in ('2b', 'both'):
        winner_aug = load_winner_aug(winner_aug_path)
        jobs_2b    = build_2b_jobs(cfg, winner, winner_aug)
        print(f"Step 2b: {len(jobs_2b)} jobs")
        write_job_configs(jobs_2b, jobs_dir, '2b')
        sbatch_2b = out_dir / 'wave2_2b.sbatch'
        write_sbatch(len(jobs_2b), '2b', jobs_dir, cfg, sbatch_2b)
        print(f"2b sbatch written → {sbatch_2b}")

        if not args.dry_run:
            r = subprocess.run(['sbatch', str(sbatch_2b)], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"sbatch FAILED [2b]:\n{r.stderr}", file=sys.stderr)
                sys.exit(1)
            job_id = r.stdout.strip().split()[-1]
            submitted_ids['2b'] = job_id
            print(f"2b submitted — SLURM array ID: {job_id}")
        else:
            print(f"--dry-run: would submit {sbatch_2b}")

    # ── Save job IDs ──────────────────────────────────────────────────────────
    if submitted_ids:
        id_file = out_dir / 'wave2_job_ids.json'
        existing = {}
        if id_file.exists():
            with open(id_file) as f:
                existing = json.load(f)
        existing.update(submitted_ids)
        with open(id_file, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"Job IDs saved → {id_file}")
        print(f"You will be emailed at {cfg['slurm']['mail_user']} on completion/failure.")


if __name__ == '__main__':
    main()
