"""
aggregator.py

Reads Wave 1/2/3 result.json files and produces paper tables.

Outputs by wave:
    Wave 1 (--wave 1, default):
        wave1_all_results.csv   — flat dump of all 5,760 results
        wave1_summary.csv       — mean ± SD per arch/encoder/weights combo
        table2_architecture.csv — architecture comparison (encoder=resnet34)
        table3_encoders_<arch>  — encoder comparison (winning arch)
        candidates.json         — top 5 per metric + consensus
        winner.json             — written via --select after review

    Wave 2a (--wave 2a):
        wave2a_all_results.csv  — flat dump of all 2a results
        table4_aug_sweep.csv    — best params per aug type (Table 4)
        Prints top settings per aug type to terminal.
        You then manually write winner_aug.json.

    Wave 2b (--wave 2b):
        table5_aug_comparison.csv — aug ON vs aug OFF per phenotype (Table 5)
        Pulls aug OFF baseline from Wave 1 SW results.

    Wave 3 (--wave 3):
        table1_learning_curve.csv — Dice ± SD vs dataset size (Table 1)

Usage:
    python aggregator.py --config analysis_config.json [--wave 1]
    python aggregator.py --config analysis_config.json --wave 2a
    python aggregator.py --config analysis_config.json --wave 2b
    python aggregator.py --config analysis_config.json --wave 3

    # After Wave 1 review — write winner.json
    python aggregator.py --config analysis_config.json --select \\
        --arch unet++ --encoder resnet34 --weights 3,1,1 \\
        [--split 67,33] --note "Best consensus across axon/myelin Dice and HD95"

Metric ranking directions:
    Higher is better: dice_*, iou_*, precision_*, recall_*
    Lower is better:  hd95_*

Background metrics included — false positive axon/myelin in connective
tissue directly corrupts morphometric outputs.
"""

from __future__ import annotations

import json
import argparse
import csv
from pathlib import Path
from collections import defaultdict
import statistics


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as f:
        return json.load(f)


# ─── Result loading ───────────────────────────────────────────────────────────

def load_results_from(results_dir: Path, label: str = '') -> list[dict]:
    """
    Recursively find all result.json files under results_dir.
    Returns list of result dicts with 'result_path' added.
    """
    results = []
    missing = []

    for p in sorted(results_dir.rglob('result.json')):
        try:
            with open(p) as f:
                r = json.load(f)
            r['result_path'] = str(p)
            results.append(r)
        except Exception as e:
            missing.append((str(p), str(e)))

    if missing:
        print(f"[WARN] Failed to load {len(missing)} result files:")
        for path, err in missing[:10]:
            print(f"  {path}: {err}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    tag = f" [{label}]" if label else ""
    print(f"Loaded {len(results)} result files from {results_dir}{tag}")
    return results


def check_completeness(results: list[dict], expected: int):
    n = len(results)
    if n < expected:
        pct = round(n / expected * 100, 1)
        print(f"[WARN] {n}/{expected} results loaded ({pct}%) — "
              f"{expected - n} jobs may have failed or not yet completed")
    else:
        print(f"[OK] All {n}/{expected} results loaded")


# ─── Shared metric constants ──────────────────────────────────────────────────

HIGHER_IS_BETTER = [
    'dice_macro', 'dice_axon', 'dice_myelin', 'dice_bg',
    'iou_macro',  'iou_axon',  'iou_myelin',  'iou_bg',
    'precision_macro', 'precision_axon', 'precision_myelin', 'precision_bg',
    'recall_macro',    'recall_axon',    'recall_myelin',    'recall_bg',
]

LOWER_IS_BETTER = [
    'hd95_macro', 'hd95_axon', 'hd95_myelin', 'hd95_bg',
]

ALL_METRICS = HIGHER_IS_BETTER + LOWER_IS_BETTER

RANKING_METRICS = [
    ('dice_macro',       'higher'),
    ('dice_axon',        'higher'),
    ('dice_myelin',      'higher'),
    ('dice_bg',          'higher'),
    ('iou_axon',         'higher'),
    ('hd95_axon',        'lower'),
    ('hd95_myelin',      'lower'),
    ('hd95_bg',          'lower'),
    ('precision_myelin', 'higher'),
    ('recall_myelin',    'higher'),
    ('precision_axon',   'higher'),
]

CONSENSUS_THRESHOLD = 4


# ─── Shared aggregation helpers ───────────────────────────────────────────────

def _mean_sd(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    mean = round(statistics.mean(vals), 4)
    sd   = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
    return mean, sd


def group_by_combo(results: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in results:
        w     = r.get('class_weights', [])
        w_str = '_'.join(str(int(x)) if x == int(x) else str(x) for x in w)
        key   = f"{r['arch']}__{r['encoder']}__{w_str}"
        groups[key].append(r)
    return dict(groups)


def aggregate_combo(results: list[dict]) -> dict:
    if not results:
        return {}
    r0      = results[0]
    summary = {
        'arch':          r0['arch'],
        'encoder':       r0['encoder'],
        'class_weights': r0.get('class_weights', []),
        'n_runs':        len(results),
    }
    for metric in ALL_METRICS:
        vals = [r[metric] for r in results if metric in r and r[metric] is not None]
        mean, sd = _mean_sd(vals)
        summary[f'{metric}_mean'] = mean
        summary[f'{metric}_sd']   = sd
    return summary


def find_optimal_weights(
    groups: dict, arch: str, encoder: str,
    primary_metric: str = 'dice_macro',
) -> dict | None:
    best_summary = None
    best_val     = -float('inf')
    for key, results in groups.items():
        r0 = results[0]
        if r0['arch'] != arch or r0['encoder'] != encoder:
            continue
        summary = aggregate_combo(results)
        val = summary.get(f'{primary_metric}_mean')
        if val is not None and val > best_val:
            best_val     = val
            best_summary = summary
    return best_summary


def find_best_split(
    results: list[dict], arch: str, encoder: str,
    weights: list[float], primary_metric: str = 'dice_macro',
) -> tuple[int, int]:
    w_str   = '_'.join(str(int(w)) if w == int(w) else str(w) for w in weights)
    matches = [
        r for r in results
        if r['arch'] == arch and r['encoder'] == encoder
        and '_'.join(str(int(x)) if x == int(x) else str(x)
                     for x in r.get('class_weights', [])) == w_str
    ]
    if not matches:
        print(f"[WARN] No results for {arch}/{encoder}/{w_str} — defaulting to 67/33")
        return 67, 33

    by_split = defaultdict(list)
    for r in matches:
        val = r.get(primary_metric)
        if val is not None:
            by_split[f"{r['train_pct']}_{r['val_pct']}"].append(
                (r['train_pct'], r['val_pct'], val)
            )

    best_mean   = -float('inf')
    best_train  = 67
    best_val_p  = 33
    for entries in by_split.values():
        m = statistics.mean(e[2] for e in entries)
        if m > best_mean:
            best_mean  = m
            best_train = entries[0][0]
            best_val_p = entries[0][1]

    print(f"Best split for {arch}/{encoder}: {best_train}/{best_val_p} "
          f"(mean {primary_metric}={round(best_mean, 4)})")
    return best_train, best_val_p


# ─── CSV writers ──────────────────────────────────────────────────────────────

def write_flat_csv(results: list[dict], out_path: Path):
    if not results:
        return
    all_keys = []
    seen     = set()
    priority = [
        'run_id', 'stage', 'wave', 'arch', 'encoder', 'class_weights',
        'n_images', 'train_pct', 'val_pct', 'seed', 'augmentation',
        'checkpoint_metric', 'best_epoch', 'epochs_completed', 'early_stopped',
        'best_val_loss',
    ] + ALL_METRICS + ['train_stems', 'val_stems', 'model_path', 'result_path']
    for k in priority:
        if k not in seen:
            all_keys.append(k); seen.add(k)
    for r in results:
        for k in r:
            if k not in seen:
                all_keys.append(k); seen.add(k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Written: {out_path} ({len(results)} rows)")


def write_summary_csv(summaries: list[dict], out_path: Path):
    if not summaries:
        return
    base_cols   = ['arch', 'encoder', 'class_weights', 'n_runs']
    metric_cols = []
    for m in ALL_METRICS:
        metric_cols += [f'{m}_mean', f'{m}_sd']
    all_cols = base_cols + metric_cols
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
        writer.writeheader()
        for s in summaries:
            row = dict(s)
            row['class_weights'] = str(s.get('class_weights', ''))
            writer.writerow(row)
    print(f"Written: {out_path} ({len(summaries)} rows)")


def write_table_csv(rows: list[dict], out_path: Path, label: str):
    write_summary_csv(rows, out_path)
    print(f"{label}: {out_path}")


def write_dict_csv(rows: list[dict], out_path: Path, label: str):
    if not rows:
        return
    all_keys = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"{label}: {out_path} ({len(rows)} rows)")


# ─── Wave 1 functions ─────────────────────────────────────────────────────────

def build_table2(groups: dict, encoder: str = 'resnet34') -> list[dict]:
    archs = sorted({r['arch'] for results in groups.values() for r in results})
    rows  = []
    for arch in archs:
        summary = find_optimal_weights(groups, arch, encoder)
        if summary:
            rows.append(summary)
    rows.sort(key=lambda x: x.get('dice_macro_mean') or 0, reverse=True)
    return rows


def build_table3(groups: dict, winning_arch: str) -> list[dict]:
    encoders = sorted({
        r['encoder']
        for results in groups.values()
        for r in results
        if r['arch'] == winning_arch
    })
    rows = []
    for encoder in encoders:
        summary = find_optimal_weights(groups, winning_arch, encoder)
        if summary:
            rows.append(summary)
    rows.sort(key=lambda x: x.get('dice_macro_mean') or 0, reverse=True)
    return rows


def build_candidates(summaries: list[dict], top_n: int = 5) -> dict:
    candidates = {}
    for metric, direction in RANKING_METRICS:
        key     = f'{metric}_mean'
        reverse = (direction == 'higher')
        ranked  = sorted(
            [s for s in summaries if s.get(key) is not None],
            key=lambda x: x[key], reverse=reverse
        )
        candidates[f'by_{metric}'] = [
            {
                'arch':          s['arch'],
                'encoder':       s['encoder'],
                'class_weights': s['class_weights'],
                'n_runs':        s['n_runs'],
                metric:          s[key],
                f'{metric}_sd':  s.get(f'{metric}_sd'),
            }
            for s in ranked[:top_n]
        ]

    appearance_count = defaultdict(int)
    for metric, _ in RANKING_METRICS:
        for entry in candidates[f'by_{metric}']:
            combo_key = (entry['arch'], entry['encoder'], str(entry['class_weights']))
            appearance_count[combo_key] += 1

    consensus = []
    for (arch, encoder, weights_str), count in sorted(
        appearance_count.items(), key=lambda x: -x[1]
    ):
        if count >= CONSENSUS_THRESHOLD:
            for s in summaries:
                if (s['arch'] == arch and s['encoder'] == encoder
                        and str(s['class_weights']) == weights_str):
                    consensus.append({
                        'arch':             arch,
                        'encoder':          encoder,
                        'class_weights':    s['class_weights'],
                        'n_rankings_top5':  count,
                        'dice_macro_mean':  s.get('dice_macro_mean'),
                        'dice_axon_mean':   s.get('dice_axon_mean'),
                        'dice_myelin_mean': s.get('dice_myelin_mean'),
                        'hd95_axon_mean':   s.get('hd95_axon_mean'),
                        'hd95_myelin_mean': s.get('hd95_myelin_mean'),
                    })
                    break

    candidates[f'consensus_top{top_n}'] = consensus[:top_n]
    candidates['_note'] = (
        f"consensus = models in top {top_n} across "
        f"{CONSENSUS_THRESHOLD}+ of {len(RANKING_METRICS)} metric rankings."
    )
    return candidates


def write_winner(
    groups: dict, results: list[dict],
    arch: str, encoder: str, weights: list[float],
    split: tuple[int, int], note: str, out_path: Path,
):
    w_str         = '_'.join(str(int(w)) if w == int(w) else str(w) for w in weights)
    key           = f"{arch}__{encoder}__{w_str}"
    combo_results = groups.get(key, [])
    summary       = aggregate_combo(combo_results) if combo_results else {}
    train_pct, val_pct = split

    winner = {
        'arch':              arch,
        'encoder':           encoder,
        'class_weights':     weights,
        'best_split':        [train_pct, val_pct],
        'best_val_fraction': round(val_pct / 100, 2),
        'selected_by':       note,
        'n_runs':            len(combo_results),
        'dice_macro_mean':   summary.get('dice_macro_mean'),
        'dice_axon_mean':    summary.get('dice_axon_mean'),
        'dice_myelin_mean':  summary.get('dice_myelin_mean'),
        'hd95_axon_mean':    summary.get('hd95_axon_mean'),
        'hd95_myelin_mean':  summary.get('hd95_myelin_mean'),
        'hd95_bg_mean':      summary.get('hd95_bg_mean'),
        '_note': (
            'Winner selected after manual review of Wave 1 results. '
            'best_split used by wave2_launcher.py and wave3_launcher.py. '
            'Wave 2b aug OFF baseline pulled from Wave 1 SW by matching '
            'arch/encoder/weights/split/seed.'
        )
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(winner, f, indent=2)
    print(f"Winner written → {out_path}")
    print(f"  arch:       {arch}")
    print(f"  encoder:    {encoder}")
    print(f"  weights:    {weights}")
    print(f"  best_split: {train_pct}/{val_pct}")
    print(f"  note:       {note}")


# ─── Wave 2a functions ────────────────────────────────────────────────────────

# Aug types and which params identify each sweep
AUG_SWEEP_KEYS = {
    'hflip_prob':       ['hflip_prob'],
    'vflip_prob':       ['vflip_prob'],
    'rotation_prob':    ['rotation_prob'],
    'rotation_deg':     ['rotation_deg'],
    'brightness':       ['brightness_prob', 'brightness_scale', 'brightness_offset'],
    'gamma':            ['gamma_prob', 'gamma_range'],
    'noise':            ['noise_prob', 'noise_sigma'],
    'gaussian_blur':    ['blur_prob', 'blur_sigma'],
    'elastic':          ['elastic_prob', 'elastic_alpha', 'elastic_sigma'],
    'clahe':            ['clahe_prob', 'clahe_clip', 'clahe_tile'],
}


def _get_aug_type(result: dict) -> str:
    """Infer aug type from stage field in result.json."""
    stage = result.get('stage', '')
    for aug_type in AUG_SWEEP_KEYS:
        if aug_type in stage:
            return aug_type
    return 'unknown'


def build_table4(results_2a: list[dict]) -> list[dict]:
    """
    Table 4 — Aug sweep matrix.
    For each aug type, find the parameter combination that maximizes
    mean dice_macro across seeds.
    One row per aug type.
    """
    by_aug_type = defaultdict(list)
    for r in results_2a:
        aug_type = _get_aug_type(r)
        by_aug_type[aug_type].append(r)

    rows = []
    for aug_type, type_results in sorted(by_aug_type.items()):
        # Group by aug param combo
        by_params = defaultdict(list)
        for r in type_results:
            aug_params = r.get('aug_params', {})
            relevant_keys = AUG_SWEEP_KEYS.get(aug_type, [])
            param_key = json.dumps(
                {k: aug_params.get(k) for k in relevant_keys}, sort_keys=True
            )
            by_params[param_key].append(r)

        best_mean   = -float('inf')
        best_params = {}
        best_n      = 0
        best_metrics = {}

        for param_key, param_results in by_params.items():
            vals = [r['dice_macro'] for r in param_results
                    if 'dice_macro' in r and r['dice_macro'] is not None]
            if not vals:
                continue
            mean = statistics.mean(vals)
            if mean > best_mean:
                best_mean    = mean
                best_params  = json.loads(param_key)
                best_n       = len(vals)
                for metric in ALL_METRICS:
                    m_vals = [r[metric] for r in param_results
                              if metric in r and r[metric] is not None]
                    mn, sd = _mean_sd(m_vals)
                    best_metrics[f'{metric}_mean'] = mn
                    best_metrics[f'{metric}_sd']   = sd

        row = {
            'aug_type':      aug_type,
            'n_seeds':       best_n,
            **best_params,
            **best_metrics,
        }
        rows.append(row)

        print(f"  {aug_type}: best params={best_params} "
              f"dice_macro={round(best_mean, 4)} (n={best_n})")

    return rows


# ─── Wave 2b functions ────────────────────────────────────────────────────────

def build_table5(results_2b: list[dict], results_sw: list[dict], winner: dict) -> list[dict]:
    """
    Table 5 — Aug ON vs OFF per phenotype.
    Aug ON: from Wave 2b results.
    Aug OFF: pulled from Wave 1 SW results matching arch/encoder/weights/split/seed.
    Phenotype inferred from val_stems in result.json.
    """
    arch    = winner['arch']
    encoder = winner['encoder']
    weights = winner['class_weights']
    split   = winner['best_split']

    w_str = '_'.join(str(int(w)) if w == int(w) else str(w) for w in weights)

    # Aug OFF baseline from Wave 1 SW
    aug_off_by_seed = {}
    for r in results_sw:
        if (r['arch'] == arch and r['encoder'] == encoder
                and r['train_pct'] == split[0] and r['val_pct'] == split[1]
                and not r.get('augmentation', True)
                and '_'.join(str(int(x)) if x == int(x) else str(x)
                             for x in r.get('class_weights', [])) == w_str):
            aug_off_by_seed[r['seed']] = r

    rows = []

    for r_on in results_2b:
        seed = r_on.get('seed')
        r_off = aug_off_by_seed.get(seed)

        # Infer phenotype balance from val_stems
        val_stems = r_on.get('val_stems', [])
        n_ctrl  = sum(1 for s in val_stems if s.startswith('ctrl_'))
        n_regen = sum(1 for s in val_stems if s.startswith('regen_'))

        row = {
            'seed':              seed,
            'n_val_ctrl':        n_ctrl,
            'n_val_regen':       n_regen,
        }

        for metric in ['dice_macro', 'dice_axon', 'dice_myelin', 'dice_bg',
                        'hd95_macro', 'hd95_axon', 'hd95_myelin']:
            row[f'{metric}_aug_on']  = r_on.get(metric)
            row[f'{metric}_aug_off'] = r_off.get(metric) if r_off else None
            on_val  = r_on.get(metric)
            off_val = r_off.get(metric) if r_off else None
            if on_val is not None and off_val is not None:
                row[f'{metric}_delta'] = round(on_val - off_val, 4)
            else:
                row[f'{metric}_delta'] = None

        rows.append(row)

    rows.sort(key=lambda x: x['seed'])
    return rows


# ─── Wave 3 functions ─────────────────────────────────────────────────────────

def build_table1(results_lc: list[dict]) -> list[dict]:
    """
    Table 1 — Learning curve.
    Mean Dice ± SD per dataset size, averaged across splits and seeds.
    Shows plateau at n=30 — dataset sufficiency argument.
    """
    by_size = defaultdict(list)
    for r in results_lc:
        if r.get('wave') == 3:  # only Wave 3 LC results
            n = r.get('n_images')
            if n is not None:
                by_size[n].append(r)

    rows = []
    for n_images in sorted(by_size.keys()):
        size_results = by_size[n_images]
        row = {
            'n_images': n_images,
            'n_runs':   len(size_results),
        }
        for metric in ALL_METRICS:
            vals = [r[metric] for r in size_results
                    if metric in r and r[metric] is not None]
            mean, sd = _mean_sd(vals)
            row[f'{metric}_mean'] = mean
            row[f'{metric}_sd']   = sd

        rows.append(row)
        print(f"  n={n_images}: dice_macro={row.get('dice_macro_mean')} "
              f"± {row.get('dice_macro_sd')} (n_runs={len(size_results)})")

    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepAxon aggregator — all waves")
    parser.add_argument('--config',   default='analysis_config.json')
    parser.add_argument('--wave',     default='1',
                        choices=['1', '2a', '2b', '3'],
                        help='Which wave to aggregate (default: 1)')
    parser.add_argument('--select',   action='store_true',
                        help='Write winner.json (Wave 1 only)')
    parser.add_argument('--arch',     help='Winning architecture (--select)')
    parser.add_argument('--encoder',  help='Winning encoder (--select)')
    parser.add_argument('--weights',  help='Class weights e.g. 3,1,1 (--select)')
    parser.add_argument('--split',    help='TT split e.g. 67,33 (--select, optional)')
    parser.add_argument('--note',     default='Manual selection after Wave 1 review')
    parser.add_argument('--expected', type=int, default=5760,
                        help='Expected SW job count for completeness check (default 5760)')
    args = parser.parse_args()

    cfg         = load_config(args.config)
    results_dir = Path(cfg['output']['results_dir'])
    out_dir     = results_dir.parent / 'aggregated'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Wave 1 ────────────────────────────────────────────────────────────────
    if args.wave == '1':
        print("\n── WAVE 1 ───────────────────────────────────────────────────────────")
        results = load_results_from(results_dir / 'sw', 'Wave 1 SW')
        check_completeness(results, args.expected)
        if not results:
            print("No results found — exiting.")
            return

        groups    = group_by_combo(results)
        summaries = [aggregate_combo(v) for v in groups.values()]
        print(f"Aggregated {len(summaries)} unique arch/encoder/weight combinations")

        write_flat_csv(results,   out_dir / 'wave1_all_results.csv')
        write_summary_csv(summaries, out_dir / 'wave1_summary.csv')

        table2_rows = build_table2(groups, encoder='resnet34')
        write_table_csv(table2_rows, out_dir / 'table2_architecture.csv', 'Table 2')

        if table2_rows:
            prov_arch   = table2_rows[0]['arch']
            table3_rows = build_table3(groups, prov_arch)
            write_table_csv(
                table3_rows,
                out_dir / f'table3_encoders_{prov_arch}.csv',
                f'Table 3 (provisional arch: {prov_arch})'
            )

        candidates = build_candidates(summaries)
        with open(out_dir / 'candidates.json', 'w') as f:
            json.dump(candidates, f, indent=2)
        print(f"Written: {out_dir / 'candidates.json'}")

        print("\n── CONSENSUS TOP 5 ──────────────────────────────────────────────────")
        for i, c in enumerate(candidates.get('consensus_top5', []), 1):
            print(
                f"  {i}. {c['arch']} + {c['encoder']} weights={c['class_weights']} "
                f"| top5 in {c['n_rankings_top5']}/{len(RANKING_METRICS)} rankings "
                f"| dice_macro={c.get('dice_macro_mean')} "
                f"| hd95_axon={c.get('hd95_axon_mean')}"
            )

        if args.select:
            if not all([args.arch, args.encoder, args.weights]):
                parser.error("--select requires --arch, --encoder, and --weights")
            weights = [float(w) for w in args.weights.split(',')]
            if args.split:
                train_pct, val_pct = [int(x) for x in args.split.split(',')]
            else:
                train_pct, val_pct = find_best_split(
                    results, args.arch, args.encoder, weights
                )
            write_winner(
                groups=groups, results=results,
                arch=args.arch, encoder=args.encoder,
                weights=weights, split=(train_pct, val_pct),
                note=args.note, out_path=out_dir / 'winner.json',
            )
            table3_rows = build_table3(groups, args.arch)
            write_table_csv(
                table3_rows,
                out_dir / f'table3_encoders_{args.arch}.csv',
                f'Table 3 (confirmed arch: {args.arch})'
            )
        else:
            print("\nReview outputs, then run:")
            print("  python aggregator.py --config analysis_config.json --select "
                  "--arch <arch> --encoder <enc> --weights <w1,w2,w3> "
                  "[--split <train,val>] --note '<rationale>'")

    # ── Wave 2a ───────────────────────────────────────────────────────────────
    elif args.wave == '2a':
        print("\n── WAVE 2a ──────────────────────────────────────────────────────────")
        results_2a = load_results_from(results_dir / 'wave2', 'Wave 2a')
        if not results_2a:
            print("No Wave 2a results found — exiting.")
            return

        write_flat_csv(results_2a, out_dir / 'wave2a_all_results.csv')

        print("\nBest parameters per aug type:")
        table4_rows = build_table4(results_2a)
        write_dict_csv(table4_rows, out_dir / 'table4_aug_sweep.csv', 'Table 4')

        print("\nNext steps:")
        print("  1. Review table4_aug_sweep.csv and wave2a_all_results.csv")
        print("  2. Write analysis/aggregated/winner_aug.json:")
        print('     { "optimized_params": { "hflip_prob": 0.75, ... } }')
        print("  3. Run: python wave2_launcher.py --config analysis_config.json --step 2b")

    # ── Wave 2b ───────────────────────────────────────────────────────────────
    elif args.wave == '2b':
        print("\n── WAVE 2b ──────────────────────────────────────────────────────────")
        winner_path = out_dir / 'winner.json'
        if not winner_path.exists():
            print(f"[ERROR] winner.json not found: {winner_path}")
            return

        with open(winner_path) as f:
            winner = json.load(f)

        results_2b = load_results_from(results_dir / 'wave2', 'Wave 2b')
        results_2b = [r for r in results_2b if r.get('stage') == '2b_aug_comparison']

        results_sw = load_results_from(results_dir / 'sw', 'Wave 1 SW baseline')

        if not results_2b:
            print("No Wave 2b results found — exiting.")
            return

        table5_rows = build_table5(results_2b, results_sw, winner)
        write_dict_csv(table5_rows, out_dir / 'table5_aug_comparison.csv', 'Table 5')

        # Summary to terminal
        on_vals  = [r['dice_macro_aug_on']  for r in table5_rows
                    if r.get('dice_macro_aug_on')  is not None]
        off_vals = [r['dice_macro_aug_off'] for r in table5_rows
                    if r.get('dice_macro_aug_off') is not None]
        if on_vals and off_vals:
            print(f"\nAug ON  dice_macro: {round(statistics.mean(on_vals), 4)} "
                  f"± {round(statistics.stdev(on_vals), 4)}")
            print(f"Aug OFF dice_macro: {round(statistics.mean(off_vals), 4)} "
                  f"± {round(statistics.stdev(off_vals), 4)}")

    # ── Wave 3 ────────────────────────────────────────────────────────────────
    elif args.wave == '3':
        print("\n── WAVE 3 — LEARNING CURVE ──────────────────────────────────────────")
        results_lc = load_results_from(results_dir / 'lc', 'Wave 3 LC')
        results_lc = [r for r in results_lc if r.get('wave') == 3]

        if not results_lc:
            print("No Wave 3 results found — exiting.")
            return

        print("\nLearning curve by dataset size:")
        table1_rows = build_table1(results_lc)
        write_dict_csv(table1_rows, out_dir / 'table1_learning_curve.csv', 'Table 1')

        print("\nTable 1 complete — plateau confirmation for dataset sufficiency.")


if __name__ == '__main__':
    main()
