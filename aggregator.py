"""
aggregator.py

Reads all Wave 1 SW result.json files and produces:

    wave1_all_results.csv   — flat dump of all 3,840 results
    wave1_summary.csv       — mean ± SD per arch/encoder/weights across seeds and splits
    table2.csv              — architecture comparison (encoder fixed resnet34, optimal weights)
    table3.csv              — encoder comparison (winning arch, optimal weights)
    candidates.json         — top 5 per metric ranking + consensus
    winner.json             — written manually via --select after review

Usage:
    # After Wave 1 completes — generate all outputs
    python aggregator.py --config analysis_config.json

    # After manual review — write winner.json
    python aggregator.py --config analysis_config.json --select \\
        --arch unet++ --encoder resnet34 --weights 3,1,1 --split 70,30 \\
        --note "Best consensus across axon/myelin Dice and HD95"

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

def load_all_results(results_sw_dir: Path) -> list[dict]:
    """
    Recursively find all result.json files under results/sw/.
    Returns list of result dicts with file path added as 'result_path'.
    """
    results = []
    missing = []

    for p in sorted(results_sw_dir.rglob('result.json')):
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

    print(f"Loaded {len(results)} result files from {results_sw_dir}")
    return results


def check_completeness(results: list[dict], expected: int):
    """Warn if result count doesn't match expected job count."""
    n = len(results)
    if n < expected:
        pct = round(n / expected * 100, 1)
        print(f"[WARN] {n}/{expected} results loaded ({pct}%) — {expected - n} jobs may have failed or not yet completed")
    else:
        print(f"[OK] All {n}/{expected} results loaded")


# ─── Aggregation ──────────────────────────────────────────────────────────────

# Metrics where higher = better
HIGHER_IS_BETTER = [
    'dice_macro', 'dice_axon', 'dice_myelin', 'dice_bg',
    'iou_macro',  'iou_axon',  'iou_myelin',  'iou_bg',
    'precision_macro', 'precision_axon', 'precision_myelin', 'precision_bg',
    'recall_macro',    'recall_axon',    'recall_myelin',    'recall_bg',
]

# Metrics where lower = better
LOWER_IS_BETTER = [
    'hd95_macro', 'hd95_axon', 'hd95_myelin', 'hd95_bg',
]

ALL_METRICS = HIGHER_IS_BETTER + LOWER_IS_BETTER

# Metrics used for candidate ranking
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

CONSENSUS_THRESHOLD = 4  # must appear in top 5 across this many rankings


def group_by_combo(results: list[dict]) -> dict:
    """
    Group results by (arch, encoder, class_weights_str).
    Returns dict: combo_key → list of result dicts
    """
    groups = defaultdict(list)
    for r in results:
        w = r.get('class_weights', [])
        w_str = '_'.join(str(int(x)) if x == int(x) else str(x) for x in w)
        key = f"{r['arch']}__{r['encoder']}__{w_str}"
        groups[key].append(r)
    return dict(groups)


def aggregate_combo(results: list[dict]) -> dict:
    """
    Compute mean ± SD for all metrics across seeds and splits for one combo.
    Returns summary dict.
    """
    if not results:
        return {}

    r0 = results[0]
    summary = {
        'arch':          r0['arch'],
        'encoder':       r0['encoder'],
        'class_weights': r0.get('class_weights', []),
        'n_runs':        len(results),
    }

    for metric in ALL_METRICS:
        vals = [r[metric] for r in results if metric in r and r[metric] is not None]
        if vals:
            summary[f'{metric}_mean'] = round(statistics.mean(vals), 4)
            summary[f'{metric}_sd']   = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        else:
            summary[f'{metric}_mean'] = None
            summary[f'{metric}_sd']   = None

    return summary


def find_optimal_weights(
    groups: dict,
    arch: str,
    encoder: str,
    primary_metric: str = 'dice_macro',
) -> dict | None:
    """
    For a given arch/encoder, find the class weight config that maximizes
    mean primary_metric across seeds and splits.
    Returns the aggregated summary at optimal weights.
    """
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
    results:        list[dict],
    arch:           str,
    encoder:        str,
    weights:        list[float],
    primary_metric: str = 'dice_macro',
) -> tuple[int, int]:
    """
    For a given arch/encoder/weights, find the split ratio that produces
    the highest mean primary_metric across seeds.
    Returns (train_pct, val_pct).
    """
    w_str   = '_'.join(str(int(w)) if w == int(w) else str(w) for w in weights)
    matches = [
        r for r in results
        if r['arch'] == arch
        and r['encoder'] == encoder
        and '_'.join(str(int(x)) if x == int(x) else str(x) for x in r.get('class_weights', [])) == w_str
    ]

    if not matches:
        print(f"[WARN] No results for {arch}/{encoder}/{w_str} — defaulting to 70/30")
        return 70, 30

    # Group by split
    by_split = defaultdict(list)
    for r in matches:
        split_key = f"{r['train_pct']}_{r['val_pct']}"
        val = r.get(primary_metric)
        if val is not None:
            by_split[split_key].append((r['train_pct'], r['val_pct'], val))

    best_split_key  = None
    best_split_mean = -float('inf')
    for split_key, entries in by_split.items():
        mean_val = statistics.mean(e[2] for e in entries)
        if mean_val > best_split_mean:
            best_split_mean = mean_val
            best_split_key  = split_key
            best_train_pct  = entries[0][0]
            best_val_pct    = entries[0][1]

    print(f"Best split for {arch}/{encoder}: {best_train_pct}/{best_val_pct} "
          f"(mean {primary_metric}={round(best_split_mean, 4)})")
    return best_train_pct, best_val_pct


# ─── Table builders ───────────────────────────────────────────────────────────

def build_table2(groups: dict, encoder: str = 'resnet34') -> list[dict]:
    """
    Table 2 — Architecture comparison at optimal weights.
    Encoder fixed at resnet34 (or specified encoder).
    One row per architecture.
    """
    archs = sorted({r['arch'] for results in groups.values() for r in results})
    rows  = []

    for arch in archs:
        summary = find_optimal_weights(groups, arch, encoder)
        if summary:
            rows.append(summary)

    # Sort by dice_macro_mean descending
    rows.sort(key=lambda x: x.get('dice_macro_mean') or 0, reverse=True)
    return rows


def build_table3(groups: dict, winning_arch: str) -> list[dict]:
    """
    Table 3 — Encoder comparison within winning architecture.
    One row per encoder at its optimal class weights.
    """
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
    """
    Build top-N candidate lists per metric and consensus list.
    summaries: list of aggregated combo dicts (one per arch/encoder/weights combo)
    """
    candidates = {}

    for metric, direction in RANKING_METRICS:
        key      = f'{metric}_mean'
        reverse  = (direction == 'higher')
        ranked   = sorted(
            [s for s in summaries if s.get(key) is not None],
            key=lambda x: x[key],
            reverse=reverse
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

    # Consensus — models appearing in top_n across CONSENSUS_THRESHOLD+ rankings
    appearance_count = defaultdict(int)
    for metric, _ in RANKING_METRICS:
        for entry in candidates[f'by_{metric}']:
            combo_key = (
                entry['arch'],
                entry['encoder'],
                str(entry['class_weights'])
            )
            appearance_count[combo_key] += 1

    consensus = []
    for (arch, encoder, weights_str), count in sorted(
        appearance_count.items(), key=lambda x: -x[1]
    ):
        if count >= CONSENSUS_THRESHOLD:
            # Find full summary for this combo
            for s in summaries:
                w_str = str(s['class_weights'])
                if s['arch'] == arch and s['encoder'] == encoder and w_str == weights_str:
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
        f"consensus = models appearing in top {top_n} across "
        f"{CONSENSUS_THRESHOLD}+ of {len(RANKING_METRICS)} metric rankings. "
        f"Higher n_rankings_top5 = more consistent across metrics."
    )

    return candidates


# ─── CSV writers ──────────────────────────────────────────────────────────────

def write_flat_csv(results: list[dict], out_path: Path):
    """Write all raw results to a flat CSV."""
    if not results:
        return

    # Collect all keys across all results
    all_keys = []
    seen = set()
    priority = [
        'run_id', 'stage', 'wave', 'arch', 'encoder', 'class_weights',
        'n_images', 'train_pct', 'val_pct', 'seed', 'augmentation',
        'checkpoint_metric', 'best_epoch', 'epochs_completed', 'early_stopped',
        'best_val_loss',
    ] + ALL_METRICS + ['train_stems', 'val_stems', 'model_path', 'result_path']

    for k in priority:
        if k not in seen:
            all_keys.append(k)
            seen.add(k)
    for r in results:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Written: {out_path} ({len(results)} rows)")


def write_summary_csv(summaries: list[dict], out_path: Path):
    """Write aggregated mean ± SD summary to CSV."""
    if not summaries:
        return

    # Build column order
    base_cols = ['arch', 'encoder', 'class_weights', 'n_runs']
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
    """Write a ranked table CSV."""
    write_summary_csv(rows, out_path)
    print(f"{label}: {out_path}")


# ─── Winner selection ─────────────────────────────────────────────────────────

def write_winner(
    groups:     dict,
    results:    list[dict],
    arch:       str,
    encoder:    str,
    weights:    list[float],
    split:      tuple[int, int],
    note:       str,
    out_path:   Path,
):
    """
    Write winner.json after manual review.
    Used by wave2_launcher.py and wave3_launcher.py.
    """
    w_str   = '_'.join(str(int(w)) if w == int(w) else str(w) for w in weights)
    key     = f"{arch}__{encoder}__{w_str}"
    combo_results = groups.get(key, [])

    if not combo_results:
        print(f"[WARN] No results found for {key} — winner.json written with config only")
        summary = {}
    else:
        summary = aggregate_combo(combo_results)

    train_pct, val_pct = split

    winner = {
        'arch':          arch,
        'encoder':       encoder,
        'class_weights': weights,
        'best_split':    [train_pct, val_pct],
        'best_val_fraction': round(val_pct / 100, 2),
        'selected_by':   note,
        'n_runs':        len(combo_results),
        'dice_macro_mean':  summary.get('dice_macro_mean'),
        'dice_axon_mean':   summary.get('dice_axon_mean'),
        'dice_myelin_mean': summary.get('dice_myelin_mean'),
        'hd95_axon_mean':   summary.get('hd95_axon_mean'),
        'hd95_myelin_mean': summary.get('hd95_myelin_mean'),
        'hd95_bg_mean':     summary.get('hd95_bg_mean'),
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepAxon Wave 1 aggregator")
    parser.add_argument('--config',  default='analysis_config.json')
    parser.add_argument('--select',  action='store_true',
                        help='Write winner.json after manual review')
    parser.add_argument('--arch',    help='Winning architecture (for --select)')
    parser.add_argument('--encoder', help='Winning encoder (for --select)')
    parser.add_argument('--weights', help='Winning class weights as comma-separated floats e.g. 3,1,1')
    parser.add_argument('--split',   help='Best TT split as train_pct,val_pct e.g. 70,30 (auto-detected if omitted)')
    parser.add_argument('--note',    default='Manual selection after Wave 1 review',
                        help='Note explaining selection rationale')
    parser.add_argument('--expected', type=int, default=5760,
                        help='Expected number of SW jobs (default 5760)')
    args = parser.parse_args()

    cfg         = load_config(args.config)
    results_dir = Path(cfg['output']['results_dir'])
    sw_dir      = results_dir / 'sw'
    out_dir     = results_dir.parent / 'aggregated'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load results ──────────────────────────────────────────────────────────
    results = load_all_results(sw_dir)
    check_completeness(results, args.expected)

    if not results:
        print("No results found — exiting.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    groups    = group_by_combo(results)
    summaries = [aggregate_combo(v) for v in groups.values()]
    print(f"Aggregated {len(summaries)} unique arch/encoder/weight combinations")

    # ── Write flat CSV ────────────────────────────────────────────────────────
    write_flat_csv(results, out_dir / 'wave1_all_results.csv')

    # ── Write summary CSV ─────────────────────────────────────────────────────
    write_summary_csv(summaries, out_dir / 'wave1_summary.csv')

    # ── Table 2 — Architecture comparison (encoder=resnet34) ──────────────────
    table2_rows = build_table2(groups, encoder='resnet34')
    write_table_csv(table2_rows, out_dir / 'table2_architecture.csv', 'Table 2')

    # ── Table 3 — Encoder comparison (winning arch) ───────────────────────────
    # Table 3 requires winning arch — use top Table 2 arch as provisional
    if table2_rows:
        provisional_winner_arch = table2_rows[0]['arch']
        table3_rows = build_table3(groups, provisional_winner_arch)
        write_table_csv(
            table3_rows,
            out_dir / f'table3_encoders_{provisional_winner_arch}.csv',
            f'Table 3 (provisional arch: {provisional_winner_arch})'
        )
        print(f"  Note: Table 3 uses provisional winner arch '{provisional_winner_arch}' — "
              f"rerun with --select to regenerate with confirmed winner")

    # ── Candidates ────────────────────────────────────────────────────────────
    candidates     = build_candidates(summaries)
    candidates_path = out_dir / 'candidates.json'
    with open(candidates_path, 'w') as f:
        json.dump(candidates, f, indent=2)
    print(f"Written: {candidates_path}")

    # ── Consensus summary to stdout ───────────────────────────────────────────
    print("\n── CONSENSUS TOP 5 ──────────────────────────────────────────────────")
    for i, c in enumerate(candidates.get('consensus_top5', []), 1):
        print(
            f"  {i}. {c['arch']} + {c['encoder']} weights={c['class_weights']} "
            f"| top5 in {c['n_rankings_top5']}/{len(RANKING_METRICS)} rankings "
            f"| dice_macro={c.get('dice_macro_mean')} "
            f"| hd95_axon={c.get('hd95_axon_mean')}"
        )

    # ── Winner selection ──────────────────────────────────────────────────────
    if args.select:
        if not all([args.arch, args.encoder, args.weights]):
            parser.error("--select requires --arch, --encoder, and --weights")
        weights = [float(w) for w in args.weights.split(',')]

        # Determine best split — use provided or auto-detect
        if args.split:
            train_pct, val_pct = [int(x) for x in args.split.split(',')]
        else:
            train_pct, val_pct = find_best_split(results, args.arch, args.encoder, weights)

        write_winner(
            groups   = groups,
            results  = results,
            arch     = args.arch,
            encoder  = args.encoder,
            weights  = weights,
            split    = (train_pct, val_pct),
            note     = args.note,
            out_path = out_dir / 'winner.json',
        )

        # Regenerate Table 3 with confirmed winner arch
        table3_rows = build_table3(groups, args.arch)
        write_table_csv(
            table3_rows,
            out_dir / f'table3_encoders_{args.arch}.csv',
            f'Table 3 (confirmed arch: {args.arch})'
        )

    print("\nReview candidates.json and table2/table3 CSVs, then run:")
    print("  python aggregator.py --config analysis_config.json --select "
          "--arch <arch> --encoder <enc> --weights <w1,w2,w3> "
          "[--split <train,val>] --note '<rationale>'")


if __name__ == '__main__':
    main()
