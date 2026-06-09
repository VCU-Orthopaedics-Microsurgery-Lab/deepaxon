"""
aggregator.py

Reads Wave 1/2/3 result.json files and produces paper tables.

Outputs by wave:
    Wave 1 (--wave 1, default):
        wave1_all_results.csv   — flat dump of all 5,760 results
        wave1_summary.csv       — mean ± SD per arch/encoder/weights combo
        wave1_collapse_report.csv — collapse analysis by arch/encoder/class weight
        table2_architecture.csv — architecture comparison (encoder=resnet34)
        table3_encoders_<arch>  — encoder comparison (winning arch)
        candidates.json         — top 5 per metric + consensus
        winner.json             — written via --select after review

    Wave 2a (--wave 2a):
        wave2a_all_results.csv  — flat dump of all 2a results
        table4_aug_sweep.csv    — best params per aug type (Table 4)

    Wave 2b (--wave 2b):
        table5_aug_comparison.csv — aug ON vs aug OFF per phenotype (Table 5)

    Wave 3 (--wave 3):
        table1_learning_curve.csv — Dice ± SD vs dataset size (Table 1)
        table1_learning_curve_by_split.csv — per-split breakdown

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


def filter_degenerate(results: list[dict],
                       dice_macro_min: float = 0.5,
                       dice_myelin_min: float = 0.5,
                       dice_axon_min: float = 0.6) -> tuple[list[dict], list[dict]]:
    """
    Separate degenerate results from valid ones.
    Degenerate: model collapsed to background-only predictions.
    Cutoffs derived from bimodal distribution analysis of Wave 1 results.
    Returns (valid, degenerate).
    """
    valid = []
    degenerate = []
    for r in results:
        if (r.get('dice_macro',  1.0) < dice_macro_min or
            r.get('dice_myelin', 1.0) < dice_myelin_min or
            r.get('dice_axon',   1.0) < dice_axon_min):
            degenerate.append(r)
        else:
            valid.append(r)
    return valid, degenerate


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
    'hd95_macro', 'hd95_myelin_axon', 'hd95_axon', 'hd95_myelin', 'hd95_bg',
]

ALL_METRICS = HIGHER_IS_BETTER + LOWER_IS_BETTER

RANKING_METRICS = [
    ('dice_macro',       'higher'),
    ('dice_axon',        'higher'),
    ('dice_myelin',      'higher'),
    ('dice_bg',          'higher'),
    ('iou_axon',         'higher'),
    ('iou_myelin',       'higher'),
    ('hd95_macro',       'lower'),
    ('hd95_myelin_axon', 'lower'),
    ('hd95_axon',        'lower'),
    ('hd95_myelin',      'lower'),
    ('hd95_bg',          'lower'),
    ('precision_myelin', 'higher'),
    ('recall_myelin',    'higher'),
    ('precision_axon',   'higher'),
]

CONSENSUS_THRESHOLD = 5


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


def compute_composite_score(summary: dict) -> float:
    """
    Myelin+axon composite score for ranking.
    Mean of dice_myelin, dice_axon, and normalized hd95_myelin_axon (cap 50px).
    Higher is better.
    """
    dm = summary.get('dice_myelin_mean') or 0
    da = summary.get('dice_axon_mean')   or 0
    hd = summary.get('hd95_myelin_axon_mean') or float('inf')
    hd_score = max(0, 1 - hd / 50)
    return round((dm + da + hd_score) / 3, 4)


def find_optimal_weights(
    groups: dict, arch: str, encoder: str,
    primary_metric: str = 'dice_macro',
) -> dict | None:
    best_macro     = None
    best_macro_val = -float('inf')
    best_composite     = None
    best_composite_val = -float('inf')

    for key, results in groups.items():
        r0 = results[0]
        if r0['arch'] != arch or r0['encoder'] != encoder:
            continue
        summary = aggregate_combo(results)

        val = summary.get(f'{primary_metric}_mean')
        if val is not None and val > best_macro_val:
            best_macro_val = val
            best_macro     = summary

        dm  = summary.get('dice_myelin_mean') or 0
        da  = summary.get('dice_axon_mean')   or 0
        hd  = summary.get('hd95_myelin_axon_mean') or float('inf')
        hd_score  = max(0, 1 - hd / 50)
        composite = (dm + da + hd_score) / 3
        if composite > best_composite_val:
            best_composite_val = composite
            best_composite     = summary

    if best_macro and best_composite:
        same = (best_macro['class_weights'] == best_composite['class_weights'])
        print(
            f"  {arch}/{encoder} — "
            f"dice_macro winner:  cw={best_macro['class_weights']} "
            f"(macro={round(best_macro_val,4)}, "
            f"myelin={round(best_macro.get('dice_myelin_mean',0),4)}, "
            f"axon={round(best_macro.get('dice_axon_mean',0),4)})"
        )
        if not same:
            print(
                f"  {arch}/{encoder} — "
                f"composite winner:   cw={best_composite['class_weights']} "
                f"(composite={round(best_composite_val,4)}, "
                f"myelin={round(best_composite.get('dice_myelin_mean',0),4)}, "
                f"axon={round(best_composite.get('dice_axon_mean',0),4)})"
            )
        else:
            print(f"  {arch}/{encoder} — both methods agree on cw={best_macro['class_weights']}")

    return best_macro  # Table 2/3 still use dice_macro winner — override via --select


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
    base_cols   = ['arch', 'encoder', 'class_weights', 'n_runs', 'composite_score']
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


# ─── Collapse diagnostics ─────────────────────────────────────────────────────

def build_collapse_report(all_results: list[dict], degenerate: list[dict], out_dir: Path):
    """
    Full collapse diagnostic report — runs before degenerate filtering.
    Produces wave1_collapse_report.csv and prints full terminal breakdown.
    Called with the unfiltered result set.
    """
    total = len(all_results)

    def classify(r):
        types = []
        myelin_ok = r.get('dice_myelin', 1.0) >= 0.5
        axon_ok   = r.get('dice_axon',   1.0) >= 0.6
        if not myelin_ok and not axon_ok:
            types.append('both_collapsed')
        elif not myelin_ok:
            types.append('myelin_only')
        elif not axon_ok:
            types.append('axon_only')
        return types

    type_counts = defaultdict(int)
    for r in all_results:
        for t in classify(r):
            type_counts[t] += 1

    print(f"\n── COLLAPSE DIAGNOSTICS ─────────────────────────────────────────────")
    print(f"  Total results:    {total}")
    print(f"  Valid:            {total - len(degenerate)} ({round((total - len(degenerate))/total*100,1)}%)")
    print(f"  Degenerate:       {len(degenerate)} ({round(len(degenerate)/total*100,1)}%)")
    print(f"\n  Collapse type (can overlap):")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<25} {count:>4} ({round(count/total*100,1)}%)")

    by_arch       = defaultdict(int)
    by_arch_total = defaultdict(int)
    for r in all_results:
        by_arch_total[r['arch']] += 1
    for r in degenerate:
        by_arch[r['arch']] += 1

    print(f"\n  By architecture:")
    for arch in sorted(by_arch_total):
        count = by_arch.get(arch, 0)
        pct   = round(count / by_arch_total[arch] * 100, 1)
        print(f"    {arch:<20} {count:>4} / {by_arch_total[arch]:>4} ({pct}%)")

    by_enc       = defaultdict(int)
    by_enc_total = defaultdict(int)
    for r in all_results:
        by_enc_total[r['encoder']] += 1
    for r in degenerate:
        by_enc[r['encoder']] += 1

    print(f"\n  By encoder:")
    for enc in sorted(by_enc_total):
        count = by_enc.get(enc, 0)
        pct   = round(count / by_enc_total[enc] * 100, 1)
        print(f"    {enc:<25} {count:>4} / {by_enc_total[enc]:>4} ({pct}%)")

    by_combo       = defaultdict(lambda: defaultdict(int))
    by_combo_total = defaultdict(int)
    for r in all_results:
        by_combo_total[f"{r['arch']}/{r['encoder']}"] += 1
    for r in degenerate:
        combo = f"{r['arch']}/{r['encoder']}"
        for t in classify(r):
            by_combo[combo][t] += 1

    print(f"\n  By arch/encoder:")
    for combo in sorted(by_combo_total):
        total_combo = by_combo_total[combo]
        collapsed   = sum(by_combo[combo].values())
        if collapsed == 0:
            print(f"    {combo:<35} {0:>3}/{total_combo:>3} (0.0%)")
        else:
            pct   = round(collapsed / total_combo * 100, 1)
            parts = []
            for t in ['both_collapsed', 'myelin_only', 'axon_only']:
                c = by_combo[combo].get(t, 0)
                if c > 0:
                    parts.append(f"{t}={c}({round(c/total_combo*100,1)}%)")
            print(f"    {combo:<35} {collapsed:>3}/{total_combo:>3} ({pct}%)  {'  '.join(parts)}")

    by_cw         = defaultdict(lambda: defaultdict(int))
    by_cw_total   = defaultdict(int)
    for r in all_results:
        by_cw_total[str(r.get('class_weights', []))] += 1
    for r in degenerate:
        cw = str(r.get('class_weights', []))
        for t in classify(r):
            by_cw[cw][t] += 1

    print(f"\n  By class weight (collapsed/total):")
    for cw in sorted(by_cw_total):
        total_cw  = by_cw_total[cw]
        collapsed = sum(by_cw[cw].values())
        pct       = round(collapsed / total_cw * 100, 1)
        if collapsed > 0:
            parts = []
            for t in ['both_collapsed', 'myelin_only', 'axon_only']:
                c = by_cw[cw].get(t, 0)
                if c > 0:
                    parts.append(f"{t}={c}")
            print(f"    cw={cw:<30} {collapsed:>4}/{total_cw:>4} ({pct}%)  {'  '.join(parts)}")

    excl = [r for r in all_results
            if r.get('encoder') not in ['efficientnet-b3', 'efficientnet-b4']]
    excl_deg = [r for r in excl if
                r.get('dice_myelin', 1.0) < 0.5 or
                r.get('dice_axon',   1.0) < 0.6]
    print(f"\n  Excluding efficientnet encoders:")
    print(f"    Total:      {len(excl)}")
    print(f"    Valid:      {len(excl) - len(excl_deg)} ({round((len(excl)-len(excl_deg))/len(excl)*100,2)}%)")
    print(f"    Degenerate: {len(excl_deg)} ({round(len(excl_deg)/len(excl)*100,2)}%)")

    csv_rows = []
    for combo in sorted(by_combo_total):
        arch, encoder = combo.split('/', 1)
        total_combo   = by_combo_total[combo]
        collapsed     = sum(by_combo[combo].values())
        csv_rows.append({
            'arch':           arch,
            'encoder':        encoder,
            'total_runs':     total_combo,
            'collapsed':      collapsed,
            'valid':          total_combo - collapsed,
            'collapse_pct':   round(collapsed / total_combo * 100, 1),
            'both_collapsed': by_combo[combo].get('both_collapsed', 0),
            'myelin_only':    by_combo[combo].get('myelin_only', 0),
            'axon_only':      by_combo[combo].get('axon_only', 0),
        })

    write_dict_csv(csv_rows, out_dir / 'wave1_collapse_report.csv', 'Collapse report')


# ─── Wave 1 functions ─────────────────────────────────────────────────────────

def build_table2(groups: dict, encoder: str = 'resnet34') -> list[dict]:
    print(f"  [NOTE] Table 2 uses encoder={encoder} as controlled comparison. "
          f"Architecture rankings may differ with other encoders.")
    archs = sorted({r['arch'] for results in groups.values() for r in results})
    rows  = []
    for arch in archs:
        summary = find_optimal_weights(groups, arch, encoder)
        if summary:
            rows.append(summary)
    for row in rows:
        row['composite_score'] = compute_composite_score(row)
    rows.sort(key=lambda x: x.get('dice_macro_mean') or 0, reverse=True)
    for row in rows:
        enc = row.get('encoder', '')
        if enc in ('efficientnet-b3', 'efficientnet-b4'):
            row['_encoder_warning'] = f'High collapse rate (~48%) for {enc} — n_runs reflects valid runs only'
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
    for row in rows:
        row['composite_score'] = compute_composite_score(row)
    rows.sort(key=lambda x: x.get('dice_macro_mean') or 0, reverse=True)
    for row in rows:
        enc = row.get('encoder', '')
        if enc in ('efficientnet-b3', 'efficientnet-b4'):
            row['_encoder_warning'] = f'High collapse rate (~48%) for {enc} — n_runs reflects valid runs only'
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
                        'arch':                  arch,
                        'encoder':               encoder,
                        'class_weights':         s['class_weights'],
                        'n_rankings_top5':       count,
                        'dice_macro_mean':       s.get('dice_macro_mean'),
                        'dice_axon_mean':        s.get('dice_axon_mean'),
                        'dice_myelin_mean':      s.get('dice_myelin_mean'),
                        'hd95_myelin_axon_mean': s.get('hd95_myelin_axon_mean'),
                        'hd95_axon_mean':        s.get('hd95_axon_mean'),
                        'hd95_myelin_mean':      s.get('hd95_myelin_mean'),
                    })
                    break
    
    # Filter consensus: must appear in top 5 for at least 1 Dice metric
    # AND at least 1 HD95 metric — prevents single-family domination
    def _in_family(c, family_metrics):
        return any(
            e['arch'] == c['arch'] and e['encoder'] == c['encoder']
            and str(e['class_weights']) == str(c['class_weights'])
            for metric in family_metrics
            for e in candidates.get(f'by_{metric}', [])
        )

    consensus = [
        c for c in consensus
        if _in_family(c, ['dice_macro', 'dice_myelin', 'dice_axon'])
        and _in_family(c, ['hd95_myelin_axon', 'hd95_axon', 'hd95_myelin'])
    ]
    
    for c in consensus:
        in_myelin_top5 = any(
            e['arch'] == c['arch'] and e['encoder'] == c['encoder']
            and str(e['class_weights']) == str(c['class_weights'])
            for e in candidates.get('by_dice_myelin', [])
        )
        in_hd_top5 = any(
            e['arch'] == c['arch'] and e['encoder'] == c['encoder']
            and str(e['class_weights']) == str(c['class_weights'])
            for e in candidates.get('by_hd95_myelin_axon', [])
        )
        c['in_dice_myelin_top5']      = in_myelin_top5
        c['in_hd95_myelin_axon_top5'] = in_hd_top5
        if not in_myelin_top5:
            c['_clinical_warning'] = 'Not in top 5 for dice_myelin specifically — review before selecting'
    
    if not consensus:
        print("  [WARN] No consensus candidates passed family balance filter "
              "(Dice + HD95 co-representation required). "
              "Consider lowering CONSENSUS_THRESHOLD or reviewing RANKING_METRICS.")

    candidates[f'consensus_top{top_n}'] = consensus[:top_n]
    candidates['_note'] = (
        f"consensus = models in top {top_n} across "
        f"{CONSENSUS_THRESHOLD}+ of {len(RANKING_METRICS)} metric rankings, "
        f"with co-representation required in both Dice and HD95 metric families."
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
    'contrast_stretch': ['contrast_stretch_prob', 'contrast_stretch_scale'],
    'random_erase':     ['erase_prob', 'erase_scale'],
}


def _get_aug_type(result: dict) -> str:
    stage = result.get('stage', '')
    for aug_type in AUG_SWEEP_KEYS:
        if aug_type in stage:
            return aug_type
    return 'unknown'


def build_table4(results_2a: list[dict]) -> list[dict]:
    by_aug_type = defaultdict(list)
    for r in results_2a:
        aug_type = _get_aug_type(r)
        by_aug_type[aug_type].append(r)

    rows = []
    for aug_type, type_results in sorted(by_aug_type.items()):
        by_params = defaultdict(list)
        for r in type_results:
            aug_params    = r.get('aug_params', {})
            relevant_keys = AUG_SWEEP_KEYS.get(aug_type, [])
            param_key     = json.dumps(
                {k: aug_params.get(k) for k in relevant_keys}, sort_keys=True
            )
            by_params[param_key].append(r)

        best_mean         = -float('inf')
        best_params       = {}
        best_n            = 0
        best_metrics      = {}
        best_myelin_mean  = -float('inf')
        best_myelin_params= {}
        best_myelin_n     = 0

        for param_key, param_results in by_params.items():
            vals = [r['dice_macro'] for r in param_results
                    if 'dice_macro' in r and r['dice_macro'] is not None]
            if not vals:
                continue
            mean = statistics.mean(vals)
            if mean > best_mean:
                best_mean   = mean
                best_params = json.loads(param_key)
                best_n      = len(vals)
                for metric in ALL_METRICS:
                    m_vals = [r[metric] for r in param_results
                              if metric in r and r[metric] is not None]
                    mn, sd = _mean_sd(m_vals)
                    best_metrics[f'{metric}_mean'] = mn
                    best_metrics[f'{metric}_sd']   = sd

            myelin_vals = [r['dice_myelin'] for r in param_results
                           if 'dice_myelin' in r and r['dice_myelin'] is not None]
            if myelin_vals:
                myelin_mean = statistics.mean(myelin_vals)
                if myelin_mean > best_myelin_mean:
                    best_myelin_mean    = myelin_mean
                    best_myelin_params  = json.loads(param_key)
                    best_myelin_n       = len(myelin_vals)

        row = {'aug_type': aug_type, 'n_seeds': best_n, **best_params, **best_metrics}
        rows.append(row)

        best_myelin = best_metrics.get('dice_myelin_mean')
        best_hd     = best_metrics.get('hd95_myelin_axon_mean')
        print(f"  {aug_type}:")
        print(f"    dice_macro winner:  params={best_params} "
              f"dice_macro={round(best_mean, 4)} "
              f"| dice_myelin={round(best_myelin, 4) if best_myelin else 'N/A'} "
              f"| hd95_myelin_axon={round(best_hd, 4) if best_hd else 'N/A'} "
              f"(n={best_n})")
        if best_myelin_params != best_params:
            print(f"    dice_myelin winner: params={best_myelin_params} "
                  f"dice_myelin={round(best_myelin_mean, 4)} (n={best_myelin_n})")
        else:
            print(f"    both methods agree on params={best_params}")

    return rows


# ─── Wave 2b functions ────────────────────────────────────────────────────────

def build_table5(results_2b: list[dict], results_sw: list[dict], winner: dict) -> list[dict]:
    arch    = winner['arch']
    encoder = winner['encoder']
    weights = winner['class_weights']
    split   = winner['best_split']
    w_str   = '_'.join(str(int(w)) if w == int(w) else str(w) for w in weights)

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
        seed      = r_on.get('seed')
        r_off     = aug_off_by_seed.get(seed)
        val_stems = r_on.get('val_stems', [])
        row = {
            'seed':        seed,
            'n_val_ctrl':  sum(1 for s in val_stems if s.startswith('ctrl_')),
            'n_val_regen': sum(1 for s in val_stems if s.startswith('regen_')),
        }
        for metric in ['dice_macro', 'dice_axon', 'dice_myelin', 'dice_bg',
                       'hd95_macro', 'hd95_myelin_axon', 'hd95_axon', 'hd95_myelin']:
            row[f'{metric}_aug_on']  = r_on.get(metric)
            row[f'{metric}_aug_off'] = r_off.get(metric) if r_off else None
            on_val  = r_on.get(metric)
            off_val = r_off.get(metric) if r_off else None
            row[f'{metric}_delta'] = round(on_val - off_val, 4) if (on_val is not None and off_val is not None) else None
        rows.append(row)

    rows.sort(key=lambda x: x['seed'])
    return rows


# ─── Wave 3 functions ─────────────────────────────────────────────────────────

def build_table1(results_lc: list[dict]) -> tuple[list[dict], list[dict]]:
    by_size = defaultdict(list)
    for r in results_lc:
        if r.get('wave') == 3:
            n = r.get('n_images')
            if n is not None:
                by_size[n].append(r)

    rows      = []
    split_rows = []

    for n_images in sorted(by_size.keys()):
        size_results = by_size[n_images]
        row = {'n_images': n_images, 'n_runs': len(size_results)}
        for metric in ALL_METRICS:
            vals = [r[metric] for r in size_results
                    if metric in r and r[metric] is not None]
            mean, sd = _mean_sd(vals)
            row[f'{metric}_mean'] = mean
            row[f'{metric}_sd']   = sd

        if rows:
            prev = rows[-1]
            for metric in ['dice_macro', 'dice_myelin', 'hd95_myelin_axon']:
                curr_val = row.get(f'{metric}_mean')
                prev_val = prev.get(f'{metric}_mean')
                if curr_val is not None and prev_val is not None and prev_val != 0:
                    if metric in LOWER_IS_BETTER:
                        pct = round((prev_val - curr_val) / prev_val * 100, 2)
                    else:
                        pct = round((curr_val - prev_val) / prev_val * 100, 2)
                    row[f'plateau_pct_{metric}'] = pct
                else:
                    row[f'plateau_pct_{metric}'] = None
        else:
            for metric in ['dice_macro', 'dice_myelin', 'hd95_myelin_axon']:
                row[f'plateau_pct_{metric}'] = None

        rows.append(row)

        print(f"  n={n_images} (n_runs={len(size_results)}):")
        print(f"    dice_macro={row.get('dice_macro_mean')} ± {row.get('dice_macro_sd')}")
        print(f"    dice_myelin={row.get('dice_myelin_mean')} ± {row.get('dice_myelin_sd')}")
        print(f"    hd95_myelin_axon={row.get('hd95_myelin_axon_mean')} ± {row.get('hd95_myelin_axon_sd')}")

        prev_n = rows[-2]['n_images'] if len(rows) >= 2 else None
        if prev_n is not None:
            for metric in ['dice_macro', 'dice_myelin', 'hd95_myelin_axon']:
                pct = row.get(f'plateau_pct_{metric}')
                if pct is not None:
                    print(f"    {metric} Δ vs n={prev_n}: {pct:+.2f}% improvement")

        by_split = defaultdict(list)
        for r in size_results:
            split_key = f"{r.get('train_pct')}_{r.get('val_pct')}"
            by_split[split_key].append(r)

        for split_key, split_results in sorted(by_split.items()):
            train_pct, val_pct = split_key.split('_')
            split_row = {
                'n_images':  n_images,
                'train_pct': int(train_pct),
                'val_pct':   int(val_pct),
                'n_runs':    len(split_results),
            }
            for metric in ALL_METRICS:
                vals = [r[metric] for r in split_results
                        if metric in r and r[metric] is not None]
                mean, sd = _mean_sd(vals)
                split_row[f'{metric}_mean'] = mean
                split_row[f'{metric}_sd']   = sd
            split_rows.append(split_row)

    return rows, split_rows


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

        results, degenerate = filter_degenerate(results)
        build_collapse_report(results + degenerate, degenerate, out_dir)

        if degenerate:
            print(f"\n[WARN] {len(degenerate)} degenerate results excluded "
                  f"(dice_macro<0.5, dice_myelin<0.5, or dice_axon<0.6)")
            by_arch = {}; by_arch_total = {}
            by_enc  = {}; by_enc_total  = {}
            by_combo= {}; by_combo_total= {}

            for r in results + degenerate:
                arch = r['arch']; enc = r['encoder']
                by_arch_total[arch] = by_arch_total.get(arch, 0) + 1
                by_enc_total[enc]   = by_enc_total.get(enc,  0) + 1
                by_combo_total[f"{arch}/{enc}"] = by_combo_total.get(f"{arch}/{enc}", 0) + 1

            for r in degenerate:
                arch = r['arch']; enc = r['encoder']
                by_arch[arch]             = by_arch.get(arch, 0) + 1
                by_enc[enc]               = by_enc.get(enc,   0) + 1
                by_combo[f"{arch}/{enc}"] = by_combo.get(f"{arch}/{enc}", 0) + 1

            for r in degenerate:
                arch = r['arch']; enc = r['encoder']
                r['collapse_pct_arch']    = round(by_arch[arch]  / by_arch_total[arch]  * 100, 1)
                r['collapse_pct_encoder'] = round(by_enc[enc]    / by_enc_total[enc]    * 100, 1)
                r['collapse_pct_combo']   = round(by_combo[f"{arch}/{enc}"] / by_combo_total[f"{arch}/{enc}"] * 100, 1)
            write_flat_csv(degenerate, out_dir / 'wave1_degenerate.csv')

        groups    = group_by_combo(results)
        summaries = [aggregate_combo(v) for v in groups.values()]
        print(f"Aggregated {len(summaries)} unique arch/encoder/weight combinations")

        write_flat_csv(results, out_dir / 'wave1_all_results.csv')
        for s in summaries:
            s['composite_score'] = compute_composite_score(s)
        write_summary_csv(summaries, out_dir / 'wave1_summary.csv')

        table2_rows = build_table2(groups, encoder='resnet34')
        write_table_csv(table2_rows, out_dir / 'table2_architecture.csv', 'Table 2')

        if table2_rows:
            prov_arch   = table2_rows[0]['arch']
            table3_rows = build_table3(groups, prov_arch)
            write_table_csv(
                table3_rows,
                out_dir / f'table3_encoders_provisional_{prov_arch}.csv',
                f'Table 3 provisional (top dice_macro arch: {prov_arch})'
            )

        candidates = build_candidates(summaries)
        with open(out_dir / 'candidates.json', 'w') as f:
            json.dump(candidates, f, indent=2)
        print(f"Written: {out_dir / 'candidates.json'}")

        print("\n── CONSENSUS TOP 5 ──────────────────────────────────────────────────")
        for i, c in enumerate(candidates.get('consensus_top5', []), 1):
            warn = ' ⚠ CLINICAL WARNING' if '_clinical_warning' in c else ''
            print(
                f"  {i}. {c['arch']} + {c['encoder']} weights={c['class_weights']} "
                f"| top5 in {c['n_rankings_top5']}/{len(RANKING_METRICS)} rankings{warn}\n"
                f"       dice_macro={c.get('dice_macro_mean')} "
                f"| dice_myelin={c.get('dice_myelin_mean')} "
                f"| dice_axon={c.get('dice_axon_mean')} "
                f"| hd95_myelin_axon={c.get('hd95_myelin_axon_mean')} "
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
                f'Table 3 confirmed arch: {args.arch}'
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

        print("\n── AUG ON vs OFF SUMMARY ────────────────────────────────────────────")
        for metric in ['dice_macro', 'dice_myelin', 'dice_axon', 'hd95_myelin_axon']:
            on_vals  = [r[f'{metric}_aug_on']  for r in table5_rows if r.get(f'{metric}_aug_on')  is not None]
            off_vals = [r[f'{metric}_aug_off'] for r in table5_rows if r.get(f'{metric}_aug_off') is not None]
            if on_vals and off_vals:
                delta_vals = [r[f'{metric}_delta'] for r in table5_rows if r.get(f'{metric}_delta') is not None]
                print(f"  {metric:<22} "
                      f"ON={round(statistics.mean(on_vals), 4)} ± {round(statistics.stdev(on_vals), 4)}  "
                      f"OFF={round(statistics.mean(off_vals), 4)} ± {round(statistics.stdev(off_vals), 4)}  "
                      f"delta={round(statistics.mean(delta_vals), 4)}")

    # ── Wave 3 ────────────────────────────────────────────────────────────────
    elif args.wave == '3':
        print("\n── WAVE 3 — LEARNING CURVE ──────────────────────────────────────────")
        results_lc = load_results_from(results_dir / 'lc', 'Wave 3 LC')
        results_lc = [r for r in results_lc if r.get('wave') == 3]

        if not results_lc:
            print("No Wave 3 results found — exiting.")
            return

        print("\nLearning curve by dataset size:")
        table1_rows, table1_split_rows = build_table1(results_lc)
        write_dict_csv(table1_rows,       out_dir / 'table1_learning_curve.csv',          'Table 1')
        write_dict_csv(table1_split_rows, out_dir / 'table1_learning_curve_by_split.csv', 'Table 1 (per-split)')
        print("\nTable 1 complete — plateau confirmation for dataset sufficiency.")


if __name__ == '__main__':
    main()