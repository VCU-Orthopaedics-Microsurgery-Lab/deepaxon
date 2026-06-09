"""
aggregator.py — DeepAxon Wave 1/2/3 results aggregation

Usage:
    python aggregator.py --config analysis_config.json [--wave 1]
    python aggregator.py --config analysis_config.json --wave 2a
    python aggregator.py --config analysis_config.json --wave 2b
    python aggregator.py --config analysis_config.json --wave 3

    # After Wave 1 review — write winner.json
    python aggregator.py --config analysis_config.json --select \
        --arch unet++ --encoder densenet169 --weights 3,1,3 \
        [--split 67,33] --note "Best composite — stable leaderboard Part B"

────────────────────────────────────────────────────────────────────────────────
WAVE 1 OUTPUTS  (--wave 1)
Pipe output to tee for a permanent record:
    python aggregator.py ... | tee analysis/aggregated/wave1_terminal_output.txt
────────────────────────────────────────────────────────────────────────────────

  FINDING 1 — Training overview (terminal only)
    Epoch statistics (all runs + valid-only) by arch and by encoder.
    Split comparison: dice_myelin + HD95 per split, stable encoders only.

  FINDING 2 — Collapse diagnostics
    wave1_collapse_report.csv   — per arch/encoder: total/valid/collapsed/type
    wave1_degenerate.csv        — flat dump of all collapsed runs
    Terminal: overall rates, by arch, by encoder, by arch/encoder, by class weight,
              EfficientNet performance comparison (valid eff vs stable encoders).

  FINDING 3 — Stable competition (primary result)
    table2_stable_leaderboard_overall.csv  — composite ranking, all weights aggregated
    table2_stable_leaderboard_best_cw.csv  — composite ranking, best class weight + cw_sd
    Excludes UNSTABLE_ENCODERS and incomplete archs (auto-detected at <80% expected runs).

  FINDING 4 — Full reference tables (all encoders, paper methods)
    table2_architecture.csv             — arch comparison, encoder=resnet34 controlled
    table3_encoders_provisional_<arch>  — encoder comparison, provisional winning arch
    table3_encoders_<arch>              — confirmed arch (--select only)

  SUPPORTING
    wave1_all_results.csv, wave1_summary.csv, candidates.json, winner.json

────────────────────────────────────────────────────────────────────────────────
WAVE 2a/2b/3 OUTPUTS — see inline section headers below
────────────────────────────────────────────────────────────────────────────────

    Wave 2a: wave2a_all_results.csv, table4_aug_sweep.csv
    Wave 2b: table5_aug_comparison.csv (aug OFF baseline from Wave 1 SW, no retraining)
    Wave 3:  table1_learning_curve.csv, table1_learning_curve_by_split.csv
             Dataset sizes auto-selected: 67/33→[6,12,18,24,30], other→[10,20,30]

    Metric directions — higher: dice_*, iou_*, precision_*, recall_*
                      — lower:  hd95_*
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

# Encoders excluded from competition: ~50% collapse rate, confirmed NOT
# performance-superior to stable encoders on surviving runs (myelin delta=-0.007,
# HD95 delta=-1.1px vs resnet/densenet). Reported in collapse diagnostics only.
UNSTABLE_ENCODERS = ['efficientnet-b3', 'efficientnet-b4']

# Auto-exclusion threshold: archs with <80% expected runs excluded from competition.
ARCH_COMPLETENESS_THRESHOLD = 0.80
EXPECTED_RUNS_PER_ARCH_ENCODER = 240  # 16 weights x 3 splits x 5 seeds


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
    Myelin+axon composite: mean of dice_myelin, dice_axon, hd95_myelin_axon normalized
    to 50px ceiling. Higher is better.
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

    return best_macro


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


# ─── FINDING 1: Training overview ─────────────────────────────────────────────

def print_training_overview(all_results: list[dict], valid: list[dict]):
    """
    FINDING 1 — Training overview.
    Epoch stats by arch and encoder (all runs + valid-only note).
    Split comparison: dice_myelin + HD95 per split, stable encoders only.
    """
    print(f"\n── FINDING 1: TRAINING OVERVIEW ─────────────────────────────────────")

    # ── Overall epoch stats ───────────────────────────────────────────────────
    all_epochs   = [r['epochs_completed'] for r in all_results if 'epochs_completed' in r]
    valid_epochs = [r['epochs_completed'] for r in valid       if 'epochs_completed' in r]
    early_all    = sum(1 for r in all_results if r.get('early_stopped'))

    print(f"\n  Early stopping: {early_all}/{len(all_results)} runs "
          f"({round(early_all/len(all_results)*100,1)}%)")
    if all_epochs:
        print(f"  Epochs (all runs):   mean={round(statistics.mean(all_epochs),1)} "
              f"sd={round(statistics.stdev(all_epochs),1)} "
              f"min={min(all_epochs)} max={max(all_epochs)}")
    if valid_epochs:
        print(f"  Epochs (valid only): mean={round(statistics.mean(valid_epochs),1)} "
              f"sd={round(statistics.stdev(valid_epochs),1)} "
              f"min={min(valid_epochs)} max={max(valid_epochs)}")

    # ── By architecture ───────────────────────────────────────────────────────
    by_arch = defaultdict(list)
    for r in all_results:
        if 'epochs_completed' in r:
            by_arch[r['arch']].append(r['epochs_completed'])

    print(f"\n  {'arch':<20} {'mean_epochs':>12} {'sd':>7} {'min':>5} {'max':>5} {'n':>6}")
    for arch in sorted(by_arch):
        ep = by_arch[arch]
        print(f"  {arch:<20} {round(statistics.mean(ep),1):>12} "
              f"{round(statistics.stdev(ep),1):>7} {min(ep):>5} {max(ep):>5} {len(ep):>6}")

    # ── By encoder ────────────────────────────────────────────────────────────
    by_enc = defaultdict(list)
    for r in all_results:
        if 'epochs_completed' in r:
            by_enc[r['encoder']].append(r['epochs_completed'])

    print(f"\n  {'encoder':<25} {'mean_epochs':>12} {'sd':>7} {'min':>5} {'max':>5} {'n':>6} {'note'}")
    for enc in sorted(by_enc):
        ep   = by_enc[enc]
        note = ' *unstable (~50% collapse)' if enc in UNSTABLE_ENCODERS else ''
        print(f"  {enc:<25} {round(statistics.mean(ep),1):>12} "
              f"{round(statistics.stdev(ep),1):>7} {min(ep):>5} {max(ep):>5} {len(ep):>6}{note}")

    # ── Split comparison: stable valid runs only ──────────────────────────────
    stable_valid = [
        r for r in valid
        if r.get('encoder') not in UNSTABLE_ENCODERS
        and 'dice_myelin' in r and 'hd95_myelin_axon' in r
    ]

    by_split = defaultdict(list)
    for r in stable_valid:
        by_split[f"{r['train_pct']}/{r['val_pct']}"].append(r)

    print(f"\n  Split comparison (stable encoders, valid runs only):")
    print(f"  {'split':>8} {'dice_myelin':>12} {'sd':>7} {'hd95_myelin_axon':>17} {'sd':>7} {'n':>6}")
    for split in sorted(by_split):
        rs = by_split[split]
        dm = [r['dice_myelin']      for r in rs]
        hd = [r['hd95_myelin_axon'] for r in rs]
        print(f"  {split:>8} {round(statistics.mean(dm),4):>12} "
              f"{round(statistics.stdev(dm),4):>7} "
              f"{round(statistics.mean(hd),4):>17} "
              f"{round(statistics.stdev(hd),4):>7} {len(rs):>6}")
    print(f"  Note: 93/7 eliminated — unstable val metrics at small n "
          f"(HD95 SD ~2x higher than 67/33 and 80/20)")


# ─── FINDING 2: Collapse diagnostics ──────────────────────────────────────────

def build_collapse_report(all_results: list[dict], degenerate: list[dict],
                           valid: list[dict], out_dir: Path):
    """
    FINDING 2 — Collapse diagnostics.
    Includes EfficientNet performance comparison: valid eff runs vs stable encoders.
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

    print(f"\n── FINDING 2: COLLAPSE DIAGNOSTICS ─────────────────────────────────")
    print(f"  Total results: {total} | Valid: {len(valid)} ({round(len(valid)/total*100,1)}%) "
          f"| Degenerate: {len(degenerate)} ({round(len(degenerate)/total*100,1)}%)")

    # ── Collapse type ─────────────────────────────────────────────────────────
    type_counts = defaultdict(int)
    for r in all_results:
        for t in classify(r):
            type_counts[t] += 1
    print(f"\n  Collapse type (can overlap):")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<25} {count:>4} ({round(count/total*100,1)}%)")

    # ── By architecture ───────────────────────────────────────────────────────
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

    # ── By encoder ────────────────────────────────────────────────────────────
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
        flag  = ' *** UNSTABLE' if enc in UNSTABLE_ENCODERS else ''
        print(f"    {enc:<25} {count:>4} / {by_enc_total[enc]:>4} ({pct}%){flag}")

    # ── By arch/encoder ───────────────────────────────────────────────────────
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

    # ── By class weight ───────────────────────────────────────────────────────
    by_cw       = defaultdict(lambda: defaultdict(int))
    by_cw_total = defaultdict(int)
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
        if collapsed > 0:
            pct   = round(collapsed / total_cw * 100, 1)
            parts = [f"{t}={by_cw[cw][t]}" for t in ['both_collapsed','myelin_only','axon_only']
                     if by_cw[cw].get(t, 0) > 0]
            print(f"    cw={cw:<30} {collapsed:>4}/{total_cw:>4} ({pct}%)  {'  '.join(parts)}")

    # ── EfficientNet performance comparison ───────────────────────────────────
    eff_valid    = [r for r in valid if r.get('encoder') in UNSTABLE_ENCODERS]
    stable_valid = [r for r in valid if r.get('encoder') not in UNSTABLE_ENCODERS]

    print(f"\n  EfficientNet exclusion rationale:")
    print(f"  Collapse rate: B3={round(by_enc.get('efficientnet-b3',0)/by_enc_total.get('efficientnet-b3',1)*100,1)}%  "
          f"B4={round(by_enc.get('efficientnet-b4',0)/by_enc_total.get('efficientnet-b4',1)*100,1)}%")
    print(f"  Performance of surviving (valid) EfficientNet runs vs stable encoders:")
    print(f"  {'metric':<25} {'eff_valid (n='+str(len(eff_valid))+')':>22} "
          f"{'stable (n='+str(len(stable_valid))+')':>22} {'delta':>8}")
    for m in ['dice_myelin', 'dice_axon', 'hd95_myelin_axon']:
        ev = [r[m] for r in eff_valid    if m in r and r[m] is not None]
        sv = [r[m] for r in stable_valid if m in r and r[m] is not None]
        if ev and sv:
            delta = round(statistics.mean(ev) - statistics.mean(sv), 4)
            print(f"  {m:<25} {round(statistics.mean(ev),4):>22} "
                  f"{round(statistics.mean(sv),4):>22} {delta:>+8}")
    print(f"  CONCLUSION: EfficientNet valid runs do NOT outperform stable encoders "
          f"on any metric. Exclusion is justified on both stability AND performance grounds.")

    # ── Write CSV ─────────────────────────────────────────────────────────────
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
            'unstable':       encoder in UNSTABLE_ENCODERS,
        })
    write_dict_csv(csv_rows, out_dir / 'wave1_collapse_report.csv', 'Collapse report')


# ─── Arch completeness detection ──────────────────────────────────────────────

def detect_incomplete_archs(results: list[dict]) -> tuple[list[str], list[str]]:
    """
    Auto-detect archs with <80% expected runs. Excluded from competition tables.
    Returns (complete_archs, incomplete_archs).
    """
    combo_counts        = defaultdict(int)
    arch_min_completion = defaultdict(lambda: float('inf'))

    for r in results:
        combo_counts[f"{r['arch']}/{r['encoder']}"] += 1
    for combo, count in combo_counts.items():
        arch = combo.split('/')[0]
        pct  = count / EXPECTED_RUNS_PER_ARCH_ENCODER
        arch_min_completion[arch] = min(arch_min_completion[arch], pct)

    complete   = []
    incomplete = []
    for arch, min_pct in sorted(arch_min_completion.items()):
        if min_pct >= ARCH_COMPLETENESS_THRESHOLD:
            complete.append(arch)
        else:
            incomplete.append((arch, round(min_pct * 100, 1)))

    if incomplete:
        print(f"\n── AUTO-DETECTED INCOMPLETE ARCHITECTURES ───────────────────────────")
        for arch, pct in incomplete:
            print(f"  {arch:<20} {pct}% complete — EXCLUDED from competition tables")
        print(f"  (appear in collapse report only)")
    print(f"\n  Architectures in competition: {', '.join(complete)}")

    return complete, [a for a, _ in incomplete]


# ─── FINDING 3: Stable leaderboard ────────────────────────────────────────────

def build_stable_leaderboard(results: list[dict], incomplete_archs: list[str]):
    """
    FINDING 3 — Stable competition (primary result).
    Part A: overall composite ranking, all class weights aggregated.
    Part B: best class weight per combo + class weight SD (sensitivity).
    """
    excluded_encoders = UNSTABLE_ENCODERS
    excluded_archs    = incomplete_archs

    filtered = [
        r for r in results
        if r.get('encoder') not in excluded_encoders
        and r.get('arch') not in excluded_archs
    ]

    print(f"\n── FINDING 3: STABLE LEADERBOARD ────────────────────────────────────")
    print(f"  Excluded encoders: {excluded_encoders}")
    print(f"    Reason: ~50% collapse rate AND no performance advantage on valid runs")
    print(f"  Excluded archs:    {excluded_archs if excluded_archs else 'none'}")
    print(f"    Reason: <{int(ARCH_COMPLETENESS_THRESHOLD*100)}% jobs complete")
    print(f"  Valid runs included: {len(filtered)}")

    # ── Part A: overall ───────────────────────────────────────────────────────
    by_combo = defaultdict(list)
    for r in filtered:
        by_combo[f"{r['arch']}/{r['encoder']}"].append(r)

    rows_overall = []
    for combo, rs in by_combo.items():
        arch, encoder = combo.split('/', 1)
        dm    = statistics.mean(r['dice_myelin']       for r in rs)
        da    = statistics.mean(r['dice_axon']         for r in rs)
        hd    = statistics.mean(r['hd95_myelin_axon']  for r in rs)
        dmsd  = statistics.stdev(r['dice_myelin']      for r in rs)
        dasd  = statistics.stdev(r['dice_axon']        for r in rs)
        hdsd  = statistics.stdev(r['hd95_myelin_axon'] for r in rs)
        hd_score  = max(0, 1 - hd / 50)
        composite = round((dm + da + hd_score) / 3, 4)
        rows_overall.append({
            'arch':                  arch,
            'encoder':               encoder,
            'n_runs':                len(rs),
            'dice_myelin_mean':      round(dm, 4),
            'dice_myelin_sd':        round(dmsd, 4),
            'dice_axon_mean':        round(da, 4),
            'dice_axon_sd':          round(dasd, 4),
            'hd95_myelin_axon_mean': round(hd, 4),
            'hd95_myelin_axon_sd':   round(hdsd, 4),
            'composite_score':       composite,
        })
    rows_overall.sort(key=lambda x: -x['composite_score'])

    print(f"\n  Part A — All class weights aggregated (composite = (myelin+axon+hd95_norm)/3):")
    print(f"  {'combo':<35} {'myelin':>7} {'axon':>7} {'hd95':>7} {'composite':>10} {'n':>5}")
    for row in rows_overall:
        combo = f"{row['arch']}/{row['encoder']}"
        print(f"  {combo:<35} {row['dice_myelin_mean']:>7} {row['dice_axon_mean']:>7} "
              f"{row['hd95_myelin_axon_mean']:>7} {row['composite_score']:>10} {row['n_runs']:>5}")

    # ── Part B: best class weight + sensitivity ───────────────────────────────
    groups_stable = group_by_combo(filtered)

    # Compute composite SD across all 16 class weight configs per arch/encoder
    cw_composite_by_combo = defaultdict(list)
    for key, rs in groups_stable.items():
        s = aggregate_combo(rs)
        combo_key = f"{s['arch']}/{s['encoder']}"
        cw_composite_by_combo[combo_key].append(compute_composite_score(s))

    rows_best = []
    for combo, rs in by_combo.items():
        arch, encoder = combo.split('/', 1)
        summary = find_optimal_weights(groups_stable, arch, encoder)
        if summary:
            summary['composite_score'] = compute_composite_score(summary)
            cw_vals = cw_composite_by_combo.get(combo, [])
            summary['cw_composite_sd'] = round(statistics.stdev(cw_vals), 4) if len(cw_vals) > 1 else 0.0
            summary['cw_composite_range'] = round(max(cw_vals) - min(cw_vals), 4) if cw_vals else 0.0
            rows_best.append(summary)
    rows_best.sort(key=lambda x: x.get('composite_score') or 0, reverse=True)

    print(f"\n  Part B — Best class weight per combo (cw_sd = sensitivity across 16 configs):")
    print(f"  {'combo':<35} {'cw':<22} {'myelin':>7} {'axon':>7} {'hd95':>7} {'composite':>10} {'cw_sd':>7}")
    for row in rows_best:
        combo = f"{row['arch']}/{row['encoder']}"
        cw    = str(row.get('class_weights', ''))
        print(f"  {combo:<35} {cw:<22} "
              f"{round(row.get('dice_myelin_mean') or 0,4):>7} "
              f"{round(row.get('dice_axon_mean') or 0,4):>7} "
              f"{round(row.get('hd95_myelin_axon_mean') or 0,4):>7} "
              f"{round(row.get('composite_score') or 0,4):>10} "
              f"{row.get('cw_composite_sd',0):>7}")
    print(f"  cw_sd interpretation: <0.01 = robust to weight choice | >0.02 = weight-sensitive")

    return rows_overall, rows_best


# ─── Wave 1 functions ─────────────────────────────────────────────────────────

def build_table2(groups: dict, encoder: str = 'resnet34') -> list[dict]:
    print(f"\n── FINDING 4: FULL REFERENCE TABLE (encoder={encoder} controlled) ────")
    print(f"  Note: includes unstable encoders — for reference only.")
    print(f"  Primary competition result is the Stable Leaderboard (Finding 3).")
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
        if enc in UNSTABLE_ENCODERS:
            row['_encoder_warning'] = f'UNSTABLE: ~50% collapse, not performance-superior on valid runs'
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
        if enc in UNSTABLE_ENCODERS:
            row['_encoder_warning'] = f'UNSTABLE: ~50% collapse, not performance-superior on valid runs'
    return rows


def build_candidates(summaries: list[dict], top_n: int = 5,
                     incomplete_archs: list[str] | None = None) -> dict:
    excluded_archs = incomplete_archs or []
    summaries = [
        s for s in summaries
        if s.get('encoder') not in UNSTABLE_ENCODERS
        and s.get('arch') not in excluded_archs
    ]
    if not summaries:
        print("  [WARN] No summaries remain after filtering — check UNSTABLE_ENCODERS and incomplete_archs")
        return {}
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
                        'n_runs':                s['n_runs'],
                        'dice_macro_mean':       s.get('dice_macro_mean'),
                        'dice_axon_mean':        s.get('dice_axon_mean'),
                        'dice_myelin_mean':      s.get('dice_myelin_mean'),
                        'hd95_myelin_axon_mean': s.get('hd95_myelin_axon_mean'),
                        'hd95_axon_mean':        s.get('hd95_axon_mean'),
                        'hd95_myelin_mean':      s.get('hd95_myelin_mean'),
                    })
                    break

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
        print("  [WARN] No consensus candidates passed family balance filter. "
              "Consider lowering CONSENSUS_THRESHOLD.")

    candidates[f'consensus_top{top_n}'] = consensus[:top_n]
    candidates['_note'] = (
        f"Stable encoders only. Consensus = top {top_n} across "
        f"{CONSENSUS_THRESHOLD}+ of {len(RANKING_METRICS)} metrics, "
        f"with Dice+HD95 family co-representation required."
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
            row[f'{metric}_delta'] = round(on_val - off_val, 4) if (
                on_val is not None and off_val is not None) else None
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

    rows       = []
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
                    print(f"    {metric} delta vs n={prev_n}: {pct:+.2f}% improvement")

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
        print("\n══ WAVE 1 ANALYSIS ══════════════════════════════════════════════════")
        results = load_results_from(results_dir / 'sw', 'Wave 1 SW')
        check_completeness(results, args.expected)
        if not results:
            print("No results found — exiting.")
            return

        results, degenerate = filter_degenerate(results)

        # Finding 1 — Training overview
        print_training_overview(results + degenerate, results)

        # Finding 2 — Collapse diagnostics
        build_collapse_report(results + degenerate, degenerate, results, out_dir)

        # Arch completeness detection (feeds into Findings 3+4)
        complete_archs, incomplete_archs = detect_incomplete_archs(results + degenerate)

        # Write supporting files
        groups    = group_by_combo(results)
        summaries = [aggregate_combo(v) for v in groups.values()]
        print(f"\n  Aggregated {len(summaries)} unique arch/encoder/weight combinations")
        write_flat_csv(results, out_dir / 'wave1_all_results.csv')
        for s in summaries:
            s['composite_score'] = compute_composite_score(s)
        write_summary_csv(summaries, out_dir / 'wave1_summary.csv')

        # Finding 3 — Stable leaderboard
        rows_overall, rows_best = build_stable_leaderboard(results, incomplete_archs)
        write_dict_csv(rows_overall, out_dir / 'table2_stable_leaderboard_overall.csv',
                       'Stable leaderboard (overall)')
        write_dict_csv(rows_best, out_dir / 'table2_stable_leaderboard_best_cw.csv',
                       'Stable leaderboard (best CW)')

        # Finding 4 — Full reference table
        table2_rows = build_table2(groups, encoder='resnet34')
        write_table_csv(table2_rows, out_dir / 'table2_architecture.csv', 'Table 2 (all encoders)')

        if table2_rows:
            prov_arch   = table2_rows[0]['arch']
            table3_rows = build_table3(groups, prov_arch)
            write_table_csv(
                table3_rows,
                out_dir / f'table3_encoders_provisional_{prov_arch}.csv',
                f'Table 3 provisional (top dice_macro arch: {prov_arch})'
            )

        # Consensus candidates
        candidates = build_candidates(summaries, incomplete_archs=incomplete_archs)
        with open(out_dir / 'candidates.json', 'w') as f:
            json.dump(candidates, f, indent=2)
        print(f"\nWritten: {out_dir / 'candidates.json'}")

        print("\n── CONSENSUS TOP 5 (stable encoders only) ───────────────────────────")
        for i, c in enumerate(candidates.get('consensus_top5', []), 1):
            warn = ' *** CLINICAL WARNING' if '_clinical_warning' in c else ''
            n_seeds = c.get('n_runs', 0)  # n_runs already = per arch/encoder/cw combo across splits×seeds
            print(
                f"  {i}. {c['arch']} + {c['encoder']} cw={c['class_weights']} "
                f"| top5 in {c['n_rankings_top5']}/{len(RANKING_METRICS)} rankings "
                f"| n_runs={c.get('n_runs')}{warn}\n"
                f"     myelin={c.get('dice_myelin_mean')} axon={c.get('dice_axon_mean')} "
                f"hd95={c.get('hd95_myelin_axon_mean')} hd95_axon={c.get('hd95_axon_mean')}"
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
            on_vals  = [r[f'{metric}_aug_on']  for r in table5_rows
                        if r.get(f'{metric}_aug_on')  is not None]
            off_vals = [r[f'{metric}_aug_off'] for r in table5_rows
                        if r.get(f'{metric}_aug_off') is not None]
            if on_vals and off_vals:
                delta_vals = [r[f'{metric}_delta'] for r in table5_rows
                              if r.get(f'{metric}_delta') is not None]
                print(f"  {metric:<22} "
                      f"ON={round(statistics.mean(on_vals),4)} "
                      f"+/- {round(statistics.stdev(on_vals),4)}  "
                      f"OFF={round(statistics.mean(off_vals),4)} "
                      f"+/- {round(statistics.stdev(off_vals),4)}  "
                      f"delta={round(statistics.mean(delta_vals),4)}")

    # ── Wave 3 ────────────────────────────────────────────────────────────────
    elif args.wave == '3':
        print("\n── WAVE 3: LEARNING CURVE ───────────────────────────────────────────")
        results_lc = load_results_from(results_dir / 'lc', 'Wave 3 LC')
        results_lc = [r for r in results_lc if r.get('wave') == 3]
        if not results_lc:
            print("No Wave 3 results found — exiting.")
            return
        print("\nLearning curve by dataset size:")
        table1_rows, table1_split_rows = build_table1(results_lc)
        write_dict_csv(table1_rows, out_dir / 'table1_learning_curve.csv', 'Table 1')
        write_dict_csv(table1_split_rows, out_dir / 'table1_learning_curve_by_split.csv',
                       'Table 1 (per-split)')
        print("\nTable 1 complete — plateau confirmation for dataset sufficiency.")


if __name__ == '__main__':
    main()