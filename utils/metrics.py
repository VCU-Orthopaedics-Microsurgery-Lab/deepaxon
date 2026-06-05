"""
utils/metrics.py

Centralized metric computation for DeepAxon.
Used by:
    train/train.py          — per-epoch lightweight metrics + post-training full suite
    segment/__main__.py     — post-segmentation evaluation against ground truth
    aggregator.py           — result validation

Two entry points:

    compute_epoch_metrics(tp, fp, fn, tn)
        Lightweight — Dice and IoU only (macro + per-class).
        Called every epoch during training loop.
        No HD95 — too expensive for per-epoch use.

    compute_all_metrics(preds, labels, device)
        Full suite — Dice, IoU, Precision, Recall, HD95 (macro + per-class).
        Called once after training on best checkpoint val set.
        Called by segment/__main__.py for ground truth evaluation.

Class order (consistent throughout):
    class 0 = background
    class 1 = myelin
    class 2 = axon
"""

from __future__ import annotations

import torch
import segmentation_models_pytorch as smp
from monai.metrics import HausdorffDistanceMetric


# ─── Class definitions ────────────────────────────────────────────────────────

NUM_CLASSES  = 3
CLASS_NAMES  = ['background', 'myelin', 'axon']
CLASS_BG     = 0
CLASS_MYELIN = 1
CLASS_AXON   = 2


# ─── Lightweight — per-epoch training metrics ─────────────────────────────────

def compute_epoch_metrics(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    tn: torch.Tensor,
) -> dict:
    """
    Compute Dice and IoU from smp confusion matrix tensors.
    Called every epoch — no HD95.

    Args:
        tp, fp, fn, tn: output of smp.metrics.get_stats()

    Returns dict with keys:
        dice_macro, iou_macro
        dice_bg, dice_myelin, dice_axon
        iou_bg,  iou_myelin,  iou_axon
    """
    dice_macro = smp.metrics.f1_score( tp, fp, fn, tn, reduction="macro").item()
    iou_macro  = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro").item()

    dice_pc = smp.metrics.f1_score( tp, fp, fn, tn, reduction="none")
    iou_pc  = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")

    return {
        'dice_macro':   round(dice_macro, 4),
        'iou_macro':    round(iou_macro,  4),
        'dice_bg':      round(dice_pc[:, CLASS_BG    ].mean().item(), 4),
        'dice_myelin':  round(dice_pc[:, CLASS_MYELIN].mean().item(), 4),
        'dice_axon':    round(dice_pc[:, CLASS_AXON  ].mean().item(), 4),
        'iou_bg':       round(iou_pc[:,  CLASS_BG    ].mean().item(), 4),
        'iou_myelin':   round(iou_pc[:,  CLASS_MYELIN].mean().item(), 4),
        'iou_axon':     round(iou_pc[:,  CLASS_AXON  ].mean().item(), 4),
    }


# ─── Full suite — post-training / post-segmentation evaluation ────────────────

def compute_all_metrics(
    preds:       torch.Tensor,
    labels:      torch.Tensor,
    device,
    num_classes: int = NUM_CLASSES,
) -> dict:
    """
    Compute full metric suite on a set of predictions and ground truth labels.
    Called once after training on best checkpoint val set.
    Called by segment/__main__.py for ground truth evaluation.

    Args:
        preds:       (N, H, W) int tensor — argmax class predictions
        labels:      (N, H, W) int tensor — ground truth class indices
        device:      torch device
        num_classes: number of segmentation classes (default 3)

    Returns flat dict with all metric fields — unpack directly into result.json:
        # Macro
        dice_macro, iou_macro, precision_macro, recall_macro, hd95_macro
        # Per-class Dice
        dice_bg, dice_myelin, dice_axon
        # Per-class IoU
        iou_bg, iou_myelin, iou_axon
        # Per-class Precision
        precision_bg, precision_myelin, precision_axon
        # Per-class Recall
        recall_bg, recall_myelin, recall_axon
        # Per-class HD95
        hd95_bg, hd95_myelin, hd95_axon

    Note: background class inflates macro Dice — flag this in reporting.
    Checkpoint metric is val_loss not val_dice — see methods.
    """
    preds  = preds.to(device)
    labels = labels.to(device)

    # ── smp confusion matrix ──────────────────────────────────────────────────
    tp, fp, fn, tn = smp.metrics.get_stats(
        preds, labels, mode="multiclass", num_classes=num_classes
    )

    # ── Macro metrics ─────────────────────────────────────────────────────────
    dice_macro      = smp.metrics.f1_score( tp, fp, fn, tn, reduction="macro").item()
    iou_macro       = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro").item()
    precision_macro = smp.metrics.precision(tp, fp, fn, tn, reduction="macro").item()
    recall_macro    = smp.metrics.recall(   tp, fp, fn, tn, reduction="macro").item()

    # ── Per-class metrics ─────────────────────────────────────────────────────
    dice_pc = smp.metrics.f1_score( tp, fp, fn, tn, reduction="none")
    iou_pc  = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
    prec_pc = smp.metrics.precision(tp, fp, fn, tn, reduction="none")
    rec_pc  = smp.metrics.recall(   tp, fp, fn, tn, reduction="none")

    def _pc(tensor, cls):
        return round(tensor[:, cls].mean().item(), 4)

    # ── HD95 via MONAI ────────────────────────────────────────────────────────
    # MONAI expects (N, C, H, W) one-hot float tensors on CPU
    preds_onehot  = torch.zeros(len(preds),  num_classes, *preds.shape[1:])
    labels_onehot = torch.zeros(len(labels), num_classes, *labels.shape[1:])
    for c in range(num_classes):
        preds_onehot[:, c]  = (preds.cpu()  == c).float()
        labels_onehot[:, c] = (labels.cpu() == c).float()

    hd_metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=95,
        reduction="mean",
    )
    
    hd_metric(y_pred=preds_onehot, y=labels_onehot)
    hd95_raw = hd_metric.aggregate(reduction="none")
    hd_metric.reset()
    if hd95_raw.ndim == 2:
        hd95_per_class = hd95_raw.nanmean(dim=0)              # (N, C) → (C,)
    else:
        hd95_per_class = hd95_raw                             # already (C,)

    hd95_bg          = round(hd95_per_class[CLASS_BG    ].item(), 4)
    hd95_myel        = round(hd95_per_class[CLASS_MYELIN].item(), 4)
    hd95_axon        = round(hd95_per_class[CLASS_AXON  ].item(), 4)
    hd95_macro       = round(hd95_per_class.nanmean().item(),     4)
    hd95_combined = round(hd95_per_class[[CLASS_MYELIN, CLASS_AXON]].nanmean().item(), 4)

    return {
        # Macro
        'dice_macro':       round(dice_macro,      4),
        'iou_macro':        round(iou_macro,       4),
        'precision_macro':  round(precision_macro, 4),
        'recall_macro':     round(recall_macro,    4),
        'hd95_macro':          hd95_macro,
        'hd95_myelin_axon':    hd95_combined,            #myelin + axon
        # Per-class Dice
        'dice_bg':          _pc(dice_pc, CLASS_BG),
        'dice_myelin':      _pc(dice_pc, CLASS_MYELIN),
        'dice_axon':        _pc(dice_pc, CLASS_AXON),
        # Per-class IoU
        'iou_bg':           _pc(iou_pc,  CLASS_BG),
        'iou_myelin':       _pc(iou_pc,  CLASS_MYELIN),
        'iou_axon':         _pc(iou_pc,  CLASS_AXON),
        # Per-class Precision
        'precision_bg':     _pc(prec_pc, CLASS_BG),
        'precision_myelin': _pc(prec_pc, CLASS_MYELIN),
        'precision_axon':   _pc(prec_pc, CLASS_AXON),
        # Per-class Recall
        'recall_bg':        _pc(rec_pc,  CLASS_BG),
        'recall_myelin':    _pc(rec_pc,  CLASS_MYELIN),
        'recall_axon':      _pc(rec_pc,  CLASS_AXON),
        # Per-class HD95
        'hd95_bg':          hd95_bg,
        'hd95_myelin':      hd95_myel,
        'hd95_axon':        hd95_axon,
    }
