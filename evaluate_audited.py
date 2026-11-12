#!/usr/bin/env python3
"""
Research audit: Entropy-Weighted COPRAS-FCM on BraTS FLAIR.
Fixed Version: Top-2 Cluster Selection + Hole Filling + Local Contrast Criteria.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import threshold_otsu

EPS = 1e-12

def normalize_flair(vol, brain):
    out = np.zeros(vol.shape, dtype=np.float32)
    vals = vol[brain]
    if vals.size == 0:
        return out
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return out
    out[brain] = np.clip((vals - lo) / (hi - lo), 0, 1)
    return out

def fcm_fit(x, clusters, m=2.0, seed=42, max_iter=50, sample_size=100000):
    if x.size < clusters:
        raise ValueError("Too few voxels for FCM")
    rng = np.random.default_rng(seed)
    sample = x
    if x.size > sample_size:
        sample = rng.choice(x.size, sample_size, replace=False)
        sample = x[sample]
        
    centers = np.percentile(sample, np.linspace(5, 95, clusters)).astype(float)
    
    for _ in range(max_iter):
        dist = np.maximum(np.abs(sample[:, None] - centers[None, :]), 1e-8)
        inv = dist ** (-2.0 / (m - 1.0))
        u = inv / np.maximum(inv.sum(axis=1, keepdims=True), EPS)
        u_m = u ** m
        new = (u_m * sample[:, None]).sum(axis=0) / np.maximum(u_m.sum(axis=0), EPS)
        if np.max(np.abs(new - centers)) < 1e-5:
            centers = new
            break
        centers = new
    return np.sort(centers)

def fcm_labels(vol, brain, centers, m=2.0, chunk=500000):
    x = vol[brain].astype(np.float64)
    labels = np.empty(x.size, dtype=np.int16)
    for start in range(0, x.size, chunk):
        xx = x[start:start + chunk]
        dist = np.maximum(np.abs(xx[:, None] - centers[None, :]), 1e-8)
        inv = dist ** (-2.0 / (m - 1.0))
        u = inv / np.maximum(inv.sum(axis=1, keepdims=True), EPS)
        labels[start:start + len(xx)] = np.argmax(u, axis=1)
    full = np.full(vol.shape, -1, dtype=np.int16)
    full[brain] = labels
    return full

def entropy_weights(matrix):
    x = np.asarray(matrix, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    n, k = x.shape
    if n <= 1:
        return np.ones(k) / k, np.ones(k)
    totals = x.sum(axis=0)
    p = np.divide(x, totals[None, :], out=np.zeros_like(x), where=totals[None, :] > EPS)
    entropy = np.zeros(k)
    for j in range(k):
        if totals[j] <= EPS:
            entropy[j] = 1.0
            continue
        pj = p[:, j]
        pos = pj > 0
        if pos.any():
            entropy[j] = -np.sum(pj[pos] * np.log(pj[pos])) / np.log(n)
    diversity = np.maximum(1.0 - entropy, 0.0)
    div_sum = diversity.sum()
    weights = diversity / div_sum if div_sum > EPS else np.ones(k) / k
    return weights, entropy

def copras_scores(matrix, weights):
    x = np.maximum(np.asarray(matrix, dtype=float), 0.0)
    totals = x.sum(axis=0, keepdims=True)
    r = np.divide(x, totals, out=np.zeros_like(x), where=totals > EPS)
    return (r * weights[None, :]).sum(axis=1)

# ==========================================
# FIX 1: ROBUST POST-PROCESSING (Hole Filling)
# ==========================================
def robust_postprocess(mask):
    """Fills holes, keeps largest component, and smooths boundaries."""
    # 1. Fill internal holes (Necrotic core)
    mask = ndimage.binary_fill_holes(mask)
    
    # 2. Keep only the largest connected component
    labeled, count = ndimage.label(mask)
    if count > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # Ignore background
        mask = (labeled == np.argmax(sizes))
        
    # 3. Morphological closing (Smooth boundaries)
    mask = ndimage.binary_closing(mask, structure=np.ones((3,3,3)))
    return mask

def dice_score(pred, truth):
    den = int(pred.sum()) + int(truth.sum())
    if den == 0:
        return 1.0
    return float(2 * np.logical_and(pred, truth).sum() / den)

def hd95_mm(pred, truth, spacing):
    if not pred.any() or not truth.any():
        return np.nan
    pg = pred ^ ndimage.binary_erosion(pred)
    tg = truth ^ ndimage.binary_erosion(truth)
    d_to_truth = ndimage.distance_transform_edt(~tg, sampling=spacing)
    d_to_pred = ndimage.distance_transform_edt(~pg, sampling=spacing)
    distances = np.concatenate([d_to_truth[pg], d_to_pred[tg]])
    return float(np.percentile(distances, 95))

def find_cases(root):
    cases = []
    for seg in sorted(Path(root).glob("*seg*.nii*")):
        candidates = [p for p in seg.parent.glob("*.nii*") if "seg" not in p.name.lower() and ("flair" in p.name.lower() or "fl" in p.name.lower())]
        if candidates:
            cases.append((seg.parent.name, candidates[0], seg))
    return cases

def evaluate_case(case_id, flair_path, seg_path, seed=42):
    img = nib.load(str(flair_path))
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    seg_img = nib.load(str(seg_path))
    seg = np.asarray(seg_img.dataobj) > 0
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])

    if vol.shape != seg.shape:
        raise ValueError("Shape mismatch")
    
    brain = np.logical_or(vol > 0, seg > 0)
    truth = seg > 0
    
    norm = normalize_flair(vol, brain)
    centers = fcm_fit(norm[brain], 4, seed=seed)
    labels = fcm_labels(norm, brain, centers)
    
    rs, criteria, diagnostics = [], [], []
    for k in range(len(centers)):
        mask = (labels == k)
        rs.append(mask)
        mean_int = float(norm[mask].mean()) if mask.any() else 0.0
        
        # ==========================================
        # FIX 2: LOCAL CONTRAST (Replaces Compactness)
        # ==========================================
        dilated = ndimage.binary_dilation(mask, iterations=3)
        ring = dilated & ~mask & brain
        ring_mean = float(norm[ring].mean()) if ring.any() else 0.0
        local_contrast = (mean_int / (ring_mean + 1e-5)) if ring_mean > 0 else 1.0
        
        criteria.append([max(mean_int, 0.0), max(local_contrast, 0.0)])
        diagnostics.append({"case_id": case_id, "cluster": k, "mean_int": mean_int, "contrast": local_contrast})
        
    matrix = np.array(criteria, dtype=float)
    weights, _ = entropy_weights(matrix)
    scores = copras_scores(matrix, weights)
    
    # ==========================================
    # FIX 3: TOP-2 CLUSTERS SELECTION
    # ==========================================
    sorted_idx = np.argsort(scores)[::-1]
    top_2 = sorted_idx[:2]
    
    # Union of top-2 clusters
    pred_raw = rs[top_2[0]] | rs[top_2[1]]
    
    # Apply the robust post-processing
    pred_copras = robust_postprocess(pred_raw)
    
    # Other baselines (Using same post-processing for fair comparison)
    pred_fcm = robust_postprocess(rs[np.argmax(centers)])
    
    smooth_vol = ndimage.gaussian_filter(norm, sigma=1.0)
    smooth_centers = fcm_fit(smooth_vol[brain], 4, seed=seed)
    smooth_labels = fcm_labels(smooth_vol, brain, smooth_centers)
    pred_sfcm = robust_postprocess(smooth_labels == np.argmax(smooth_centers))
    
    try:
        th = threshold_otsu(norm[brain])
        pred_otsu = robust_postprocess((norm > th) & brain)
    except Exception:
        pred_otsu = np.zeros(vol.shape, dtype=bool)

    row = {"case_id": case_id, "selected_clusters": str(top_2.tolist())}
    for name, pred in [("copras", pred_copras), ("fcm", pred_fcm), ("sfcm", pred_sfcm), ("otsu", pred_otsu)]:
        row[f"{name}_dice"] = dice_score(pred, truth)
        row[f"{name}_hd95"] = hd95_mm(pred, truth, spacing)

    return row, diagnostics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/kaggle/working/brats_100")
    parser.add_argument("--out", default="/kaggle/working/results_audited_fixed")
    parser.add_argument("-n", type=int, default=100) # Change this to 10 for quick test
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = find_cases(args.input)[:args.n]
    if not cases:
        raise RuntimeError("No cases found.")

    rows, diags = [], []
    print(f"Running on {len(cases)} cases...")
    
    for i, (cid, f_img, f_seg) in enumerate(cases, 1):
        try:
            row, diag = evaluate_case(cid, f_img, f_seg, seed=args.seed)
            rows.append(row)
            diags.extend(diag)
            print(f"[{i}/{len(cases)}] {cid} - COPRAS Dice: {row['copras_dice']:.4f}")
        except Exception as e:
            print(f"[{i}/{len(cases)}] {cid} - ERROR: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    pd.DataFrame(diags).to_csv(out / "diagnostics.csv", index=False)

    print("\n--- FINAL MEAN RESULTS ---")
    summary = df[["copras_dice", "fcm_dice", "sfcm_dice", "otsu_dice"]].mean()
    print(summary)
    df[["copras_hd95", "fcm_hd95", "sfcm_hd95", "otsu_hd95"]].mean().to_csv(out / "hd95_summary.csv")

if __name__ == "__main__":
    main()
