
#!/usr/bin/env python3
"""
Research audit: Entropy-Weighted COPRAS-FCM on BraTS FLAIR.

Evaluates real FLAIR/SEG pairs. Ground-truth labels are used only
for evaluation, never for selecting the predicted cluster.

Important:
- This is a reproducible research implementation, not clinical software.
- COPRAS with all-benefit criteria reduces to a weighted normalized sum.
- The proposed intensity/compactness criteria require validation.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes, mesh_surface_area


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


def fcm_fit(x, clusters=4, m=2.0, seed=42,
            max_iter=150, tol=1e-6, sample_size=30000):
    """Fit 1-D FCM centres on a deterministic voxel sample."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < clusters:
        raise ValueError("Too few voxels for FCM")

    rng = np.random.default_rng(seed)
    sample = x
    if x.size > sample_size:
        sample = x[rng.choice(x.size, sample_size, replace=False)]

    centers = np.percentile(
        sample, np.linspace(5, 95, clusters)
    ).astype(float)

    for _ in range(max_iter):
        dist = np.maximum(
            np.abs(sample[:, None] - centers[None, :]), 1e-8
        )
        inv = dist ** (-2.0 / (m - 1.0))
        u = inv / np.maximum(inv.sum(axis=1, keepdims=True), EPS)
        um = u ** m

        new = (
            (um * sample[:, None]).sum(axis=0)
            / np.maximum(um.sum(axis=0), EPS)
        )

        if np.max(np.abs(new - centers)) < tol:
            centers = new
            break
        centers = new

    return np.sort(centers)


def fcm_labels(vol, brain, centers, m=2.0, chunk=500000):
    """Assign each brain voxel to its maximum-membership FCM cluster."""
    x = vol[brain].astype(np.float64)
    labels = np.empty(x.size, dtype=np.int16)

    for start in range(0, x.size, chunk):
        xx = x[start:start + chunk, None]
        dist = np.maximum(
            np.abs(xx - centers[None, :]), 1e-8
        )
        inv = dist ** (-2.0 / (m - 1.0))
        u = inv / np.maximum(inv.sum(axis=1, keepdims=True), EPS)
        labels[start:start + len(xx)] = np.argmax(u, axis=1)

    full = np.full(vol.shape, -1, dtype=np.int16)
    full[brain] = labels
    return full


def compactness_3d(mask, spacing):
    """
    Dimensionless compactness: 36*pi*V^2/A^3.
    Volume and surface area use physical voxel spacing.
    """
    if mask.sum() < 5:
        return 0.0, 0.0, 0.0

    volume = float(mask.sum() * np.prod(spacing))

    try:
        verts, faces, _, _ = marching_cubes(
            np.pad(mask.astype(np.uint8), 1),
            level=0.5,
            spacing=spacing
        )
        area = float(mesh_surface_area(verts, faces))
        compact = 36.0 * np.pi * volume**2 / max(area**3, EPS)
        return volume, area, float(max(compact, 0.0))
    except Exception:
        return volume, np.nan, 0.0


def entropy_weights(matrix):
    """
    Standard entropy weights.
    Rows = candidate clusters; columns = criteria.
    p_ij = x_ij / sum_i(x_ij)
    E_j = -sum_i(p_ij ln(p_ij)) / ln(n)
    w_j = (1-E_j) / sum_j(1-E_j)
    """
    x = np.asarray(matrix, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)

    n, k = x.shape
    if n <= 1:
        return np.ones(k) / k, np.ones(k)

    totals = x.sum(axis=0)
    p = np.divide(
        x, totals[None, :],
        out=np.zeros_like(x),
        where=totals[None, :] > EPS
    )

    for j in range(k):
        if totals[j] <= EPS:
            p[:, j] = 1.0 / n

    terms = np.zeros_like(p)
    positive = p > 0
    terms[positive] = p[positive] * np.log(p[positive])

    entropy = -terms.sum(axis=0) / np.log(n)
    diversity = np.maximum(1.0 - entropy, 0.0)

    weights = (
        diversity / diversity.sum()
        if diversity.sum() > EPS
        else np.ones(k) / k
    )
    return weights, entropy


def copras_scores(matrix, weights):
    """
    COPRAS for all-benefit criteria:
    r_ij = x_ij / sum_i(x_ij)
    y_ij = r_ij * w_j
    Q_i = sum_j(y_ij)

    With all criteria beneficial, this is a weighted normalized sum.
    """
    x = np.maximum(np.asarray(matrix, dtype=float), 0.0)
    totals = x.sum(axis=0, keepdims=True)
    r = np.divide(
        x, totals,
        out=np.zeros_like(x),
        where=totals > EPS
    )
    return (r * weights[None, :]).sum(axis=1)


def largest_component(mask):
    labels, count = ndi.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def dice_score(pred, truth):
    den = int(pred.sum()) + int(truth.sum())
    if den == 0:
        return 1.0
    return float(
        2 * np.logical_and(pred, truth).sum() / den
    )


def hd95_mm(pred, truth, spacing):
    if not pred.any() or not truth.any():
        return np.nan

    ps = pred ^ ndi.binary_erosion(pred)
    ts = truth ^ ndi.binary_erosion(truth)

    d_to_truth = ndi.distance_transform_edt(
        ~ts, sampling=spacing
    )
    d_to_pred = ndi.distance_transform_edt(
        ~ps, sampling=spacing
    )

    distances = np.concatenate([
        d_to_truth[ps],
        d_to_pred[ts]
    ])
    return float(np.percentile(distances, 95))


def find_cases(root):
    cases = []
    for seg in sorted(root.rglob("*seg.nii*")):
        candidates = [
            p for p in seg.parent.glob("*.nii*")
            if "seg" not in p.name.lower()
            and (
                "flair" in p.name.lower()
                or "t2f" in p.name.lower()
            )
        ]
        if candidates:
            cases.append((seg.parent.name, candidates[0], seg))

    unique = []
    seen = set()
    for case in cases:
        key = (str(case[1]), str(case[2]))
        if key not in seen:
            unique.append(case)
            seen.add(key)
    return unique


def evaluate_case(case_id, flair_path, seg_path, seed=42):
    img = nib.load(str(flair_path))
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    seg_img = nib.load(str(seg_path))
    seg = np.asarray(seg_img.dataobj)

    if vol.shape != seg.shape:
        raise ValueError("FLAIR and SEG shapes differ")

    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    if len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError("Invalid voxel spacing")

    seg_spacing = tuple(
        float(s) for s in seg_img.header.get_zooms()[:3]
    )
    if not np.allclose(spacing, seg_spacing, rtol=1e-3, atol=1e-3):
        warnings.warn(f"{case_id}: FLAIR/SEG spacing differs")

    brain = np.isfinite(vol) & (vol > 0)
    if brain.sum() < 1000:
        raise ValueError("Too few nonzero brain voxels")

    # Whole-tumour ground truth; never used for cluster selection.
    truth = seg > 0
    if not truth.any():
        raise ValueError("Ground-truth tumour mask is empty")

    norm = normalize_flair(vol, brain)
    centers = fcm_fit(norm[brain], seed=seed)
    labels = fcm_labels(norm, brain, centers)

    masks = []
    criteria = []
    diagnostics = []

    for k in range(len(centers)):
        mask = labels == k
        masks.append(mask)

        mean_intensity = (
            float(norm[mask].mean()) if mask.any() else 0.0
        )
        volume, area, compact = compactness_3d(mask, spacing)

        # The paper proposes mean intensity and compactness as benefits.
        criteria.append([
            max(mean_intensity, 0.0),
            max(compact, 0.0)
        ])

        diagnostics.append({
            "case_id": case_id,
            "cluster": k,
            "center": float(centers[k]),
            "voxels": int(mask.sum()),
            "mean_intensity": mean_intensity,
            "volume_mm3": volume,
            "surface_area_mm2": area,
            "compactness": compact
        })

    matrix = np.asarray(criteria, dtype=float)
    weights, entropies = entropy_weights(matrix)
    scores = copras_scores(matrix, weights)

    selected = int(np.argmax(scores))
    pred_copras = largest_component(masks[selected])

    # Baselines use the same brain mask and same whole-tumour ground truth.
    pred_fcm = largest_component(masks[int(np.argmax(centers))])

    smooth = ndi.gaussian_filter(norm, sigma=1.0)
    smooth_centers = fcm_fit(smooth[brain], seed=seed)
    smooth_labels = fcm_labels(smooth, brain, smooth_centers)
    smooth_mask = smooth_labels == int(np.argmax(smooth_centers))
    pred_smooth = largest_component(smooth_mask)

    try:
        threshold = threshold_otsu(norm[brain])
        pred_otsu = largest_component((norm >= threshold) & brain)
    except Exception:
        pred_otsu = np.zeros(vol.shape, dtype=bool)

    row = {
        "case_id": case_id,
        "selected_cluster": selected,
        "cluster_centers": json.dumps(centers.tolist()),
        "copras_scores": json.dumps(scores.tolist()),
        "entropy_weights": json.dumps(weights.tolist()),
        "entropy_values": json.dumps(entropies.tolist()),
        "gt_voxels": int(truth.sum()),
        "spacing_mm": json.dumps(spacing)
    }

    for name, pred in [
        ("copras_fcm", pred_copras),
        ("fcm_brightest_cc", pred_fcm),
        ("smoothed_fcm_cc", pred_smooth),
        ("otsu_cc", pred_otsu)
    ]:
        row[name + "_dice"] = dice_score(pred, truth)
        row[name + "_hd95_mm"] = hd95_mm(pred, truth, spacing)
        row[name + "_pred_voxels"] = int(pred.sum())

    return row, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/kaggle/working/brats_100"
    )
    parser.add_argument(
        "--out",
        default="/kaggle/working/results_audited"
    )
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise FileNotFoundError(f"Input folder not found: {root}")

    cases = find_cases(root)[:args.n]
    print("Input:", root)
    print("Matched cases:", len(cases))

    if not cases:
        raise RuntimeError("No FLAIR/SEG pairs found.")

    rows, diagnostics, errors = [], [], []
    start = time.time()

    for i, (case_id, flair, seg) in enumerate(cases, 1):
        try:
            row, diag = evaluate_case(
                case_id, flair, seg, seed=args.seed
            )
            rows.append(row)
            diagnostics.extend(diag)
            print(
                f"{i}/{len(cases)} {case_id}: "
                f"COPRAS Dice={row['copras_fcm_dice']:.4f}"
            )
        except Exception as exc:
            errors.append({
                "case_id": case_id,
                "error": repr(exc)
            })
            print(f"{i}/{len(cases)} {case_id}: ERROR {exc}")

    pd.DataFrame(rows).to_csv(
        out / "per_case_results.csv", index=False
    )
    pd.DataFrame(diagnostics).to_csv(
        out / "cluster_diagnostics.csv", index=False
    )
    pd.DataFrame(errors).to_csv(
        out / "errors.csv", index=False
    )

    if not rows:
        raise RuntimeError("No cases completed successfully.")

    df = pd.DataFrame(rows)
    methods = [
        "copras_fcm", "fcm_brightest_cc",
        "smoothed_fcm_cc", "otsu_cc"
    ]
    summary = {}

    for method in methods:
        vals = df[method + "_dice"].dropna().to_numpy()
        summary[method] = {
            "n": int(len(vals)),
            "mean_dice": float(vals.mean()) if len(vals) else None,
            "sd_dice": (
                float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            )
        }

    info = {
        "input": str(root),
        "output": str(out),
        "requested_cases": args.n,
        "matched_cases": len(cases),
        "successful_cases": len(rows),
        "failed_cases": len(errors),
        "runtime_seconds": round(time.time() - start, 2),
        "summary": summary,
        "entropy_method": "standard Shannon entropy over candidate clusters",
        "criteria": ["mean normalized FLAIR intensity", "3D compactness"],
        "warning": (
            "This implementation requires scientific validation. "
            "All-benefit COPRAS reduces to a weighted normalized sum; "
            "performance and novelty are not guaranteed."
        )
    }

    (out / "summary.json").write_text(
        json.dumps(info, indent=2)
    )
    print("\nSUMMARY")
    print(json.dumps(info, indent=2))
    print("\nSaved output files to:", out)


if __name__ == "__main__":
    main()
