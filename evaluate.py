#!/usr/bin/env python3
"""
BraTS FLAIR evaluation: FCM baselines and COPRAS-FCM ablations.

Run in Kaggle:
  !python mri-fuzzy-segmentation/evaluate.py --n 100 --out-name results_dev
  !python mri-fuzzy-segmentation/evaluate.py --start 100 --n 300 --out-name results_test

IMPORTANT:
- Development cases may be used for design decisions; held-out cases must not be
  used to tune parameters.
- This script does not guarantee COPRAS-FCM outperforms any baseline.
- Existing output CSV is resumed only within the chosen output directory.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.cluster.vq import kmeans2
from scipy.stats import wilcoxon

# FIX (Bug 5): removed useless "t2f" / "t2w" keys; kept only real BraTS substrings.
MODALITY_KEYS = {
    "flair": ("flair",),
    "t2": ("_t2", "-t2"),
}
DIAGNOSTICS = []

# FIX (Bug 22): single source of truth for FCM sample size, shared by fit and COPRAS.
FCM_SAMPLE_SIZE = 30000


def normalise(vol, brain):
    vals = vol[brain]
    # FIX (Bug 1): guard against <2 voxels so np.percentile never misbehaves.
    if vals.size < 2:
        return np.zeros_like(vol, dtype=np.float32)
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - lo) / (hi - lo), 0, 1).astype(np.float32)


def memberships(x, centres, m):
    d = np.abs(x[:, None] - centres[None, :]) + 1e-8
    inv = d ** (-2.0 / (m - 1.0))
    return inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-12)


def subsample(x, n, seed):
    if x.size <= n:
        return x
    return x[np.random.default_rng(seed).choice(x.size, size=n, replace=False)]


def fcm_fit(x, c, m, seed, iters=150, tol=1e-6, n_sample=FCM_SAMPLE_SIZE):
    """
    FIX (Bug 22): returns (centres, sample) so callers can reuse the exact
    sample that produced the centres. This removes the 30k-vs-40k mismatch
    between fcm_fit and copras_mask.
    """
    sample = subsample(np.asarray(x), n_sample, seed)
    centres = np.percentile(sample, np.linspace(5, 95, c))
    for _ in range(iters):
        w = memberships(sample, centres, m) ** m
        new = (w * sample[:, None]).sum(0) / np.maximum(w.sum(0), 1e-12)
        converged = np.max(np.abs(new - centres)) < tol
        centres = new
        if converged:
            break
    return np.sort(centres), sample


def brightest_mask(x, centres, m, threshold=0.0, chunk=500000):
    result = np.zeros(x.size, dtype=bool)
    k = int(np.argmax(centres))
    for start in range(0, x.size, chunk):
        u = memberships(x[start:start + chunk], centres, m)
        result[start:start + chunk] = (
            (u.argmax(axis=1) == k) & (u[:, k] >= threshold)
        )
    return result


def otsu_threshold(x, bins=256):
    hist, edges = np.histogram(x, bins=bins, range=(0, 1))
    prob = hist.astype(float) / max(hist.sum(), 1)
    weight = np.cumsum(prob)
    means = np.cumsum(prob * ((edges[:-1] + edges[1:]) / 2))
    denom = weight * (1 - weight)
    score = (means[-1] * weight - means) ** 2 / (denom + 1e-12)
    score[(weight <= 0) | (weight >= 1)] = 0
    idx = int(np.argmax(score))
    # FIX (Bug 2): explicit guard against off-by-one at the last bin.
    return edges[idx + 1] if idx + 1 < len(edges) else edges[-1]


def postprocess(mask, keep_frac=0.15):
    if not mask.any():
        return mask
    labels, count = ndi.label(mask)
    if count <= 1:
        return mask
    sizes = ndi.sum(mask, labels, range(1, count + 1))
    keep = np.where(sizes >= keep_frac * sizes.max())[0] + 1
    return np.isin(labels, keep)


def copras_select_cluster(x_sample, u_sample, centres):
    """
    Fixed-weight COPRAS decision score.
    Criteria: intensity (+), mean membership (+), intensity SD (-),
    distance of cluster fraction from target fraction (-).
    This is not entropy weighting.
    """
    c = len(centres)
    labels = u_sample.argmax(axis=1)
    rows, valid_ids = [], []
    for k in range(c):
        chosen = labels == k
        if chosen.sum() < 10:
            continue
        frac = float(chosen.mean())
        rows.append([
            float(centres[k]) + 1e-6,
            float(u_sample[chosen, k].mean()) + 1e-6,
            float(x_sample[chosen].std()) + 1e-4,
            abs(frac - 0.06) + (10.0 if (frac > 0.35 or frac < 0.002) else 0.001),
        ])
        valid_ids.append(k)

    scores = np.full(c, -np.inf)
    if not rows:
        return int(np.argmax(centres)), scores

    X = np.asarray(rows, dtype=np.float64)
    weights = np.array([0.40, 0.20, 0.20, 0.20])
    D = X / (X.sum(axis=0, keepdims=True) + 1e-12) * weights
    benefit = D[:, :2].sum(axis=1)
    cost = D[:, 2:].sum(axis=1)
    inv_cost = 1.0 / (cost + 1e-12)
    q = benefit + (cost.sum() * inv_cost) / (inv_cost.sum() + 1e-12)
    scores[valid_ids] = q / (q.max() + 1e-12) * 100
    return int(np.argmax(scores)), scores


def copras_mask(vol, brain, args, threshold, record_diagnostic=False):
    smoothed = ndi.gaussian_filter(vol, args.sigma) if args.sigma > 0 else vol
    norm = normalise(smoothed, brain)
    x = norm[brain].astype(np.float64)
    # FIX (Bug 22): reuse the SAME sample that fcm_fit used to produce centres.
    centres, x_sample = fcm_fit(
        x, args.c, args.m, args.seed, n_sample=FCM_SAMPLE_SIZE
    )
    u_sample = memberships(x_sample, centres, args.m)
    selected, _ = copras_select_cluster(x_sample, u_sample, centres)

    if record_diagnostic:
        DIAGNOSTICS.append(selected == int(np.argmax(centres)))

    result = np.zeros(vol.shape, dtype=bool)
    labels = np.zeros(x.size, dtype=bool)
    for start in range(0, x.size, 500000):
        xx = x[start:start + 500000]
        u = memberships(xx, centres, args.m)
        labels[start:start + xx.size] = (
            (u.argmax(axis=1) == selected) & (u[:, selected] >= threshold)
        )
    result[brain] = labels
    return result


def masked_fcm(vol, brain, args, smooth=False, threshold=0.0):
    source = ndi.gaussian_filter(vol, args.sigma) if smooth and args.sigma > 0 else vol
    norm = normalise(source, brain)
    x = norm[brain].astype(np.float64)
    centres, _ = fcm_fit(x, args.c, args.m, args.seed)
    result = np.zeros(vol.shape, dtype=bool)
    result[brain] = brightest_mask(x, centres, args.m, threshold)
    return result


def method_otsu(vol, brain, args):
    norm = normalise(vol, brain)
    return brain & (norm > otsu_threshold(norm[brain]))


def method_kmeans(vol, brain, args):
    norm = normalise(vol, brain)
    x = norm[brain].astype(np.float64)
    sample = subsample(x, 30000, args.seed)
    centres, _ = kmeans2(sample[:, None], args.c, minit="++", seed=args.seed)
    centres = centres.ravel()
    chosen_centre = int(np.argmax(centres))
    result = np.zeros(vol.shape, dtype=bool)
    labels = np.zeros(x.size, dtype=bool)
    for start in range(0, x.size, 500000):
        xx = x[start:start + 500000]
        labels[start:start + xx.size] = (
            np.abs(xx[:, None] - centres[None, :]).argmin(axis=1) == chosen_centre
        )
    result[brain] = labels
    return result


def method_fcm_unmasked(vol, brain, args):
    """
    FIX (Bug 10): by design this method ignores `brain` so it can over-segment
    into the background. Kept for ablation; documented explicitly.
    """
    norm = normalise(vol, brain).ravel().astype(np.float64)
    centres, _ = fcm_fit(norm, args.c, args.m, args.seed)
    return brightest_mask(norm, centres, args.m).reshape(vol.shape)


def method_masked_fcm(vol, brain, args):
    return masked_fcm(vol, brain, args)


def method_sfcm(vol, brain, args):
    return masked_fcm(vol, brain, args, smooth=True)


def method_sfcm_thr(vol, brain, args):
    return masked_fcm(vol, brain, args, smooth=True, threshold=args.thr)


def method_copras_nothr(vol, brain, args):
    return copras_mask(vol, brain, args, 0.0, False)


def method_copras_fcm(vol, brain, args):
    return copras_mask(vol, brain, args, args.thr, True)


METHODS = {
    "otsu": method_otsu,
    "kmeans": method_kmeans,
    "fcm_unmasked": method_fcm_unmasked,
    "masked_fcm": method_masked_fcm,
    "sfcm": method_sfcm,
    "sfcm_thr": method_sfcm_thr,
    "copras_nothr": method_copras_nothr,
    "copras_fcm": method_copras_fcm,
}


def dice_iou(pred, truth):
    total = pred.sum() + truth.sum()
    if total == 0:
        return 1.0, 1.0
    intersection = np.logical_and(pred, truth).sum()
    return float(2 * intersection / total), float(intersection / (total - intersection))


def hd95(pred, truth, spacing):
    if not pred.any() or not truth.any():
        return np.nan
    pred_surface = pred ^ ndi.binary_erosion(pred)
    truth_surface = truth ^ ndi.binary_erosion(truth)
    d_truth = ndi.distance_transform_edt(~truth_surface, sampling=spacing)
    d_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([d_truth[pred_surface], d_pred[truth_surface]])
    return float(np.percentile(distances, 95)) if distances.size else np.nan


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def find_cases(root, modality):
    keys = MODALITY_KEYS[modality]
    root = Path(root)
    cases = []
    for directory in sorted(p for p in root.rglob("*") if p.is_dir()):
        files = list(directory.glob("*.nii")) + list(directory.glob("*.nii.gz"))
        images = [f for f in files if "seg" not in f.name.lower()
                  and any(k in f.name.lower() for k in keys)]
        segs = [f for f in files if "seg" in f.name.lower()]
        if images and segs:
            cases.append((str(directory.relative_to(root)), images[0], segs[0]))
    unique = {}
    for case in cases:
        unique[case[0]] = case
    return [unique[k] for k in sorted(unique)]


def bootstrap_ci(values, reps=2000, seed=12345):
    """
    FIX (Bug 14): fresh RNG per call so each method's CI is independent
    and reproducible regardless of call order.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(values, size=values.size, replace=True).mean()
        for _ in range(reps)
    ])
    return np.percentile(means, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--work", default="/kaggle/working")
    parser.add_argument("--n", type=int, default=100, help="cases; 0 means all available")
    parser.add_argument("--start", type=int, default=0, help="start index in sorted case list")
    parser.add_argument("--out-name", default="results")
    parser.add_argument("--modality", choices=list(MODALITY_KEYS), default="flair")
    parser.add_argument("--c", type=int, default=4)
    parser.add_argument("--m", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--thr", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-frac", type=float, default=0.15)
    parser.add_argument("--no-postproc", action="store_true")
    args = parser.parse_args()

    if args.m <= 1:
        parser.error("--m must be greater than 1")
    if args.c < 2:
        parser.error("--c must be at least 2")
    if not 0 <= args.thr <= 1:
        parser.error("--thr must be between 0 and 1")
    if not 0 <= args.keep_frac <= 1:
        parser.error("--keep-frac must be between 0 and 1")

    work = Path(args.work)
    out = work / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    cases = find_cases(args.input, args.modality)
    if not cases:
        cases = find_cases(work / "brats_extracted", args.modality)
    if not cases:
        raise SystemExit("No complete cases found. Attach the BraTS dataset and check modality filenames.")

    selected_cases = cases[args.start:args.start + args.n] if args.n else cases[args.start:]
    if not selected_cases:
        raise SystemExit(f"No cases in requested range: start={args.start}, n={args.n}")

    print(f"Available complete cases: {len(cases)}")
    print(f"Selected indices: {args.start}..{args.start + len(selected_cases) - 1}")
    print(f"First: {selected_cases[0][0]} | Last: {selected_cases[-1][0]}")
    print(f"Evaluating {len(selected_cases)} cases x {len(METHODS)} methods: {', '.join(METHODS)}")

    csv_path = out / "per_case.csv"
    settings_path = out / "run_info.json"
    settings = vars(args).copy()
    settings["methods"] = list(METHODS)
    if csv_path.exists() and settings_path.exists():
        old = json.loads(settings_path.read_text())
        if old.get("settings") != settings:
            raise SystemExit(
                f"Output folder {out} has results from different settings. "
                "Use a new --out-name to avoid mixing results."
            )
        rows = pd.read_csv(csv_path).to_dict("records")
    elif csv_path.exists() and not settings_path.exists():
        raise SystemExit(
            f"{csv_path} exists without run_info.json. Use a new --out-name to avoid mixing results."
        )
    else:
        rows = []

    done = {(r["case"], r["method"]) for r in rows}
    # FIX (Bug 8/23): count only cases skipped in THIS run; resume runs report honestly.
    skipped = 0
    processed_now = 0
    for idx, (case_id, image_path, seg_path) in enumerate(selected_cases, 1):
        todo = [name for name in METHODS if (case_id, name) not in done]
        if not todo:
            print(f"[{idx}/{len(selected_cases)}] {case_id}: already complete")
            continue
        start_time = time.time()
        image = nib.load(str(image_path))
        vol = np.asarray(image.get_fdata(dtype=np.float32))
        truth = np.asarray(nib.load(str(seg_path)).dataobj) > 0
        spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
        brain = np.isfinite(vol) & (vol > 0)

        if truth.shape != vol.shape or not truth.any() or brain.sum() < 1000:
            skipped += 1
            print(f"[{idx}/{len(selected_cases)}] {case_id}: skipped (invalid image/label)")
            continue

        for name in todo:
            pred = METHODS[name](vol, brain, args)
            if not args.no_postproc:
                pred = postprocess(pred, args.keep_frac)
            dice, iou = dice_iou(pred, truth)
            rows.append({
                "case": case_id, "method": name, "dice": dice, "iou": iou,
                "hd95_mm": hd95(pred, truth, spacing),
                "pred_cm3": float(pred.sum() * np.prod(spacing) / 1000),
                "gt_cm3": float(truth.sum() * np.prod(spacing) / 1000),
                "vox_x": spacing[0], "vox_y": spacing[1], "vox_z": spacing[2],
            })
        processed_now += 1
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        settings_path.write_text(json.dumps({
            "settings": settings, "n_selected": len(selected_cases),
            "start_index": args.start, "case_ids": [c[0] for c in selected_cases],
        }, indent=2))
        print(f"[{idx}/{len(selected_cases)}] {case_id}: {time.time() - start_time:.1f}s")

    df = pd.DataFrame(rows)
    df = df[df["case"].isin({c[0] for c in selected_cases})].copy()
    if df.empty:
        raise SystemExit("No valid evaluation rows were produced.")
    pivot = df.pivot_table(index="case", columns="method", values="dice", aggfunc="first")
    method_names = list(METHODS)
    if "copras_fcm" not in pivot.columns:
        raise SystemExit("No copras_fcm results found in this output.")

    comparisons, p_values = [], []
    for name in method_names:
        if name == "copras_fcm" or name not in pivot.columns:
            continue
        pair = pivot[["copras_fcm", name]].dropna()
        diff = pair["copras_fcm"] - pair[name]
        p = float(wilcoxon(diff).pvalue) if len(diff) and np.any(np.abs(diff) > 0) else 1.0
        comparisons.append(name)
        p_values.append(p)
    adjusted = holm_adjust(p_values) if p_values else []
    adjusted_map = dict(zip(comparisons, adjusted))

    lines = [
        "| Method | Dice mean ± SD | Dice 95% bootstrap CI | Δ Dice vs COPRAS-FCM | IoU mean | HD95 mm mean (NaN count) | p (Holm) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in method_names:
        subset = df[df.method == name]
        if subset.empty:
            continue
        lo, hi = bootstrap_ci(subset.dice)
        delta = "-" if name == "copras_fcm" else (
            f"{(pivot['copras_fcm'] - pivot[name]).mean():+.4f}" if name in pivot.columns else "NA"
        )
        p_label = "reference" if name == "copras_fcm" else (
            f"{adjusted_map[name]:.4g}" if name in adjusted_map else "NA"
        )
        lines.append(
            f"| {name} | {subset.dice.mean():.4f} ± {subset.dice.std():.4f} "
            f"| [{lo:.3f}, {hi:.3f}] | {delta} | {subset.iou.mean():.4f} "
            f"| {subset.hd95_mm.mean():.2f} ({int(subset.hd95_mm.isna().sum())}) | {p_label} |"
        )

    diagnostic = (
        f"COPRAS selected the brightest cluster in {np.mean(DIAGNOSTICS):.1%} "
        f"of copras_fcm cases computed in this run ({len(DIAGNOSTICS)} cases)."
        if DIAGNOSTICS else "COPRAS cluster-selection diagnostic unavailable for this run."
    )
    summary = "\n".join(lines) + (
        f"\n\nValid cases: {pivot.shape[0]}; skipped in this run: {skipped}; "
        f"processed in this run: {processed_now}; modality: {args.modality}; "
        f"post-processing: {'off' if args.no_postproc else 'on for all methods'}.\n\n"
        + diagnostic + "\n\n"
        "Method note: COPRAS uses fixed weights [0.40, 0.20, 0.20, 0.20], not entropy-derived weights. "
        "The target cluster fraction (0.06), penalty bounds, threshold, smoothing sigma, and component "
        "retention fraction are design choices and must be fixed using development data only."
    )
    (out / "summary.md").write_text(summary)
    print("\n" + summary)

    available = [m for m in method_names if m in pivot.columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([pivot[m].dropna().to_numpy() for m in available], tick_labels=available)
    ax.set_ylabel("Dice (whole tumour)")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out / "dice_boxplot.png", dpi=200)
    plt.close(fig)

    print(f"\nSaved results to: {out}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {out / 'summary.md'}")
    print(f"Plot: {out / 'dice_boxplot.png'}")


if __name__ == "__main__":
    main()
