#!/usr/bin/env python3
"""
Batch evaluation of FCM variants against baselines on BraTS (whole-tumour task).
Includes COPRAS-Guided FCM (BS Mathematics thesis multi-criteria optimisation).
Patched version: fair post-processing for all methods, COPRAS sentinel bug fixed,
NaN-aware HD95 reporting, unique case IDs, diagnostics for COPRAS cluster choice.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from scipy import ndimage as ndi
from scipy.cluster.vq import kmeans2
from scipy.stats import wilcoxon

MOD_KEYS = {"flair": ("flair", "t2f"), "t2": ("t2w", "_t2.nii", "-t2.nii")}

# diagnostics: was the COPRAS-chosen cluster just the brightest one?
DIAG = []


# ------------------------------------------------------------------ core maths
def normalise(vol, brain):
    lo, hi = np.percentile(vol[brain], [1, 99])
    return np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def _memberships(x, centres, m):
    d = np.abs(x[:, None] - centres[None, :]) + 1e-8
    inv = d ** (-2.0 / (m - 1.0))
    return inv / inv.sum(axis=1, keepdims=True)


def _subsample(x, n, seed):
    if x.size <= n:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.integers(0, x.size, n)]  # fast, avoids full permutation


def fcm_fit(x, c, m, seed, iters=150, tol=1e-6, n_sample=30000):
    """Standard 1-D Fuzzy C-Means on a random voxel sample. Returns ascending centres."""
    s = _subsample(x, n_sample, seed)
    cn = np.percentile(s, np.linspace(5, 95, c))
    for _ in range(iters):
        w = _memberships(s, cn, m) ** m
        new = (w * s[:, None]).sum(0) / w.sum(0)
        done = np.max(np.abs(new - cn)) < tol
        cn = new
        if done:
            break
    return np.sort(cn)


def bright_cluster(x, centres, m, chunk=500_000):
    """Hard label: voxel belongs to the brightest cluster (highest membership)."""
    out = np.empty(x.size, dtype=bool)
    last = len(centres) - 1
    for i in range(0, x.size, chunk):
        out[i:i + chunk] = _memberships(x[i:i + chunk], centres, m).argmax(1) == last
    return out


def otsu(x, bins=256):
    h, e = np.histogram(x, bins=bins, range=(0, 1))
    p = h / max(h.sum(), 1)
    w = np.cumsum(p)
    mu = np.cumsum(p * (e[:-1] + e[1:]) / 2)
    between = (mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return e[np.argmax(between) + 1]


def postprocess(mask, keep_frac=0.15):
    """Keep connected components >= keep_frac of the largest one. Applied to ALL methods."""
    if not mask.any():
        return mask
    labeled, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, labeled, range(1, n + 1))
    keep = np.where(sizes >= keep_frac * sizes.max())[0] + 1
    return np.isin(labeled, keep)


# ------------------------------------------------------------------ COPRAS
def copras_select_cluster(x, u_mat, centres):
    """
    COPRAS multi-criteria selection of the tumour cluster.
    Criteria: intensity (+), sharpness (+), variance (-), volume penalty (-).
    Invalid (tiny) clusters are excluded BEFORE normalisation.
    """
    c = len(centres)
    labels = u_mat.argmax(axis=1)
    valid = np.zeros(c, dtype=bool)
    rows = []

    for k in range(c):
        mask = labels == k
        if mask.sum() < 10:
            continue
        valid[k] = True
        c1_intensity = float(centres[k]) + 1e-6
        c2_sharpness = float(u_mat[mask, k].mean()) + 1e-6
        c3_variance = float(x[mask].std()) + 1e-4
        frac = float(mask.mean())
        c4_vol_penalty = abs(frac - 0.06) + (10.0 if (frac > 0.35 or frac < 0.002) else 0.001)
        rows.append([c1_intensity, c2_sharpness, c3_variance, c4_vol_penalty])

    U = np.full(c, -np.inf)
    if not valid.any():
        return int(np.argmax(centres)), U

    X = np.array(rows, dtype=np.float64)
    R = X / (X.sum(axis=0, keepdims=True) + 1e-12)
    weights = np.array([0.40, 0.20, 0.20, 0.20])
    D = R * weights

    S_plus = D[:, :2].sum(axis=1)
    S_minus = D[:, 2:].sum(axis=1)

    inv_s_min = 1.0 / (S_minus + 1e-12)
    Q = S_plus + (S_minus.sum() * inv_s_min) / (inv_s_min.sum() + 1e-12)
    U[valid] = (Q / (Q.max() + 1e-12)) * 100.0

    return int(np.argmax(U)), U


def m_copras_fcm(vol, brain, a):
    """Proposed COPRAS-Guided Spatial FCM."""
    vol_smooth = ndi.gaussian_filter(vol, a.sigma)
    n = normalise(vol_smooth, brain)
    x = n[brain].astype(np.float64)
    cn = fcm_fit(x, a.c, a.m, a.seed)

    rng = np.random.default_rng(a.seed)
    idx = np.arange(x.size) if x.size <= 40000 else rng.integers(0, x.size, 40000)
    u_sample = _memberships(x[idx], cn, a.m)
    tumor_k, _ = copras_select_cluster(x[idx], u_sample, cn)
    DIAG.append(tumor_k == len(cn) - 1)

    lab = np.empty(x.size, dtype=bool)
    for i in range(0, x.size, 500_000):
        u_chunk = _memberships(x[i:i + 500_000], cn, a.m)
        lab[i:i + 500_000] = (u_chunk.argmax(1) == tumor_k) & (u_chunk[:, tumor_k] >= a.thr)

    out = np.zeros(vol.shape, bool)
    out[brain] = lab
    return out


# ------------------------------------------------------------------ baselines
def m_otsu(vol, brain, a):
    n = normalise(vol, brain)
    return brain & (n > otsu(n[brain]))


def m_kmeans(vol, brain, a):
    n = normalise(vol, brain)
    x = n[brain].astype(np.float64)
    s = _subsample(x, 30000, a.seed)
    cent, _ = kmeans2(s[:, None], a.c, minit="++", seed=a.seed)
    cent = cent.ravel()
    lab = np.empty(x.size, dtype=bool)
    for i in range(0, x.size, 500_000):
        lab[i:i + 500_000] = np.abs(x[i:i + 500_000, None] - cent[None, :]).argmin(1) == cent.argmax()
    out = np.zeros(vol.shape, bool)
    out[brain] = lab
    return out


def m_fcm_unmasked(vol, brain, a):
    n = normalise(vol, brain).ravel().astype(np.float64)
    cn = fcm_fit(n, a.c, a.m, a.seed)
    return bright_cluster(n, cn, a.m).reshape(vol.shape)


def _masked_fcm(vol, brain, a):
    n = normalise(vol, brain)
    x = n[brain].astype(np.float64)
    cn = fcm_fit(x, a.c, a.m, a.seed)
    out = np.zeros(vol.shape, bool)
    out[brain] = bright_cluster(x, cn, a.m)
    return out


def m_masked_fcm(vol, brain, a):
    return _masked_fcm(vol, brain, a)


def m_sfcm(vol, brain, a):
    return _masked_fcm(ndi.gaussian_filter(vol, a.sigma), brain, a)


METHODS = {
    "otsu": m_otsu,
    "kmeans": m_kmeans,
    "fcm_unmasked": m_fcm_unmasked,
    "masked_fcm": m_masked_fcm,
    "sfcm": m_sfcm,
    "copras_fcm": m_copras_fcm,
}


# ------------------------------------------------------------------ metrics
def dice_iou(p, g):
    inter = np.logical_and(p, g).sum()
    s = p.sum() + g.sum()
    if s == 0:
        return 1.0, 1.0
    return 2 * inter / s, inter / (s - inter)


def hd95(p, g, spacing):
    if not p.any() or not g.any():
        return np.nan
    sp = p ^ ndi.binary_erosion(p)
    sg = g ^ ndi.binary_erosion(g)
    d_to_g = ndi.distance_transform_edt(~sg, sampling=spacing)
    d_to_p = ndi.distance_transform_edt(~sp, sampling=spacing)
    return float(np.percentile(np.concatenate([d_to_g[sp], d_to_p[sg]]), 95))


# ------------------------------------------------------------------ data
def find_cases(root, modality):
    keys = MOD_KEYS[modality]
    root = Path(root)
    cases = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        files = list(d.glob("*.nii")) + list(d.glob("*.nii.gz"))
        img = [f for f in files if "seg" not in f.name.lower() and any(k in f.name.lower() for k in keys)]
        seg = [f for f in files if "seg" in f.name.lower()]
        if img and seg:
            cases.append((str(d.relative_to(root)), img[0], seg[0]))  # unique ID
    return cases


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--modality", default="flair", choices=list(MOD_KEYS))
    ap.add_argument("--max-cases", type=int, default=0, help="0 = all cases")
    ap.add_argument("--c", type=int, default=4)
    ap.add_argument("--m", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--thr", type=float, default=0.55, help="membership threshold for copras_fcm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-postproc", action="store_true",
                    help="disable connected-component filtering (applied to ALL methods by default)")
    ap.add_argument("--proposed", default="copras_fcm", choices=list(METHODS))
    a = ap.parse_args()

    if a.m <= 1.0:
        raise SystemExit("--m must be > 1 for Fuzzy C-Means.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = find_cases(a.data, a.modality)
    if a.max_cases:
        cases = cases[:a.max_cases]
    if not cases:
        raise SystemExit("No cases found. Each case folder needs a modality file and a *seg* file.")
    print(f"{len(cases)} cases, methods: {', '.join(METHODS)}")

    rows = []
    for k, (cid, f_img, f_seg) in enumerate(cases, 1):
        t0 = time.time()
        img = nib.load(str(f_img))
        vol = np.asarray(img.get_fdata(dtype=np.float32))
        gt = np.asarray(nib.load(str(f_seg)).dataobj) > 0
        spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
        brain = vol > 0
        if not gt.any() or brain.sum() < 1000:
            print(f"[{k}/{len(cases)}] {cid}: skipped (empty mask or brain)")
            continue
        for name, fn in METHODS.items():
            pred = fn(vol, brain, a)
            if not a.no_postproc:
                pred = postprocess(pred)
            d, i = dice_iou(pred, gt)
            rows.append(dict(case=cid, method=name, dice=d, iou=i, hd95_mm=hd95(pred, gt, spacing),
                             pred_cm3=pred.sum() * np.prod(spacing) / 1000,
                             gt_cm3=gt.sum() * np.prod(spacing) / 1000))
        print(f"[{k}/{len(cases)}] {cid}: {time.time() - t0:.1f}s")
        pd.DataFrame(rows).to_csv(out / "per_case.csv", index=False)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("All cases were skipped; nothing to evaluate.")
    df.to_csv(out / "per_case.csv", index=False)

    piv = df.pivot(index="case", columns="method", values="dice")
    others = [m for m in METHODS if m != a.proposed]
    pv = [wilcoxon(piv[a.proposed], piv[o]).pvalue if (piv[a.proposed] - piv[o]).abs().sum() > 0 else 1.0
          for o in others]
    padj = dict(zip(others, holm(np.array(pv))))
    lines = ["| Method | Dice | IoU | HD95 (mm) | HD95 NaN | p vs proposed (Holm) |",
             "|---|---|---|---|---|---|"]
    summ = []
    for name in METHODS:
        s = df[df.method == name]
        n_nan = int(s.hd95_mm.isna().sum())
        row = dict(method=name, n=len(s), dice_mean=s.dice.mean(), dice_sd=s.dice.std(),
                   iou_mean=s.iou.mean(), iou_sd=s.iou.std(),
                   hd95_mean=s.hd95_mm.mean(), hd95_sd=s.hd95_mm.std(), hd95_nan=n_nan,
                   p_holm=padj.get(name, np.nan))
        summ.append(row)
        p = "reference" if name == a.proposed else f"{padj[name]:.4g}"
        lines.append(f"| {name} | {row['dice_mean']:.3f} ± {row['dice_sd']:.3f} | "
                     f"{row['iou_mean']:.3f} ± {row['iou_sd']:.3f} | "
                     f"{row['hd95_mean']:.2f} ± {row['hd95_sd']:.2f} | {n_nan} | {p} |")
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)

    brightest_frac = float(np.mean(DIAG)) if DIAG else float("nan")
    footer = (f"\nN = {piv.shape[0]} cases, modality = {a.modality}, "
              f"postprocessing = {'off' if a.no_postproc else 'on (all methods)'}\n"
              f"COPRAS picked the brightest cluster in {brightest_frac:.1%} of cases\n")
    (out / "summary.md").write_text("\n".join(lines) + "\n" + footer)
    print("\n" + "\n".join(lines) + footer)

    fig, ax = plt.subplots(figsize=(7, 4))
    data = [piv[m].dropna() for m in METHODS]
    try:
        ax.boxplot(data, tick_labels=list(METHODS), showfliers=True)  # matplotlib >= 3.9
    except TypeError:
        ax.boxplot(data, labels=list(METHODS), showfliers=True)       # older matplotlib
    ax.set_ylabel("Dice (whole tumour)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "dice_boxplot.png", dpi=200)
    plt.close(fig)

    (out / "run_info.json").write_text(json.dumps(dict(
        args=vars(a), python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        nibabel=nib.__version__, n_cases=int(piv.shape[0]),
        copras_brightest_fraction=brightest_frac), indent=2))
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
