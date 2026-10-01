#!/usr/bin/env python3
"""
Batch evaluation of FCM variants against baselines on BraTS (whole-tumour task).

Usage:
    python evaluate.py --data /path/to/BraTS --out results --modality flair --max-cases 100

Expected layout: one folder per case, each holding the modality file and *seg*.nii(.gz).
Works with BraTS 2021/2023/2024 naming (flair / t2f). Whole tumour = any non-zero label.

Outputs (in --out): per_case.csv, summary.csv, summary.md, dice_boxplot.png, run_info.json
To add your own method (TP-FCM, fractional FCM...), write a function f(vol, brain, a) -> bool volume
and register it in METHODS. Nothing else needs to change.
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


# ------------------------------------------------------------------ core maths
def normalise(vol, brain):
    lo, hi = np.percentile(vol[brain], [1, 99])
    return np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def _memberships(x, centres, m):
    d = np.abs(x[:, None] - centres[None, :]) + 1e-8
    inv = d ** (-2.0 / (m - 1.0))
    return inv / inv.sum(axis=1, keepdims=True)


def fcm_fit(x, c, m, seed, iters=150, tol=1e-6, n_sample=30000):
    """Standard 1-D Fuzzy C-Means on a random voxel sample. Returns ascending centres."""
    rng = np.random.default_rng(seed)
    s = x if x.size <= n_sample else rng.choice(x, n_sample, replace=False)
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


# ------------------------------------------------------------------ methods
def m_otsu(vol, brain, a):
    n = normalise(vol, brain)
    return brain & (n > otsu(n[brain]))


def m_kmeans(vol, brain, a):
    n = normalise(vol, brain)
    x = n[brain].astype(np.float64)
    rng = np.random.default_rng(a.seed)
    s = x if x.size <= 30000 else rng.choice(x, 30000, replace=False)
    cent, _ = kmeans2(s[:, None], a.c, minit="++", seed=a.seed)
    cent = cent.ravel()
    lab = np.empty(x.size, dtype=bool)
    for i in range(0, x.size, 500_000):
        lab[i:i + 500_000] = np.abs(x[i:i + 500_000, None] - cent[None, :]).argmin(1) == cent.argmax()
    out = np.zeros(vol.shape, bool)
    out[brain] = lab
    return out


def m_fcm_unmasked(vol, brain, a):
    """Plain FCM over ALL voxels, background included (no skull-stripping logic)."""
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
    """Gaussian spatial regularisation before masked FCM (sigma from --sigma)."""
    return _masked_fcm(ndi.gaussian_filter(vol, a.sigma), brain, a)


# def m_my_variant(vol, brain, a):   # <- plug in TP-FCM / fractional FCM here
#     ...
#     return bool_volume

METHODS = {
    "otsu": m_otsu,
    "kmeans": m_kmeans,
    "fcm_unmasked": m_fcm_unmasked,
    "masked_fcm": m_masked_fcm,
    "sfcm": m_sfcm,
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
    cases = []
    for d in sorted(p for p in Path(root).rglob("*") if p.is_dir()):
        files = list(d.glob("*.nii")) + list(d.glob("*.nii.gz"))
        img = [f for f in files if "seg" not in f.name.lower() and any(k in f.name.lower() for k in keys)]
        seg = [f for f in files if "seg" in f.name.lower()]
        if img and seg:
            cases.append((d.name, img[0], seg[0]))
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proposed", default="sfcm", choices=list(METHODS))
    a = ap.parse_args()

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
        brain = vol > 0  # BraTS is skull-stripped: background is exactly zero
        if not gt.any() or brain.sum() < 1000:
            print(f"[{k}/{len(cases)}] {cid}: skipped (empty mask or brain)")
            continue
        for name, fn in METHODS.items():
            pred = fn(vol, brain, a)
            d, i = dice_iou(pred, gt)
            rows.append(dict(case=cid, method=name, dice=d, iou=i, hd95_mm=hd95(pred, gt, spacing),
                             pred_cm3=pred.sum() * np.prod(spacing) / 1000,
                             gt_cm3=gt.sum() * np.prod(spacing) / 1000))
        print(f"[{k}/{len(cases)}] {cid}: {time.time() - t0:.1f}s")
        pd.DataFrame(rows).to_csv(out / "per_case.csv", index=False)  # safe to interrupt

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_case.csv", index=False)

    # summary with paired Wilcoxon (Holm-corrected) against the proposed method
    piv = df.pivot(index="case", columns="method", values="dice")
    others = [m for m in METHODS if m != a.proposed]
    pv = [wilcoxon(piv[a.proposed], piv[o]).pvalue if (piv[a.proposed] - piv[o]).abs().sum() > 0 else 1.0
          for o in others]
    padj = dict(zip(others, holm(np.array(pv))))
    lines = ["| Method | Dice | IoU | HD95 (mm) | p vs proposed (Holm) |", "|---|---|---|---|---|"]
    summ = []
    for name in METHODS:
        s = df[df.method == name]
        row = dict(method=name, n=len(s), dice_mean=s.dice.mean(), dice_sd=s.dice.std(),
                   iou_mean=s.iou.mean(), iou_sd=s.iou.std(),
                   hd95_mean=s.hd95_mm.mean(), hd95_sd=s.hd95_mm.std(),
                   p_holm=padj.get(name, np.nan))
        summ.append(row)
        p = "reference" if name == a.proposed else f"{padj[name]:.4g}"
        lines.append(f"| {name} | {row['dice_mean']:.3f} ± {row['dice_sd']:.3f} | "
                     f"{row['iou_mean']:.3f} ± {row['iou_sd']:.3f} | "
                     f"{row['hd95_mean']:.2f} ± {row['hd95_sd']:.2f} | {p} |")
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text("\n".join(lines) + f"\n\nN = {piv.shape[0]} cases, modality = {a.modality}\n")
    print("\n" + "\n".join(lines))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([piv[m].dropna() for m in METHODS], tick_labels=list(METHODS), showfliers=True)
    ax.set_ylabel("Dice (whole tumour)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "dice_boxplot.png", dpi=200)

    (out / "run_info.json").write_text(json.dumps(dict(
        args=vars(a), python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        nibabel=nib.__version__, n_cases=int(piv.shape[0])), indent=2))
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
