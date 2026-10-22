#!/usr/bin/env python3
"""
One-command BraTS evaluation for Kaggle (whole-tumour, FLAIR by default), with the user's
COPRAS-Guided FCM as the proposed method.

    !pip -q install nibabel
    !git clone https://github.com/yawarmk123/mri-fuzzy-segmentation
    # 1) development run (the cases you have already looked at)
    !python mri-fuzzy-segmentation/kaggle_eval.py --n 100
    # 2) FINAL run on cases you have never looked at, with all constants frozen
    !python mri-fuzzy-segmentation/kaggle_eval.py --start 100 --n 300 --out-name results_test

Fairness rules built in
  * Methods receive only (image, brain_mask, args): ground truth is never visible to them.
  * The same connected-component clean-up is applied to EVERY method.
  * Ablations show where the gain comes from: sfcm_thr (threshold alone) and copras_nothr (COPRAS alone).
Resumable per (case, method): re-running skips finished work.
"""
import argparse
import json
import os
import shutil
import tarfile
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

KEYS = {"flair": ("flair", "t2f"), "t2": ("t2w", "_t2.nii", "-t2.nii")}
KAGGLEHUB_SLUG = "dschettler8845/brats-2021-task1"
DIAG = []  # was the COPRAS-chosen cluster simply the brightest one?


# ------------------------------------------------------------------ maths
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
    return x[np.random.default_rng(seed).integers(0, x.size, n)]


def fcm_fit(x, c, m, seed, iters=150, tol=1e-6, n_sample=30000):
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


def bright_cluster(x, centres, m, chunk=500_000, thr=0.0):
    """Voxel belongs to the brightest cluster (and has membership >= thr there)."""
    out = np.empty(x.size, dtype=bool)
    last = len(centres) - 1
    for i in range(0, x.size, chunk):
        u = _memberships(x[i:i + chunk], centres, m)
        out[i:i + chunk] = (u.argmax(1) == last) & (u[:, last] >= thr)
    return out


def otsu(x, bins=256):
    h, e = np.histogram(x, bins=bins, range=(0, 1))
    p = h / max(h.sum(), 1)
    w = np.cumsum(p)
    mu = np.cumsum(p * (e[:-1] + e[1:]) / 2)
    return e[np.argmax((mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)) + 1]


def postprocess(mask, keep_frac=0.15):
    """Keep connected components >= keep_frac of the largest one. Applied to ALL methods."""
    if not mask.any():
        return mask
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, np.where(sizes >= keep_frac * sizes.max())[0] + 1)


# ------------------------------------------------------------------ COPRAS (user's design)
def copras_select_cluster(x, u_mat, centres):
    """Criteria: intensity (+), sharpness (+), variance (-), volume penalty (-). Fixed weights."""
    c = len(centres)
    labels = u_mat.argmax(axis=1)
    valid = np.zeros(c, dtype=bool)
    rows = []
    for k in range(c):
        mask = labels == k
        if mask.sum() < 10:
            continue
        valid[k] = True
        frac = float(mask.mean())
        rows.append([float(centres[k]) + 1e-6,
                     float(u_mat[mask, k].mean()) + 1e-6,
                     float(x[mask].std()) + 1e-4,
                     abs(frac - 0.06) + (10.0 if (frac > 0.35 or frac < 0.002) else 0.001)])
    U = np.full(c, -np.inf)
    if not valid.any():
        return int(np.argmax(centres)), U
    X = np.array(rows, dtype=np.float64)
    D = X / (X.sum(axis=0, keepdims=True) + 1e-12) * np.array([0.40, 0.20, 0.20, 0.20])
    s_plus, s_minus = D[:, :2].sum(1), D[:, 2:].sum(1)
    inv = 1.0 / (s_minus + 1e-12)
    Q = s_plus + (s_minus.sum() * inv) / (inv.sum() + 1e-12)
    U[valid] = (Q / (Q.max() + 1e-12)) * 100.0
    return int(np.argmax(U)), U


def _copras(vol, brain, a, thr, diag):
    n = normalise(ndi.gaussian_filter(vol, a.sigma), brain)
    x = n[brain].astype(np.float64)
    cn = fcm_fit(x, a.c, a.m, a.seed)
    u_s = _memberships(_subsample(x, 40000, a.seed), cn, a.m)
    k, _ = copras_select_cluster(_subsample(x, 40000, a.seed), u_s, cn)
    if diag:
        DIAG.append(k == len(cn) - 1)
    lab = np.empty(x.size, dtype=bool)
    for i in range(0, x.size, 500_000):
        u = _memberships(x[i:i + 500_000], cn, a.m)
        lab[i:i + 500_000] = (u.argmax(1) == k) & (u[:, k] >= thr)
    out = np.zeros(vol.shape, bool)
    out[brain] = lab
    return out


def m_copras_fcm(vol, brain, a):
    return _copras(vol, brain, a, a.thr, True)


def m_copras_nothr(vol, brain, a):
    return _copras(vol, brain, a, 0.0, False)


# ------------------------------------------------------------------ baselines / ablations
def m_otsu(vol, brain, a):
    n = normalise(vol, brain)
    return brain & (n > otsu(n[brain]))


def m_kmeans(vol, brain, a):
    x = normalise(vol, brain)[brain].astype(np.float64)
    cent, _ = kmeans2(_subsample(x, 30000, a.seed)[:, None], a.c, minit="++", seed=a.seed)
    cent = cent.ravel()
    lab = np.empty(x.size, bool)
    for i in range(0, x.size, 500_000):
        lab[i:i + 500_000] = np.abs(x[i:i + 500_000, None] - cent[None, :]).argmin(1) == cent.argmax()
    out = np.zeros(vol.shape, bool)
    out[brain] = lab
    return out


def m_fcm_unmasked(vol, brain, a):
    n = normalise(vol, brain).ravel().astype(np.float64)
    return bright_cluster(n, fcm_fit(n, a.c, a.m, a.seed), a.m).reshape(vol.shape)


def _masked(vol, brain, a, thr=0.0):
    x = normalise(vol, brain)[brain].astype(np.float64)
    out = np.zeros(vol.shape, bool)
    out[brain] = bright_cluster(x, fcm_fit(x, a.c, a.m, a.seed), a.m, thr=thr)
    return out


def m_masked_fcm(vol, brain, a):
    return _masked(vol, brain, a)


def m_sfcm(vol, brain, a):
    return _masked(ndi.gaussian_filter(vol, a.sigma), brain, a)


def m_sfcm_thr(vol, brain, a):
    """Ablation: sFCM + the same membership threshold as copras_fcm, WITHOUT COPRAS."""
    return _masked(ndi.gaussian_filter(vol, a.sigma), brain, a, thr=a.thr)


METHODS = {"otsu": m_otsu, "kmeans": m_kmeans, "fcm_unmasked": m_fcm_unmasked,
           "masked_fcm": m_masked_fcm, "sfcm": m_sfcm, "sfcm_thr": m_sfcm_thr,
           "copras_nothr": m_copras_nothr, "copras_fcm": m_copras_fcm}


# ------------------------------------------------------------------ metrics
def dice_iou(p, g):
    s = p.sum() + g.sum()
    if s == 0:
        return 1.0, 1.0
    i = np.logical_and(p, g).sum()
    return 2 * i / s, i / (s - i)


def hd95(p, g, spacing):
    if not p.any() or not g.any():
        return np.nan
    sp, sg = p ^ ndi.binary_erosion(p), g ^ ndi.binary_erosion(g)
    d_g = ndi.distance_transform_edt(~sg, sampling=spacing)
    d_p = ndi.distance_transform_edt(~sp, sampling=spacing)
    return float(np.percentile(np.concatenate([d_g[sp], d_p[sg]]), 95))


def holm(p):
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ data discovery
def find_cases(root, modality):
    keys, root, cases = KEYS[modality], Path(root), []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        files = list(d.glob("*.nii")) + list(d.glob("*.nii.gz"))
        img = [f for f in files if "seg" not in f.name.lower() and any(k in f.name.lower() for k in keys)]
        seg = [f for f in files if "seg" in f.name.lower()]
        if img and seg:
            cases.append((str(d.relative_to(root)), img[0], seg[0]))
    return cases


def extract_from_tars(tars, out, n, modality):
    keys = KEYS[modality]
    out.mkdir(parents=True, exist_ok=True)
    big = [t for t in tars if "training" in t.lower()] or tars
    seen = []
    for tpath in big:
        print(f"Streaming {tpath} ...")
        with tarfile.open(tpath, "r|") as t:
            for m in t:
                low = m.name.lower()
                if not m.isfile() or not low.endswith((".nii", ".nii.gz")):
                    continue
                if not ("seg" in low or any(k in low for k in keys)):
                    continue
                case = os.path.basename(os.path.dirname(m.name)) or Path(m.name).stem
                if case not in seen:
                    if n and len(seen) >= n:
                        break
                    seen.append(case)
                    if len(seen) % 25 == 0:
                        print(f"  {len(seen)} cases extracted")
                try:
                    t.extract(m, out, filter="data")
                except TypeError:
                    t.extract(m, out)
        if n and len(seen) >= n:
            break


def locate_data(a, work):
    need = a.start + a.n if a.n else 0
    src = Path(a.input)
    cases = find_cases(src, a.modality)
    if cases:
        print(f"Found {len(cases)} loose cases under {src}")
        return cases
    tars = [str(p) for p in src.rglob("*.tar")]
    if not tars and a.allow_download:
        try:
            import kagglehub
            print(f"No data attached; trying kagglehub download of {KAGGLEHUB_SLUG} ...")
            src = Path(kagglehub.dataset_download(KAGGLEHUB_SLUG))
            cases = find_cases(src, a.modality)
            if cases:
                return cases
            tars = [str(p) for p in src.rglob("*.tar")]
        except Exception as e:
            print("kagglehub download failed:", repr(e)[:200])
    if not tars:
        raise SystemExit("\nNo BraTS data found. Attach 'BRaTS 2021 Task 1 Dataset' via Add Input > Datasets.")
    out = work / "brats_extracted"
    have = find_cases(out, a.modality) if out.exists() else []
    if need and len(have) >= need:
        print(f"Reusing {len(have)} already-extracted cases in {out}")
        return have
    extract_from_tars(tars, out, need, a.modality)
    return find_cases(out, a.modality)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--work", default="/kaggle/working")
    ap.add_argument("--n", type=int, default=100, help="number of cases; 0 = all")
    ap.add_argument("--start", type=int, default=0, help="skip the first START cases (held-out split)")
    ap.add_argument("--out-name", default="results")
    ap.add_argument("--modality", default="flair", choices=list(KEYS))
    ap.add_argument("--c", type=int, default=4)
    ap.add_argument("--m", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--thr", type=float, default=0.55, help="membership threshold (copras_fcm, sfcm_thr)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-postproc", action="store_true")
    ap.add_argument("--proposed", default="copras_fcm")
    ap.add_argument("--no-download", dest="allow_download", action="store_false")
    a, _ = ap.parse_known_args()
    if a.m <= 1.0:
        raise SystemExit("--m must be > 1.")
    if a.proposed not in METHODS:
        raise SystemExit(f"--proposed must be one of {list(METHODS)}")

    work = Path(a.work)
    out = work / a.out_name
    out.mkdir(parents=True, exist_ok=True)
    cases = locate_data(a, work)
    cases = cases[a.start:a.start + a.n] if a.n else cases[a.start:]
    if not cases:
        raise SystemExit("No complete cases (modality + seg) in the requested range.")
    print(f"\nCases {a.start}..{a.start + len(cases) - 1}: first {cases[0][0]}, last {cases[-1][0]}")
    print(f"Evaluating {len(cases)} cases x {len(METHODS)} methods: {', '.join(METHODS)}\n")

    csv = out / "per_case.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["case"], r["method"]) for r in rows}
    skipped = 0
    for k, (cid, f_img, f_seg) in enumerate(cases, 1):
        todo = [m for m in METHODS if (cid, m) not in done]
        if not todo:
            continue
        t0 = time.time()
        img = nib.load(str(f_img))
        vol = np.asarray(img.get_fdata(dtype=np.float32))
        gt = np.asarray(nib.load(str(f_seg)).dataobj) > 0
        sp = tuple(float(z) for z in img.header.get_zooms()[:3])
        brain = vol > 0
        if not gt.any() or brain.sum() < 1000 or gt.shape != vol.shape:
            skipped += 1
            print(f"[{k}/{len(cases)}] {cid}: skipped (empty/mismatched mask or brain)")
            continue
        for name in todo:
            pred = METHODS[name](vol, brain, a)  # ground truth is NOT passed in
            if not a.no_postproc:
                pred = postprocess(pred)
            d, i = dice_iou(pred, gt)
            rows.append(dict(case=cid, method=name, dice=d, iou=i, hd95_mm=hd95(pred, gt, sp),
                             pred_cm3=pred.sum() * np.prod(sp) / 1000, gt_cm3=gt.sum() * np.prod(sp) / 1000,
                             vox_x=sp[0], vox_y=sp[1], vox_z=sp[2]))
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"[{k}/{len(cases)}] {cid}: {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df = df[df.case.isin({c[0] for c in cases})]
    piv = df.pivot(index="case", columns="method", values="dice")
    names = list(METHODS)
    others = [m for m in names if m != a.proposed]
    pv = [wilcoxon(piv[a.proposed], piv[o]).pvalue if (piv[a.proposed] - piv[o]).abs().sum() > 0 else 1.0
          for o in others]
    padj = dict(zip(others, holm(np.array(pv))))
    rng = np.random.default_rng(0)

    def ci(v):
        v = v.dropna().to_numpy()
        return np.percentile([rng.choice(v, v.size).mean() for _ in range(2000)], [2.5, 97.5])

    lines = ["| Method | Dice mean ± SD | Dice 95% CI | proposed − method (Dice) | IoU | HD95 (mm) | p (Holm) |",
             "|---|---|---|---|---|---|---|"]
    for name in names:
        s = df[df.method == name]
        lo, hi = ci(s.dice)
        delta = "-" if name == a.proposed else f"{(piv[a.proposed] - piv[name]).mean():+.4f}"
        p = "reference" if name == a.proposed else f"{padj[name]:.4g}"
        lines.append(f"| {name} | {s.dice.mean():.4f} ± {s.dice.std():.4f} | [{lo:.3f}, {hi:.3f}] | {delta} | "
                     f"{s.iou.mean():.4f} ± {s.iou.std():.4f} | {s.hd95_mm.mean():.2f} ± {s.hd95_mm.std():.2f} "
                     f"(NaN {int(s.hd95_mm.isna().sum())}) | {p} |")
    bf = f"COPRAS picked the brightest cluster in {np.mean(DIAG):.1%} of cases this run\n" if DIAG else ""
    summary = ("\n".join(lines) + f"\n\nN = {piv.shape[0]} cases, modality = {a.modality}, proposed = {a.proposed}, "
               f"post-processing = {'off' if a.no_postproc else 'on, all methods'}\n" + bf)
    (out / "summary.md").write_text(summary)
    print("\n" + summary)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([piv[m].dropna() for m in names], tick_labels=names, showfliers=True)
    ax.set_ylabel("Dice (whole tumour)")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out / "dice_boxplot.png", dpi=200)

    print("=" * 60 + "\nAUDIT (read before trusting any number)\n" + "=" * 60)
    first = df[df.method == names[0]]
    print(f"Cases evaluated: {piv.shape[0]}   skipped: {skipped}")
    print(f"Voxel sizes (mm): {sorted({(round(r.vox_x, 2), round(r.vox_y, 2), round(r.vox_z, 2)) for r in first.itertuples()})}")
    print(f"Ground-truth tumour cm3: median {first.gt_cm3.median():.1f}, range {first.gt_cm3.min():.1f} to {first.gt_cm3.max():.1f}")
    print("Ground truth access: methods are called as f(vol, brain, args); the mask is never passed in.")
    dup = [(x, y) for i, x in enumerate(names) for y in names[i + 1:] if np.allclose(piv[x], piv[y], atol=1e-12)]
    print("Identical-score method pairs:", dup or "none")
    for m in names:
        if piv[m].mean() > 0.95:
            print(f"WARNING: {m} mean Dice {piv[m].mean():.3f} is far above published whole-tumour results; re-check.")
    if piv.shape[0] < 100:
        print(f"WARNING: only {piv.shape[0]} cases; use at least 100 (ideally several hundred) for a paper.")
    print("REMINDER: weights (0.4/0.2/0.2/0.2), target fraction 0.06, thr, keep_frac and sigma are hand-set. "
          "If you chose them after seeing results on these cases, report only on a held-out range (--start).")
    (out / "run_info.json").write_text(json.dumps(dict(args=vars(a), n_cases=int(piv.shape[0])), indent=2))
    shutil.make_archive(str(work / f"{a.out_name}_bundle"), "zip", out)
    print(f"\nSaved: {out}  and  {work / (a.out_name + '_bundle.zip')}")
    try:
        from IPython.display import FileLink, display
        display(FileLink(str(work / f"{a.out_name}_bundle.zip")))
    except Exception:
        pass


if __name__ == "__main__":
    main()
