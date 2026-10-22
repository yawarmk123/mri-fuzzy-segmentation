import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.ndimage as ndi
from scipy.cluster.vq import kmeans2
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="BraTS Evaluation for Entropy-COPRAS-FCM")
parser.add_argument("-n", "--num_cases", type=int, default=100)
parser.add_argument("--out-name", type=str, default="results_v2", dest="out_name")
parser.add_argument("--input-dir", type=str, default="/kaggle/input")
parser.add_argument("--output-dir", type=str,
                    default="/kaggle/working" if Path("/kaggle/working").exists() else ".")
args, unknown = parser.parse_known_args()

# ------------------------------------------------------------------
# 2. Core maths & preprocessing
# ------------------------------------------------------------------
def normalise(vol, brain):
    lo, hi = np.percentile(vol[brain], [1, 99])
    return np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)

def _memberships(x, centres, m):
    d = np.abs(x[:, None] - centres[None, :]) + 1e-8
    inv = d ** (-2.0 / (m - 1.0))
    return inv / inv.sum(axis=1, keepdims=True)

def fcm_fit(x, c=4, m=2.0, iters=50, tol=1e-4):
    rng = np.random.default_rng(0)
    s = x if x.size <= 25000 else x[rng.integers(0, x.size, 25000)]
    cn = np.percentile(s, np.linspace(5, 95, c))
    for _ in range(iters):
        w = _memberships(s, cn, m) ** m
        new = (w * s[:, None]).sum(0) / (w.sum(0) + 1e-12)
        if np.max(np.abs(new - cn)) < tol:
            cn = new
            break
        cn = new
    return np.sort(cn)

def otsu(x, bins=256):
    h, e = np.histogram(x, bins=bins, range=(0, 1))
    p = h / max(h.sum(), 1)
    w = np.cumsum(p)
    mu = np.cumsum(p * (e[:-1] + e[1:]) / 2)
    between = (mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return e[np.argmax(between) + 1]

def fill_holes_3d(mask):
    return ndi.binary_fill_holes(mask)

def extract_largest_component(mask):
    labeled, nf = ndi.label(mask)
    if nf == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == int(np.argmax(counts))

# ------------------------------------------------------------------
# 3. Entropy-weighted COPRAS
# ------------------------------------------------------------------
def entropy_weights(R):
    m, n = R.shape
    if m <= 1:
        return np.ones(n) / max(n, 1)
    k_ent = 1.0 / np.log(m)
    E = -k_ent * np.sum(R * np.log(R + 1e-12), axis=0)
    div = np.maximum(1.0 - E, 1e-6)
    return div / div.sum()

def copras_rank(X, is_beneficial):
    X = np.maximum(np.asarray(X, dtype=np.float64), 1e-6)
    R = X / (X.sum(axis=0, keepdims=True) + 1e-12)
    weights = entropy_weights(R)
    D = R * weights[None, :]

    ben = np.asarray(is_beneficial, dtype=bool)
    cost = ~ben
    S_plus = D[:, ben].sum(axis=1) if ben.any() else np.zeros(X.shape[0])
    S_minus = D[:, cost].sum(axis=1) if cost.any() else np.zeros(X.shape[0])

    if cost.any() and S_minus.sum() > 1e-12:
        s_safe = np.maximum(S_minus, 1e-8)
        inv_s = 1.0 / s_safe
        Q = S_plus + (s_safe.sum() * inv_s) / (inv_s.sum() + 1e-12)
    else:
        Q = S_plus.copy()

    U = (Q / (Q.max() + 1e-12)) * 100.0
    return int(np.argmax(U)), U

# Log: which cluster COPRAS picked vs. the brightest one (for diagnostics)
SELECTION_LOG = {"picked_brightest": 0, "total": 0}

def copras_fcm_segment(vol, brain):
    vol_smooth = ndi.gaussian_filter(vol, 1.0)
    n_vol = normalise(vol_smooth, brain)
    x_1d = n_vol[brain].astype(np.float64)

    centres = fcm_fit(x_1d, c=4, m=2.0)
    u_1d = _memberships(x_1d, centres, m=2.0)
    labels_1d = u_1d.argmax(axis=1)

    grid = np.zeros(vol.shape, dtype=np.int16)
    grid[brain] = labels_1d + 1

    rows, cluster_ids = [], []
    brain_com = ndi.center_of_mass(brain)

    for k in range(1, 5):
        m3d = (grid == k)
        cnt = int(m3d.sum())
        if cnt < 10:
            continue
        c1 = float(centres[k - 1])
        surface = float((m3d ^ ndi.binary_erosion(m3d)).sum())
        c2 = float(cnt) / (surface + 1e-3)
        ccom = ndi.center_of_mass(m3d)
        c3 = float(np.sqrt(sum((ccom[i] - brain_com[i]) ** 2 for i in range(3))))
        c4 = float(x_1d[labels_1d == (k - 1)].std())
        rows.append([max(c1, 1e-3), max(c2, 1e-3), max(c3, 1e-3), max(c4, 1e-3)])
        cluster_ids.append(k)

    if not cluster_ids:
        out = np.zeros(vol.shape, dtype=bool)
        out[brain] = (labels_1d == 3)
        return out

    X = np.array(rows)
    best_idx, _ = copras_rank(X, [True, True, False, False])
    best_k = cluster_ids[best_idx]

    SELECTION_LOG["total"] += 1
    SELECTION_LOG["picked_brightest"] += int(best_k == max(cluster_ids))

    out_mask = (grid == best_k)
    out_mask = extract_largest_component(out_mask)
    out_mask = fill_holes_3d(out_mask)
    return out_mask

# ------------------------------------------------------------------
# 4. Baselines
# ------------------------------------------------------------------
def m_otsu(vol, brain):
    n = normalise(vol, brain)
    return brain & (n > otsu(n[brain]))

def m_kmeans(vol, brain):
    n = normalise(vol, brain)
    x = n[brain].astype(np.float64)
    s = x if x.size <= 25000 else x[np.random.default_rng(0).integers(0, x.size, 25000)]
    cent, _ = kmeans2(s[:, None], 4, minit="++", seed=0)
    out = np.zeros(vol.shape, bool)
    out[brain] = np.abs(x[:, None] - cent.ravel()[None, :]).argmin(1) == cent.argmax()
    return out

def m_masked_fcm(vol, brain):
    n = normalise(vol, brain)
    x = n[brain].astype(np.float64)
    cn = fcm_fit(x, 4, 2.0)
    out = np.zeros(vol.shape, bool)
    out[brain] = _memberships(x, cn, 2.0).argmax(1) == 3
    return out

def m_sfcm(vol, brain):
    return m_masked_fcm(ndi.gaussian_filter(vol, 1.0), brain)

def m_sfcm_cc(vol, brain):
    out = m_sfcm(vol, brain)
    return fill_holes_3d(extract_largest_component(out))

METHODS = {
    "otsu": m_otsu,
    "kmeans": m_kmeans,
    "masked_fcm": m_masked_fcm,
    "sfcm": m_sfcm,
    "sfcm_cc": m_sfcm_cc,
    "copras_fcm": copras_fcm_segment,
}

# ------------------------------------------------------------------
# 5. Metrics
# ------------------------------------------------------------------
def dice_score(p, g):
    inter = np.logical_and(p, g).sum()
    s = p.sum() + g.sum()
    return float(2 * inter / s) if s > 0 else 1.0

def hd95(p, g, spacing=(1.0, 1.0, 1.0)):
    """HD95 in mm (uses voxel spacing). NaN if either mask is empty."""
    if not p.any() or not g.any():
        return np.nan
    sp = p ^ ndi.binary_erosion(p)
    sg = g ^ ndi.binary_erosion(g)
    d_to_g = ndi.distance_transform_edt(~sg, sampling=spacing)
    d_to_p = ndi.distance_transform_edt(~sp, sampling=spacing)
    return float(np.percentile(np.concatenate([d_to_g[sp], d_to_p[sg]]), 95))

def worst_case_hd(shape, spacing):
    """Volume diagonal in mm: penalty used when a method returns an empty mask."""
    return float(np.sqrt(sum((s * z) ** 2 for s, z in zip(shape, spacing))))

# ------------------------------------------------------------------
# 6. Data helpers
# ------------------------------------------------------------------
def get_brats_cases(base_dir, limit):
    cases = []
    if not Path(base_dir).exists():
        return cases
    for d in sorted(Path(base_dir).rglob("*")):
        if d.is_dir():
            files = list(d.glob("*.nii*"))
            flair = [f for f in files if "flair" in f.name.lower()]
            seg = [f for f in files if "seg" in f.name.lower()]
            if flair and seg:
                cases.append((d.name, flair[0], seg[0]))
                if len(cases) >= limit:
                    break
    return cases

def crop_to_brain(vol, gt, brain, margin=5):
    idx = np.where(brain)
    sl = tuple(slice(max(i.min() - margin, 0), i.max() + margin + 1) for i in idx)
    return vol[sl], gt[sl]

def make_synthetic(seed):
    rng = np.random.default_rng(seed)
    shape = (96, 96, 48)
    X, Y, Z = np.ogrid[:96, :96, :48]
    brain = np.broadcast_to((X - 48) ** 2 + (Y - 48) ** 2 <= 30 ** 2, shape).copy()
    vol = np.zeros(shape, dtype=np.float32)              # background = 0
    vol[brain] = rng.normal(0.5, 0.1, int(brain.sum()))
    gt = np.zeros(shape, dtype=bool)
    gt[40:55, 40:55, 20:30] = True
    vol[gt] = rng.normal(0.9, 0.05, int(gt.sum()))
    return vol, gt

# ------------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting evaluation on {args.num_cases} cases...")

    cases = []
    for sdir in [args.input_dir, args.output_dir]:
        cases = get_brats_cases(sdir, args.num_cases)
        if cases:
            print(f"Found {len(cases)} cases in {sdir}")
            break

    if not cases:
        print("No MRI data found! Generating 2 synthetic cases for testing code...")
        cases = [("Synth_1", None, None), ("Synth_2", None, None)]

    rows = []
    for k, (cid, f_img, f_seg) in enumerate(cases, 1):
        t0 = time.time()
        spacing = (1.0, 1.0, 1.0)

        if f_img and f_seg:
            img = nib.load(str(f_img))
            vol = np.asarray(img.get_fdata(dtype=np.float32))
            gt = np.asarray(nib.load(str(f_seg)).dataobj) > 0
            spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
            if vol.shape != gt.shape:
                print(f"[{k}] {cid}: shape mismatch, skipped")
                continue
            b0 = vol > 1e-3
            if not b0.any():
                continue
            vol, gt = crop_to_brain(vol, gt, b0)
        else:
            vol, gt = make_synthetic(k)

        brain_mask = vol > 1e-3
        if not brain_mask.any() or not gt.any():
            continue

        penalty = worst_case_hd(vol.shape, spacing)
        for m_name, fn in METHODS.items():
            pred = fn(vol, brain_mask)
            d = dice_score(pred, gt)
            h = hd95(pred, gt, spacing)
            failed = bool(np.isnan(h))
            rows.append(dict(case=cid, method=m_name, dice=d, hd95=h,
                             hd95_penalised=penalty if failed else h,
                             failed=failed))

        print(f"[{k}/{len(cases)}] Processed {cid} in {time.time() - t0:.2f}s")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No valid cases were processed (all skipped). Check data paths.")

    out_dir = Path(args.output_dir) / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)

    print("\n--- FINAL RESULTS ---")
    summary = (df.groupby("method")
                 .agg(dice=("dice", "mean"),
                      hd95_mm=("hd95", "mean"),
                      hd95_penalised_mm=("hd95_penalised", "mean"),
                      failures=("failed", "sum"))
                 .sort_values("dice", ascending=False))
    print(summary)
    summary.to_csv(out_dir / "summary.csv")

    if SELECTION_LOG["total"]:
        print(f"\nCOPRAS picked the brightest cluster in "
              f"{SELECTION_LOG['picked_brightest']}/{SELECTION_LOG['total']} cases")

    # Wilcoxon signed-rank: copras_fcm vs sfcm_cc (paired by case)
    piv = df.pivot_table(index="case", columns="method", values="dice")
    if {"copras_fcm", "sfcm_cc"} <= set(piv.columns):
        pair = piv[["copras_fcm", "sfcm_cc"]].dropna()
        if len(pair) >= 6:
            try:
                stat, p = wilcoxon(pair["copras_fcm"], pair["sfcm_cc"])
                print(f"\nWilcoxon copras_fcm vs sfcm_cc (n={len(pair)}): "
                      f"W={stat:.3f}, p={p:.4g}")
            except ValueError as e:
                print(f"\nWilcoxon skipped: {e}")
        else:
            print("\nWilcoxon skipped: need at least 6 paired cases.")

    # Plot (version-safe)
    try:
        names = [m for m in METHODS if m in piv.columns]
        data = [piv[m].dropna().values for m in names]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(data, showfliers=True)
        ax.set_xticklabels(names)
        ax.set_title("3D Brain MRI Segmentation: Entropy-COPRAS-FCM vs Baselines")
        ax.set_ylabel("Dice Score")
        fig.tight_layout()
        fig.savefig(out_dir / "boxplot.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Plotting failed: {e!r}")

    print(f"Results saved to {out_dir}")
