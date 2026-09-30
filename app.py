import io
import os
import tempfile

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import skfuzzy as fuzz
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Neuro Segmentation Workbench", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
footer, #MainMenu { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1280px; }
h1.title { font-family: 'IBM Plex Serif', serif; font-size: 2rem; margin: 0 0 .2rem 0; color: #14202B; }
p.sub { color: #55636F; margin: 0 0 1.2rem 0; }
.kpi { background: #fff; border: 1px solid #D9E0E6; border-left: 4px solid #0E7C86;
       border-radius: 6px; padding: .8rem 1rem; }
.kpi .v { font-size: 1.6rem; font-weight: 600; color: #14202B; line-height: 1.2; }
.kpi .l { font-size: .82rem; color: #55636F; }
.kpi.warn { border-left-color: #C77700; } .kpi.high { border-left-color: #B3261E; }
.notice { background: #FFF6E5; border: 1px solid #F0D9A8; border-radius: 6px;
          padding: .7rem 1rem; color: #5C4200; font-size: .9rem; }
section[data-testid="stSidebar"] { background: #F3F6F8; }
</style>
""",
    unsafe_allow_html=True,
)


def kpi(label, value, tone=""):
    st.markdown(f'<div class="kpi {tone}"><div class="v">{value}</div><div class="l">{label}</div></div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- data loading
@st.cache_data(show_spinner=False)
def load_nifti(data: bytes):
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        img = nib.load(path)
        vol = np.asarray(img.get_fdata(), dtype=np.float32)
        spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    finally:
        os.remove(path)
    return vol, spacing


def make_demo():
    rng = np.random.default_rng(0)
    vol = rng.normal(0.2, 0.05, (64, 64, 30)).astype(np.float32)
    x, y, z = np.ogrid[:64, :64, :30]
    vol[(x - 32) ** 2 + (y - 32) ** 2 + ((z - 15) * 2) ** 2 < 600] += 0.4
    gt = (x - 40) ** 2 + (y - 38) ** 2 + ((z - 15) * 2) ** 2 < 80
    vol[gt] += 0.7
    return vol, gt.astype(np.uint8), (1.0, 1.0, 1.0)


# ---------------------------------------------------------------- segmentation
@st.cache_data(show_spinner=False)
def segment(vol, clusters, fuzz_m, bg_thr, seed=0):
    """Masked FCM: fit centres on a brain-voxel sample, predict membership for the full volume."""
    lo, hi = np.percentile(vol, [1, 99.5])
    norm = np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1)
    brain = norm > bg_thr
    vox = norm[brain]
    if vox.size < clusters * 10:
        return None
    rng = np.random.default_rng(seed)
    sample = vox if vox.size <= 30000 else rng.choice(vox, 30000, replace=False)
    cntr, *_ = fuzz.cluster.cmeans(sample.reshape(1, -1), clusters, fuzz_m, error=0.005, maxiter=100)
    u, *_ = fuzz.cluster.cmeans_predict(vox.reshape(1, -1), cntr, fuzz_m, error=0.005, maxiter=100)
    tumor = int(np.argmax(cntr))
    membership = np.zeros(vol.shape, dtype=np.float32)
    membership[brain] = u[tumor]
    fpc = float(np.mean(np.sum(u ** 2, axis=0)))
    return dict(norm=norm, brain=brain, membership=membership, fpc=fpc, centres=np.sort(cntr.ravel()))


def dice_iou(pred, gt):
    inter = float(np.sum(pred & gt))
    p, g = float(pred.sum()), float(gt.sum())
    if p + g == 0:
        return 1.0, 1.0
    return 2 * inter / (p + g), inter / (p + g - inter)


def slice_figure(img, overlay=None, thr=0.6):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
    ax.imshow(img.T, cmap="gray", origin="lower")
    if overlay is not None:
        m = overlay.T >= thr
        ax.imshow(np.ma.masked_where(~m, overlay.T), cmap="autumn", alpha=0.45, origin="lower", vmin=thr, vmax=1)
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors="#FF4D4D", linewidths=1.1, origin="lower")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Scan input")
    source = st.radio("Source", ["Demo case", "BraTS NIfTI (.nii / .nii.gz)", "2D image (.png / .jpg)"],
                      label_visibility="collapsed")
    st.markdown("### Segmentation settings")
    clusters = st.slider("Clusters (c)", 2, 6, 4)
    fuzz_m = st.slider("Fuzziness (m)", 1.2, 3.0, 2.0, 0.1)
    bg_thr = st.slider("Background cut-off", 0.02, 0.4, 0.12, 0.01,
                       help="Voxels below this normalised intensity are excluded as background / skull region.")
    thr = st.slider("Tumour membership threshold", 0.3, 0.95, 0.6, 0.05)

vol, gt, spacing, name = None, None, (1.0, 1.0, 1.0), ""

if source == "Demo case":
    vol, gt, spacing = make_demo()
    name = "Synthetic demo"
elif source.startswith("BraTS"):
    up = st.sidebar.file_uploader("Scan volume", type=["nii", "gz"])
    up_gt = st.sidebar.file_uploader("Ground-truth mask (optional)", type=["nii", "gz"])
    if up:
        vol, spacing = load_nifti(up.getvalue())
        name = up.name
        if up_gt:
            gt = (load_nifti(up_gt.getvalue())[0] > 0).astype(np.uint8)
            if gt.shape != vol.shape:
                st.sidebar.error(f"Mask shape {gt.shape} does not match scan {vol.shape}.")
                gt = None
else:
    up = st.sidebar.file_uploader("Image", type=["png", "jpg", "jpeg"])
    if up:
        vol = np.array(Image.open(up).convert("L").resize((256, 256)), dtype=np.float32)[:, :, None]
        name = up.name

# ---------------------------------------------------------------- header
st.markdown('<h1 class="title">Neuro Segmentation Workbench</h1>'
            '<p class="sub">Masked Fuzzy C-Means tumour segmentation for 3D MRI, with FPC, Dice and IoU validation.</p>',
            unsafe_allow_html=True)
st.markdown('<div class="notice"><b>Research prototype.</b> Outputs are algorithmic estimates for study and '
            'publication work. They are not a diagnosis and must not guide treatment.</div>', unsafe_allow_html=True)
st.write("")

if vol is None:
    st.info("Choose a scan in the sidebar to begin. The demo case needs no upload.")
    st.stop()

with st.spinner("Running masked FCM on the full volume..."):
    res = segment(vol, clusters, fuzz_m, bg_thr)
if res is None:
    st.error("Too few brain voxels after background removal. Lower the background cut-off in the sidebar.")
    st.stop()

mem = res["membership"]
pred = mem >= thr
n_brain, n_tumor = int(res["brain"].sum()), int(pred.sum())
burden = 100 * n_tumor / max(n_brain, 1)
vol_cm3 = n_tumor * float(np.prod(spacing)) / 1000
tone = "high" if burden > 30 else "warn" if burden > 10 else ""

# ---------------------------------------------------------------- KPIs
k = st.columns(4)
with k[0]: kpi("Segmented volume", f"{vol_cm3:.2f} cm³", tone)
with k[1]: kpi("Share of brain voxels", f"{burden:.2f}%", tone)
with k[2]: kpi("Fuzzy partition coefficient", f"{res['fpc']:.4f}")
with k[3]: kpi("Tensor shape", "×".join(map(str, vol.shape)))
st.write("")

# ---------------------------------------------------------------- tabs
tab_view, tab_metrics, tab_report = st.tabs(["Slice viewer", "Validation", "Report"])

with tab_view:
    z = 0
    if vol.shape[2] > 1:
        z = st.slider("Axial slice", 0, vol.shape[2] - 1, vol.shape[2] // 2)
    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"Original, slice {z}")
        f1 = slice_figure(res["norm"][:, :, z])
        st.pyplot(f1, use_container_width=True)
    with c2:
        st.caption(f"Masked FCM overlay, slice {z}")
        f2 = slice_figure(res["norm"][:, :, z], mem[:, :, z], thr)
        st.pyplot(f2, use_container_width=True)
        buf = io.BytesIO()
        f2.savefig(buf, format="png", dpi=200, facecolor="black", bbox_inches="tight", pad_inches=0)
        st.download_button("Download overlay (PNG)", buf.getvalue(), f"fcm_overlay_slice_{z}.png", "image/png")
    plt.close("all")

with tab_metrics:
    if gt is not None:
        d3, i3 = dice_iou(pred, gt.astype(bool))
        ds, is_ = dice_iou(pred[:, :, z], gt[:, :, z].astype(bool))
        m = st.columns(4)
        with m[0]: kpi("Dice, whole volume", f"{d3:.4f}")
        with m[1]: kpi("IoU, whole volume", f"{i3:.4f}")
        with m[2]: kpi(f"Dice, slice {z}", f"{ds:.4f}")
        with m[3]: kpi(f"IoU, slice {z}", f"{is_:.4f}")
    else:
        st.info("Add a ground-truth mask in the sidebar, or use the demo case, to compute Dice and IoU.")
    st.write("")
    st.markdown("**Cluster centres** (normalised intensity, ascending). The brightest centre is treated as tumour, "
                "which suits FLAIR and T2 scans.")
    st.bar_chart({f"C{i + 1}": float(v) for i, v in enumerate(res["centres"])})

with tab_report:
    if burden > 30:
        level = "Large segmented region"
    elif burden > 10:
        level = "Moderate segmented region"
    elif burden > 2:
        level = "Small segmented region"
    else:
        level = "Minimal or no segmented region"
    report = (
        f"NEURO SEGMENTATION REPORT (research use only)\n"
        f"Scan: {name}\nShape: {vol.shape}   Voxel size (mm): {tuple(round(s, 2) for s in spacing)}\n"
        f"Method: Masked FCM, c={clusters}, m={fuzz_m}, background cut-off={bg_thr}, threshold={thr}\n"
        f"Segmented volume: {vol_cm3:.2f} cm3 ({burden:.2f}% of brain voxels)\n"
        f"FPC: {res['fpc']:.4f}\nCategory: {level}\n"
    )
    st.markdown(f"**{level}.** About {burden:.1f}% of brain voxels ({vol_cm3:.2f} cm³) exceed the tumour "
                f"membership threshold of {thr}.")
    st.markdown(f"An FPC of {res['fpc']:.3f} describes how crisp the clustering is (1.0 means fully crisp). "
                "It measures the algorithm, not the pathology.")
    st.markdown("Any finding here needs confirmation by a radiologist on the original scans.")
    st.code(report, language="text")
    st.download_button("Download report (.txt)", report, "segmentation_report.txt")
