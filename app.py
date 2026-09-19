import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skfuzzy as fuzz
import nibabel as nib
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
import requests
import os

st.set_page_config(page_title="Universal Medical AI Engine", layout="wide")
st.title("🧠 Universal Medical AI: Multi-Format MRI Processor")
st.write("Processing via Gaussian-Smoothed FCM, Otsu Background Masking, and Direct Web Fetching.")

# --- Helper Function for URL Downloading ---
def download_from_url(url, save_path):
    try:
        with st.spinner("Downloading file from cloud... Please wait."):
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Failed to download URL: {e}")
        return False

# --- Sidebar: Ground Truth Upload / URL ---
st.sidebar.header("🔬 Academic Validation (Ground Truth)")
gt_file = st.sidebar.file_uploader("Upload Ground Truth Mask", type=['nii', 'nii.gz'])
gt_url = st.sidebar.text_input("OR Paste Ground Truth URL (.nii.gz link):")

gt_volume = None
if gt_file is not None:
    with open("temp_gt.nii.gz", "wb") as f:
        f.write(gt_file.read())
    gt_volume = nib.load("temp_gt.nii.gz").get_fdata()
    st.sidebar.success("Ground Truth loaded from file!")
elif gt_url:
    if download_from_url(gt_url, "temp_gt_url.nii.gz"):
        gt_volume = nib.load("temp_gt_url.nii.gz").get_fdata()
        st.sidebar.success("Ground Truth loaded from URL!")

# --- Main Page: MRI Upload / URL ---
st.subheader("📥 Data Input Module")
col_upload, col_url = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("Upload Input MRI", type=['nii', 'nii.gz', 'png', 'jpg', 'jpeg'])
with col_url:
    mri_url = st.text_input("OR Paste MRI File URL (.nii.gz link):")

volume_3d = None

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(('.nii', '.nii.gz')):
        with open("temp_scan.nii.gz", "wb") as f:
            f.write(uploaded_file.read())
        volume_3d = nib.load("temp_scan.nii.gz").get_fdata()
        st.success("3D NIfTI loaded from file!")
    else:
        grid_img = Image.open(uploaded_file).convert('L')
        arr = np.array(grid_img.resize((128, 128), Image.Resampling.LANCZOS))
        volume_3d = np.stack([arr] * 5, axis=-1)
        st.success("2D Plate converted to Tensor!")
        
elif mri_url:
    if download_from_url(mri_url, "temp_scan_url.nii.gz"):
        volume_3d = nib.load("temp_scan_url.nii.gz").get_fdata()
        st.success("3D NIfTI loaded from URL!")

# --- CORE ACADEMIC PIPELINE ---
if volume_3d is not None:
    st.markdown("---")
    st.subheader("🔬 Volumetric Analysis & Gaussian-Smoothed FCM")
    
    if len(volume_3d.shape) == 3:
        max_z = volume_3d.shape[2] - 1
        z_idx = st.slider("Navigate Z-Axis Depth", 0, max_z, max_z // 2)
        current_slice = volume_3d[:, :, z_idx]
    else:
        current_slice = volume_3d
        z_idx = 0

    smoothed_slice = gaussian_filter(current_slice, sigma=1.0)
    
    try:
        thresh = threshold_otsu(smoothed_slice)
        brain_mask = smoothed_slice > thresh
    except:
        brain_mask = smoothed_slice > 0.1
        
    brain_pixels = smoothed_slice[brain_mask]
    
    if len(brain_pixels) > 0:
        norm = (brain_pixels - np.min(brain_pixels)) / (np.max(brain_pixels) - np.min(brain_pixels) + 1e-8)
        
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            norm.reshape(1, -1), c=3, m=2.0, error=0.005, maxiter=50, init=None
        )
        
        tumor_idx = np.argmax(cntr)
        membership = np.zeros_like(current_slice, dtype=float)
        membership[brain_mask] = u[tumor_idx]
        
        tumor_pixel_count = np.sum(u[tumor_idx] > 0.5)
        total_brain_count = np.sum(brain_mask)
        anomaly_ratio = (tumor_pixel_count / total_brain_count) * 100
        
        dice_score_text = "N/A (Ground Truth mask not provided)"
        if gt_volume is not None:
            if gt_volume.shape == volume_3d.shape:
                gt_slice = gt_volume[:, :, z_idx]
                pred_binary = membership > 0.5
                gt_binary = gt_slice > 0
                
                intersection = np.logical_and(pred_binary, gt_binary).sum()
                dice = (2. * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
                dice_score_text = f"{dice:.4f}"
            else:
                dice_score_text = "Error: Input and GT Dimensions mismatch!"

        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write(f"**Original Slice (Gaussian Smoothed)**")
            fig1, ax1 = plt.subplots()
            ax1.imshow(smoothed_slice, cmap='gray', interpolation='bicubic')
            ax1.axis('off')
            st.pyplot(fig1)
            
        with col_b:
            st.write("**Fuzzy Segregation Mask (Anomaly Map)**")
            fig2, ax2 = plt.subplots()
            ax2.imshow(smoothed_slice, cmap='gray', interpolation='bicubic')
            ax2.imshow(membership, cmap='jet', alpha=0.55, interpolation='bicubic')
            ax2.axis('off')
            st.pyplot(fig2)
            
        st.markdown("---")
        st.markdown(f"""
        ### 📋 Graduate-Level Validation Metrics:
        - **Pipeline Architecture:** Gaussian-Smoothed FCM with Otsu's Thresholding isolation.
        - **Tissue Proportion:** Anomaly cluster occupies **{anomaly_ratio:.2f}%** of the isolated brain volume at Z = {z_idx}.
        - **Internal Validity:** Fuzzy Partition Coefficient (FPC) = **{fpc:.3f}**.
        - **External Clinical Validity (Dice Coefficient):** **{dice_score_text}**
        """)
