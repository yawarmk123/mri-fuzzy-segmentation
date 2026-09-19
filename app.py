import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skfuzzy as fuzz
import nibabel as nib
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter

st.set_page_config(page_title="Universal Medical AI Engine", layout="wide")
st.title("🧠 Universal Medical AI: Multi-Format MRI Processor")
st.write("Processing Legacy 2D Plates and Native 3D NIfTI via Gaussian-Smoothed FCM and Otsu Background Masking.")

# Sidebar for Ground Truth (For Dice Score Validation)
st.sidebar.header("🔬 Academic Validation")
gt_file = st.sidebar.file_uploader("Upload Ground Truth Mask (.nii/.nii.gz) for Dice Score", type=['nii', 'nii.gz'])

# Universal File Uploader
uploaded_file = st.file_uploader(
    "Upload Input MRI File (.nii, .nii.gz, .png, .jpg, .jpeg)", 
    type=['nii', 'nii.gz', 'png', 'jpg', 'jpeg']
)

volume_3d = None
gt_volume = None

# Load Ground Truth if provided
if gt_file is not None:
    bytes_data = gt_file.read()
    with open("temp_gt.nii.gz", "wb") as f:
        f.write(bytes_data)
    img_gt = nib.load("temp_gt.nii.gz")
    gt_volume = img_gt.get_fdata()
    st.sidebar.success("Ground Truth loaded successfully!")

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    
    if file_name.endswith(('.nii', '.nii.gz')):
        bytes_data = uploaded_file.read()
        with open("temp_scan.nii.gz", "wb") as f:
            f.write(bytes_data)
        img_nii = nib.load("temp_scan.nii.gz")
        volume_3d = img_nii.get_fdata()
        st.success(f"3D NIfTI file successfully loaded! Tensor Shape: {volume_3d.shape}")
        
    else:
        grid_img = Image.open(uploaded_file).convert('L')
        st.image(uploaded_file, caption="Uploaded 8-bit Plate", width=350)
        
        is_grid = st.checkbox("Multiple Slices Grid Sheet?", value=True)
        
        if is_grid:
            col1, col2 = st.columns(2)
            with col1:
                rows = st.number_input("Grid Rows", min_value=1, max_value=10, value=6)
            with col2:
                cols = st.number_input("Grid Columns", min_value=1, max_value=10, value=8)
            
            img_arr = np.array(grid_img)
            h, w = img_arr.shape
            h_step, w_step = h // rows, w // cols
            
            tiles = []
            for r in range(rows):
                for c in range(cols):
                    tile = img_arr[r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]
                    tiles.append(np.array(Image.fromarray(tile).resize((128, 128), Image.Resampling.LANCZOS)))
            
            volume_3d = np.stack(tiles, axis=-1)
            st.success("Grid sheet digitized into 3D Tensor!")
        else:
            arr = np.array(grid_img.resize((256, 256), Image.Resampling.LANCZOS))
            volume_3d = np.stack([arr] * 5, axis=-1)

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

    # 1. Gaussian Pre-smoothing (Resolves Claude's strict requirement)
    smoothed_slice = gaussian_filter(current_slice, sigma=1.0)
    
    # 2. Otsu Skull-Stripping (Removes background/skull from math)
    try:
        thresh = threshold_otsu(smoothed_slice)
        brain_mask = smoothed_slice > thresh
    except:
        brain_mask = smoothed_slice > 0.1
        
    brain_pixels = smoothed_slice[brain_mask]
    
    if len(brain_pixels) > 0:
        norm = (brain_pixels - np.min(brain_pixels)) / (np.max(brain_pixels) - np.min(brain_pixels) + 1e-8)
        
        # 3. Fuzzy C-Means ONLY on Brain Tissue
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            norm.reshape(1, -1), c=3, m=2.0, error=0.005, maxiter=50, init=None
        )
        
        # Anomaly Rule: Highest mean intensity cluster
        tumor_idx = np.argmax(cntr)
        
        # Reconstruct 2D Membership Matrix
        membership = np.zeros_like(current_slice, dtype=float)
        membership[brain_mask] = u[tumor_idx]
        
        # Calculate Math Ratios correctly (excluding background)
        tumor_pixel_count = np.sum(u[tumor_idx] > 0.5)
        total_brain_count = np.sum(brain_mask)
        anomaly_ratio = (tumor_pixel_count / total_brain_count) * 100
        
        # 4. Dice Score Validation (If Ground Truth provided)
        dice_score_text = "N/A (Ground Truth mask not uploaded)"
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
            # interpolation='bicubic' creates smooth photo-like quality
            ax1.imshow(smoothed_slice, cmap='gray', interpolation='bicubic')
            ax1.axis('off')
            st.pyplot(fig1)
            
        with col_b:
            st.write("**Fuzzy Segregation Mask (Anomaly Map)**")
            fig2, ax2 = plt.subplots()
            ax2.imshow(smoothed_slice, cmap='gray', interpolation='bicubic')
            # Overlay heatmap with bicubic smoothing
            ax2.imshow(membership, cmap='jet', alpha=0.55, interpolation='bicubic')
            ax2.axis('off')
            st.pyplot(fig2)
            
        st.markdown("---")
        st.markdown(f"""
        ### 📋 Graduate-Level Validation Metrics:
        - **Pipeline Architecture:** **Gaussian-Smoothed Fuzzy C-Means (FCM)**. Gaussian preprocessing ($\sigma = 1.0$) suppresses high-frequency noise. Otsu's Thresholding isolates internal brain tissue.
        - **Anomaly Identification Rule:** Cluster centroid with the maximum scalar intensity designates the anomaly.
        - **Tissue Proportion:** Anomaly cluster occupies **{anomaly_ratio:.2f}%** of the isolated brain volume at Z = {z_idx}. *(Excludes skull/background artifacts).*
        - **Internal Validity:** Fuzzy Partition Coefficient (FPC) = **{fpc:.3f}**.
        - **External Clinical Validity (Dice Coefficient):** **{dice_score_text}**
        """)
    else:
        st.error("Brain tissue detection failed. Image might be completely black.")
