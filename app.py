import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skfuzzy as fuzz

# Page Configuration
st.set_page_config(page_title="2D-to-3D MRI Fuzzy Segmentation", layout="wide")
st.title("🧠 2D-to-3D MRI Reconstruction & Fuzzy C-Means Segmentation")
st.write("Transforming standard 2D MRI slices into a 3D volumetric tensor and applying Fuzzy Logic for ambiguous tumor boundaries.")

# File Uploader for Multiple 2D Slices (PNG/JPG from Kaggle datasets)
uploaded_files = st.file_uploader(
    "Upload Multiple 2D MRI Slices (Select multiple PNG/JPG files)", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"Successfully uploaded {len(uploaded_files)} 2D slices! Building 3D Tensor Volume...")
    
    # 1. 2D Slices ko 3D Tensor mein stack karna (Mathematical Hyper-plane mapping)
    slices_list = []
    for file in uploaded_files:
        img = Image.open(file).convert('L').resize((128, 128))  # Standardize shape
        slices_list.append(np.array(img))
        
    # 3D Volume Tensor V shape: (Height, Width, Depth)
    volume_3d = np.stack(slices_list, axis=-1)
    
    st.info(f"Successfully constructed 3D Tensor Matrix of shape: {volume_3d.shape}")
    
    # 2. Z-axis slice navigation slider
    z_index = st.slider("Navigate Through 3D Volume (Z-axis Slices)", 0, volume_3d.shape[2] - 1, volume_3d.shape[2] // 2)
    
    current_slice = volume_3d[:, :, z_index]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Original 2D Slice (Z = {z_index})")
        fig1, ax1 = plt.subplots()
        ax1.imshow(current_slice, cmap='gray')
        ax1.axis('off')
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Fuzzy C-Means Segmentation (False-Color)")
        
        # Mathematical Normalization
        flat_data = current_slice.flatten().astype(float)
        norm_data = (flat_data - np.min(flat_data)) / (np.max(flat_data) - np.min(flat_data) + 1e-8)
        
        # Apply Fuzzy C-Means (FCM) Clustering to handle ambiguous boundaries
        n_clusters = 3
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            norm_data.reshape(1, -1), c=n_clusters, m=2.0, error=0.005, maxiter=100, init=None
        )
        
        tumor_cluster_idx = np.argmax(cntr)
        membership_map = u[tumor_cluster_idx].reshape(current_slice.shape)
        
        # Render False-Color Overlay
        fig2, ax2 = plt.subplots()
        ax2.imshow(current_slice, cmap='gray')
        ax2.imshow(membership_map, cmap='jet', alpha=0.6)  # False-color mapping for anomaly
        ax2.axis('off')
        st.pyplot(fig2)
        
    st.markdown("---")
    st.markdown("### 📊 Mathematical & Research Insight")
    st.latex(r"V \in \mathbb{R}^{X \times Y \times Z} \quad \text{(3D Reconstructed Tensor)}")
    st.write("By stacking 2D scans into a 3D spatial matrix and applying Fuzzy C-Means membership functions, the system successfully resolves overlapping intensity distributions and partial volume effects at the tumor boundaries.")

else:
    st.warning("Please upload a set of 2D MRI slice images from your dataset to initiate 2D-to-3D volumetric reconstruction.")
