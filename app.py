import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skfuzzy as fuzz

# Page Configuration
st.set_page_config(page_title="Clinical MRI Grid to 3D Tensor Segmentation", layout="wide")
st.title("🧠 Clinical MRI Grid Plate to 3D Tensor & Fuzzy Segmentation")
st.write("Extracting individual brain slices from radiological grid sheets, building a 3D volumetric tensor, and applying Fuzzy C-Means for ambiguous tumor boundaries.")

# File Uploader for Clinical MRI Grid Sheet
uploaded_file = st.file_uploader("Upload a Clinical MRI Grid Image Plate", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    grid_img = Image.open(uploaded_file).convert('L')
    st.image(uploaded_file, caption="Uploaded Radiological Grid Sheet", use_container_width=True)
    
    # Grid dimensions configuration for user
    st.subheader("Grid Slicing Configuration")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        rows = st.number_input("Number of Rows in Grid Plate", min_value=1, max_value=10, value=6)
    with col_r2:
        cols = st.number_input("Number of Columns in Grid Plate", min_value=1, max_value=10, value=8)
        
    img_arr = np.array(grid_img)
    img_h, img_w = img_arr.shape
    
    # Automatically chop the grid sheet into individual sub-slice tiles
    h_step = img_h // rows
    w_step = img_w // cols
    
    extracted_slices = []
    for r in range(rows):
        for c in range(cols):
            tile = img_arr[r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]
            # Standardize tile size
            tile_resized = np.array(Image.fromarray(tile).resize((64, 64)))
            extracted_slices.append(tile_resized)
            
    # Build 3D Tensor Volume V from chopped 2D tiles
    volume_3d = np.stack(extracted_slices, axis=-1)
    st.success(f"Successfully chopped grid into {len(extracted_slices)} individual slices & built 3D Tensor Volume of shape: {volume_3d.shape}!")
    
    # Z-axis Slider to navigate through reconstructed slices
    z_index = st.slider("Navigate Through Reconstructed 3D Slices (Z-axis)", 0, volume_3d.shape[2] - 1, volume_3d.shape[2] // 2)
    current_slice = volume_3d[:, :, z_index]
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader(f"Extracted Slice (Index {z_index})")
        fig1, ax1 = plt.subplots()
        ax1.imshow(current_slice, cmap='gray')
        ax1.axis('off')
        st.pyplot(fig1)
        
    with c2:
        st.subheader("Fuzzy C-Means Segmentation (False-Color)")
        
        # Mathematical Normalization
        flat_data = current_slice.flatten().astype(float)
        norm_data = (flat_data - np.min(flat_data)) / (np.max(flat_data) - np.min(flat_data) + 1e-8)
        
        # Apply Fuzzy C-Means (FCM) Clustering
        n_clusters = 3
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            norm_data.reshape(1, -1), c=n_clusters, m=2.0, error=0.005, maxiter=50, init=None
        )
        
        tumor_cluster_idx = np.argmax(cntr)
        membership_map = u[tumor_cluster_idx].reshape(current_slice.shape)
        
        # Render False-Color Overlay
        fig2, ax2 = plt.subplots()
        ax2.imshow(current_slice, cmap='gray')
        ax2.imshow(membership_map, cmap='jet', alpha=0.6)  # False-color mapping for uncertainty
        ax2.axis('off')
        st.pyplot(fig2)
        
    st.markdown("---")
    st.markdown("### 📊 Research & Mathematical Insight")
    st.latex(r"V \in \mathbb{R}^{X \times Y \times Z} \quad \text{(3D Tensor reconstructed from 2D grid plates)}")
    st.write("By programmatically parsing radiological grid sheets into spatial hyper-planes and executing Fuzzy C-Means membership functions, the pipeline effectively handles overlapping intensity bounds and partial volume effects.")

else:
    st.warning("Please upload one of your downloaded clinical MRI grid sheets to initiate the 3D pipeline.")
