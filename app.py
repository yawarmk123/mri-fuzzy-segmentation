import streamlit as st
import nibabel as nib
import numpy as np
import skfuzzy as fuzz
import plotly.express as px
import tempfile
import os

# Page ka basic setup
st.set_page_config(page_title="3D MRI Tumor Segmentation", layout="wide")
st.title("🧠 3D Brain Tumor MRI Segmentation")
st.write("Applying Fuzzy C-Means Logic to handle ambiguous tumor boundaries for MS Biomedical Engineering Research.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload an MRI Scan (.nii or .nii.gz)", type=['nii', 'nii.gz'])

if uploaded_file is not None:
    st.success("File successfully uploaded! Running Fuzzy Processing Engine...")
    
    # File ko temporary server memory me save karna
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
        
    try:
        # 1. Load the NIfTI Tensor Matrix
        mri_image = nib.load(tmp_path)
        volume_3d = mri_image.get_fdata()
        
        # 2. Extract a 2D Slice from the middle of the brain (Z-axis)
        mid_z = volume_3d.shape[2] // 2
        slice_2d = volume_3d[:, :, mid_z]
        
        # Normalize the tensor values between 0 and 1
        slice_norm = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)
        
        # 3. Apply Fuzzy C-Means (FCM) Clustering
        st.info("Applying Mathematical Fuzzy Logic (FCM) on Non-Binary Tumor Borders...")
        data = slice_norm.reshape(1, -1)
        n_clusters = 3
        
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data, c=n_clusters, m=2.0, error=0.005, maxiter=50, init=None
        )
        
        tumor_idx = np.argmax(cntr)
        tumor_membership = u[tumor_idx].reshape(slice_norm.shape)
        
        # 4. Display Results Side-by-Side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original MRI Slice")
            fig1 = px.imshow(slice_norm, color_continuous_scale='gray')
            fig1.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.subheader("Fuzzy Segmented Tumor (False Color)")
            fig2 = px.imshow(tumor_membership, color_continuous_scale='jet')
            fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)
            
    finally:
        os.remove(tmp_path)
else:
    st.warning("Please upload an MRI NIfTI file to begin.")
