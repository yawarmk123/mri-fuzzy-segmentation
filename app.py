import streamlit as st
import nibabel as nib
import numpy as np
import skfuzzy as fuzz
import plotly.express as px
import tempfile
import os
import urllib.request

# Page setup
st.set_page_config(page_title="3D MRI Tumor Segmentation", layout="wide")
st.title("🧠 3D Brain Tumor MRI Segmentation")
st.write("Applying Fuzzy C-Means Logic to handle ambiguous tumor boundaries for MS Biomedical Engineering Research.")
st.markdown("---")

# Option to upload or use default sample
st.subheader("Select MRI Source")
option = st.radio("Choose how to load the MRI scan:", ["Upload your own .nii/.nii.gz file", "Use Built-in Sample Brain MRI"])

target_path = None
temp_file_obj = None

if option == "Upload your own .nii/.nii.gz file":
    uploaded_file = st.file_uploader("Upload an MRI Scan", type=['nii', 'nii.gz'])
    if uploaded_file is not None:
        temp_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz')
        temp_file_obj.write(uploaded_file.read())
        target_path = temp_file_obj.name
else:
    if st.button("Load and Process Sample MRI"):
        with st.spinner("Downloading sample brain MRI from public repository..."):
            sample_url = "https://raw.githubusercontent.com/miykael/nipype-tutorial/master/notebooks/data/subject1_T1.nii.gz"
            temp_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz')
            urllib.request.urlretrieve(sample_url, temp_file_obj.name)
            target_path = temp_file_obj.name

if target_path and os.path.exists(target_path):
    st.success("MRI Scan loaded successfully! Running Fuzzy Processing Engine...")
    
    try:
        # 1. Load the NIfTI Tensor Matrix
        mri_image = nib.load(target_path)
        volume_3d = mri_image.get_fdata()
        
        # 2. Extract a 2D Slice from the middle of the brain (Z-axis)
        mid_z = volume_3d.shape[2] // 2
        slice_2d = volume_3d[:, :, mid_z]
        
        # Normalize tensor values between 0 and 1
        slice_norm = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)
        
        # 3. Apply Fuzzy C-Means (FCM) Clustering
        st.info("Applying Mathematical Fuzzy Logic (FCM) on Non-Binary Boundaries...")
        data = slice_norm.reshape(1, -1)
        n_clusters = 3
        
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data, c=n_clusters, m=2.0, error=0.005, maxiter=50, init=None
        )
        
        cluster_idx = np.argmax(cntr)
        segmented_membership = u[cluster_idx].reshape(slice_norm.slice_shape if hasattr(slice_norm, 'slice_shape') else slice_norm.shape)
        
        # 4. Display Results Side-by-Side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original MRI Slice")
            fig1 = px.imshow(slice_norm, color_continuous_scale='gray')
            fig1.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.subheader("Fuzzy Segmented Region (False Color)")
            fig2 = px.imshow(segmented_membership, color_continuous_scale='jet')
            fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)
            
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        
    finally:
        if temp_file_obj:
            temp_file_obj.close()
            try:
                os.remove(target_path)
            except:
                pass
else:
    st.warning("Please select a sample scan or upload a file to begin.")
