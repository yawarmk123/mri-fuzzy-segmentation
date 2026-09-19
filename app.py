import streamlit as st
import nibabel as nib
import numpy as np
import skfuzzy as fuzz
import plotly.express as px
import tempfile
import os

# Page setup
st.set_page_config(page_title="3D MRI Tumor Segmentation", layout="wide")
st.title("🧠 3D Brain Tumor MRI Segmentation")
st.write("Applying Fuzzy C-Means Logic to handle ambiguous tumor boundaries for MS Biomedical Engineering Research.")
st.markdown("---")

st.subheader("MRI Scan Source")
option = st.radio("Choose option:", ["Upload your own .nii/.nii.gz file", "Generate Synthetic 3D Brain Matrix (No Download Needed)"])

target_path = None

if option == "Upload your own .nii/.nii.gz file":
    uploaded_file = st.file_uploader("Upload an MRI Scan", type=['nii', 'nii.gz'])
    if uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz')
        tmp.write(uploaded_file.read())
        target_path = tmp.name
else:
    if st.button("Generate & Process Synthetic Brain Volume"):
        with st.spinner("Generating 3D mathematical brain tensor and tumor simulation..."):
            # Create synthetic 3D brain volume with a simulated tumor region
            shape = (64, 64, 30)
            vol = np.random.normal(0.2, 0.05, shape)
            z, y, x = np.ogrid[:64, :64, :30]
            
            # Brain tissue mask
            brain_mask = (x - 32)**2 + (y - 32)**2 + (z - 15)**2 < 600
            vol[brain_mask] += 0.4
            
            # Tumor anomaly mask (high intensity region with ambiguous borders)
            tumor_mask = (x - 40)**2 + (y - 38)**2 + (z - 15)**2 < 80
            vol[tumor_mask] += 0.7
            
            # Save as temporary NIfTI file
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(vol, affine)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz')
            nib.save(nifti_img, tmp.name)
            target_path = tmp.name

if target_path and os.path.exists(target_path):
    st.success("MRI Matrix loaded successfully! Running Fuzzy Processing Engine...")
    
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
        segmented_membership = u[cluster_idx].reshape(slice_norm.shape)
        
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
        try:
            os.remove(target_path)
        except:
            pass
else:
    st.warning("Please select an option and click the button to begin.")
