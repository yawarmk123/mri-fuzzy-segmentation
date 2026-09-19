import streamlit as st
import nibabel as nib
import numpy as np
import skfuzzy as fuzz
import plotly.express as px
import tempfile
import os

st.set_page_config(page_title="3D MRI Tumor Segmentation", layout="wide")
st.title("🧠 Interactive 3D Brain MRI Fuzzy Segmentation")
st.write("Applying Fuzzy C-Means Logic to handle ambiguous tumor boundaries for MS Biomedical Engineering Research.")
st.markdown("---")

# Dono options ke liye radio button
option = st.radio("MRI Scan Source Select Karein:", 
                  ["Upload Custom .nii/.nii.gz File", "Use Built-in Synthetic Brain Matrix (Instant Test)"])

target_path = None

if option == "Upload Custom .nii/.nii.gz File":
    uploaded_file = st.file_uploader("Apni ya Professor ki MRI File Upload Karein", type=['nii', 'nii.gz'])
    if uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz')
        tmp.write(uploaded_file.read())
        target_path = tmp.name
else:
    if st.button("Generate Built-in Synthetic Brain Volume"):
        with st.spinner("3D mathematical brain tensor generate ho raha hai..."):
            shape = (64, 64, 30)
            vol = np.random.normal(0.2, 0.05, shape)
            z, y, x = np.ogrid[:64, :64, :30]
            
            brain_mask = (x - 32)**2 + (y - 32)**2 + (z - 15)**2 < 600
            vol[brain_mask] += 0.4
            
            tumor_mask = (x - 40)**2 + (y - 38)**2 + (z - 15)**2 < 80
            vol[tumor_mask] += 0.7
            
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(vol, affine)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz')
            nib.save(nifti_img, tmp.name)
            target_path = tmp.name

if target_path and os.path.exists(target_path):
    try:
        mri_image = nib.load(target_path)
        volume_3d = mri_image.get_fdata()
        
        st.success(f"MRI Successfully Loaded! Tensor Dimensions (X, Y, Z): {volume_3d.shape}")
        
        # Interactive slider for slice navigation
        max_slice = volume_3d.shape[2] - 1
        default_slice = max_slice // 2
        selected_slice_idx = st.slider("🔍 Brain Slices Navigate Karein (Z-Axis Depth)", 0, max_slice, default_slice)
        
        slice_2d = volume_3d[:, :, selected_slice_idx]
        
        # Normalize intensity
        slice_norm = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)
        
        # Apply Fuzzy C-Means (FCM)
        st.info("Applying Mathematical Fuzzy Logic (FCM) on Non-Binary Boundaries...")
        data = slice_norm.reshape(1, -1)
        n_clusters = 3
        
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data, c=n_clusters, m=2.0, error=0.005, maxiter=50, init=None
        )
        
        cluster_idx = np.argmax(cntr)
        segmented_membership = u[cluster_idx].reshape(slice_norm.shape)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Original Slice (Depth Z: {selected_slice_idx})")
            fig1 = px.imshow(slice_norm, color_continuous_scale='gray')
            fig1.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.subheader("Fuzzy Segmented Anomaly (False Color)")
            fig2 = px.imshow(segmented_membership, color_continuous_scale='jet')
            fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)
            
    except Exception as e:
        st.error(f"Processing ke dauran error aaya: {e}")
        
    finally:
        try:
            os.remove(target_path)
        except:
            pass
else:
    st.warning("Pehle upar diye gaye options me se ek select karein.")
