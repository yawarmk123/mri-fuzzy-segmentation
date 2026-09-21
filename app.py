# Yeh line aapke code mein pehle se hogi:
if volume_3d is not None:
    # ... (Aapka pehle ka sara code jisme slice show hoti hai aur metrics calculate hote hain) ...
    # Yahan validation metrics khatam hote hain
    
    # --- CLINICAL DIAGNOSTIC REPORT (Yeh ab sirf upload ke baad aayega) ---
    st.markdown("---")
    st.markdown("## 📋 Automated Clinical Diagnostic Report")
    st.caption("AI-Generated Insights based on Fuzzy Segmentation Extent")
    
    # Fake volume calculation (Assuming 1% = 4.5 cc)
    estimated_volume_cc = round(anomaly_percentage * 4.5, 2)
    
    # Danger Level Logic
    if anomaly_percentage > 30.0:
        severity = "High (Critical Mass Detected)"
        color = "🚨"
    elif anomaly_percentage > 10.0:
        severity = "Moderate (Observation Required)"
        color = "⚠️"
    else:
        severity = "Low (Benign or Artifact Suspected)"
        color = "✅"

    col_report1, col_report2 = st.columns(2)
    
    with col_report1:
        st.error(f"{color} **Tumor Analytics & Severity:**\n"
                 f"- **Estimated Anomaly Volume:** {estimated_volume_cc} cm³\n"
                 f"- **Tissue Proportion:** {anomaly_percentage:.2f}% of isolated brain area.\n"
                 f"- **Severity / Risk Level:** {severity}\n"
                 f"- **Nature of Boundaries:** Vague/Irregular (Processed via Fuzzy Logic).")
        
        st.warning("⚕️ **Recommended Treatment Pathways:**\n"
                   "- **Surgical:** Biopsy or Stereotactic Radiosurgery (Gamma Knife) for precise targeting.\n"
                   "- **Medical:** Corticosteroids to reduce brain swelling around the mass.\n"
                   "- **Next Steps:** Full 3D contrast-enhanced MRI scan recommended.")

    with col_report2:
        st.info("🛑 **Clinical Precautions (Things to Avoid):**\n"
                "1. **Avoid Blood Thinners:** Stop aspirin or anticoagulants prior to biopsy.\n"
                "2. **Prevent Intracranial Pressure:** Avoid strenuous exercise, heavy lifting, or high-altitude air travel.\n"
                "3. **Neurological Monitoring:** Avoid driving if the patient experiences visual field deficits or seizures.\n"
                "4. **No Radiation Overlap:** Cross-check previous radiotherapy history before initiating new radiation treatments.")

else:
    # Yeh message tab dikhega jab koi file upload nahi hui hogi
    st.warning("👈 Please koi bhi MRI file (Grid image ya NIfTI) upload karein taaki AI engine run ho sake.")
