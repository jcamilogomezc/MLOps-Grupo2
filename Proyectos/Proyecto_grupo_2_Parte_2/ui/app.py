import streamlit as st
import requests
import os
from typing import Dict, Any
import json

# Configuration
API_URL = os.getenv("API_URL", "http://diabetes-api:8000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Page configuration
st.set_page_config(
    page_title="Diabetes Readmission Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-top: 1rem;
    }
    .model-info {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e8f4f8;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Default/predefined values
DEFAULT_VALUES = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[70-80)",
    "weight": "?",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 1,
    "payer_code": "?",
    "medical_specialty": "Emergency/Trauma",
    "num_lab_procedures": 41,
    "num_procedures": 0,
    "num_medications": 1,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "diag_1": "250.83",
    "diag_2": "?",
    "diag_3": "?",
    "number_diagnoses": 1,
    "max_glu_serum": "None",
    "A1Cresult": "None",
    "metformin": "No",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "glipizide": "No",
    "glyburide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "examide": "No",
    "citoglipton": "No",
    "insulin": "No",
    "glyburide-metformin": "No",
    "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No",
    "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
    "change": "No",
    "diabetesMed": "No"
}

# Possible values for categorical fields
RACE_OPTIONS = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"]
GENDER_OPTIONS = ["Male", "Female", "Unknown/Invalid"]
AGE_OPTIONS = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
WEIGHT_OPTIONS = ["?", "[0-25)", "[25-50)", "[50-75)", "[75-100)", "[100-125)", "[125-150)", "[150-175)", "[175-200)", ">200"]
MEDICATION_OPTIONS = ["No", "Up", "Down", "Steady"]
CHANGE_OPTIONS = ["No", "Ch"]
DIABETES_MED_OPTIONS = ["No", "Yes"]
GLU_SERUM_OPTIONS = ["None", "Norm", ">200", ">300"]
A1C_OPTIONS = ["None", "Norm", ">7", ">8"]

def get_model_info():
    """Fetch model information from API"""
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching model info: {str(e)}")
        return None

def predict_readmission(features: Dict[str, Any]):
    """Send prediction request to API"""
    try:
        payload = {"features": features}
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"Error making prediction: {str(e)}"}

def main():
    # Initialize session state with default values if not set
    if 'values_loaded' not in st.session_state:
        for key, value in DEFAULT_VALUES.items():
            if key not in st.session_state:
                st.session_state[key] = value
        st.session_state.values_loaded = True
    
    # Header
    st.markdown('<div class="main-header">🏥 Diabetes Readmission Prediction</div>', unsafe_allow_html=True)
    
    # Load model info
    model_info = get_model_info()
    
    # Sidebar with model information
    with st.sidebar:
        st.header("📊 Model Information")
        if model_info:
            st.markdown('<div class="model-info">', unsafe_allow_html=True)
            st.write(f"**Model Name:** {model_info.get('model_name', 'N/A')}")
            st.write(f"**Stage/Version:** {model_info.get('model_stage_or_version', 'N/A')}")
            st.write(f"**Model Version:** {model_info.get('model_version', 'N/A')}")
            if model_info.get('expected_features_count'):
                st.write(f"**Features Count:** {model_info.get('expected_features_count')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show preprocessing status
            preprocessing = model_info.get('preprocessing_status', {})
            if preprocessing.get('label_encoders_loaded'):
                st.success("✅ Label encoders loaded")
            else:
                st.warning("⚠️ Label encoders not loaded")
            
            if preprocessing.get('scaler_loaded'):
                st.success("✅ Scaler loaded")
            else:
                st.warning("⚠️ Scaler not loaded")
        else:
            st.error("⚠️ Could not load model information")
            st.info(f"API URL: {API_URL}")
        
        st.markdown("---")
        st.header("ℹ️ Instructions")
        st.write("""
        1. Fill in the patient information below
        2. Use 'Load Default Values' to populate with example data
        3. Click 'Predict Readmission' to get the prediction
        4. Review the prediction and probabilities
        """)
    
    # Main content area
    st.header("Patient Information Form")
    
    # Button to load default values
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 Load Default Values", type="secondary"):
            for key, value in DEFAULT_VALUES.items():
                st.session_state[key] = value
            st.rerun()
    
    # Create form with tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Admission Details", "Medical Information", "Medications"])
    
    features = {}
    
    with tab1:
        st.subheader("Demographic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            race_val = st.session_state.get("race", DEFAULT_VALUES["race"])
            race_idx = RACE_OPTIONS.index(race_val) if race_val in RACE_OPTIONS else 0
            features["race"] = st.selectbox("Race", RACE_OPTIONS, index=race_idx, key="race")
            
            gender_val = st.session_state.get("gender", DEFAULT_VALUES["gender"])
            gender_idx = GENDER_OPTIONS.index(gender_val) if gender_val in GENDER_OPTIONS else (1 if len(GENDER_OPTIONS) > 1 else 0)
            features["gender"] = st.selectbox("Gender", GENDER_OPTIONS, index=gender_idx, key="gender")
            
            age_val = st.session_state.get("age", DEFAULT_VALUES["age"])
            age_idx = AGE_OPTIONS.index(age_val) if age_val in AGE_OPTIONS else (7 if len(AGE_OPTIONS) > 7 else 0)
            features["age"] = st.selectbox("Age Range", AGE_OPTIONS, index=age_idx, key="age")
            
            weight_val = st.session_state.get("weight", DEFAULT_VALUES["weight"])
            weight_idx = WEIGHT_OPTIONS.index(weight_val) if weight_val in WEIGHT_OPTIONS else 0
            features["weight"] = st.selectbox("Weight Range (kg)", WEIGHT_OPTIONS, index=weight_idx, key="weight")
        
        with col2:
            features["payer_code"] = st.text_input("Payer Code", value=st.session_state.get("payer_code", DEFAULT_VALUES["payer_code"]), key="payer_code")
            features["medical_specialty"] = st.text_input("Medical Specialty", value=st.session_state.get("medical_specialty", DEFAULT_VALUES["medical_specialty"]), key="medical_specialty")
    
    with tab2:
        st.subheader("Admission Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            features["admission_type_id"] = st.number_input("Admission Type ID", min_value=1, max_value=9, value=st.session_state.get("admission_type_id", DEFAULT_VALUES["admission_type_id"]), step=1, key="admission_type_id")
            features["admission_source_id"] = st.number_input("Admission Source ID", min_value=1, max_value=25, value=st.session_state.get("admission_source_id", DEFAULT_VALUES["admission_source_id"]), step=1, key="admission_source_id")
        
        with col2:
            features["discharge_disposition_id"] = st.number_input("Discharge Disposition ID", min_value=1, max_value=30, value=st.session_state.get("discharge_disposition_id", DEFAULT_VALUES["discharge_disposition_id"]), step=1, key="discharge_disposition_id")
            features["time_in_hospital"] = st.number_input("Time in Hospital (days)", min_value=1, max_value=14, value=st.session_state.get("time_in_hospital", DEFAULT_VALUES["time_in_hospital"]), step=1, key="time_in_hospital")
        
        with col3:
            features["number_outpatient"] = st.number_input("Number of Outpatient Visits (past year)", min_value=0, value=st.session_state.get("number_outpatient", DEFAULT_VALUES["number_outpatient"]), step=1, key="number_outpatient")
            features["number_emergency"] = st.number_input("Number of Emergency Visits (past year)", min_value=0, value=st.session_state.get("number_emergency", DEFAULT_VALUES["number_emergency"]), step=1, key="number_emergency")
            features["number_inpatient"] = st.number_input("Number of Inpatient Visits (past year)", min_value=0, value=st.session_state.get("number_inpatient", DEFAULT_VALUES["number_inpatient"]), step=1, key="number_inpatient")
    
    with tab3:
        st.subheader("Medical Information")
        col1, col2 = st.columns(2)
        
        with col1:
            features["num_lab_procedures"] = st.number_input("Number of Lab Procedures", min_value=0, value=st.session_state.get("num_lab_procedures", DEFAULT_VALUES["num_lab_procedures"]), step=1, key="num_lab_procedures")
            features["num_procedures"] = st.number_input("Number of Procedures", min_value=0, value=st.session_state.get("num_procedures", DEFAULT_VALUES["num_procedures"]), step=1, key="num_procedures")
            features["num_medications"] = st.number_input("Number of Medications", min_value=0, value=st.session_state.get("num_medications", DEFAULT_VALUES["num_medications"]), step=1, key="num_medications")
            features["number_diagnoses"] = st.number_input("Number of Diagnoses", min_value=1, value=st.session_state.get("number_diagnoses", DEFAULT_VALUES["number_diagnoses"]), step=1, key="number_diagnoses")
        
        with col2:
            features["diag_1"] = st.text_input("Primary Diagnosis (ICD-9)", value=st.session_state.get("diag_1", DEFAULT_VALUES["diag_1"]), key="diag_1")
            features["diag_2"] = st.text_input("Secondary Diagnosis (ICD-9)", value=st.session_state.get("diag_2", DEFAULT_VALUES["diag_2"]), key="diag_2")
            features["diag_3"] = st.text_input("Additional Diagnosis (ICD-9)", value=st.session_state.get("diag_3", DEFAULT_VALUES["diag_3"]), key="diag_3")
            
            glu_val = st.session_state.get("max_glu_serum", DEFAULT_VALUES["max_glu_serum"])
            glu_idx = GLU_SERUM_OPTIONS.index(glu_val) if glu_val in GLU_SERUM_OPTIONS else 0
            features["max_glu_serum"] = st.selectbox("Max Glucose Serum", GLU_SERUM_OPTIONS, index=glu_idx, key="max_glu_serum")
            
            a1c_val = st.session_state.get("A1Cresult", DEFAULT_VALUES["A1Cresult"])
            a1c_idx = A1C_OPTIONS.index(a1c_val) if a1c_val in A1C_OPTIONS else 0
            features["A1Cresult"] = st.selectbox("A1C Result", A1C_OPTIONS, index=a1c_idx, key="A1Cresult")
    
    with tab4:
        st.subheader("Diabetes Medications")
        col1, col2, col3 = st.columns(3)
        
        medications = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide",
            "glimepiride", "acetohexamide", "glipizide", "glyburide",
            "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
            "miglitol", "troglitazone", "tolazamide", "examide",
            "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
        ]
        
        medication_values = {}
        for i, med in enumerate(medications):
            col_idx = i % 3
            if col_idx == 0:
                current_col = col1
            elif col_idx == 1:
                current_col = col2
            else:
                current_col = col3
            
            with current_col:
                default_med_value = st.session_state.get(med, DEFAULT_VALUES.get(med, "No"))
                try:
                    med_idx = MEDICATION_OPTIONS.index(default_med_value) if default_med_value in MEDICATION_OPTIONS else 0
                except (ValueError, AttributeError):
                    med_idx = 0
                medication_values[med] = st.selectbox(
                    med.replace("-", " ").title(),
                    MEDICATION_OPTIONS,
                    index=med_idx,
                    key=med
                )
        
        features.update(medication_values)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            change_val = st.session_state.get("change", DEFAULT_VALUES["change"])
            try:
                change_idx = CHANGE_OPTIONS.index(change_val) if change_val in CHANGE_OPTIONS else 0
            except (ValueError, AttributeError):
                change_idx = 0
            features["change"] = st.selectbox("Change in Medications", CHANGE_OPTIONS, index=change_idx, key="change")
        with col2:
            diabetes_med_val = st.session_state.get("diabetesMed", DEFAULT_VALUES["diabetesMed"])
            try:
                diabetes_med_idx = DIABETES_MED_OPTIONS.index(diabetes_med_val) if diabetes_med_val in DIABETES_MED_OPTIONS else 0
            except (ValueError, AttributeError):
                diabetes_med_idx = 0
            features["diabetesMed"] = st.selectbox("Diabetes Medication", DIABETES_MED_OPTIONS, index=diabetes_med_idx, key="diabetesMed")
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Readmission", type="primary", use_container_width=True)
    
    # Make prediction
    if predict_button:
        with st.spinner("Making prediction..."):
            result = predict_readmission(features)
            
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                # Display prediction results
                st.markdown("---")
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.header("📋 Prediction Results")
                
                # Main prediction
                prediction = result.get("readmission_prediction", "Unknown")
                probabilities = result.get("probabilities", {})
                
                # Color code based on prediction
                if prediction == "NO":
                    st.success(f"## ✅ Prediction: **{prediction}**")
                    st.write("The model predicts that the patient will **NOT** be readmitted.")
                elif prediction == "<30":
                    st.warning(f"## ⚠️ Prediction: **{prediction}**")
                    st.write("The model predicts that the patient will be readmitted within **30 days**.")
                elif prediction == ">30":
                    st.info(f"## ℹ️ Prediction: **{prediction}**")
                    st.write("The model predicts that the patient will be readmitted after **30 days**.")
                else:
                    st.write(f"## Prediction: **{prediction}**")
                
                # Display probabilities
                if probabilities:
                    st.subheader("📊 Prediction Probabilities")
                    prob_cols = st.columns(len(probabilities))
                    
                    for idx, (label, prob) in enumerate(probabilities.items()):
                        with prob_cols[idx]:
                            st.metric(
                                label=f"{label}",
                                value=f"{prob:.2%}",
                                delta=None
                            )
                    
                    # Probability bar chart
                    import pandas as pd
                    prob_df = pd.DataFrame([
                        {"Readmission Status": label, "Probability": prob}
                        for label, prob in probabilities.items()
                    ])
                    st.bar_chart(prob_df.set_index("Readmission Status"))
                
                # Model information in results
                if model_info:
                    st.markdown("---")
                    st.caption(f"**Model Used:** {model_info.get('model_name', 'N/A')} | "
                             f"**Version:** {model_info.get('model_version', 'N/A')} | "
                             f"**Stage:** {model_info.get('model_stage_or_version', 'N/A')}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show input features (collapsible)
                with st.expander("🔍 View Input Features"):
                    st.json(features)

if __name__ == "__main__":
    main()

