# ─────────────────────────────────────────────────────────────────────────────
# maternal_UI_ATTRACTIVE.py — Maternal Health Risk Predictor (Attractive UI)
# Run with: streamlit run maternal_UI_ATTRACTIVE.py
# ─────────────────────────────────────────────────────────────────────────────

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for attractive styling (keeping same structure)
st.markdown("""
    <style>
    /* Page background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar text color */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: white;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 10px;
        font-weight: bold;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Card style for content */
    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        margin: 15px 0;
    }
    
    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 15px;
        border-radius: 10px;
        color: #1a5f3f;
        font-weight: bold;
    }
    
    /* Warning message */
    .warning-message {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 15px;
        border-radius: 10px;
        color: #5f3a1a;
        font-weight: bold;
    }
    
    /* Error message */
    .error-message {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    
    /* Title styling */
    h1, h2, h3 {
        color: #333;
    }
    
    /* Table styling */
    table {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    table th {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        border: 2px solid #667eea;
        border-radius: 8px;
        padding: 10px;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #764ba2;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model   = joblib.load("ml_capstone_deploy/trained_models/maternal_risk_model.pkl")
    encoder = joblib.load("ml_capstone_deploy/trained_models/label_encoder.pkl")
    return model, encoder

model, encoder = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# UI ENHANCEMENT 1: ADD SIDEBAR WITH ABOUT & MODEL INFO (SAME STRUCTURE)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 About This App")
    st.markdown("""
    **Maternal Health Risk Predictor**
    
    This application uses an AI model trained on maternal health data to predict 
    risk levels for pregnant patients.
    
    **Model Information:**
    - Algorithm: Gradient Boosting
    - Training Accuracy: 82.16%
    - Features: 6 vital signs
    - Classes: 3 risk levels
    
    Created: 2026 | Version: 1.0
    """)
    
    st.markdown("---")
    st.markdown("## 🔗 Resources")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[GitHub Repository](https://github.com/adupatil161/maternal-health-app)")
    with col2:
        st.markdown("[Documentation](https://github.com/adupatil161/maternal-health-app)")

# ─────────────────────────────────────────────────────────────────────────────
# UI ENHANCEMENT 2: ADD "HOW THIS WORKS" EXPANDABLE SECTION (SAME STRUCTURE)
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏥 Maternal Health Risk Predictor")
st.markdown("Enter the patient's vital signs and click **Predict Risk Level**.")

with st.expander("ℹ️ How This Works"):
    st.markdown("""
    This app uses a machine learning model trained on maternal health data to predict risk levels.
    
    **The model evaluates:**
    - Age of the patient
    - Systolic Blood Pressure
    - Diastolic Blood Pressure
    - Blood Sugar Level
    - Body Temperature
    - Heart Rate
    
    **Risk Levels:**
    - 🟢 **Low Risk**: Healthy pregnancy, routine monitoring
    - 🟡 **Mid Risk**: Some concerns, regular checkups recommended
    - 🔴 **High Risk**: Significant concerns, immediate medical attention needed
    
    **Disclaimer:** This is an educational tool. Always consult qualified healthcare professionals for medical decisions.
    """)

st.markdown("---")

st.subheader("📋 Enter Patient Information")
col1, col2, col3 = st.columns(3)

with col1:
    age    = st.number_input("👤 Age (years)",                     min_value=10,  max_value=70,  value=30)
    sys_bp = st.number_input("❤️ Systolic BP (mmHg)", min_value=50,  max_value=200, value=120)

with col2:
    dia_bp = st.number_input("💓 Diastolic BP (mmHg)", min_value=30,  max_value=150, value=80)
    bs     = st.number_input("🩸 Blood Sugar (mg/dL)",            min_value=1.0, max_value=30.0, value=7.0, step=0.1)

with col3:
    temp       = st.number_input("🌡️ Body Temperature (°F)",  min_value=95.0, max_value=106.0, value=98.6, step=0.1)
    heart_rate = st.number_input("💗 Heart Rate (bpm)",      min_value=40,   max_value=150,   value=76)

# Display Input Summary
st.markdown("### 📊 Input Summary:")
summary_col1, summary_col2, summary_col3 = st.columns(3)
with summary_col1:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.9;">Age</p>
        <p style="margin: 0; font-size: 1.8em; font-weight: bold;">{age}</p>
        <p style="margin: 5px 0; font-size: 0.8em; opacity: 0.8;">years</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 10px;">
        <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.9;">Systolic BP</p>
        <p style="margin: 0; font-size: 1.8em; font-weight: bold;">{sys_bp}</p>
        <p style="margin: 5px 0; font-size: 0.8em; opacity: 0.8;">mmHg</p>
    </div>
    """, unsafe_allow_html=True)

with summary_col2:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.9;">Diastolic BP</p>
        <p style="margin: 0; font-size: 1.8em; font-weight: bold;">{dia_bp}</p>
        <p style="margin: 5px 0; font-size: 0.8em; opacity: 0.8;">mmHg</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 10px;">
        <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.9;">Blood Sugar</p>
        <p style="margin: 0; font-size: 1.8em; font-weight: bold;">{bs:.1f}</p>
        <p style="margin: 5px 0; font-size: 0.8em; opacity: 0.8;">mg/dL</p>
    </div>
    """, unsafe_allow_html=True)

with summary_col3:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.9;">Temperature</p>
        <p style="margin: 0; font-size: 1.8em; font-weight: bold;">{temp:.1f}</p>
        <p style="margin: 5px 0; font-size: 0.8em; opacity: 0.8;">°F</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 10px;">
        <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.9;">Heart Rate</p>
        <p style="margin: 0; font-size: 1.8em; font-weight: bold;">{heart_rate}</p>
        <p style="margin: 5px 0; font-size: 0.8em; opacity: 0.8;">bpm</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
predict_clicked = st.button("🔍 Predict Risk Level", use_container_width=True)

if predict_clicked:
    data = pd.DataFrame({
        "Age"        : [age],
        "SystolicBP" : [sys_bp],
        "DiastolicBP": [dia_bp],
        "BS"         : [bs],
        "BodyTemp"   : [temp],
        "HeartRate"  : [heart_rate]
    })

    prediction_num = model.predict(data)[0]
    probabilities  = model.predict_proba(data)[0]
    risk_label = encoder.classes_[prediction_num]

    st.subheader("📊 Prediction Results")

    m1, m2, m3 = st.columns(3)
    for col_widget, cls, prob in zip([m1, m2, m3], encoder.classes_, probabilities):
        with col_widget:
            if cls.lower() == "high risk":
                color_start = "#eb3349"
                color_end = "#f45c43"
                icon = "🔴"
            elif cls.lower() == "mid risk":
                color_start = "#fa709a"
                color_end = "#fee140"
                icon = "🟡"
            else:
                color_start = "#84fab0"
                color_end = "#8fd3f4"
                icon = "🟢"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color_start} 0%, {color_end} 100%);
                        padding: 20px; border-radius: 12px; text-align: center; color: white;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <p style="margin: 0; font-size: 1.8em;">{icon}</p>
                <p style="margin: 10px 0 5px 0; font-size: 0.95em; opacity: 0.95;">{cls.title()}</p>
                <p style="margin: 0; font-size: 2em; font-weight: bold;">{prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    if risk_label == "high risk":
        st.markdown("""
        <div class="error-message">
            🔴 HIGH RISK — Immediate medical attention recommended.
        </div>
        """, unsafe_allow_html=True)
    elif risk_label == "mid risk":
        st.markdown("""
        <div class="warning-message">
            🟡 MID RISK — Increased monitoring and care advised.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-message">
            🟢 LOW RISK — Patient vitals appear within safe range.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🏥 Clinical Recommendation")
    if risk_label == "high risk":
        st.markdown("""
        <div class="card" style="border-left-color: #eb3349;">
            <p><b>Urgent: Refer to specialist, monitor BP/BS continuously, check for preeclampsia.</b></p>
        </div>
        """, unsafe_allow_html=True)
    elif risk_label == "mid risk":
        st.markdown("""
        <div class="card" style="border-left-color: #fa709a;">
            <p><b>Schedule follow-up in 1-2 weeks, daily home BP monitoring, dietary guidance.</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="border-left-color: #84fab0;">
            <p><b>Routine care: Continue regular prenatal check-ups and healthy lifestyle.</b></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Patient Summary")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"""
        <div class="card">
        <h4>Vital Signs</h4>
        
| Vital | Value | Normal Range |
|---|---|---|
| Age | {age} years | 18-40 |
| Systolic BP | {sys_bp} mmHg | 90-120 |
| Diastolic BP | {dia_bp} mmHg | 60-80 |
        </div>
        """, unsafe_allow_html=True)
    
    with s2:
        st.markdown(f"""
        <div class="card">
        <h4>Biochemical Parameters</h4>
        
| Vital | Value | Normal Range |
|---|---|---|
| Blood Sugar | {bs:.1f} mg/dL | 70-100 |
| Body Temp | {temp:.1f} °F | 97-99 |
| Heart Rate | {heart_rate} bpm | 60-100 |
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: white; border-radius: 12px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-top: 20px;">
    <small>🏥 Maternal Health Risk Predictor - Gradient Boosting + SMOTE - Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)
