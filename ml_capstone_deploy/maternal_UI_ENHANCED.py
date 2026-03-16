# ─────────────────────────────────────────────────────────────────────────────
# maternal_UI_POLISHED.py — Maternal Health Risk Predictor (Polished UI)
# Run with: streamlit run maternal_UI_POLISHED.py
# ─────────────────────────────────────────────────────────────────────────────

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for polished styling (NO background changes)
st.markdown("""
    <style>
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        border: 2px solid #667eea !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-weight: 500;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #764ba2 !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        padding: 12px !important;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.35);
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 16px;
        border-radius: 10px;
        color: #1a5f3f;
        font-weight: bold;
        font-size: 1.05em;
        box-shadow: 0 4px 12px rgba(132, 250, 176, 0.25);
        border-left: 5px solid #84fab0;
    }
    
    /* Warning message */
    .warning-message {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 16px;
        border-radius: 10px;
        color: #5f3a1a;
        font-weight: bold;
        font-size: 1.05em;
        box-shadow: 0 4px 12px rgba(250, 112, 154, 0.25);
        border-left: 5px solid #fa709a;
    }
    
    /* Error message */
    .error-message {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 16px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        font-size: 1.05em;
        box-shadow: 0 4px 12px rgba(235, 51, 73, 0.25);
        border-left: 5px solid #eb3349;
    }
    
    /* Title styling */
    h1 {
        color: #333;
        font-weight: 700;
    }
    
    h2 {
        color: #333;
        font-weight: 600;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    
    h3 {
        color: #333;
        font-weight: 600;
    }
    
    h4 {
        color: #667eea;
        font-weight: 600;
    }
    
    /* Table styling */
    table {
        border-collapse: collapse;
        width: 100%;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    table th {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 14px;
        font-weight: bold;
        text-align: left;
    }
    
    table td {
        padding: 12px 14px;
        border-bottom: 1px solid #e8e8e8;
        color: #333;
        font-weight: 500;
    }
    
    table tr:hover td {
        background-color: #f8f9ff;
    }
    
    table tr:last-child td {
        border-bottom: none;
    }
    
    /* Markdown links */
    a {
        color: #667eea !important;
        font-weight: 600;
        text-decoration: none;
    }
    
    a:hover {
        color: #764ba2 !important;
        text-decoration: underline;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        padding: 20px;
    }
    
    [data-testid="stSidebar"] h2 {
        border-bottom-color: rgba(255, 255, 255, 0.3);
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
# SIDEBAR
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
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title(" Maternal Health Risk Predictor")
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
    age    = st.number_input(" Age (years)",                     min_value=10,  max_value=70,  value=30)
    sys_bp = st.number_input( Systolic BP (mmHg)", min_value=50,  max_value=200, value=120)

with col2:
    dia_bp = st.number_input(" Diastolic BP (mmHg)", min_value=30,  max_value=150, value=80)
    bs     = st.number_input(" Blood Sugar (mg/dL)",            min_value=1.0, max_value=30.0, value=7.0, step=0.1)

with col3:
    temp       = st.number_input(" Body Temperature (°F)",  min_value=95.0, max_value=106.0, value=98.6, step=0.1)
    heart_rate = st.number_input(" Heart Rate (bpm)",      min_value=40,   max_value=150,   value=76)

# Display Input Summary
st.markdown("### 📊 Input Summary:")
summary_col1, summary_col2, summary_col3 = st.columns(3)
with summary_col1:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 8px 0; font-size: 0.85em; opacity: 0.95;">Age</p>
        <p style="margin: 5px 0; font-size: 2em; font-weight: bold;">{age}</p>
        <p style="margin: 8px 0; font-size: 0.75em; opacity: 0.85;">years</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 12px;">
        <p style="margin: 8px 0; font-size: 0.85em; opacity: 0.95;">Systolic BP</p>
        <p style="margin: 5px 0; font-size: 2em; font-weight: bold;">{sys_bp}</p>
        <p style="margin: 8px 0; font-size: 0.75em; opacity: 0.85;">mmHg</p>
    </div>
    """, unsafe_allow_html=True)

with summary_col2:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 8px 0; font-size: 0.85em; opacity: 0.95;">Diastolic BP</p>
        <p style="margin: 5px 0; font-size: 2em; font-weight: bold;">{dia_bp}</p>
        <p style="margin: 8px 0; font-size: 0.75em; opacity: 0.85;">mmHg</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 12px;">
        <p style="margin: 8px 0; font-size: 0.85em; opacity: 0.95;">Blood Sugar</p>
        <p style="margin: 5px 0; font-size: 2em; font-weight: bold;">{bs:.1f}</p>
        <p style="margin: 8px 0; font-size: 0.75em; opacity: 0.85;">mg/dL</p>
    </div>
    """, unsafe_allow_html=True)

with summary_col3:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 8px 0; font-size: 0.85em; opacity: 0.95;">Temperature</p>
        <p style="margin: 5px 0; font-size: 2em; font-weight: bold;">{temp:.1f}</p>
        <p style="margin: 8px 0; font-size: 0.75em; opacity: 0.85;">°F</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 12px;">
        <p style="margin: 8px 0; font-size: 0.85em; opacity: 0.95;">Heart Rate</p>
        <p style="margin: 5px 0; font-size: 2em; font-weight: bold;">{heart_rate}</p>
        <p style="margin: 8px 0; font-size: 0.75em; opacity: 0.85;">bpm</p>
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
                        padding: 22px; border-radius: 12px; text-align: center; color: white;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15); transition: all 0.3s ease;"
                 onmouseover="this.style.transform='translateY(-4px)'; this.style.boxShadow='0 8px 16px rgba(0,0,0,0.2)'"
                 onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)'">
                <p style="margin: 0; font-size: 2em; font-weight: bold;">{icon}</p>
                <p style="margin: 12px 0 6px 0; font-size: 0.95em; opacity: 0.95; font-weight: 600;">{cls.title()}</p>
                <p style="margin: 0; font-size: 2.2em; font-weight: bold;">{prob*100:.1f}%</p>
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
        <div class="error-message" style="background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%); color: #eb3349; border-left: 5px solid #eb3349;">
            <b>Urgent:</b> Refer to specialist, monitor BP/BS continuously, check for preeclampsia.
        </div>
        """, unsafe_allow_html=True)
    elif risk_label == "mid risk":
        st.markdown("""
        <div class="warning-message" style="background: linear-gradient(135deg, #fffaf0 0%, #fff9e6 100%); color: #5f3a1a; border-left: 5px solid #fa709a;">
            <b>Enhanced Monitoring:</b> Schedule follow-up in 1-2 weeks, daily home BP monitoring, dietary guidance.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-message" style="background: linear-gradient(135deg, #f0fdf4 0%, #e6f9f0 100%); color: #1a5f3f; border-left: 5px solid #84fab0;">
            <b>Routine Care:</b> Continue regular prenatal check-ups and healthy lifestyle.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Patient Summary")

    st.markdown(f"""
| Vital | Value | Normal Range |
|---|---|---|
| Age | {age} years | 18-40 |
| Systolic BP | {sys_bp} mmHg | 90-120 |
| Diastolic BP | {dia_bp} mmHg | 60-80 |
| Blood Sugar | {bs:.1f} mg/dL | 70-100 |
| Body Temp | {temp:.1f} °F | 97-99 |
| Heart Rate | {heart_rate} bpm | 60-100 |
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 16px; margin-top: 20px;">
    <small style="color: #666; font-weight: 500;">🏥 Maternal Health Risk Predictor - Gradient Boosting + SMOTE - Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)
