# ─────────────────────────────────────────────────────────────────────────────
# maternal_UI_GORGEOUS.py — Maternal Health Risk Predictor (Gorgeous UI)
# Run with: streamlit run maternal_UI_GORGEOUS.py
# ─────────────────────────────────────────────────────────────────────────────

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for gorgeous styling with glow effects
st.markdown("""
    <style>
    /* Page background with subtle gradient */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
    }
    
    /* Button styling with glow */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4),
                    0 0 20px rgba(102, 126, 234, 0.3);
        font-size: 1.05em;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.5),
                    0 0 30px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Number input styling with glow */
    .stNumberInput > div > div > input {
        border: 2px solid #667eea !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        font-weight: 600 !important;
        background: white !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 0 0 rgba(102, 126, 234, 0) !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #764ba2 !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2),
                    0 0 15px rgba(102, 126, 234, 0.3) !important;
        background: white !important;
    }
    
    /* Expander styling with glow */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 14px 18px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.25);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35),
                    0 0 20px rgba(102, 126, 234, 0.25);
        transform: translateY(-2px);
    }
    
    /* Metric card styling with glow and glassmorphism */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 22px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35),
                    0 0 25px rgba(102, 126, 234, 0.25),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.1) 50%, transparent 70%);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .metric-card:hover {
        transform: translateY(-6px) scale(1.03);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.5),
                    0 0 35px rgba(102, 126, 234, 0.35);
    }
    
    /* Success message with glow */
    .success-message {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 18px;
        border-radius: 12px;
        color: #1a5f3f;
        font-weight: 700;
        font-size: 1.08em;
        box-shadow: 0 8px 28px rgba(132, 250, 176, 0.35),
                    0 0 20px rgba(132, 250, 176, 0.25);
        border-left: 6px solid #84fab0;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .success-message:hover {
        box-shadow: 0 12px 36px rgba(132, 250, 176, 0.45),
                    0 0 30px rgba(132, 250, 176, 0.35);
        transform: translateY(-2px);
    }
    
    /* Warning message with glow */
    .warning-message {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 18px;
        border-radius: 12px;
        color: #5f3a1a;
        font-weight: 700;
        font-size: 1.08em;
        box-shadow: 0 8px 28px rgba(250, 112, 154, 0.35),
                    0 0 20px rgba(250, 112, 154, 0.25);
        border-left: 6px solid #fa709a;
        transition: all 0.3s ease;
    }
    
    .warning-message:hover {
        box-shadow: 0 12px 36px rgba(250, 112, 154, 0.45),
                    0 0 30px rgba(250, 112, 154, 0.35);
        transform: translateY(-2px);
    }
    
    /* Error message with glow */
    .error-message {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 18px;
        border-radius: 12px;
        color: white;
        font-weight: 700;
        font-size: 1.08em;
        box-shadow: 0 8px 28px rgba(235, 51, 73, 0.35),
                    0 0 20px rgba(235, 51, 73, 0.25);
        border-left: 6px solid #eb3349;
        transition: all 0.3s ease;
    }
    
    .error-message:hover {
        box-shadow: 0 12px 36px rgba(235, 51, 73, 0.45),
                    0 0 30px rgba(235, 51, 73, 0.35);
        transform: translateY(-2px);
    }
    
    /* Title styling */
    h1 {
        color: #2d3748;
        font-weight: 800;
        font-size: 2.5em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 2px 10px rgba(102, 126, 234, 0.1);
    }
    
    h2 {
        color: #2d3748;
        font-weight: 700;
        border-bottom: 3px solid #667eea;
        padding-bottom: 12px;
        position: relative;
    }
    
    h2::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, transparent);
        border-radius: 2px;
        animation: slideIn 0.6s ease-out;
    }
    
    @keyframes slideIn {
        from { width: 0; }
        to { width: 100%; }
    }
    
    h3 {
        color: #2d3748;
        font-weight: 700;
    }
    
    h4 {
        color: #667eea;
        font-weight: 700;
    }
    
    /* Table styling with glow */
    table {
        border-collapse: collapse;
        width: 100%;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.08),
                    0 0 15px rgba(102, 126, 234, 0.1);
        background: white;
    }
    
    table th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px;
        font-weight: 700;
        text-align: left;
        font-size: 1.05em;
        letter-spacing: 0.3px;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
    }
    
    table td {
        padding: 14px 16px;
        border-bottom: 1px solid #e8e8e8;
        color: #2d3748;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    table tr:hover td {
        background: linear-gradient(90deg, #f8f9ff 0%, white 100%);
        box-shadow: inset 0 1px 0 rgba(102, 126, 234, 0.1);
    }
    
    table tr:last-child td {
        border-bottom: none;
    }
    
    /* Links with glow */
    a {
        color: #667eea !important;
        font-weight: 700 !important;
        text-decoration: none;
        transition: all 0.3s ease;
        position: relative;
    }
    
    a::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 0;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.3s ease;
    }
    
    a:hover {
        color: #764ba2 !important;
        text-decoration: none;
        text-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
    }
    
    a:hover::after {
        width: 100%;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        padding: 20px;
        background: linear-gradient(180deg, #f5f7fa 0%, #e9ecef 100%);
    }
    
    [data-testid="stSidebar"] h2 {
        border-bottom-color: #667eea;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 20px 0;
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
        <p style="margin: 8px 0; font-size: 0.85em; opacity: 0.95; font-weight: 600;">Age</p>
        <p style="margin: 8px 0; font-size: 2.2em; font-weight: 900;">{age}</p>
        <p style="margin: 8px 0; font-size: 0.75em; opacity: 0.85; font-weight: 600;">years</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 14px;">
        <p style="margin: 8px 0; font-size: 0.85em; opacity: 0.95; font-weight: 600;">Systolic BP</p>
        <p style="margin: 8px 0; font-size: 2.2em; font-weight: 900;">{sys_bp}</p>
        <p style="margin: 8px 0; font-size: 0.75em; opacity: 0.85; font-weight: 600;">mmHg</p>
    </div>
    """, unsafe_allow_html=True)

with summary_col2:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 8px 0; font-size: 0.85em; opacity: 0.95; font-weight: 600;">Diastolic BP</p>
        <p style="margin: 8px 0; font-size: 2.2em; font-weight: 900;">{dia_bp}</p>
        <p style="margin: 8px 0; font-size: 0.75em; opacity: 0.85; font-weight: 600;">mmHg</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 14px;">
        <p style="margin: 8px 0; font-size: 0.85em; opacity: 0.95; font-weight: 600;">Blood Sugar</p>
        <p style="margin: 8px 0; font-size: 2.2em; font-weight: 900;">{bs:.1f}</p>
        <p style="margin: 8px 0; font-size: 0.75em; opacity: 0.85; font-weight: 600;">mg/dL</p>
    </div>
    """, unsafe_allow_html=True)

with summary_col3:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 8px 0; font-size: 0.85em; opacity: 0.95; font-weight: 600;">Temperature</p>
        <p style="margin: 8px 0; font-size: 2.2em; font-weight: 900;">{temp:.1f}</p>
        <p style="margin: 8px 0; font-size: 0.75em; opacity: 0.85; font-weight: 600;">°F</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 14px;">
        <p style="margin: 8px 0; font-size: 0.85em; opacity: 0.95; font-weight: 600;">Heart Rate</p>
        <p style="margin: 8px 0; font-size: 2.2em; font-weight: 900;">{heart_rate}</p>
        <p style="margin: 8px 0; font-size: 0.75em; opacity: 0.85; font-weight: 600;">bpm</p>
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
                        padding: 24px; border-radius: 16px; text-align: center; color: white;
                        box-shadow: 0 10px 32px rgba(0,0,0,0.2),
                                    0 0 30px {color_start}40;
                        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        cursor: pointer;"
                 onmouseover="this.style.transform='translateY(-6px) scale(1.04)'; this.style.boxShadow='0 14px 40px rgba(0,0,0,0.25), 0 0 40px {color_start}60'"
                 onmouseout="this.style.transform='translateY(0) scale(1)'; this.style.boxShadow='0 10px 32px rgba(0,0,0,0.2), 0 0 30px {color_start}40'">
                <p style="margin: 0; font-size: 2.5em; font-weight: 900;">{icon}</p>
                <p style="margin: 14px 0 8px 0; font-size: 1em; opacity: 0.98; font-weight: 700; letter-spacing: 1px;">{cls.title()}</p>
                <p style="margin: 0; font-size: 2.4em; font-weight: 900;">{prob*100:.1f}%</p>
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
        <div class="error-message" style="background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%); color: #eb3349; border-left: 6px solid #eb3349;">
            <b>Urgent:</b> Refer to specialist, monitor BP/BS continuously, check for preeclampsia.
        </div>
        """, unsafe_allow_html=True)
    elif risk_label == "mid risk":
        st.markdown("""
        <div class="warning-message" style="background: linear-gradient(135deg, #fffaf0 0%, #fff9e6 100%); color: #5f3a1a; border-left: 6px solid #fa709a;">
            <b>Enhanced Monitoring:</b> Schedule follow-up in 1-2 weeks, daily home BP monitoring, dietary guidance.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-message" style="background: linear-gradient(135deg, #f0fdf4 0%, #e6f9f0 100%); color: #1a5f3f; border-left: 6px solid #84fab0;">
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
<div style="text-align: center; padding: 20px; margin-top: 20px;">
    <small style="color: #667eea; font-weight: 700; letter-spacing: 0.5px;">🏥 MATERNAL HEALTH RISK PREDICTOR — Gradient Boosting + SMOTE — Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)
