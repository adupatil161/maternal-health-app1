# ─────────────────────────────────────────────────────────────────────────────
# maternal_UI_BEAUTIFUL_FINAL.py — Maternal Health Risk Predictor
# Run with: streamlit run maternal_UI_BEAUTIFUL_FINAL.py
# ─────────────────────────────────────────────────────────────────────────────

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# AGGRESSIVE CSS OVERRIDES FOR BACKGROUND & SMALL BOXES
st.markdown("""
    <style>
    * { margin: 0; padding: 0; }
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"], .appViewContainer {
        background-color: #d5d7e0 !important;
        background: #d5d7e0 !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #d5d7e0 !important;
    }
    
    [data-testid="stMainBlockContainer"] {
        background-color: #d5d7e0 !important;
    }
    
    .main {
        background-color: #d5d7e0 !important;
    }
    
    section[data-testid="stAppViewContainer"] {
        background-color: #d5d7e0 !important;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #5b6db8 0%, #6b3c8f 100%) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] a {
        color: #e0e0ff !important;
        font-weight: bold !important;
    }
    
    [data-testid="stSidebar"] a:hover {
        color: white !important;
    }
    
    /* TEXT & LABELS - SMALLER */
    label {
        font-size: 0.75em !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
    }
    
    /* NUMBER INPUT BOXES - VERY SMALL */
    .stNumberInput {
        margin-bottom: 6px !important;
    }
    
    .stNumberInput > div {
        margin-bottom: 4px !important;
    }
    
    .stNumberInput input {
        height: 28px !important;
        padding: 4px 8px !important;
        font-size: 0.75em !important;
        border: 2px solid #5b6db8 !important;
        border-radius: 6px !important;
    }
    
    .stNumberInput input:focus {
        border-color: #6b3c8f !important;
        box-shadow: 0 0 0 2px rgba(91, 109, 184, 0.2) !important;
    }
    
    /* TITLES */
    h1 {
        color: #2d3748 !important;
        font-size: 1.8em !important;
        font-weight: 800 !important;
        margin-bottom: 6px !important;
    }
    
    h2 {
        color: #2d3748 !important;
        font-size: 1.1em !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #5b6db8 !important;
        padding-bottom: 4px !important;
        margin: 8px 0 6px 0 !important;
    }
    
    h3 {
        color: #2d3748 !important;
        font-size: 0.95em !important;
        font-weight: 700 !important;
        margin: 8px 0 6px 0 !important;
    }
    
    /* PARAGRAPHS */
    p {
        font-size: 0.85em !important;
        color: #2d3748 !important;
        margin-bottom: 4px !important;
    }
    
    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(90deg, #5b6db8 0%, #6b3c8f 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 0.85em !important;
        padding: 8px 20px !important;
        border-radius: 6px !important;
        height: 36px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(91, 109, 184, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(91, 109, 184, 0.4) !important;
    }
    
    /* EXPANDER */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #5b6db8 0%, #6b3c8f 100%) !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        padding: 8px 12px !important;
        font-size: 0.9em !important;
    }
    
    /* METRIC CARDS - VERY SMALL */
    .metric-card {
        background: linear-gradient(135deg, #5b6db8 0%, #6b3c8f 100%) !important;
        color: white !important;
        padding: 10px !important;
        border-radius: 8px !important;
        text-align: center !important;
        box-shadow: 0 3px 10px rgba(91, 109, 184, 0.25) !important;
        margin: 4px 0 !important;
    }
    
    .metric-card p {
        color: white !important;
        margin: 1px 0 !important;
    }
    
    .metric-card p:nth-child(1) {
        font-size: 0.65em !important;
        opacity: 0.95 !important;
    }
    
    .metric-card p:nth-child(2) {
        font-size: 1.3em !important;
        font-weight: 900 !important;
        margin: 3px 0 !important;
    }
    
    .metric-card p:nth-child(3) {
        font-size: 0.6em !important;
        opacity: 0.85 !important;
    }
    
    /* MESSAGE BOXES */
    .success-message {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%) !important;
        padding: 10px !important;
        border-radius: 6px !important;
        color: #1a5f3f !important;
        font-weight: 700 !important;
        font-size: 0.85em !important;
        box-shadow: 0 3px 10px rgba(132, 250, 176, 0.3) !important;
        border-left: 4px solid #84fab0 !important;
    }
    
    .warning-message {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%) !important;
        padding: 10px !important;
        border-radius: 6px !important;
        color: #5f3a1a !important;
        font-weight: 700 !important;
        font-size: 0.85em !important;
        box-shadow: 0 3px 10px rgba(250, 112, 154, 0.3) !important;
        border-left: 4px solid #fa709a !important;
    }
    
    .error-message {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%) !important;
        padding: 10px !important;
        border-radius: 6px !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 0.85em !important;
        box-shadow: 0 3px 10px rgba(235, 51, 73, 0.3) !important;
        border-left: 4px solid #eb3349 !important;
    }
    
    /* RECOMMENDATION CARD */
    .recommendation-card {
        background: white !important;
        padding: 10px !important;
        border-radius: 6px !important;
        border-left: 4px solid #5b6db8 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
        margin: 6px 0 !important;
        font-size: 0.85em !important;
        color: #2d3748 !important;
    }
    
    /* TABLE */
    table {
        font-size: 0.8em !important;
        border-collapse: collapse !important;
        width: 100% !important;
        border-radius: 6px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        background: white !important;
    }
    
    table th {
        background: linear-gradient(90deg, #5b6db8 0%, #6b3c8f 100%) !important;
        color: white !important;
        padding: 8px !important;
        font-weight: 700 !important;
        text-align: left !important;
    }
    
    table td {
        padding: 6px 8px !important;
        border-bottom: 1px solid #e0e0e0 !important;
        color: #2d3748 !important;
        font-weight: 500 !important;
    }
    
    table tr:hover td {
        background: #f8f9ff !important;
    }
    
    /* SUBHEADER */
    .stSubheader {
        font-size: 0.95em !important;
        font-weight: 700 !important;
    }
    
    /* MARKDOWN */
    .stMarkdown {
        font-size: 0.85em !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load("ml_capstone_deploy/trained_models/maternal_risk_model.pkl")
    encoder = joblib.load("ml_capstone_deploy/trained_models/label_encoder.pkl")
    return model, encoder

model, encoder = load_model()

# SIDEBAR
with st.sidebar:
    st.markdown("## About This App")
    st.markdown("""
    **Maternal Health Risk Predictor**
    
    AI-powered prediction system for maternal health risk assessment.
    
    **Model Details:**
    - Algorithm: Gradient Boosting
    - Accuracy: 82.16%
    - Features: 6 vital signs
    - Risk Levels: 3 categories
    
    Version 1.0 | 2026
    """)
    
    st.markdown("---")
    st.markdown("## Resources")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[GitHub](https://github.com)")
    with col2:
        st.markdown("[Docs](https://github.com)")

# MAIN CONTENT
st.title("Maternal Health Risk Predictor")
st.markdown("Enter vital signs and click **Predict Risk Level**")

with st.expander("How This Works"):
    st.markdown("ML model predicts pregnancy risk. **Evaluates:** Age | BP | Blood Sugar | Temperature | Heart Rate. **Risk Levels:** Low | Mid | High")

st.markdown("---")

st.subheader("Patient Information")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=10, max_value=70, value=30)
    sys_bp = st.number_input("Systolic", min_value=50, max_value=200, value=120)

with col2:
    dia_bp = st.number_input("Diastolic", min_value=30, max_value=150, value=80)
    bs = st.number_input("Blood Sugar", min_value=1.0, max_value=30.0, value=7.0, step=0.1)

with col3:
    temp = st.number_input("Temp (F)", min_value=95.0, max_value=106.0, value=98.6, step=0.1)
    heart_rate = st.number_input("Heart Rate", min_value=40, max_value=150, value=76)

st.markdown("### Input Summary")
sc1, sc2, sc3 = st.columns(3)

with sc1:
    st.markdown(f'<div class="metric-card"><p>Age</p><p>{age}</p><p>yrs</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card"><p>Systolic</p><p>{sys_bp}</p><p>mmHg</p></div>', unsafe_allow_html=True)

with sc2:
    st.markdown(f'<div class="metric-card"><p>Diastolic</p><p>{dia_bp}</p><p>mmHg</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card"><p>Sugar</p><p>{bs:.1f}</p><p>mg/dL</p></div>', unsafe_allow_html=True)

with sc3:
    st.markdown(f'<div class="metric-card"><p>Temp</p><p>{temp:.1f}</p><p>F</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card"><p>Heart Rate</p><p>{heart_rate}</p><p>bpm</p></div>', unsafe_allow_html=True)

st.markdown("---")
predict_clicked = st.button("Predict Risk Level", use_container_width=True)

if predict_clicked:
    data = pd.DataFrame({
        "Age": [age], "SystolicBP": [sys_bp], "DiastolicBP": [dia_bp],
        "BS": [bs], "BodyTemp": [temp], "HeartRate": [heart_rate]
    })

    prediction_num = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]
    risk_label = encoder.classes_[prediction_num]

    st.subheader("Prediction Results")

    m1, m2, m3 = st.columns(3)
    for col_widget, cls, prob in zip([m1, m2, m3], encoder.classes_, probabilities):
        with col_widget:
            if cls.lower() == "high risk":
                colors = "#eb3349", "#f45c43"
            elif cls.lower() == "mid risk":
                colors = "#fa709a", "#fee140"
            else:
                colors = "#84fab0", "#8fd3f4"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {colors[0]} 0%, {colors[1]} 100%);
                        padding: 12px; border-radius: 8px; text-align: center; color: white;
                        box-shadow: 0 3px 10px rgba(0,0,0,0.15);">
                <p style="margin: 2px 0; font-size: 0.9em; font-weight: bold;">{cls.title()}</p>
                <p style="margin: 4px 0; font-size: 1.5em; font-weight: bold;">{prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    if risk_label == "high risk":
        st.markdown('<div class="error-message">HIGH RISK - Immediate medical attention required</div>', unsafe_allow_html=True)
    elif risk_label == "mid risk":
        st.markdown('<div class="warning-message">MID RISK - Increased monitoring advised</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-message">LOW RISK - Vitals within safe range</div>', unsafe_allow_html=True)

    st.markdown("### Clinical Recommendation")
    if risk_label == "high risk":
        st.markdown('<div class="recommendation-card"><b>Urgent:</b> Refer to specialist, continuous monitoring</div>', unsafe_allow_html=True)
    elif risk_label == "mid risk":
        st.markdown('<div class="recommendation-card"><b>Monitor:</b> Follow-up in 1-2 weeks, home BP monitoring</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="recommendation-card"><b>Routine:</b> Continue prenatal check-ups</div>', unsafe_allow_html=True)

    st.markdown("### Patient Summary")
    st.markdown(f"""
| Vital | Value | Normal |
|---|---|---|
| Age | {age} yrs | 18-40 |
| Systolic | {sys_bp} | 90-120 |
| Diastolic | {dia_bp} | 60-80 |
| Sugar | {bs:.1f} | 70-100 |
| Temp | {temp:.1f}F | 97-99 |
| HR | {heart_rate} | 60-100 |
    """)

st.markdown("---")
st.markdown('<div style="text-align: center; padding: 6px; font-size: 0.75em;"><small>Maternal Health Risk Predictor - Gradient Boosting</small></div>', unsafe_allow_html=True)
