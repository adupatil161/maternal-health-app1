# ─────────────────────────────────────────────────────────────────────────────
# maternal_UI_FINAL_CLEAN.py — Maternal Health Risk Predictor
# Run with: streamlit run maternal_UI_FINAL_CLEAN.py
# ─────────────────────────────────────────────────────────────────────────────

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS - BACKGROUND COLOR & SMALL TEXT BOXES
st.markdown("""
    <style>
    /* MAIN PAGE BACKGROUND - LIGHT GRAY */
    [data-testid="stAppViewContainer"] {
        background-color: #e8eaef !important;
    }
    
    .main {
        background-color: #e8eaef !important;
    }
    
    /* SIDEBAR BACKGROUND - PURPLE */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: white !important;
    }
    
    /* BUTTON STYLING */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 0.85em;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* NUMBER INPUT - SMALLER */
    .stNumberInput {
        margin-bottom: 8px;
    }
    
    .stNumberInput > label {
        font-size: 0.85em !important;
        margin-bottom: 4px !important;
    }
    
    .stNumberInput > div > div > input {
        border: 2px solid #667eea !important;
        border-radius: 6px !important;
        padding: 6px 10px !important;
        font-weight: 500 !important;
        background: white !important;
        font-size: 0.85em !important;
        height: 32px !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #764ba2 !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* EXPANDER STYLING */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 8px 12px !important;
        font-size: 0.9em !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    
    /* METRIC CARD - VERY SMALL */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* SUCCESS MESSAGE */
    .success-message {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 12px;
        border-radius: 8px;
        color: #1a5f3f;
        font-weight: 600;
        font-size: 0.9em;
        box-shadow: 0 2px 8px rgba(132, 250, 176, 0.2);
        border-left: 4px solid #84fab0;
    }
    
    /* WARNING MESSAGE */
    .warning-message {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 12px;
        border-radius: 8px;
        color: #5f3a1a;
        font-weight: 600;
        font-size: 0.9em;
        box-shadow: 0 2px 8px rgba(250, 112, 154, 0.2);
        border-left: 4px solid #fa709a;
    }
    
    /* ERROR MESSAGE */
    .error-message {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 12px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        font-size: 0.9em;
        box-shadow: 0 2px 8px rgba(235, 51, 73, 0.2);
        border-left: 4px solid #eb3349;
    }
    
    /* TITLES */
    h1 {
        color: #2d3748;
        font-weight: 700;
        font-size: 2em;
        margin-bottom: 8px;
    }
    
    h2 {
        color: #2d3748;
        font-weight: 600;
        border-bottom: 2px solid #667eea;
        padding-bottom: 6px;
        font-size: 1.2em;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    
    h3 {
        color: #2d3748;
        font-weight: 600;
        font-size: 1em;
        margin-top: 10px;
        margin-bottom: 8px;
    }
    
    h4 {
        color: #667eea;
        font-weight: 600;
        font-size: 0.95em;
    }
    
    /* TABLE STYLING */
    table {
        border-collapse: collapse;
        width: 100%;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        background: white;
        font-size: 0.85em;
    }
    
    table th {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px;
        font-weight: 600;
        text-align: left;
        font-size: 0.85em;
    }
    
    table td {
        padding: 8px 10px;
        border-bottom: 1px solid #e8e8e8;
        color: #2d3748;
        font-weight: 500;
        font-size: 0.85em;
    }
    
    table tr:hover td {
        background-color: #f8f9ff;
    }
    
    table tr:last-child td {
        border-bottom: none;
    }
    
    /* LINKS */
    a {
        color: white !important;
        font-weight: 600 !important;
        text-decoration: none;
    }
    
    a:hover {
        color: #e8eaef !important;
        text-decoration: underline;
    }
    
    /* RECOMMENDATION CARD */
    .recommendation-card {
        background: white;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin: 8px 0;
        font-size: 0.9em;
    }
    
    /* PARAGRAPH STYLING */
    p {
        font-size: 0.9em;
        margin-bottom: 8px;
    }
    
    /* MARKDOWN STYLING */
    .stMarkdown {
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model   = joblib.load("ml_capstone_deploy/trained_models/maternal_risk_model.pkl")
    encoder = joblib.load("ml_capstone_deploy/trained_models/label_encoder.pkl")
    return model, encoder

model, encoder = load_model()

# SIDEBAR
with st.sidebar:
    st.markdown("## About This App")
    st.markdown("""
    **Maternal Health Risk Predictor**
    
    AI model trained on maternal health data to predict risk levels for pregnant patients.
    
    **Model Info:**
    - Algorithm: Gradient Boosting
    - Accuracy: 82.16%
    - Features: 6 vital signs
    - Classes: 3 risk levels
    
    2026 | Version: 1.0
    """)
    
    st.markdown("---")
    st.markdown("## Resources")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[GitHub](https://github.com)")
    with col2:
        st.markdown("[Docs](https://github.com)")

# HEADER
st.title("Maternal Health Risk Predictor")
st.markdown("Enter patient vital signs and click **Predict Risk Level**")

with st.expander("How This Works"):
    st.markdown("""
    ML model predicts pregnancy risk levels.
    
    **Evaluates:** Age | Systolic BP | Diastolic BP | Blood Sugar | Body Temperature | Heart Rate
    
    **Levels:** Green-Low | Yellow-Mid | Red-High
    """)

st.markdown("---")

st.subheader("Patient Information")
col1, col2, col3 = st.columns(3)

with col1:
    age    = st.number_input("Age (years)", min_value=10, max_value=70, value=30)
    sys_bp = st.number_input("Systolic BP", min_value=50, max_value=200, value=120)

with col2:
    dia_bp = st.number_input("Diastolic BP", min_value=30, max_value=150, value=80)
    bs     = st.number_input("Blood Sugar", min_value=1.0, max_value=30.0, value=7.0, step=0.1)

with col3:
    temp       = st.number_input("Body Temp (F)", min_value=95.0, max_value=106.0, value=98.6, step=0.1)
    heart_rate = st.number_input("Heart Rate", min_value=40, max_value=150, value=76)

# INPUT SUMMARY - VERY SMALL
st.markdown("### Input Summary")
summary_col1, summary_col2, summary_col3 = st.columns(3)
with summary_col1:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 2px 0; font-size: 0.7em;">Age</p>
        <p style="margin: 2px 0; font-size: 1.4em; font-weight: bold;">{age}</p>
        <p style="margin: 2px 0; font-size: 0.65em;">yrs</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 6px;">
        <p style="margin: 2px 0; font-size: 0.7em;">Systolic</p>
        <p style="margin: 2px 0; font-size: 1.4em; font-weight: bold;">{sys_bp}</p>
        <p style="margin: 2px 0; font-size: 0.65em;">mmHg</p>
    </div>
    """, unsafe_allow_html=True)

with summary_col2:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 2px 0; font-size: 0.7em;">Diastolic</p>
        <p style="margin: 2px 0; font-size: 1.4em; font-weight: bold;">{dia_bp}</p>
        <p style="margin: 2px 0; font-size: 0.65em;">mmHg</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 6px;">
        <p style="margin: 2px 0; font-size: 0.7em;">Sugar</p>
        <p style="margin: 2px 0; font-size: 1.4em; font-weight: bold;">{bs:.1f}</p>
        <p style="margin: 2px 0; font-size: 0.65em;">mg/dL</p>
    </div>
    """, unsafe_allow_html=True)

with summary_col3:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 2px 0; font-size: 0.7em;">Temp</p>
        <p style="margin: 2px 0; font-size: 1.4em; font-weight: bold;">{temp:.1f}</p>
        <p style="margin: 2px 0; font-size: 0.65em;">F</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 6px;">
        <p style="margin: 2px 0; font-size: 0.7em;">Heart Rate</p>
        <p style="margin: 2px 0; font-size: 1.4em; font-weight: bold;">{heart_rate}</p>
        <p style="margin: 2px 0; font-size: 0.65em;">bpm</p>
    </div>
    """, unsafe_allow_html=True)

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
                color_start = "#eb3349"
                color_end = "#f45c43"
            elif cls.lower() == "mid risk":
                color_start = "#fa709a"
                color_end = "#fee140"
            else:
                color_start = "#84fab0"
                color_end = "#8fd3f4"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color_start} 0%, {color_end} 100%);
                        padding: 14px; border-radius: 8px; text-align: center; color: white;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        transition: all 0.3s ease;">
                <p style="margin: 2px 0; font-size: 1.4em; font-weight: bold;">{cls.title()}</p>
                <p style="margin: 4px 0; font-size: 1.6em; font-weight: bold;">{prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    if risk_label == "high risk":
        st.markdown('<div class="error-message">HIGH RISK - Immediate medical attention recommended</div>', unsafe_allow_html=True)
    elif risk_label == "mid risk":
        st.markdown('<div class="warning-message">MID RISK - Increased monitoring and care advised</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-message">LOW RISK - Patient vitals appear within safe range</div>', unsafe_allow_html=True)

    st.markdown("### Clinical Recommendation")
    if risk_label == "high risk":
        st.markdown('<div class="recommendation-card"><b>Urgent:</b> Refer to specialist, monitor BP/BS continuously</div>', unsafe_allow_html=True)
    elif risk_label == "mid risk":
        st.markdown('<div class="recommendation-card"><b>Monitor:</b> Follow-up in 1-2 weeks, daily home BP monitoring</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="recommendation-card"><b>Routine:</b> Continue prenatal check-ups and healthy lifestyle</div>', unsafe_allow_html=True)

    st.markdown("### Patient Summary")
    st.markdown(f"""
| Vital | Value | Normal |
|---|---|---|
| Age | {age} yrs | 18-40 |
| Systolic | {sys_bp} | 90-120 |
| Diastolic | {dia_bp} | 60-80 |
| Blood Sugar | {bs:.1f} | 70-100 |
| Temp | {temp:.1f}F | 97-99 |
| Heart Rate | {heart_rate} | 60-100 |
    """)

st.markdown("---")
st.markdown('<div style="text-align: center; padding: 8px; font-size: 0.8em;"><small>Maternal Health Risk Predictor - Gradient Boosting</small></div>', unsafe_allow_html=True)
