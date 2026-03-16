# ─────────────────────────────────────────────────────────────────────────────
# maternal_UI_BEAUTIFUL.py — Maternal Health Risk Predictor (Beautiful UI)
# Run with: streamlit run maternal_UI_BEAUTIFUL.py
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

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .header-title {
        text-align: center;
        font-size: 3em;
        font-weight: 700;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    
    .header-subtitle {
        text-align: center;
        font-size: 1.1em;
        color: #e0e0e0;
        margin-bottom: 30px;
    }
    
    /* Card styling */
    .info-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
        border-left: 5px solid #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 12px 30px;
        border-radius: 25px;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        font-size: 1.1em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 10px 15px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    /* Success/Warning/Error boxes */
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 20px;
        border-radius: 15px;
        color: #1a5f3f;
        font-weight: bold;
        font-size: 1.1em;
        box-shadow: 0 8px 32px rgba(132, 250, 176, 0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 20px;
        border-radius: 15px;
        color: #5f3a1a;
        font-weight: bold;
        font-size: 1.1em;
        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
    }
    
    .error-box {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        font-size: 1.1em;
        box-shadow: 0 8px 32px rgba(235, 51, 73, 0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    
    /* Table styling */
    table {
        border-collapse: collapse;
        width: 100%;
    }
    
    table th {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px;
        border-radius: 8px 8px 0 0;
    }
    
    table td {
        padding: 10px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Sidebar text */
    .sidebar-text {
        color: white;
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
# ENHANCED SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
        <h2 style="color: #667eea; text-align: center;">📋 About This App</h2>
        <p style="color: #333; font-size: 0.95em;">
            <b>Maternal Health Risk Predictor</b><br><br>
            An AI-powered application that predicts pregnancy risk levels using advanced machine learning.
        </p>
        <hr style="border: 1px solid #ddd;">
        <h3 style="color: #667eea; font-size: 0.95em;">📊 Model Information:</h3>
        <ul style="color: #333; font-size: 0.9em; line-height: 1.8;">
            <li>🤖 Algorithm: Gradient Boosting</li>
            <li>📈 Accuracy: 82.16%</li>
            <li>🔢 Features: 6 vital signs</li>
            <li>📋 Categories: 3 risk levels</li>
            <li>✅ Status: Production Ready</li>
        </ul>
        <hr style="border: 1px solid #ddd;">
        <p style="color: #667eea; font-size: 0.85em; text-align: center;">
            <b>v1.0 | 2026</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 15px;">
        <h3 style="color: #667eea; text-align: center;">🔗 Quick Links</h3>
        <p style="text-align: center; color: #333; font-size: 0.9em;">
            <a href="https://github.com/adupatil161/maternal-health-app" target="_blank" style="color: #667eea; text-decoration: none; font-weight: bold;">
                🐙 GitHub Repository
            </a>
            <br><br>
            <a href="https://github.com/adupatil161/maternal-health-app" target="_blank" style="color: #667eea; text-decoration: none; font-weight: bold;">
                📚 Documentation
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BEAUTIFUL HEADER
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="header-title">🏥 Maternal Health Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">AI-Powered Clinical Decision Support System</div>', unsafe_allow_html=True)

# HOW IT WORKS SECTION
with st.expander("ℹ️ **How This Works** - Click to expand", expanded=False):
    st.markdown("""
    <div class="info-card">
        <h3 style="color: #667eea;">🔍 What This Tool Does:</h3>
        <p>This application uses machine learning to analyze maternal vital signs and predict pregnancy risk levels in real-time.</p>
        
        <h3 style="color: #667eea;">📊 Analysis Parameters:</h3>
        <ul>
            <li><b>Age</b> - Patient's age in years</li>
            <li><b>Systolic BP</b> - Upper blood pressure reading</li>
            <li><b>Diastolic BP</b> - Lower blood pressure reading</li>
            <li><b>Blood Sugar</b> - Glucose level measurement</li>
            <li><b>Body Temperature</b> - Core body temperature</li>
            <li><b>Heart Rate</b> - Pulse measurement</li>
        </ul>
        
        <h3 style="color: #667eea;">⚠️ Risk Classifications:</h3>
        <div style="background: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 10px;">
            <p><span style="font-size: 1.5em;">🟢</span> <b>Low Risk</b> - Healthy pregnancy with routine monitoring recommended</p>
            <p><span style="font-size: 1.5em;">🟡</span> <b>Mid Risk</b> - Some concerns present, increased monitoring advised</p>
            <p><span style="font-size: 1.5em;">🔴</span> <b>High Risk</b> - Significant concerns, immediate medical attention required</p>
        </div>
        
        <hr style="margin-top: 20px;">
        <p style="font-size: 0.9em; color: #666; font-style: italic;">
            <b>⚕️ Medical Disclaimer:</b> This tool is for educational purposes only. Always consult with qualified healthcare professionals for medical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# INPUT SECTION WITH BEAUTIFUL LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; margin: 30px 0;">
    <h2 style="color: white; font-size: 2em;">👩‍⚕️ Enter Patient Vital Signs</h2>
    <p style="color: #e0e0e0; font-size: 1.1em;">Fill in the fields below to get a risk assessment</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<p style="color: white; font-weight: bold; font-size: 1.1em;">👤 Age & Vitals</p>', unsafe_allow_html=True)
    age    = st.number_input("Age (years)", min_value=10, max_value=70, value=30, 
                             help="Patient's age in years")
    sys_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=200, value=120,
                             help="Upper blood pressure reading")

with col2:
    st.markdown('<p style="color: white; font-weight: bold; font-size: 1.1em;">❤️ Blood & Pressure</p>', unsafe_allow_html=True)
    dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80,
                             help="Lower blood pressure reading")
    bs     = st.number_input("Blood Sugar (mg/dL)", min_value=1.0, max_value=30.0, value=7.0, step=0.1,
                             help="Blood glucose level")

with col3:
    st.markdown('<p style="color: white; font-weight: bold; font-size: 1.1em;">🌡️ Temperature & Heart</p>', unsafe_allow_html=True)
    temp       = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=106.0, value=98.6, step=0.1,
                                help="Core body temperature")
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=76,
                                help="Pulse rate in beats per minute")

# INPUT SUMMARY
st.markdown("---")
st.markdown("""
<div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; border: 2px solid rgba(255,255,255,0.3); margin: 20px 0;">
    <h3 style="color: white; text-align: center;">📋 Current Input Summary</h3>
</div>
""", unsafe_allow_html=True)

summary_col1, summary_col2, summary_col3 = st.columns(3)
with summary_col1:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Age</p>
        <p style="margin: 5px 0; font-size: 1.8em; font-weight: bold;">{age}</p>
        <p style="margin: 0; font-size: 0.85em; opacity: 0.8;">years</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 10px;">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Systolic BP</p>
        <p style="margin: 5px 0; font-size: 1.8em; font-weight: bold;">{sys_bp}</p>
        <p style="margin: 0; font-size: 0.85em; opacity: 0.8;">mmHg</p>
    </div>
    """, unsafe_allow_html=True)

with summary_col2:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Diastolic BP</p>
        <p style="margin: 5px 0; font-size: 1.8em; font-weight: bold;">{dia_bp}</p>
        <p style="margin: 0; font-size: 0.85em; opacity: 0.8;">mmHg</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 10px;">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Blood Sugar</p>
        <p style="margin: 5px 0; font-size: 1.8em; font-weight: bold;">{bs:.1f}</p>
        <p style="margin: 0; font-size: 0.85em; opacity: 0.8;">mg/dL</p>
    </div>
    """, unsafe_allow_html=True)

with summary_col3:
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Temperature</p>
        <p style="margin: 5px 0; font-size: 1.8em; font-weight: bold;">{temp:.1f}</p>
        <p style="margin: 0; font-size: 0.85em; opacity: 0.8;">°F</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 10px;">
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">Heart Rate</p>
        <p style="margin: 5px 0; font-size: 1.8em; font-weight: bold;">{heart_rate}</p>
        <p style="margin: 0; font-size: 0.85em; opacity: 0.8;">bpm</p>
    </div>
    """, unsafe_allow_html=True)

# PREDICT BUTTON
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_clicked = st.button("🔍 Predict Risk Level", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION RESULTS
# ─────────────────────────────────────────────────────────────────────────────
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

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin: 30px 0;">
        <h2 style="color: white; font-size: 2.2em;">📊 Risk Assessment Results</h2>
    </div>
    """, unsafe_allow_html=True)

    # Probability metrics
    m1, m2, m3 = st.columns(3, gap="medium")
    
    for col_widget, cls, prob in zip([m1, m2, m3], encoder.classes_, probabilities):
        with col_widget:
            if cls.lower() == "high risk":
                color = "#eb3349"
                icon = "🔴"
            elif cls.lower() == "mid risk":
                color = "#fa709a"
                icon = "🟡"
            else:
                color = "#84fab0"
                icon = "🟢"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                        padding: 25px; border-radius: 15px; color: white; text-align: center;
                        box-shadow: 0 8px 32px rgba(0,0,0,0.2);">
                <p style="font-size: 2em; margin: 0;">{icon}</p>
                <p style="margin: 10px 0 5px 0; font-size: 1em; opacity: 0.9;">{cls.title()}</p>
                <p style="margin: 0; font-size: 2.2em; font-weight: bold;">{prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Risk assessment box
    if risk_label == "high risk":
        st.markdown("""
        <div class="error-box">
            🔴 HIGH RISK - IMMEDIATE MEDICAL ATTENTION REQUIRED
        </div>
        """, unsafe_allow_html=True)
    elif risk_label == "mid risk":
        st.markdown("""
        <div class="warning-box">
            🟡 MID RISK - INCREASED MONITORING ADVISED
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
            🟢 LOW RISK - PATIENT APPEARS HEALTHY
        </div>
        """, unsafe_allow_html=True)

    # Clinical recommendations
    st.markdown("""
    <div style="text-align: center; margin: 30px 0;">
        <h3 style="color: white; font-size: 1.8em;">⚕️ Clinical Recommendations</h3>
    </div>
    """, unsafe_allow_html=True)

    if risk_label == "high risk":
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #eb3349;">🔴 Urgent Action Required:</h4>
            <ul style="font-size: 1.05em; line-height: 2;">
                <li>Refer to specialist immediately</li>
                <li>Continuous BP and blood sugar monitoring</li>
                <li>Screen for preeclampsia</li>
                <li>Consider hospitalization for observation</li>
                <li>Increase follow-up frequency to weekly or more</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif risk_label == "mid risk":
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #fa709a;">🟡 Enhanced Monitoring Needed:</h4>
            <ul style="font-size: 1.05em; line-height: 2;">
                <li>Schedule follow-up appointment in 1-2 weeks</li>
                <li>Daily home blood pressure monitoring</li>
                <li>Dietary modification and guidance</li>
                <li>Moderate exercise if approved by provider</li>
                <li>Bi-weekly clinical check-ups</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #84fab0;">🟢 Routine Care Plan:</h4>
            <ul style="font-size: 1.05em; line-height: 2;">
                <li>Continue regular prenatal check-ups</li>
                <li>Maintain healthy lifestyle habits</li>
                <li>Balanced diet with prenatal vitamins</li>
                <li>Moderate physical activity as tolerated</li>
                <li>Standard follow-up schedule (monthly then bi-weekly)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Patient summary table
    st.markdown("""
    <div style="text-align: center; margin: 30px 0;">
        <h3 style="color: white; font-size: 1.8em;">📋 Detailed Patient Summary</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #667eea;">Vital Signs & Assessment</h4>
            <table>
                <tr>
                    <th>Vital</th>
                    <th>Value</th>
                    <th>Normal</th>
                </tr>
        """ + f"""
                <tr>
                    <td><b>Age</b></td>
                    <td>{age} yrs</td>
                    <td>18-40</td>
                </tr>
                <tr>
                    <td><b>Systolic BP</b></td>
                    <td>{sys_bp} mmHg</td>
                    <td>90-120</td>
                </tr>
                <tr>
                    <td><b>Diastolic BP</b></td>
                    <td>{dia_bp} mmHg</td>
                    <td>60-80</td>
                </tr>
        """ + """
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color: #667eea;">Biochemical Parameters</h4>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                    <th>Normal</th>
                </tr>
                <tr>
                    <td><b>Blood Sugar</b></td>
                    <td>{bs:.1f} mg/dL</td>
                    <td>70-100</td>
                </tr>
                <tr>
                    <td><b>Temperature</b></td>
                    <td>{temp:.1f} °F</td>
                    <td>97-99</td>
                </tr>
                <tr>
                    <td><b>Heart Rate</b></td>
                    <td>{heart_rate} bpm</td>
                    <td>60-100</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 30px;">
    <p style="color: #e0e0e0; font-size: 0.9em;">
        <b>🏥 Maternal Health Risk Predictor</b> | Powered by Gradient Boosting ML<br>
        <small>Built with ❤️ using Streamlit | v1.0 | 2026</small>
    </p>
</div>
""", unsafe_allow_html=True)
