import streamlit as st
import pandas as pd
import altair as alt
import joblib
import os

# Page config
st.set_page_config(page_title="SmartCare Risk Predictor", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, 'disease_risk_model.pkl'))
    encoder_unscaled = joblib.load(os.path.join(MODELS_DIR, 'encoder_unscaled.pkl'))
    target_encoding = joblib.load(os.path.join(MODELS_DIR, 'target_encoding.pkl'))
    raw_input_columns = joblib.load(os.path.join(MODELS_DIR, 'raw_input_columns.pkl'))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    return model, encoder_unscaled, target_encoding, raw_input_columns, feature_columns

model, encoder_unscaled, target_encoding, raw_input_columns, feature_columns = load_artifacts()
inverse_target = {v: k for k, v in target_encoding.items()}

# Numeric vs categorical split, and valid category options — both derived directly
# from the fitted encoder, so there's no separate category_values file to keep in sync
numeric_cols_final = list(encoder_unscaled.transformers_[0][2])
categorical_cols_final = list(encoder_unscaled.transformers_[1][2])
cat_encoder = encoder_unscaled.named_transformers_['cat']
category_values = dict(zip(categorical_cols_final, [list(c) for c in cat_encoder.categories_]))

# Styling
st.markdown("""
    <style>
    .main { background-color: #F4F6FB; }

    .header-banner {
        background: linear-gradient(120deg, #1B4965 0%, #2C7A9E 50%, #0891A5 100%);
        border-radius: 16px;
        padding: 28px 20px 24px 20px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 6px 18px rgba(27,73,101,0.25);
    }
    .title-text {
        color: white;
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 4px;
        text-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .subtitle-text {
        color: #EFF3FF;
        margin-bottom: 0px;
        font-size: 15px;
    }

    .card {
        background-color: var(--bg, white);
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.07);
        margin-bottom: 18px;
        border-top: 5px solid var(--accent, #2C7A9E);
    }
    .card-header {
        font-weight: 700;
        color: #2D3436;
        font-size: 17px;
        margin-bottom: 12px;
    }
    .card-demo     { --accent: #1B4965; --bg: rgba(27,73,101,0.06); }
    .card-vitals   { --accent: #2C7A9E; --bg: rgba(44,122,158,0.06); }
    .card-clinical { --accent: #0891A5; --bg: rgba(8,145,165,0.06); }
    .card-case     { --accent: #27AE60; --bg: rgba(39,174,96,0.06); }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #1B4965, #0891A5);
        color: white;
        border-radius: 10px;
        padding: 11px 24px;
        font-weight: 700;
        font-size: 15px;
        border: none;
        width: 100%;
        transition: 0.25s;
        box-shadow: 0 3px 10px rgba(27,73,101,0.35);
    }
    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(90deg, #163a52, #076f80);
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 5px 14px rgba(27,73,101,0.45);
    }
    div.stButton > button:focus,
    div.stButton > button:active,
    div[data-testid="stFormSubmitButton"] > button:focus,
    div[data-testid="stFormSubmitButton"] > button:active {
        color: white;
        background: linear-gradient(90deg, #1B4965, #0891A5);
        box-shadow: 0 3px 10px rgba(27,73,101,0.35);
    }

    .warning-pill {
        background-color: #FFF4E5;
        border-left: 4px solid #E9A319;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 13px;
        color: #7A4F01;
        margin-top: 6px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #F4F6FB, #EAF0FF);
        border-radius: 10px;
        padding: 12px 8px;
        border: 1px solid #E2E8F5;
    }

    html, body, [class*="css"] {
        font-size: 18px;
    }
    .stMarkdown p, .stMarkdown li {
        font-size: 18px;
    }
    label, .stSelectbox label, .stNumberInput label {
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 16px !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
    }
    .card-header {
        font-size: 20px;
    }

    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header-banner">
        <p class="title-text">SmartCare Disease Risk Predictor</p>
        <p class="subtitle-text">AI-powered clinical decision support prototype - not a substitute for professional diagnosis</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1B4965, #0891A5);
                    padding: 16px; border-radius: 12px; color: white; margin-bottom: 10px;">
            <b>About this tool</b><br><br>
            <span style="font-size:13px;">
            This prototype estimates a patient's disease risk level
            (<b>Low / Medium / High</b>) from vitals and clinical inputs,
            using a model trained on the SmartCare Hospital dataset.
            </span>
        </div>
    """, unsafe_allow_html=True)
    st.divider()
    if st.button("Reset form"):
        st.session_state.clear()
        st.rerun()
    st.divider()
    st.caption("SmartCare Disease Risk Predictor")

tab_predict, tab_about = st.tabs(["Predict", "How it works"])

with tab_predict:
    with st.form("risk_form"):
        st.markdown('<div class="card card-demo"><div class="card-header">Demographics</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 100, 45, help="Patient age in years")
        with c2:
            gender = st.selectbox("Gender", category_values['gender'])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card card-vitals"><div class="card-header">Vitals</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            systolic_bp = st.number_input("Systolic BP", 80, 220, 120, help="Normal: ~90–120 mmHg")
        with c2:
            diastolic_bp = st.number_input("Diastolic BP", 50, 140, 80, help="Normal: ~60–80 mmHg")
        with c3:
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0, help="Normal range: 18.5–24.9")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card card-clinical"><div class="card-header">Clinical Values</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            blood_sugar = st.number_input("Blood Sugar (mg/dL)", 50, 300, 110, help="Normal fasting: 70–99 mg/dL")
        with c2:
            cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 350, 200, help="Desirable: below 200 mg/dL")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card card-case"><div class="card-header">Case Details</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            department = st.selectbox("Department", category_values['department'])
        with c2:
            diagnosis = st.selectbox("Diagnosis", category_values['diagnosis'])
        c3, c4, c5 = st.columns(3)
        with c3:
            previous_appointments = st.number_input("Previous Appointments", 0, 20, 2)
        with c4:
            missed_previous_appointments = st.number_input("Missed Appointments", 0, 10, 0)
        with c5:
            previous_admissions = st.number_input("Previous Admissions", 0, 20, 0)
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("Predict Risk Level")

    if submitted:
        warnings = []
        if systolic_bp >= 140 or diastolic_bp >= 90:
            warnings.append("Blood pressure is in a hypertensive range.")
        if blood_sugar >= 126:
            warnings.append("Blood sugar is above the typical diabetic threshold.")
        if bmi >= 30:
            warnings.append("BMI falls in the obese range.")
        for w in warnings:
            st.markdown(f'<div class="warning-pill">⚠️ {w}</div>', unsafe_allow_html=True)

        # Build the raw input row using exactly the columns the encoder expects,
        # in FINAL_PREDICTORS order — then let encoder_unscaled do all encoding at once
        raw_row = pd.DataFrame([{
            'age': age, 'gender': gender, 'blood_group': category_values.get('blood_group', ['A+'])[0],
            'department': department, 'diagnosis': diagnosis,
            'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
            'blood_sugar_mg_dl': blood_sugar, 'cholesterol_mg_dl': cholesterol, 'bmi': bmi,
            'previous_appointments': previous_appointments,
            'missed_previous_appointments': missed_previous_appointments,
            'previous_admissions': previous_admissions,
        }])[raw_input_columns]

        encoded = encoder_unscaled.transform(raw_row)
        row = pd.DataFrame(encoded, columns=encoder_unscaled.get_feature_names_out())[feature_columns]

        pred = model.predict(row)[0]
        pred_label = inverse_target[pred]
        proba = model.predict_proba(row)[0]
        classes = [inverse_target[i] for i in range(len(proba))]

        st.session_state["result"] = {
            "label": pred_label,
            "proba": proba,
            "classes": classes,
        }

    if "result" in st.session_state:
        res = st.session_state["result"]
        pred_label = res["label"]
        proba = res["proba"]
        classes = res["classes"]

        risk_colors = {"Low": "#27AE60", "Medium": "#E9A319", "High": "#E63946"}
        risk_icons = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}
        risk_advice = {
            "Low": "No immediate concerns indicated. Routine monitoring is advised.",
            "Medium": "Some risk factors are elevated. Consider a follow-up consultation.",
            "High": "Multiple risk factors detected. Prompt clinical review is recommended.",
        }
        color = risk_colors.get(pred_label, "#1B4965")
        icon = risk_icons.get(pred_label, "")
        advice = risk_advice.get(pred_label, "")

        st.write("")
        st.markdown(f"""
            <div style="background-color:{color}18; border-left: 6px solid {color};
                        padding: 22px; border-radius: 10px;">
                <h3 style="color:{color}; margin:0;">{icon} Predicted Risk Level: {pred_label}</h3>
                <p style="margin:8px 0 0 0; color:#333;">{advice}</p>
            </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="card" style="--accent:#E9A319; --bg:rgba(233,163,25,0.06);"><div class="card-header">Prediction Confidence</div>', unsafe_allow_html=True)
        mcols = st.columns(len(classes))
        for i, cls in enumerate(classes):
            with mcols[i]:
                st.metric(label=cls, value=f"{proba[i]*100:.1f}%")

        proba_df = pd.DataFrame({"Risk Level": classes, "Probability": proba})
        risk_order = [c for c in ["Low", "Medium", "High"] if c in list(classes)]
        color_domain = risk_order
        color_range = [risk_colors.get(lvl, "#1B4965") for lvl in risk_order]

        chart = (
            alt.Chart(proba_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Risk Level:N", sort=risk_order, title=None),
                y=alt.Y("Probability:Q", title="Probability", axis=alt.Axis(format="%")),
                color=alt.Color("Risk Level:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
                tooltip=[alt.Tooltip("Risk Level:N"), alt.Tooltip("Probability:Q", format=".1%")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab_about:
    st.markdown("""
    ### How this predictor works
    The model was trained on the **SmartCare Hospital dataset** to classify patients into
    three disease risk categories - **Low, Medium, High** - based on demographic and
    clinical inputs (age, gender, BMI, blood pressure, blood sugar, cholesterol,
    department, diagnosis, and appointment/admission history).

    **Note:** This is an academic prototype for coursework purposes. It is not validated
    for real clinical use and should not be used to make actual medical decisions.
    Predicted probabilities have not been formally calibration-tested; relative ranking
    between classes should be treated as more reliable than the exact percentage shown.
    """)