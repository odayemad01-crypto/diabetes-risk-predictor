import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config & global styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#2563EB"
PRIMARY_DARK = "#1E3A8A"
ACCENT = "#0EA5A4"
DANGER = "#DC2626"
SAFE = "#059669"

st.markdown(f"""
<style>
    #MainMenu, footer, header {{visibility: hidden;}}

    .stApp {{
        background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%);
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    .hero {{
        background: linear-gradient(135deg, {PRIMARY_DARK} 0%, {PRIMARY} 60%, {ACCENT} 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.25);
    }}
    .hero h1 {{ margin: 0 0 0.3rem 0; font-size: 2.1rem; font-weight: 800; }}
    .hero p {{ margin: 0; opacity: 0.92; font-size: 1.02rem; }}

    .metric-card {{
        background: white;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
        border: 1px solid #E2E8F0;
        height: 100%;
    }}
    .metric-card .label {{
        color: #64748B; font-size: 0.82rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.04em;
    }}
    .metric-card .value {{ color: #0F172A; font-size: 1.75rem; font-weight: 800; margin-top: 0.2rem; }}

    .section-title {{
        font-size: 1.3rem; font-weight: 700; color: #0F172A;
        margin: 0.4rem 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid #E2E8F0;
    }}

    .result-banner {{
        border-radius: 16px; padding: 1.6rem 2rem; margin-top: 1rem;
        display: flex; align-items: center; gap: 1.2rem;
    }}
    .result-high {{ background: linear-gradient(135deg, #FEF2F2, #FEE2E2); border: 1px solid #FCA5A5; }}
    .result-low {{ background: linear-gradient(135deg, #ECFDF5, #D1FAE5); border: 1px solid #6EE7B7; }}
    .result-banner .icon {{ font-size: 2.6rem; }}
    .result-banner .title {{ font-size: 1.35rem; font-weight: 800; margin-bottom: 0.15rem; }}
    .result-banner.result-high .title {{ color: {DANGER}; }}
    .result-banner.result-low .title {{ color: {SAFE}; }}
    .result-banner .subtitle {{ color: #334155; font-size: 0.95rem; }}

    section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {PRIMARY_DARK}, #0F172A); }}
    section[data-testid="stSidebar"] * {{ color: #E2E8F0 !important; }}

    div.stButton > button {{
        background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
        color: white; border: none; border-radius: 10px;
        padding: 0.7rem 1.5rem; font-weight: 700; font-size: 1rem; width: 100%;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    }}
    div.stButton > button:hover {{ box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45); }}

    .footer-note {{ text-align: center; color: #94A3B8; font-size: 0.82rem; margin-top: 2.5rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_bundle():
    return joblib.load("diabetes_model.joblib")


@st.cache_data
def load_artifacts():
    with open("model_artifacts.json") as f:
        return json.load(f)


@st.cache_data
def load_sample():
    return pd.read_csv("diabetes_sample.csv")


bundle = load_model_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]

artifacts = load_artifacts()
sample_df = load_sample()


def build_feature_row(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose):
    """Builds a one-hot row matching the exact training columns (get_dummies, drop_first)."""
    row = pd.DataFrame(0, index=[0], columns=feature_columns, dtype=float)
    row["age"] = age
    row["hypertension"] = hypertension
    row["heart_disease"] = heart_disease
    row["bmi"] = bmi
    row["HbA1c_level"] = hba1c
    row["blood_glucose_level"] = glucose

    gender_col = f"gender_{gender}"
    if gender_col in row.columns:
        row[gender_col] = 1

    smoking_col = f"smoking_history_{smoking_history}"
    if smoking_col in row.columns:
        row[smoking_col] = 1

    return row


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 Diabetes Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔮 Prediction", "📊 Data Insights", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.85rem; opacity:0.8;'>"
        f"Model in use:<br><b>{artifacts['model_name']}</b><br><br>"
        f"Trained on {artifacts['n_train']:,} records<br>"
        f"Tested on {artifacts['n_test']:,} records"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("""
<div class="hero">
    <h1>Diabetes Risk Prediction System</h1>
    <p>An ML-powered clinical decision-support tool that estimates diabetes risk
    from routine health indicators.</p>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# PAGE: Prediction
# ===========================================================================
if page == "🔮 Prediction":
    st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)

    gender_options = artifacts["gender_categories"]
    smoking_options = artifacts["smoking_categories"]

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", gender_options)
            age = st.slider("Age", 1, 100, 40)
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])

        with c2:
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            smoking_history = st.selectbox("Smoking History", smoking_options)
            bmi = st.slider("BMI", 10.0, 60.0, 24.5, step=0.1)

        with c3:
            hba1c = st.slider("HbA1c Level (%)", 3.5, 9.0, 5.5, step=0.1,
                               help="Average blood sugar over ~3 months. Normal < 5.7%")
            glucose = st.slider("Blood Glucose Level (mg/dL)", 70, 300, 110,
                                 help="Fasting or random glucose reading")

        submitted = st.form_submit_button("🔍 Predict Diabetes Risk")

    if submitted:
        row = build_feature_row(
            gender, age,
            1 if hypertension == "Yes" else 0,
            1 if heart_disease == "Yes" else 0,
            smoking_history, bmi, hba1c, glucose,
        )
        row_scaled = scaler.transform(row)
        proba = model.predict_proba(row_scaled)[0, 1]
        pred = int(proba >= 0.5)

        if pred == 1:
            st.markdown(f"""
            <div class="result-banner result-high">
                <div class="icon">⚠️</div>
                <div>
                    <div class="title">Higher Diabetes Risk Detected</div>
                    <div class="subtitle">Estimated probability: <b>{proba*100:.1f}%</b> —
                    consider consulting a healthcare professional for further evaluation.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-banner result-low">
                <div class="icon">✅</div>
                <div>
                    <div class="title">Lower Diabetes Risk</div>
                    <div class="subtitle">Estimated probability: <b>{proba*100:.1f}%</b> —
                    indicators are currently within a lower-risk range.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        gcol1, gcol2 = st.columns([1, 1.2])

        with gcol1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": PRIMARY_DARK},
                    "steps": [
                        {"range": [0, 30], "color": "#D1FAE5"},
                        {"range": [30, 60], "color": "#FEF3C7"},
                        {"range": [60, 100], "color": "#FEE2E2"},
                    ],
                    "threshold": {"line": {"color": DANGER, "width": 4}, "thickness": 0.8, "value": 50},
                },
                title={"text": "Predicted Risk Probability"},
            ))
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with gcol2:
            st.markdown("##### How the key inputs compare to healthy reference ranges")
            ref = pd.DataFrame({
                "Metric": ["BMI", "HbA1c (%)", "Glucose (mg/dL)"],
                "Your Value": [bmi, hba1c, glucose],
                "Healthy Upper Bound": [25, 5.7, 140],
            })
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(y=ref["Metric"], x=ref["Healthy Upper Bound"],
                                   orientation="h", name="Healthy upper bound",
                                   marker_color="#CBD5E1"))
            fig2.add_trace(go.Scatter(y=ref["Metric"], x=ref["Your Value"],
                                       mode="markers", name="Your value",
                                       marker=dict(size=16, color=PRIMARY, symbol="diamond")))
            fig2.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=10),
                                barmode="overlay", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "⚠️ This tool is for educational/demonstration purposes only and is not a "
            "substitute for professional medical diagnosis."
        )


# ===========================================================================
# PAGE: Data Insights (EDA)
# ===========================================================================
elif page == "📊 Data Insights":
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    total = len(sample_df)
    diabetic_pct = sample_df["diabetes"].mean() * 100
    avg_age = sample_df["age"].mean()
    avg_bmi = sample_df["bmi"].mean()

    m1, m2, m3, m4 = st.columns(4)
    for col, label, value in zip(
        [m1, m2, m3, m4],
        ["Sample Records", "Diabetes Rate", "Avg. Age", "Avg. BMI"],
        [f"{total:,}", f"{diabetic_pct:.1f}%", f"{avg_age:.0f} yrs", f"{avg_bmi:.1f}"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Distributions</div>', unsafe_allow_html=True)

    row1c1, row1c2 = st.columns(2)
    with row1c1:
        fig = px.histogram(sample_df, x="age", color="diabetes", barmode="overlay", nbins=30,
                            color_discrete_map={0: "#93C5FD", 1: "#F87171"},
                            labels={"diabetes": "Diabetes"}, title="Age Distribution by Diabetes Status")
        fig.update_layout(height=380, legend_title_text="Diabetes")
        st.plotly_chart(fig, use_container_width=True)

    with row1c2:
        fig = px.histogram(sample_df, x="HbA1c_level", color="diabetes", barmode="overlay", nbins=30,
                            color_discrete_map={0: "#93C5FD", 1: "#F87171"},
                            labels={"diabetes": "Diabetes"}, title="HbA1c Level Distribution by Diabetes Status")
        fig.update_layout(height=380, legend_title_text="Diabetes")
        st.plotly_chart(fig, use_container_width=True)

    row2c1, row2c2 = st.columns(2)
    with row2c1:
        fig = px.box(sample_df, x="diabetes", y="bmi", color="diabetes",
                     color_discrete_map={0: "#93C5FD", 1: "#F87171"},
                     labels={"diabetes": "Diabetes (0=No, 1=Yes)"}, title="BMI by Diabetes Status")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with row2c2:
        fig = px.box(sample_df, x="diabetes", y="blood_glucose_level", color="diabetes",
                     color_discrete_map={0: "#93C5FD", 1: "#F87171"},
                     labels={"diabetes": "Diabetes (0=No, 1=Yes)"}, title="Blood Glucose by Diabetes Status")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Categorical Breakdown</div>', unsafe_allow_html=True)
    row3c1, row3c2 = st.columns(2)
    with row3c1:
        gdf = sample_df.groupby(["gender", "diabetes"]).size().reset_index(name="count")
        fig = px.bar(gdf, x="gender", y="count", color="diabetes", barmode="group",
                     color_discrete_map={0: "#93C5FD", 1: "#F87171"}, title="Diabetes Cases by Gender")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with row3c2:
        sdf = sample_df.groupby(["smoking_history", "diabetes"]).size().reset_index(name="count")
        fig = px.bar(sdf, x="smoking_history", y="count", color="diabetes", barmode="group",
                     color_discrete_map={0: "#93C5FD", 1: "#F87171"}, title="Diabetes Cases by Smoking History")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE: Model Performance
# ===========================================================================
elif page == "📈 Model Performance":
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

    m = artifacts["metrics"]
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    for col, label, value in zip(
        [mc1, mc2, mc3, mc4, mc5],
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        [m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"]],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:12px;
                padding:0.9rem 1.2rem; margin: 1.2rem 0 1.5rem 0;">
        🏆 <b>Model: {artifacts['model_name']}</b> — evaluated on a held-out 20% test split.
        The dataset is imbalanced (only {artifacts['class_balance'].get('1', 0)*100:.1f}% positive cases),
        so ROC-AUC and Recall are the more informative metrics here alongside Accuracy.
    </div>
    """, unsafe_allow_html=True)

    p1, p2 = st.columns(2)

    with p1:
        fpr = artifacts["roc_curve"]["fpr"]
        tpr = artifacts["roc_curve"]["tpr"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve", line=dict(color=PRIMARY, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#CBD5E1", dash="dash")))
        fig.update_layout(title=f"ROC Curve — {artifacts['model_name']}",
                           xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with p2:
        cm = np.array(artifacts["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         labels=dict(x="Predicted", y="Actual", color="Count"),
                         x=["No Diabetes", "Diabetes"], y=["No Diabetes", "Diabetes"],
                         title=f"Confusion Matrix — {artifacts['model_name']}")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    if artifacts.get("feature_importances"):
        st.markdown('<div class="section-title">Feature Importance (|coefficient|)</div>', unsafe_allow_html=True)
        fi = pd.Series(artifacts["feature_importances"]).sort_values(ascending=True)
        fig = px.bar(fi, orientation="h", labels={"index": "Feature", "value": "|Coefficient|"},
                     title="What drives predictions — Logistic Regression",
                     color_discrete_sequence=[PRIMARY])
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE: About
# ===========================================================================
elif page == "ℹ️ About":
    st.markdown('<div class="section-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown(f"""
    This application predicts an individual's risk of diabetes based on key
    health indicators, using the machine learning pipeline built in the
    accompanying notebook (`oday_emadeldin_day_13.ipynb`).

    **Pipeline overview**
    - **Cleaning:** duplicate rows dropped, rows with a missing target dropped
    - **Encoding:** one-hot encoding via `pd.get_dummies(drop_first=True)` on gender and smoking history
    - **Split:** 80% train / 20% test (`random_state=42`)
    - **Scaling:** `StandardScaler` fit on the training set
    - **Model:** Logistic Regression (`max_iter=1000`)

    **Features used**
    | Feature | Description |
    |---|---|
    | Gender | {", ".join(artifacts['gender_categories'])} |
    | Age | Patient age in years |
    | Hypertension | Diagnosed hypertension (Yes/No) |
    | Heart Disease | Diagnosed heart disease (Yes/No) |
    | Smoking History | {", ".join(artifacts['smoking_categories'])} |
    | BMI | Body Mass Index |
    | HbA1c Level | Average blood sugar over ~3 months (%) |
    | Blood Glucose Level | Fasting/random glucose reading (mg/dL) |

    ---
    ⚠️ **Disclaimer:** This tool is built for educational and demonstration
    purposes as part of a data science coursework project. It is **not** a
    certified medical device and should never replace professional medical
    advice, diagnosis, or treatment.
    """)

st.markdown('<div class="footer-note">Diabetes Risk Prediction System · Built with Streamlit & scikit-learn</div>',
            unsafe_allow_html=True)
