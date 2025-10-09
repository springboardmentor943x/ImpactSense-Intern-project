import os
import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="ImpactSense", layout="wide")

# Utility functions
def create_damage_potential(mag, depth):
    return 0.6 * mag + 0.2 * (700 - depth) / 700 * 10

def create_risk_category(score):
    if score < 4.0: return 0
    if score < 6.0: return 1
    return 2

def create_urban_risk(lat, lon, score):
    return score * (1 + (abs(lat) + abs(lon)) / 360)

def shap_explain(model, X):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    fig, _ = plt.subplots(figsize=(8,5))
    shap.summary_plot(sv, X, show=False)
    return fig

# Sidebar
with st.sidebar:
    data_path  = st.text_input("CSV path", "data/preprocessed_earthquake_data.csv")
    model_path = st.text_input("Model file", "models/lgb_damage_model.pkl")
    page       = st.radio("Page", ["📊 Data", "🔮 Predict", "🗺️ Map", "ℹ️ About"], index=0)

# Load and preprocess
if not os.path.exists(data_path):
    st.error("CSV missing"); st.stop()
df = pd.read_csv(data_path).dropna(
    subset=["Latitude", "Longitude", "Depth", "Magnitude", "Root Mean Square"]
).copy()

features = ["Latitude", "Longitude", "Depth", "Magnitude", "Root Mean Square"]

df["Damage_Potential"] = create_damage_potential(df.Magnitude, df.Depth)
df["Risk_Category"]    = df["Damage_Potential"].apply(create_risk_category)
df["Urban_Risk"]       = df.apply(
    lambda r: create_urban_risk(r.Latitude, r.Longitude, r.Damage_Potential),
    axis=1
)

if not os.path.exists(model_path):
    st.error("Model missing"); st.stop()
saved = joblib.load(model_path)
model, feat_list = saved["model"], saved["features"]

if page == "📊 Data":
    st.title("🌏 ImpactSense – Earthquake Impact Prediction & Risk Visualization")
    st.markdown("""
    Welcome to **ImpactSense**, a machine learning project that predicts:
    - **Damage Potential**: severity score based on magnitude & depth  
    - **Risk Category**: Low / Moderate / High classification  
    - **Urban Risk Score**: adjusts damage potential by location proxy
    """)
    st.subheader("Dataset Overview")
    st.write(df[features + ["Damage_Potential", "Urban_Risk"]].describe())

    st.markdown("### Distributions")
    cols = st.multiselect(
        "Select columns",
        features + ["Damage_Potential", "Urban_Risk"],
        default=features[:3]
    )
    for c in cols:
        fig, ax = plt.subplots()
        ax.hist(df[c], bins=30, color="skyblue", edgecolor="black")
        ax.set_title(f"{c} Distribution")
        st.pyplot(fig)

    st.markdown("### Correlation Matrix")
    corr = df[features + ["Damage_Potential", "Urban_Risk"]].corr()
    fig, ax = plt.subplots(figsize=(6,5))
    import seaborn as sns
    sns.heatmap(corr, annot=True, cmap="vlag", ax=ax)
    st.pyplot(fig)

elif page == "🔮 Predict":
    st.subheader("Single Prediction")
    vals = {c: st.slider(c, float(df[c].min()), float(df[c].max()), float(df[c].median())) for c in features}
    if st.button("Predict"):
        X = pd.DataFrame([vals])
        X_enc = X.reindex(columns=feat_list, fill_value=0)
        dp = model.predict(X_enc)[0]
        rc = create_risk_category(dp)
        ur = create_urban_risk(vals["Latitude"], vals["Longitude"], dp)
        st.metric("Damage Potential", f"{dp:.2f}")
        st.metric("Risk Category", ["Low", "Moderate", "High"][rc])
        st.metric("Urban Risk Score", f"{ur:.2f}")
        with st.expander("🔍 SHAP Explainability"):
            st.pyplot(shap_explain(model, X_enc))

elif page == "🗺️ Map":
    st.subheader("Risk Map")
    df_map = df.copy()
    df_map["Risk_Label"] = df_map.Risk_Category.map({0: "Low", 1: "Moderate", 2: "High"})
    fig = px.scatter_mapbox(
        df_map, lat="Latitude", lon="Longitude",
        color="Risk_Label", size="Damage_Potential", size_max=15,
        zoom=1, mapbox_style="carto-positron",
        hover_data=features + ["Urban_Risk"]
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Sample Data")
    st.dataframe(df_map[features + ["Damage_Potential", "Risk_Label"]].head(10))

else:
    st.subheader("About ImpactSense")
    st.markdown("""
### What is Damage Potential?
A numeric score estimating the earthquake's potential to cause destruction using magnitude and depth — higher means more damage expected.

### What is Risk Category?
A qualitative classification into:
- **Low** (green): minimal damage potential
- **Moderate** (yellow): potential for noticeable damage
- **High** (red): severe damage likely

### What is Urban Risk Score?
An adjusted damage potential factoring in location geography as a proxy for population density and infrastructure, highlighting areas where impact could affect more people.
    
This project helps visualize and predict earthquake impact to aid decision-making in disaster management and urban planning.
    """)
