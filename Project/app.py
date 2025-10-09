import os
import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="ImpactSense", layout="wide")

# Utility functions
def create_risk_category(score):
    if score < 4.0:
        return 0
    if score < 6.0:
        return 1
    return 2

def create_urban_risk(lat, lon, score):
    return score * (1 + (abs(lat) + abs(lon)) / 360)

def shap_explain(model, X):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    fig, _ = plt.subplots(figsize=(8,5))
    shap.summary_plot(sv, X, show=False)
    return fig

# Preprocessing helper functions & classes
class DropNaNRows(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.dropna(axis=0)

class DamagePotentialCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        mag = X['Magnitude']
        depth = X['Depth']
        damage_potential = 0.6 * mag + 0.2 * (700 - depth) / 700 * 10
        X = X.copy()
        X['Damage_Potential'] = damage_potential
        return X

def select_features(X):
    return X[['Latitude', 'Longitude', 'Depth', 'Magnitude', 'Type', 'Magnitude Type', 'Status', 'Root Mean Square']]

def impute_root_mean_square(df):
    known_df = df[df['Root Mean Square'].notna()]
    unknown_df = df[df['Root Mean Square'].isna()]

    if unknown_df.empty:
        return df

    features = ['Latitude', 'Longitude', 'Depth', 'Magnitude']
    X_train = known_df[features]
    y_train = known_df['Root Mean Square']
    X_pred = unknown_df[features]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predicted = model.predict(X_pred)
    df.loc[df['Root Mean Square'].isna(), 'Root Mean Square'] = predicted
    return df

def filter_earthquake(X):
    return X[X['Type'] == 'Earthquake']

def drop_type(X):
    return X.drop('Type', axis=1)

def encode_categoricals(X):
    categorical_cols = ['Magnitude Type', 'Status']
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(X[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)
    X = X.drop(columns=categorical_cols).reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    return pd.concat([X, encoded_df], axis=1)

def transform_input(input_dict, feat_list, df_raw):
    X = pd.DataFrame([input_dict])

    if pd.isna(X.loc[0, 'Root Mean Square']):
        known_df = df_raw[df_raw['Root Mean Square'].notna()]
        unknown_df = X
        features = ['Latitude', 'Longitude', 'Depth', 'Magnitude']
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(known_df[features], known_df['Root Mean Square'])
        predicted = rf.predict(unknown_df[features])
        X.loc[0, 'Root Mean Square'] = predicted[0]

    mag = X.loc[0, 'Magnitude']
    depth = X.loc[0, 'Depth']
    damage_potential = 0.6 * mag + 0.2 * (700 - depth) / 700 * 10
    X['Damage_Potential'] = damage_potential

    if X.loc[0, 'Type'] != 'Earthquake':
        return pd.DataFrame([[0] * len(feat_list)], columns=feat_list)

    categorical_cols = ['Magnitude Type', 'Status']
    for cat_col in categorical_cols:
        dummies = pd.get_dummies(X[cat_col], prefix=cat_col, drop_first=True)
        for col in dummies.columns:
            X[col] = dummies[col]
        X.drop(columns=[cat_col], inplace=True)

    X.drop(columns=['Type'], inplace=True)

    for col in feat_list:
        if col not in X.columns:
            X[col] = 0

    X = X[feat_list]

    return X

# Sidebar interface
with st.sidebar:
    uploaded_file = st.file_uploader("Upload earthquake CSV file", type=["csv"])
    page = st.radio("Page", ["📊 Data", "🔮 Predict", "🗺️ Map", "ℹ️ About"], index=0)

# Fixed paths for preprocessing pipeline and model
preprocessing_path = "models/data_preprocessing_pipeline.pkl"
model_path = "models/lgb_damage_model.pkl"

if uploaded_file is None:
    st.warning("Please upload an earthquake CSV file to proceed.")
    st.stop()

# Load raw data from uploaded file
df_raw = pd.read_csv(uploaded_file).dropna(subset=["Latitude", "Longitude", "Depth", "Magnitude"]).copy()

# Load preprocessing pipeline
if not os.path.exists(preprocessing_path):
    st.error("Preprocessing pipeline missing on server.")
    st.stop()
preprocessing_pipeline = joblib.load(preprocessing_path)

# Preprocess raw data
df_processed = preprocessing_pipeline.transform(df_raw)

features = list(df_processed.columns)
if 'Damage_Potential' in features:
    features.remove('Damage_Potential')

# Load model and features
if not os.path.exists(model_path):
    st.error("Model missing on server.")
    st.stop()
saved = joblib.load(model_path)
model, feat_list = saved["model"], saved["features"]

# Add Risk Category only for display (no Urban Risk column)
if 'Damage_Potential' in df_processed.columns:
    df_processed["Risk_Category"] = df_processed["Damage_Potential"].apply(create_risk_category)
else:
    df_processed["Risk_Category"] = None

# Pages
if page == "📊 Data":
    st.title("🌏 ImpactSense – Earthquake Impact Prediction & Risk Visualization")
    
    st.markdown("""
    Welcome to **ImpactSense**, a comprehensive project designed to predict and visualize the impact of earthquakes.
    
    This application leverages machine learning models to estimate:
    
    - **Damage Potential**: a severity score based on earthquake magnitude and depth.
    - **Risk Category**: classification of earthquakes into low, moderate, or high risk.
    - **Urban Risk Score**: an adjusted damage potential factoring in geographical location as a proxy for population exposure.
    
    Use the navigation panel to explore earthquake data, make predictions, and visualize risk maps.
    """)
    st.subheader("Dataset Overview")
    display_cols = features.copy()
    if 'Damage_Potential' in df_processed.columns:
        display_cols += ['Damage_Potential']
    st.write(df_processed[display_cols].describe())

    st.markdown("### Distributions")
    cols = st.multiselect("Select columns", display_cols, default=features[:3])
    for c in cols:
        fig, ax = plt.subplots()
        ax.hist(df_processed[c], bins=30, color="skyblue", edgecolor="black")
        ax.set_title(f"{c} Distribution")
        st.pyplot(fig)

    st.markdown("### Correlation Matrix")
    corr = df_processed[display_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
      corr,
      annot=True,
      fmt=".2f",
      cmap="vlag",
      ax=ax,
      annot_kws={"size": 10},
      xticklabels=True,
      yticklabels=True
    )
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    st.pyplot(fig)




elif page == "🔮 Predict":
    st.subheader("Single Prediction")

    base_numeric = ["Latitude", "Longitude", "Depth", "Magnitude", "Root Mean Square"]
    numeric_vals = {c: st.slider(c, float(df_raw[c].min()), float(df_raw[c].max()), float(df_raw[c].median())) for c in base_numeric}

    # type_options = df_raw['Type'].dropna().unique().tolist() if 'Type' in df_raw.columns else []
    magnitude_type_options = df_raw['Magnitude Type'].dropna().unique().tolist() if 'Magnitude Type' in df_raw.columns else []
    status_options = df_raw['Status'].dropna().unique().tolist() if 'Status' in df_raw.columns else []

    type_val = 'Earthquake' # Fixed as only 'Earthquake' is processed
    magnitude_type_val = st.selectbox('Magnitude Type', options=magnitude_type_options if magnitude_type_options else ['ML'])
    status_val = st.selectbox('Status', options=status_options if status_options else ['Reviewed'])

    input_vals = numeric_vals.copy()
    input_vals['Type'] = type_val
    input_vals['Magnitude Type'] = magnitude_type_val
    input_vals['Status'] = status_val

    if st.button("Predict"):
        X_input = transform_input(input_vals, feat_list, df_raw)

        dp = model.predict(X_input)[0]
        rc = create_risk_category(dp)
        ur = create_urban_risk(input_vals["Latitude"], input_vals["Longitude"], dp)

        st.metric("Damage Potential", f"{dp:.2f}")
        st.metric("Risk Category", ["Low", "Moderate", "High"][rc])
        st.metric("Urban Risk Score", f"{ur:.2f}")

        with st.expander("🔍 SHAP Explainability"):
            st.pyplot(shap_explain(model, X_input))

elif page == "🗺️ Map":
    st.subheader("Risk Map")
    df_map = df_processed.copy()
    if 'Damage_Potential' in df_map.columns:
        df_map["Risk_Label"] = df_map.Risk_Category.map({0: "Low", 1: "Moderate", 2: "High"})
        fig = px.scatter_mapbox(
            df_map, lat="Latitude", lon="Longitude",
            color="Risk_Label", size="Damage_Potential", size_max=15,
            zoom=1, mapbox_style="carto-positron",
            hover_data=features
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Sample Data")
        st.dataframe(df_map[features + ["Damage_Potential", "Risk_Label"]].head(10))
    else:
        st.warning("Damage_Potential not found in processed data, cannot display map.")

else:
    st.subheader("About ImpactSense")
    st.markdown("""
### What is Damage Potential?
A numeric score estimating the earthquake's potential to cause destruction using magnitude and depth — higher means more damage expected.

### What is Risk Category?
A qualitative classification into:
- **Low**: minimal damage potential
- **Moderate**: potential for noticeable damage
- **High**: severe damage likely

### What is Urban Risk Score?
An adjusted damage potential factoring in location geography as a proxy for population density and infrastructure, highlighting areas where impact could affect more people.

This project helps visualize and predict earthquake impact to aid decision-making in disaster management and urban planning.
""")
