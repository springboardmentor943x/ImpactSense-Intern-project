import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset and model
iris = load_iris()
model = joblib.load("iris_model.pkl")
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# --- App Config ---
st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="wide")

# Sidebar navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_versicolor_3.jpg", caption="Iris Flower", use_container_width=True)
    st.title("🌸 Iris Classifier")
    st.markdown("Navigate through the app:")
    mode = st.radio("Choose Mode:", ["🔮 Prediction", "📊 Data Exploration"])

# Main Layout
st.title("🌼 Interactive Iris Flower Classifier")
st.markdown("Built with **Streamlit** + **Scikit-learn**. Explore the Iris dataset and predict flower species with ease.")

st.markdown("---")

# --- Prediction Mode ---
if mode == "🔮 Prediction":
    st.subheader("Enter Flower Features")
    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider("Sepal length (cm)", float(df["sepal length (cm)"].min()), float(df["sepal length (cm)"].max()), help="Length of the sepal in centimeters.")
        sepal_width = st.slider("Sepal width (cm)", float(df["sepal width (cm)"].min()), float(df["sepal width (cm)"].max()), help="Width of the sepal in centimeters.")
    with col2:
        petal_length = st.slider("Petal length (cm)", float(df["petal length (cm)"].min()), float(df["petal length (cm)"].max()), help="Length of the petal in centimeters.")
        petal_width = st.slider("Petal width (cm)", float(df["petal width (cm)"].min()), float(df["petal width (cm)"].max()), help="Width of the petal in centimeters.")

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("🌟 Prediction Result")

    # Color-coded result
    result_color = {
        0: "🟢 Setosa",
        1: "🟡 Versicolor",
        2: "🔴 Virginica"
    }
    st.markdown(f"### Predicted Class: **{result_color[prediction]}**")

    st.progress(int(max(prediction_proba) * 100))

    # Show probability table
    proba_df = pd.DataFrame(prediction_proba.reshape(1, -1), columns=iris.target_names)
    st.write("Prediction Probabilities:")
    st.dataframe(proba_df.style.highlight_max(axis=1, color="lightgreen"))

# --- Data Exploration Mode ---
elif mode == "📊 Data Exploration":
    st.subheader("Explore the Iris Dataset")

    tab1, tab2 = st.tabs(["📈 Histogram", "🌐 Scatter Plot"])

    with tab1:
        feature = st.selectbox("Select feature for histogram", iris.feature_names)
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax, color="skyblue")
        st.pyplot(fig)

    with tab2:
        x_axis = st.selectbox("X-axis", iris.feature_names, index=2)
        y_axis = st.selectbox("Y-axis", iris.feature_names, index=3)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue="target", palette="Set1", s=100, ax=ax)
        st.pyplot(fig)

    st.markdown("---")
    st.write("### Dataset Preview")
    st.dataframe(df.head())
