# importing the libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load our trained model we previously trained
model = joblib.load('iris_model.joblib')
target_names = joblib.load('iris_target_names.joblib')

# Load the Iris dataset for exploration
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species'] = iris_df['species'].apply(lambda x: iris.target_names[x])

# Set page configuration
st.set_page_config(page_title="Iris Species Predictor", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode",
    ["Predict Iris Species", "Explore the Iris Dataset"]
)

# prediction mode
if app_mode == "Predict Iris Species":
    st.title("Flower Predictor")
    st.markdown("This app predicts the species of an Iris flower based on its sepal and petal measurements.")
    st.markdown("---")

    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Features")
        st.markdown("Use the sliders below to input the flower's measurements.")

        # Sliders for user input
        sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.4, 0.1)
        sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.4, 0.1)
        petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.3, 0.1)
        petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2, 0.1)
        
        # Create a DataFrame from user inputs
        input_data = pd.DataFrame({
            'sepal length (cm)': [sepal_length],
            'sepal width (cm)': [sepal_width],
            'petal length (cm)': [petal_length],
            'petal width (cm)': [petal_width]
        })

    with col2:
        st.subheader("Prediction")
        st.markdown("The model's prediction and confidence level are shown below.")

        # Make prediction and get probabilities
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Get the predicted species name
        predicted_species = target_names[prediction[0]]
        
        # Display the prediction with styling
        st.success(f"**Predicted Species: {predicted_species.title()}**")
        
        st.write("---")
        
        # Display prediction probabilities
        st.subheader("Prediction Probability")
        proba_df = pd.DataFrame(prediction_proba, columns=target_names)
        st.dataframe(proba_df.style.format("{:.2%}").highlight_max(axis=1, color='darkgreen'))


# exploration of dataset

elif app_mode == "Explore the Iris Dataset":
    st.title("Exploration of Iris Dataset")
    st.markdown("Explore the relationships between different features of the Iris dataset.")
    
    # Display the full dataset as a reference
    st.subheader("Full Dataset")
    st.dataframe(iris_df)
    st.markdown("---")
    
    # Create two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Histograms")
        # Allow user to select a feature
        feature_to_plot = st.selectbox("Select a feature for histogram", iris.feature_names)
        
        # Plot histogram
        fig, ax = plt.subplots()
        sns.histplot(data=iris_df, x=feature_to_plot, hue='species', kde=True, ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Feature Scatter Plot")
        # Allow user to select two features for the scatter plot
        x_feature = st.selectbox("Select X-axis feature", iris.feature_names, index=0)
        y_feature = st.selectbox("Select Y-axis feature", iris.feature_names, index=1)
        
        # Plot scatter plot
        fig, ax = plt.subplots()
        sns.scatterplot(data=iris_df, x=x_feature, y=y_feature, hue='species', ax=ax)
        st.pyplot(fig)