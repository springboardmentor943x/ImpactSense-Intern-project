import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title
st.title("Titanic Dataset Data Analysis")

# Load Titanic dataset
@st.cache_data
def load_data():
    url = "titanic.csv"
    data = pd.read_csv(url)
    return data

df = load_data()

# Sidebar controls for filtering and customization
st.sidebar.header("Filter & Customize")

# Filter by passenger class
pclass_filter = st.sidebar.multiselect(
    "Select Passenger Class(es):",
    options=sorted(df["Pclass"].unique()),
    default=sorted(df["Pclass"].unique())
)
filtered_df = df[df["Pclass"].isin(pclass_filter)]

# Filter by survival status
survival_filter = st.sidebar.multiselect(
    "Select Survival Status:",
    options=[0, 1],
    format_func=lambda x: "Survived" if x == 1 else "Did not survive",
    default=[0, 1]
)
filtered_df = filtered_df[filtered_df["Survived"].isin(survival_filter)]

# Select column for histogram plot
numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns.to_list()
hist_col = st.sidebar.selectbox("Select column for histogram:", numerical_columns, index=numerical_columns.index("Age"))

# Show original or filtered data preview
if st.sidebar.checkbox("Show Filtered Data Preview", value=True):
    st.subheader("Filtered Data Preview")
    st.write(filtered_df.head())
else:
    st.subheader("Data Preview")
    st.write(df.head())

# Show basic statistics for filtered or full data
if st.sidebar.checkbox("Show Data Summary Statistics", value=True):
    st.subheader("Summary Statistics")
    st.write(filtered_df.describe())

# Survival Count plot with filter applied
st.subheader("Survival Count")
fig1, ax1 = plt.subplots()
sns.countplot(x="Survived", data=filtered_df, ax=ax1)
ax1.set_xticklabels(["Did not survive", "Survived"])
st.pyplot(fig1)

# Age (or selected numerical column) distribution with KDE toggle
st.subheader(f"{hist_col} Distribution")
show_kde = st.checkbox("Show KDE plot", value=True)
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df[hist_col].dropna(), kde=show_kde, bins=30, ax=ax2)
st.pyplot(fig2)

# Survival by Passenger Class bar plot with filter applied
st.subheader("Survival by Passenger Class")
fig3, ax3 = plt.subplots()
sns.countplot(x="Pclass", hue="Survived", data=filtered_df, ax=ax3)
ax3.legend(title="Survived", labels=["No", "Yes"])
st.pyplot(fig3)
