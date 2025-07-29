import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

# --------------------
# PAGE CONFIG
# --------------------
st.set_page_config(page_title="Group Five - Employment Status Classifier", layout="wide")

st.title("📊 Employment Status Prediction App")
st.markdown("This app allows you to explore the cleaned dataset and predict employment status based on a trained Logistic Regression model.")

# --------------------
# LOAD DATA
# --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Group Five_data.csv")
    return df

df = load_data()

# --------------------
# SIDEBAR
# --------------------
st.sidebar.header("Navigation")
option = st.sidebar.radio("Select a page:", ["📁 Dataset Overview", "📈 Visualizations", "🤖 Model Prediction"])

# --------------------
# PAGE 1: Dataset Overview
# --------------------
if option == "📁 Dataset Overview":
    st.subheader("Preview of Dataset")
    st.dataframe(df.head(20))

    st.subheader("Dataset Info")
    st.write(df.info())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(include='all').T)

# --------------------
# PAGE 2: Visualizations
# --------------------
elif option == "📈 Visualizations":
    st.subheader("Distribution Plots")

    with st.expander("⿡ Gender Distribution"):
        gender_count = df['Gender'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(gender_count, labels=gender_count.index, autopct='%1.0f%%', colors=['#ff9999','#66b3ff'])
        ax1.set_title("Gender Proportion")
        st.pyplot(fig1)

    with st.expander("⿢ Province Distribution"):
        prov_count = df['Province'].value_counts().sort_values()
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        bars = ax2.barh(prov_count.index, prov_count.values, color='brown')
        ax2.set_title("Province Distribution")
        for bar in bars:
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center')
        st.pyplot(fig2)

    with st.expander("⿣ Status by Gender"):
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.countplot(data=df, x='Status', hue='Gender', ax=ax3)
        for container in ax3.containers:
            ax3.bar_label(container, padding=3)
        ax3.set_title("Status Distribution by Gender")
        st.pyplot(fig3)

# --------------------
# PAGE 3: Model Prediction
# --------------------
elif option == "🤖 Model Prediction":
    st.subheader("Predict Employment Status")

    # Load model artifacts
    model = joblib.load("logistic_model.pkl")
    scaler = joblib.load("scaler.pkl")
    X_columns = joblib.load("X_columns.pkl")

    input_dict = {}
    for col in X_columns:
        if col in ['Matric', 'Degree', 'Diploma']:
            input_dict[col] = st.selectbox(f"{col} (Yes/No)", ['Yes', 'No'])
        elif col in ['Sa_citizen']:
            input_dict[col] = st.selectbox("Are you a SA Citizen?", ['Citizen', 'Non-citizen'])
        elif col in ['Gender']:
            input_dict[col] = st.selectbox("Gender", ['Female', 'Male'])
        elif col in ['Highest_Education']:
            input_dict[col] = st.selectbox("Highest Education", ['Degree', 'Diploma', 'Matric', 'None'])
        elif col in ['Math', 'Mathlit', 'Additional_lang', 'Home_lang', 'Science', 'Geography', 'Province', 'Status']:
            input_dict[col] = st.selectbox(f"{col}", df[col].unique())
        else:
            input_dict[col] = st.number_input(f"{col}", value=0.0)

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Encode categorical
    for col in input_df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        input_df[col] = le.fit(df[col].astype(str)).transform(input_df[col].astype(str))

    # Scale numeric
    input_df_scaled = scaler.transform(input_df)

    # Predict
    if st.button("Predict"):
        pred = model.predict(input_df_scaled)[0]
        prob = model.predict_proba(input_df_scaled)[0][int(pred)]

        st.success(f"🎯 Predicted Employment Status: *{'Unemployed' if pred == 1 else 'Employed'}*")
        st.write(f"📌 Confidence: *{round(prob * 100, 2)}%*")

# --------------------
# END OF APP
# --------------------
