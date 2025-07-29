import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from streamlit_option_menu import option_menu

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------
st.set_page_config(page_title="Employment Status Classifier", layout="wide")

# ------------------------------------------------
# CUSTOM STYLING (New Theme)
# ------------------------------------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #F0F4EF !important;
        }
        html, body, [data-testid="stAppViewContainer"] > .main {
            background-color: #FBFBFB !important;
            color: #222222 !important;
        }
        .stSelectbox > div,
        .stSelectbox div[data-baseweb="select"] > div,
        input[type="number"] {
            background-color: #F9F9F9 !important;
            border-radius: 6px;
            border: 1px solid #CED4DA;
            padding: 0.3rem;
        }
        div[data-baseweb="slider"] [role="slider"] {
            background-color: #006666 !important;
        }
        div[data-baseweb="slider"] > div > div > div:nth-child(2) {
            background: #006666 !important;
        }
        div.stButton > button {
            background-color: #006666 !important;
            color: white !important;
            border-radius: 6px !important;
            border: none;
            padding: 0.5rem 1.2rem;
        }
        div.stButton > button:hover {
            background-color: #004C4C !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# MODEL LOADING
# ------------------------------------------------
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
X_columns = joblib.load("X_columns.pkl")

# ------------------------------------------------
# SIDEBAR MENU
# ------------------------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Predictor", "About"],
        icons=["house-door-fill", "cpu-fill", "info-square-fill"],
        default_index=1,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#F0F4EF"},
            "icon": {"color": "#006666", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#D9F2EF",
                "color": "#333333"
            },
            "nav-link-selected": {
                "background-color": "#B2DFDB",
                "color": "#000000"
            },
        },
    )

# ------------------------------------------------
# HOME TAB
# ------------------------------------------------
if selected == "Home":
    st.title("🏡 Employment Status Prediction System")
    st.write("Welcome! Use the sidebar to navigate and explore employment prediction based on educational and socio-economic features.")

# ------------------------------------------------
# PREDICTOR TAB
# ------------------------------------------------
elif selected == "Predictor":
    st.title("🔍 Employment Status Predictor")
    st.write("Provide individual data below to predict employment classification.")

    input_dict = {}
    for col in X_columns:
        if col in ['Matric', 'Degree', 'Diploma']:
            input_dict[col] = st.selectbox(f"{col} (Yes/No)", ['Yes', 'No'])
        elif col == 'Sa_citizen':
            input_dict[col] = st.selectbox("Are you a SA Citizen?", ['Citizen', 'Non-citizen'])
        elif col == 'Gender':
            input_dict[col] = st.selectbox("Gender", ['Female', 'Male'])
        elif col == 'Highest_Education':
            input_dict[col] = st.selectbox("Highest Education", ['Degree', 'Diploma', 'Matric', 'None'])
        elif col in ['Math', 'Mathlit', 'Additional_lang', 'Home_lang', 'Science', 'Geography']:
            input_dict[col] = st.selectbox(col, ['0% - 49%', '50% - 79%', '80% - 100%'])
        elif col == 'Province':
            input_dict[col] = st.selectbox(col, [
                'Gauteng', 'Mpumalanga', 'North West', 'Free State',
                'Eastern Cape', 'Limpopo', 'KwaZulu-Natal', 'Northern Cape', 'Western Cape'])
        elif col == 'Status':
            input_dict[col] = st.selectbox(col, [
                'Unemployed', 'Studying', 'Wage employed', 'Self employed',
                'Wage and self employed', 'Employment program', 'Other'])
        else:
            input_dict[col] = st.number_input(col, value=0)

    input_df = pd.DataFrame([input_dict])

    for col in input_df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        input_df[col] = le.fit(input_df[col]).transform(input_df[col])

    input_df_scaled = scaler.transform(input_df)

    if st.button("Predict Employment Status"):
        pred = model.predict(input_df_scaled)[0]
        prob = model.predict_proba(input_df_scaled)[0][int(pred)]
        status = 'Unemployed' if pred == 1 else 'Employed'
        st.success(f"🧾 Prediction: {status}")
        st.info(f"Confidence Level: {prob * 100:.2f}%")

# ------------------------------------------------
# ABOUT TAB
# ------------------------------------------------
elif selected == "About":
    st.title("📘 About the App")
    st.markdown("""
        This application predicts the employment status of individuals using a Logistic Regression model
        trained on educational and demographic attributes.

        *Team:* Group Five  
        *Built with:* Python, Streamlit, scikit-learn
    """)
