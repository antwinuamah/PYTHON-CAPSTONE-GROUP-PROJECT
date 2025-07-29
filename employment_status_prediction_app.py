import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from streamlit_option_menu import option_menu

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------
st.set_page_config(page_title="Employment Status Predictor", layout="wide")

# ------------------------------------------------
# CUSTOM STYLING (Modern & Professional Look)
# ------------------------------------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #f0f2f6 !important;
        }
        html, body, [data-testid="stAppViewContainer"] > .main {
            background-color: #ffffff !important;
            color: #1e1e1e !important;
        }
        .stSelectbox > div,
        .stSelectbox div[data-baseweb="select"] > div,
        input[type="number"] {
            background-color: #f7f9fc !important;
            border-radius: 10px;
            padding: 0.5rem;
            font-size: 16px;
        }
        div[data-baseweb="slider"] [role="slider"] {
            background-color: #004080 !important;
        }
        div.stButton > button {
            background-color: #004080 !important;
            color: white !important;
            border-radius: 8px !important;
            height: 3em;
            width: 100%;
            font-size: 16px;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #002b59 !important;
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
        menu_title="",
        options=["Home", "Predictor", "About"],
        icons=["house", "bar-chart", "info-circle"],
        default_index=1,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6"},
            "icon": {"color": "#004080", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#cbd9ec",
                "color": "#1e1e1e"
            },
            "nav-link-selected": {
                "background-color": "#cbd9ec",
                "color": "#000000"
            },
        },
    )

# ------------------------------------------------
# HOME TAB
# ------------------------------------------------
if selected == "Home":
    st.title("🏠 Welcome to the Employment Status Predictor")
    st.write("Use the navigation on the left to explore the app and predict employment status.")

# ------------------------------------------------
# PREDICTOR TAB
# ------------------------------------------------
elif selected == "Predictor":
    st.title("🤖 Employment Status Prediction")
    st.markdown("""
    Provide the relevant details below. The model will analyze your inputs and predict whether the individual is likely to be **Employed** or **Unemployed**.
    """)

    input_dict = {}
    for col in X_columns:
        if col in ['Matric', 'Degree', 'Diploma']:
            input_dict[col] = st.selectbox(f"{col} (Yes/No)", ['Yes', 'No'])
        elif col == 'Round':
            input_dict[col] = st.selectbox(col, ['1','2','3','4'])
        elif col == 'Sa_citizen':
            input_dict[col] = st.selectbox("Are you a SA Citizen?", ['Citizen', 'Non-citizen'])
        elif col == 'Gender':
            input_dict[col] = st.selectbox("Gender", ['Female', 'Male'])
        elif col == 'Highest_Education':
            input_dict[col] = st.selectbox("Highest Education", ['Degree', 'Diploma', 'Matric', 'None'])
        elif col in ['Math', 'Mathlit', 'Additional_lang', 'Home_lang', 'Science', 'Geography']:
            input_dict[col] = st.selectbox(col, ['0% - 49%', '50% - 79%','80% - 100%'])
        elif col == 'Province':
            input_dict[col] = st.selectbox(col, ['Gauteng' , 'Mpumalanga' , 'North West', 'Free State', 'Eastern Cape', 'Limpopo','KwaZulu-Natal','Northern Cape','Western Cape'])
        elif col == 'Status':
            input_dict[col] = st.selectbox(col, ['Unemployed', 'Studying','Wage employed','Self employed','Wage and self employed','Employment program','Other'])
        else:
            input_dict[col] = st.number_input(col, value=0)

    input_df = pd.DataFrame([input_dict])

    # Encode categoricals
    for col in input_df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        input_df[col] = le.fit(input_df[col]).transform(input_df[col])

    # Scale numeric
    input_df_scaled = scaler.transform(input_df)

    if st.button("Predict Employment Status"):
        pred = model.predict(input_df_scaled)[0]
        prob = model.predict_proba(input_df_scaled)[0][int(pred)]
        status = 'Unemployed' if pred == 1 else 'Employed'
        st.success(f"Prediction: *{status}*")
        st.info(f"Confidence: *{prob * 100:.2f}%*")

# ------------------------------------------------
# ABOUT TAB
# ------------------------------------------------
elif selected == "About":
    st.title("ℹ About This App")
    st.markdown("""
        This app predicts whether a person is employed or unemployed based on various demographic and academic factors.

        - **Built by:** Group Five Team  
        - **Model Used:** Logistic Regression  
        - **Technologies:** Python, Streamlit, Scikit-learn  
    """)
