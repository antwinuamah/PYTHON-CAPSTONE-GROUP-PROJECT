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
# CUSTOM STYLING
# ------------------------------------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #E3E7F7 !important;
        }
        html, body, [data-testid="stAppViewContainer"] > .main {
            background-color: white !important;
            color: black !important;
        }
        .stSelectbox > div,
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #E3E7F7 !important;
            border-radius: 8px;
        }
        input[type="number"] {
            background-color: #E3E7F7 !important;
            border-radius: 8px;
            padding: 0.4rem;
        }
        div[data-baseweb="slider"] > div > div > div:nth-child(2) {
            background: #4B0082 !important;
        }
        div[data-baseweb="slider"] > div > div > div:nth-child(3) {
            background: #e6e6e6 !important;
        }
        div[data-baseweb="slider"] [role="slider"] {
            background-color: #4B0082 !important;
        }
        div.stButton > button {
            background-color: #4B0082 !important;
            color: white !important;
            border-radius: 8px !important;
            height: 3em;
            width: auto;
            padding: 0.6rem 1.5rem;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #3a006b !important;
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
        menu_title=None,
        options=["Home", "Predictor", "About"],
        icons=["house", "bar-chart", "info-circle"],
        default_index=1,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#E6E6E6"},
            "icon": {"color": "#000000", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#007847",
                "color": "#007847"
            },
            "nav-link-selected": {
                "background-color": "#007847",
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
    st.write("Fill in the details below to predict employment status.")

    # Dynamic input form based on training columns
    input_dict = {}
    for col in X_columns:
        if col in ['Matric', 'Degree', 'Diploma']:
            input_dict[col] = st.selectbox(f"{col} (Yes/No)", ['Yes', 'No'])
        elif col in ['Round']:
            input_dict[col] = st.selectbox(col, ['1','2','3','4'])
        elif col == 'Sa_citizen':
            input_dict[col] = st.selectbox("Are you a SA Citizen?", ['Citizen', 'Non-citizen'])
        elif col == 'Gender':
            input_dict[col] = st.selectbox("Gender", ['Female', 'Male'])
        elif col == 'Highest_Education':
            input_dict[col] = st.selectbox("Highest Education", ['Degree', 'Diploma', 'Matric', 'None'])
        elif col in ['Math', 'Mathlit', 'Additional_lang', 'Home_lang', 'Science', 'Geography']:
            input_dict[col] = st.selectbox(col, ['0% - 49%', '50% - 79%','80% - 100%'])
        elif col in ['Province']:
            input_dict[col] = st.selectbox(col, ['Gauteng' , 'Mpumalanga' , 'North West', 'Free State', 'Eastern Cape', 'Limpopo','KwaZulu-Natal','Northern Cape','Western Cape'])
        elif col in ['Status']:
            input_dict[col] = st.selectbox(col, ['Unemployed', 'Studying','Wage employed','Self employed','Wage and self employed','Employment program','Other'])
        else:
            input_dict[col] = st.number_input(col, value=0)

    # Convert to DataFrame
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
        st.write(f"Confidence: *{prob * 100:.2f}%*")

# ------------------------------------------------
# ABOUT TAB
# ------------------------------------------------
elif selected == "About":
    st.title("ℹ About This App")
    st.markdown("""
        This Streamlit app predicts whether an individual is likely to be employed or unemployed
        based on socio-economic and educational inputs.

        *Built by:* Group Five Team  
        *Model:* Logistic Regression  
        *Tools:* Python, Streamlit, Scikit-learn
    """)
