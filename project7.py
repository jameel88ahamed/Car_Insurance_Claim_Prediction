#REQUIRED LIBRARIES
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np

def double_log_transform(x):
    return np.log1p(np.log1p(x))

def triple_log_transform(x):
    return np.log1p(np.log1p(np.log1p(x)))

st.set_page_config(page_title= "Car Insurance Claim Prediction| By Jameel",
                   layout= "wide",
                   initial_sidebar_state= "expanded",
                   menu_items={'About': """# This Application is created by *Jameel*!"""})
st.markdown("<h1 style='text-align: center; color: #E5F246;'>CAR INSURANCE CLAIM PREDICTION</h1>", unsafe_allow_html=True)

# SETTING-UP BACKGROUND IMAGE
def setting_bg():
    st.markdown(f""" <style>.stApp {{
                        background: url("https://www.solidbackgrounds.com/images/2560x1600/2560x1600-true-blue-solid-color-background.jpg");
                        background-size: cover}}
                     </style>""",unsafe_allow_html=True)    
setting_bg()

with st.sidebar:
    Option = option_menu('Menu', ["Home","Predictions"], 
                        icons=["house-door-fill","gear"],
                        default_index=0,
                        styles={"nav-link": {"font-size": "30px", "text-align": "centre", "margin": "0px", "--hover-color": "#0C5991"},
                                "icon": {"font-size": "25px"},
                                "container" : {"max-width": "3000px"},
                                "nav-link-selected": {"background-color": "#159BFF"}})
    
if Option=="Home":
    st.markdown("## <span style='color:#E5F246'>**Technologies Used :**</span> Python, Streamlit, Pandas, Matplotlib.pyplot, Scikit-learn, Gradient Boosting Regressor, MSE, RMSE, R2-Score, Model Deployment, Pickling", unsafe_allow_html=True)
    st.markdown("## <span style='color:#E5F246'>**Description :**</span> This project aims to develop a machine learning model and deploy it as a user-friendly online application to predict car insurance claims. The predictive model will utilize historical data on car insurance policies and claims, aiming to assist insurance companies in assessing the likelihood of a claim being filed. Various factors influence the likelihood of a claim, such as the policy holder's age, no of kids, their home value, income, education, occupation, travel time,  car make and model, claim frequency, claim flag and urbanicity. Providing insurers with an estimated claim probability based on these factors is one way a predictive model can help address these challenges.", unsafe_allow_html=True)

elif Option=="Predictions":
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('rf_model.pkl', 'rb') as f:
        RF_Model = pickle.load(f)

    # Define the input fields
    col1,col2,col3 = st.columns([5,2,5])
    with col1:
        kidsdriv = st.number_input('Enter the number of kids who drive', min_value=0, step=1)
        age = st.number_input('Enter your age', min_value=0, step=1)
        homekids = st.number_input('Enter number of kids in your home', min_value=0, step=1)
        yoj = st.number_input('Enter your years of experience in job', min_value=0, step=1)
        income = st.number_input('Enter your income (in dollars)', min_value=0.0, step=0.01)
        parent1 = st.selectbox('Select Parent1 Status', ['No', 'Yes'])
        home_val = st.number_input('Enter your home value (in dollars)', min_value=0.0, step=0.01)
        mstatus = st.selectbox('Select Marital Status', ['z_No', 'Yes'])
        gender = st.selectbox('Select Gender', ['M', 'z_F'])
        education = st.selectbox('Select Max Education', ['PhD', 'z_High School', 'Bachelors', '<High School', 'Masters'])
        occupation = st.selectbox('Select Occupation Type', ['Professional', 'z_Blue Collar', 'Manager', 'Clerical', 'Doctor', 'Lawyer', 'Home Maker', 'Student'])
        travtime = st.number_input('Enter travel time value', min_value=0, step=1)

    with col3:
        car_use = st.selectbox('Select type of Car use', ['Private', 'Commercial'])
        tif = st.number_input('Select value of tenure in force', min_value=0, step=1)
        car_type = st.selectbox('Select Car Type', ['Minivan', 'Van', 'z_SUV', 'Sports Car', 'Panel Truck', 'Pickup'])
        red_car = st.selectbox('Is it Red Car?', ['yes', 'no'])
        oldclaim = st.number_input('Enter value of old claim (in dollars)', min_value=0.0, step=0.01)
        clm_freq = st.number_input('Enter claim frequency', min_value=0, step=1)
        revoked = st.selectbox('Select Revoked status', ['Yes', 'No'])
        mvr_pts = st.number_input('Enter your motor vehicle record points', min_value=0, step=1)
        car_age = st.number_input('Enter your car age', min_value=0, step=1)
        claim_flag = st.number_input('Enter claim flag', min_value=0, step=1)
        urbanicity = st.selectbox('Select the Status of Urbanicity', ['Highly Urban/ Urban', 'z_Highly Rural/ Rural'])

        #Prediction button
        submit_button = st.button(label="PREDICT CLAIM AMOUNT")
        st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #04B027;
                color: white;
                width: 100%;
            }
            </style>
        """, unsafe_allow_html=True)

    if submit_button:
        # Collect user inputs into a single list
        user_input = [kidsdriv, age, homekids, yoj, income, parent1, home_val, mstatus, gender,
                    education, occupation, travtime, car_use, tif, car_type, red_car, oldclaim,
                    clm_freq, revoked, mvr_pts, car_age, claim_flag, urbanicity]

        user_input_transformed = user_input.copy()
        user_input_transformed[2] = double_log_transform(user_input_transformed[2])
        user_input_transformed[3] = np.sqrt(user_input_transformed[3])
        user_input_transformed[4] = np.log1p(user_input_transformed[4])
        user_input_transformed[6] = np.sqrt(user_input_transformed[6])
        user_input_transformed[11] = np.sqrt(user_input_transformed[11])
        user_input_transformed[13] = double_log_transform(user_input_transformed[13])
        user_input_transformed[16] = triple_log_transform(user_input_transformed[16])
        user_input_transformed[17] = double_log_transform(user_input_transformed[17])
        user_input_transformed[19] = double_log_transform(user_input_transformed[19])
        user_input_transformed[20] = np.sqrt(user_input_transformed[20])
        user_input_transformed[21] = double_log_transform(user_input_transformed[21])

        # Encode the categorical variables
        parent1_mapping = {'Yes': 1, 'No': 0}
        mstatus_mapping = {'Yes': 0, 'z_No': 1}
        gender_mapping = {'M': 0, 'z_F': 1}
        car_use_mapping = {'Private': 1, 'Commercial': 0}
        red_car_mapping = {'yes': 1, 'no': 0}
        revoked_mapping = {'Yes': 1, 'No': 0}
        urbanicity_mapping = {'Highly Urban/ Urban': 0, 'z_Highly Rural/ Rural': 1}

        education_mapping = {
        "<High School": 0,
        "z_High School": 1,
        "Bachelors": 2,
        "Masters": 3,
        "PhD": 4
        }

        occupation_mapping = {
        "Home Maker": 0,
        "Student": 1,
        "z_Blue Collar": 2,
        "Clerical": 3,
        "Manager": 4,
        "Professional": 5,
        "Lawyer": 6,
        "Doctor": 7
        }

        car_type_mapping = {
        "Panel Truck": 0,
        "Van": 1,
        "Pickup": 2,
        "Minivan": 3,
        "z_SUV": 4,
        "Sports Car": 5
        }

        user_input_transformed[5] = parent1_mapping[user_input_transformed[5]]
        user_input_transformed[7] = mstatus_mapping[user_input_transformed[7]]
        user_input_transformed[8] = gender_mapping[user_input_transformed[8]]
        user_input_transformed[12] = car_use_mapping[user_input_transformed[12]]
        user_input_transformed[15] = red_car_mapping[user_input_transformed[15]]
        user_input_transformed[18] = revoked_mapping[user_input_transformed[18]]
        user_input_transformed[22] = urbanicity_mapping[user_input_transformed[22]]

        # Encode the categorical variables
        # Map the categorical variables for OrdinalEncoders
        user_input_transformed[9] = education_mapping[user_input_transformed[9]]
        user_input_transformed[10] = occupation_mapping[user_input_transformed[10]]
        user_input_transformed[14] = car_type_mapping[user_input_transformed[14]]

        # Feature scaling
        user_input_scaled = scaler.transform([user_input_transformed])

        # Predict the claim amount
        prediction_log_transformed = RF_Model.predict(user_input_scaled)

        # Inverse transform the prediction
        prediction = np.expm1(np.expm1(prediction_log_transformed[0]))

        # Display the prediction
        st.markdown('<h2 style="color:#E5F246; display: inline; font-size: 36px;">Predicted Claim Amount: {:.2f}</h2>'.format(prediction), unsafe_allow_html=True)