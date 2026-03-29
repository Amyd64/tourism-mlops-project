import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="Amyd64/tourism-package-model", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase the **Wellness Tourism Package**
based on their profile and interaction details.
Please enter the customer details below to get a prediction.
""")

# User input
type_of_contact   = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation        = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender            = st.selectbox("Gender", ["Male", "Female"])
product_pitched   = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
marital_status    = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation       = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

city_tier              = st.selectbox("City Tier", [1, 2, 3])
passport               = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
own_car                = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
pitch_satisfaction     = st.slider("Pitch Satisfaction Score", 1, 5, 3)

age                    = st.number_input("Age", min_value=18, max_value=80, value=35)
duration_of_pitch      = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10)
n_person_visiting      = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
n_followups            = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=3)
preferred_property_star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
n_trips                = st.number_input("Number of Trips per Year", min_value=1, max_value=20, value=3)
n_children_visiting    = st.number_input("Number of Children Visiting (< 5 yrs)", min_value=0, max_value=5, value=0)
monthly_income         = st.number_input("Monthly Income (INR)", min_value=1000, max_value=100000, value=20000, step=500)

# Label encoding maps (matching training)
contact_map    = {"Company Invited": 0, "Self Enquiry": 1}
occ_map        = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
gender_map     = {"Female": 0, "Male": 1}
product_map    = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
marital_map    = {"Divorced": 0, "Married": 1, "Single": 2}
desig_map      = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age':                    age,
    'TypeofContact':          contact_map[type_of_contact],
    'CityTier':               city_tier,
    'DurationOfPitch':        duration_of_pitch,
    'Occupation':             occ_map[occupation],
    'Gender':                 gender_map[gender],
    'NumberOfPersonVisiting': n_person_visiting,
    'NumberOfFollowups':      n_followups,
    'ProductPitched':         product_map[product_pitched],
    'PreferredPropertyStar':  preferred_property_star,
    'MaritalStatus':          marital_map[marital_status],
    'NumberOfTrips':          n_trips,
    'Passport':               passport,
    'PitchSatisfactionScore': pitch_satisfaction,
    'OwnCar':                 own_car,
    'NumberOfChildrenVisiting': n_children_visiting,
    'Designation':            desig_map[designation],
    'MonthlyIncome':          monthly_income
}])

# Predict button
if st.button("Predict"):
    prediction   = model.predict(input_data)[0]
    probability  = model.predict_proba(input_data)[0][1]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"✅ Customer is **LIKELY** to purchase the Wellness Tourism Package  (Probability: {probability:.1%})")
    else:
        st.error(f"❌ Customer is **UNLIKELY** to purchase the Wellness Tourism Package  (Probability: {probability:.1%})")
