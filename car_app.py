import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Load the trained model and preprocessing components
model = joblib.load('car_price_prediction_model_2025-01-18_23-41-07.pkl')
processing_objects = joblib.load('preprocessing_objects.pkl')

label_encoders = processing_objects["label_encoders"]
scaler = processing_objects["scaler"]

# Set up the page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load and resize banner image
image = Image.open("assets/sell-car-hero-banner.png")
image = image.resize((350, 150))  # Resize image to make it smaller
st.sidebar.image(image, use_container_width=False)

# Page title
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: #2c3e50;">Car Price Prediction System</h1>
        <p style="font-size: 18px; color: #34495e;">Get an accurate estimate of your car's resale value instantly!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar input section
st.sidebar.header("Enter Car Details")
st.sidebar.markdown(
    """
    <div style="background-color: #f1f1f1; padding: 10px; border-radius: 8px;">
        <p style="text-align: center; color: #34495e; font-weight: bold;">Fill in the details below:</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Group inputs into a form
with st.sidebar.form("car_details_form"):
    mileage = st.number_input("Mileage (MPG)", min_value=0, max_value=500000, value=10000)
    standard_colour = st.text_input("Car Color", "Enter color")
    standard_make = st.text_input("Car Make", "Enter make")
    standard_model = st.text_input("Car Model", "Enter model name")
    year_of_registration = st.number_input("Year of Registration", min_value=1980, max_value=2024, value=2015)
    body_type = st.text_input("Body Type", "Enter body type")
    crossover_car_and_van = st.radio("Crossover Vehicle?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    fuel_type = st.text_input("Fuel Type", "Enter fuel type")
    submit_button = st.form_submit_button("Predict Price (₤)")

# Derived Fields
current_year = 2024
age_of_vehicle = current_year - year_of_registration

# Calculate condition of vehicle
condition_of_vehicle = "OLD" if year_of_registration <= 2013 else "MID-AGE" if year_of_registration <= 2017 else "NEW"

# Calculate usage of vehicle
usage_of_vehicle = "LOW" if mileage < 15129 else "AVERAGE" if mileage <= 57525 else "HIGH"


# Function to handle unseen labels
# def handle_unseen_labels(value, encoder):
#     try:
#         # Try to transform the value
#         return encoder.transform([value])[0]
#     except ValueError:
#         # Return a fallback value or handle it gracefully
#         st.warning(f"Unrecognized value: {value}. Using fallback value.")
#         return encoder.transform(["Other"])[0]  # Fallback to a common label
def handle_unseen_labels(value, encoder):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        fallback = encoder.classes_[0]
        st.warning(f"Unrecognized value: {value}. Using '{fallback}' instead.")
        return encoder.transform([fallback])[0]

# Prediction only triggered on button click
if submit_button:
    # Validate and preprocess inputs
    data_dict = {
        "mileage": mileage,
        "standard_colour": standard_colour,
        "standard_make": standard_make,
        "standard_model": standard_model,
        "year_of_registration": year_of_registration,
        "body_type": body_type,
        "crossover_car_and_van": crossover_car_and_van,
        "fuel_type": fuel_type,
        "age_of_vehicle": age_of_vehicle,
        "condition_of_vehicle": condition_of_vehicle,
        "usage_of_vehicle": usage_of_vehicle
    }

    # Preprocessing Function
    def preprocess_inputs(data_dict):
        categorical_columns = ['standard_colour', 'standard_make', 'standard_model', 
                               'body_type', 'fuel_type', 'condition_of_vehicle', 'usage_of_vehicle']
        for col in categorical_columns:
            # Use the handle_unseen_labels function to check for unseen labels
            data_dict[col] = handle_unseen_labels(data_dict[col], label_encoders[col])

        data_df = pd.DataFrame([data_dict])
        data_df = scaler.transform(data_df)
        return data_df

    try:
        # Preprocess and predict
        processed_data = preprocess_inputs(data_dict)
        predicted_price = model.predict(processed_data)
        
        # Display results
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px; background-color: #ecf0f1; padding: 20px; border-radius: 12px;">
                <h2 style="color: #27ae60;">Predicted Selling Price</h2>
                <p style="font-size: 36px; font-weight: bold; color: #e74c3c;">₤{predicted_price[0]:,.2f}</p>
                <p style="color: #7f8c8d;">Car Age: {age_of_vehicle} years</p>
                <p style="color: #7f8c8d;">Condition: {condition_of_vehicle}</p>
                <p style="color: #7f8c8d;">Usage: {usage_of_vehicle}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Footer Section
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px;">
        <p style="color: #95a5a6;">Developed by Ruchi Rathod with ❤️ using Streamlit </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Lets use following test data for prediction: 
# 64000.0 -> Grey -> Land Rover -> Range Rover Sport -> 2015 -> SUV -> False -> Diesel -> Price: 26995 