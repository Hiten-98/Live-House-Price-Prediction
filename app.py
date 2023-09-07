import numpy as np
import pandas as pd
import streamlit as st
import pickle

st.set_page_config(
    page_title = 'House Price Prediction',
    page_icon = '✅',
    layout = 'wide'
)
st.title("Predict your 🏠 price")
st.header("Welcome to Gurugram's Premier House Price Prediction Tool")
st.write('---')

pickle_in = open('rf.pkl', 'rb')
rf = pickle.load(pickle_in)

left_column, right_column = st.columns(2)
with left_column:
    property_type = st.selectbox('Property Type:', ['flat','house'])
    price_per_sqft = st.number_input("Price per Sqft", value=5000)
    bedRoom = st.sidebar.slider("Bedrooms", min_value=1, max_value=10, value=3)
    furnishing_type = st.selectbox("Furnishing Type", ['No furnished', 'Semi Furnished', 'Full Furnished'])

with right_column:
    agePossession = st.selectbox("Age of Possession (Years)", ['relatively new  ','new Property  ','moderately old ','undefined ','old property ','under construction'])
    floor_category = st.selectbox("Floor Category", ['High Floor', 'Low Floor', 'Mid Floor', None])

bathroom = st.sidebar.slider("Bathrooms", min_value=1, max_value=10, value=2)
balcony = st.sidebar.slider("Balconies", min_value=0, max_value=5, value=1)
super_built_up_area = st.sidebar.slider("Super Built-up Area (sqft)", min_value=0, max_value=5000, value=300)
built_up_area = st.sidebar.slider("Built-up Area (sqft)", min_value=0, max_value=10000, value=400)
carpet_area = st.sidebar.slider("Carpet Area (sqft)", min_value=0, max_value=10000, value=500)

left_column1, right_column1 = st.columns(2)
with left_column1:
    study_room = st.checkbox("Study Room")
    servant_room = st.checkbox("Servant Room")
    store_room = st.checkbox("Store Room")

with right_column1:
    pooja_room = st.checkbox("Pooja Room")
    others = st.checkbox("Others")

def preprocess_input(property_type, agePossession, floor_category, furnishing_type):
    # Encode categorical features
    property_type_encoded = 0 if property_type == 'flat' else 1

    age_possession_mapping = {
        'relatively new': 0,
        'new Property': 1,
        'moderately old': 2,
        'undefined': 3,
        'old property': 4,
        'under construction': 5
    }
    age_possession_encoded = age_possession_mapping.get(agePossession, 0)  # Default to 0 if not found

    furnishing_type_mapping = {
        'No furnished': 0,
        'Semi Furnished': 1,
        'Full Furnished': 2
    }
    furnishing_type_encoded = furnishing_type_mapping.get(furnishing_type, 0)  # Default to 0 if not found
    floor_category_mapping = {
        'High Floor': 0,
        'Low Floor': 1,
        'Mid Floor': 2
    }
    floor_category_encoded = floor_category_mapping.get(floor_category, -1)  # Default to -1 if not found

    return property_type_encoded, age_possession_encoded, floor_category_encoded, furnishing_type_encoded


def prediction(property_type_encoded, price_per_sqft, bedRoom, bathroom, balcony, age_possession_encoded,
               super_built_up_area, built_up_area, carpet_area, study_room, servant_room, store_room, pooja_room,
               others, furnishing_type_encoded, floor_category_encoded):
    prediction = rf.predict(
        [[property_type_encoded, price_per_sqft, bedRoom, bathroom, balcony, age_possession_encoded,
          super_built_up_area, built_up_area, carpet_area, study_room, servant_room, store_room, pooja_room, others,
          furnishing_type_encoded, floor_category_encoded]])
    return prediction

def format_price(price):
    if price < 1:
        return f"{price * 100:.0f} lakhs"
    else:
        return f"{price:.2f} crores"

# Preprocess input data
property_type_encoded, age_possession_encoded, floor_category_encoded, furnishing_type_encoded = preprocess_input(
    property_type, agePossession, floor_category, furnishing_type)



if st.button('Predict Price'):
    price = prediction(property_type_encoded, price_per_sqft, bedRoom, bathroom, balcony, age_possession_encoded,
                       super_built_up_area, built_up_area, carpet_area, study_room, servant_room, store_room, pooja_room,
                       others, furnishing_type_encoded, floor_category_encoded)
    formatted_price = format_price(price[0])
    st.success(f'The predicted price of the house is ₹{formatted_price} ')



