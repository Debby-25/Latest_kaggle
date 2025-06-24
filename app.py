import streamlit as st
import pandas as pd
import pickle

# --- Load model and preprocessors ---
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("spaceship_model.pkl", "rb"))
    scaler = pickle.load(open("spaceship_scaler.pkl", "rb"))
    columns = pickle.load(open("spaceship_columns.pkl", "rb"))
    return model, scaler, columns

model, scaler, columns = load_artifacts()

# --- App Layout ---
st.title("🚀 Spaceship Titanic Passenger Prediction")
st.markdown("Predict whether a passenger was transported to another dimension.")

# --- User Input Section ---
st.header("🧍 Enter Passenger Details")

# Categorical Inputs
home_planet = st.selectbox("Home Planet", ["Earth", "Europa", "Mars"])
cryosleep = st.selectbox("CryoSleep Status", ["True", "False"])
destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
vip = st.selectbox("VIP Status", ["True", "False"])

# Numerical Inputs
age = st.slider("Age", 0, 100, 30)
room_service = st.number_input("Room Service ($)", min_value=0.0, value=0.0)
food_court = st.number_input("Food Court ($)", min_value=0.0, value=0.0)
shopping_mall = st.number_input("Shopping Mall ($)", min_value=0.0, value=0.0)
spa = st.number_input("Spa ($)", min_value=0.0, value=0.0)
vr_deck = st.number_input("VR Deck ($)", min_value=0.0, value=0.0)

# --- Data Preprocessing ---
# Build input dataframe
input_df = pd.DataFrame({
    "HomePlanet": [home_planet],
    "CryoSleep": [cryosleep == "True"],
    "Destination": [destination],
    "VIP": [vip == "True"],
    "Age": [age],
    "RoomService": [room_service],
    "FoodCourt": [food_court],
    "ShoppingMall": [shopping_mall],
    "Spa": [spa],
    "VRDeck": [vr_deck],
})

# Create TotalSpending feature
input_df["TotalSpending"] = input_df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)

# One-hot encode using same columns as training
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

# Scale numeric features
numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpending']
input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

### Make Prediction 
if st.button("🚀 Predict Transported"):
    prediction = model.predict(input_encoded)[0]
    st.success("✅ Transported" if prediction else "❌ Not Transported")
