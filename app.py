import streamlit as st 
import joblib 
import pandas as pd 
model = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")
st.title("Disease Prediction Using Machine Learing ")
st.write("enter symptoms below to predict the disease")
st.subheader("Data analysis")

st.image("output1.png", caption =  "feature importance")
st.image("output2.png", caption = "multiclass disease prediction using confusion matrix")
st.image("output3.png", caption = "dataset class balance")
st.image("output4.png", caption = "distribution of binary symptoms accross dataset")

symptoms = pd.read_csv("symptoms.csv")["Symptom"].tolist()


patient_data = []

for symptom in symptoms:
    value = st.selectbox(symptom.replace("_", " ").title(), [0, 1])
    patient_data.append(value)
if st.button("Predict"):
    prediction = model.predict([patient_data])
    disease = le.inverse_transform(prediction)
    st.success(f"Predicted Disease: {disease[0]}") 