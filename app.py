import streamlit as st
import pandas as pd
import joblib

model = joblib.load("rfiris.pkl")

st.title("IRIS FLOWER CLASSIFICATION APPLICATION")
st.write("Predict the species of an Iris Flower Using a Random Forest Model")

with st.form("iris_form"):

    st.subheader("Enter Flower Measurement")

    sepal_length = st.number_input(
        "sepal_length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=5.1
    )

    sepal_width = st.number_input(
        "sepal_width (cm)",
        min_value=2.0,
        max_value=5.0,
        value=3.5
    )

    petal_length = st.number_input(
        "petal_length (cm)",
        min_value=1.0,
        max_value=7.0,
        value=1.4
    )

    petal_width = st.number_input(
        "petal_width (cm)",
        min_value=0.1,
        max_value=2.5,
        value=0.2
    )

    submit_button = st.form_submit_button("Predict")

if submit_button:
    input_data = pd.DataFrame({
        "sepal length (cm)": [sepal_length],
        "sepal width (cm)": [sepal_width],
        "petal length (cm)": [petal_length],
        "petal width (cm)": [petal_width]
    })

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")
    st.success(f"Predicted species: {prediction[0]}")




