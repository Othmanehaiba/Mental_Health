
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Mental Health Classification",
    page_icon="📊"
)

st.title("Student Mental Health Classification")

st.write(
    "Educational machine learning demonstration "
    "using student academic and lifestyle data."
)

model = joblib.load("models/best_model.pkl")

train_data = pd.read_csv("data/processed/train.csv")

X_train = train_data.drop(
    columns=["Mental_Health_Risk"]
)

st.subheader("Enter Student Information")

student_data = {}

for column in X_train.columns:

    if pd.api.types.is_numeric_dtype(X_train[column]):

        default_value = float(X_train[column].median())

        student_data[column] = st.number_input(
            column,
            value=default_value
        )

    else:

        options = X_train[column].dropna().unique().tolist()

        student_data[column] = st.selectbox(
            column,
            options
        )


if st.button("Predict"):

    input_data = pd.DataFrame(
        [student_data]
    )

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    st.success(
        f"Predicted class: {prediction[0]}"
    )