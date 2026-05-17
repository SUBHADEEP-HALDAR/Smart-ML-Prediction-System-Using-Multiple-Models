import streamlit as st
import numpy as np
import pickle

try:
    house_model = pickle.load(open('models/house_model.pkl', 'rb'))
    bmi_model = pickle.load(open('models/bmi_model.pkl', 'rb'))
    insurance_model = pickle.load(open('models/insurance_model.pkl', 'rb'))
    titanic_model = pickle.load(open('models/titanic_model.pkl', 'rb'))
except:
    st.warning(" Models not found. Using dummy predictions.")


st.title("🔥 Smart ML Prediction System")


option = st.sidebar.selectbox(
    "Select Prediction Model",
    ("House Price", "BMI", "Insurance", "Titanic Survival")
)

# 🏠 HOUSE PRICE

if option == "House Price":
    st.header("🏠 House Price Prediction")

    area = st.number_input("Enter Area (sq ft)")
    bedrooms = st.number_input("Bedrooms", 1, 10)

    if st.button("Predict Price"):
        try:
            result = house_model.predict([[area, bedrooms]])
            st.success(f"💰 Predicted Price: {result[0]:,.2f}")
        except:
            st.success(f"💰 Dummy Price: {area * 3000}")

# 🧍 BMI

elif option == "BMI":
    st.header("🧍 BMI Calculator")

    height = st.number_input("Height (meters)")
    weight = st.number_input("Weight (kg)")

    if st.button("Calculate BMI"):
        bmi = weight / (height ** 2)

        st.success(f"📊 BMI: {bmi:.2f}")

        if bmi < 18.5:
            st.warning("Underweight")
        elif bmi < 25:
            st.success("Normal")
        elif bmi < 30:
            st.warning("Overweight")
        else:
            st.error("Obese")

# 💰 INSURANCE

elif option == "Insurance":
    st.header("💰 Insurance Cost Prediction")

    age = st.number_input("Age")
    bmi = st.number_input("BMI")
    children = st.number_input("Children", 0, 10)
    smoker = st.selectbox("Smoker", ("No", "Yes"))

    smoker_val = 1 if smoker == "Yes" else 0

    if st.button("Predict Insurance Cost"):
        try:
            result = insurance_model.predict([[age, bmi, children, smoker_val]])
            st.success(f"💵 Cost: {result[0]:,.2f}")
        except:
            st.success(f"💵 Dummy Cost: {(age + bmi * 2) * 100}")


# 🚢 TITANIC

elif option == "Titanic Survival":
    st.header("🚢 Titanic Survival Prediction")

    pclass = st.selectbox("Passenger Class", (1, 2, 3))
    sex = st.selectbox("Sex", ("Male", "Female"))
    age = st.number_input("Age")

    sex_val = 1 if sex == "Male" else 0

    if st.button("Predict Survival"):
        try:
            result = titanic_model.predict([[pclass, sex_val, age]])

            if result[0] == 1:
                st.success("✅ Survived")
            else:
                st.error("❌ Did Not Survive")
        except:
            st.success("⚠️ Dummy Result: Survived")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning")