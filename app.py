import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)
# -----------------------------
# LOAD DATA AND MODEL
# -----------------------------
df = pd.read_csv("processed_churn.csv")

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

X = df.drop("Churn", axis=1)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Customer Churn Prediction")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict Churn"]
)

# ==================================================
# DASHBOARD PAGE
# ==================================================

if page == "Dashboard":

    st.title("📊 Customer Churn Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Customers",
            len(df)
        )

    with col2:
        st.metric(
            "Churn Rate",
            f"{df['Churn'].mean()*100:.2f}%"
        )

    # -------------------------
    # Churn by Gender
    # -------------------------

    st.subheader("Churn by Gender")

    gender_churn = (
        df.groupby("gender_Male")["Churn"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        gender_churn,
        x="gender_Male",
        y="Churn",
        title="Churn Rate by Gender"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Contract Analysis
    # -------------------------

    st.subheader("Contract Analysis")

    contract_cols = [
        "Contract_One year",
        "Contract_Two year"
    ]

    contract_churn = (
        df.groupby(contract_cols)["Churn"]
        .mean()
        .reset_index()
    )

    st.dataframe(contract_churn)

    # -------------------------
    # Internet Service Analysis
    # -------------------------

    st.subheader("Internet Service Analysis")

    internet_cols = [
        "InternetService_Fiber optic",
        "InternetService_No"
    ]

    internet_churn = (
        df.groupby(internet_cols)["Churn"]
        .mean()
        .reset_index()
    )

    st.dataframe(internet_churn)

    # -------------------------
    # Monthly Charges Analysis
    # -------------------------

    st.subheader("Monthly Charges Analysis")

    fig = px.histogram(
        df,
        x="MonthlyCharges",
        color="Churn",
        title="Monthly Charges Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Feature Importance
    # -------------------------

    st.subheader("Top 10 Important Features")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# PREDICTION PAGE
# ==================================================

elif page == "Predict Churn":

    st.title("🤖 Predict Customer Churn")

    tenure = st.number_input(
        "Tenure (Months)",
        0,
        100,
        12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        0.0,
        1000.0,
        50.0
    )

    total_charges = st.number_input(
        "Total Charges",
        0.0,
        100000.0,
        500.0
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    if st.button("Predict Churn"):

        # Create blank row with all features

        input_df = pd.DataFrame(
            0,
            index=[0],
            columns=X.columns
        )

        input_df["SeniorCitizen"] = senior
        input_df["tenure"] = tenure
        input_df["MonthlyCharges"] = monthly_charges
        input_df["TotalCharges"] = total_charges

        input_df["gender_Male"] = (
            1 if gender == "Male" else 0
        )

        input_df["Partner_Yes"] = (
            1 if partner == "Yes" else 0
        )

        input_df["Dependents_Yes"] = (
            1 if dependents == "Yes" else 0
        )

        input_df["PhoneService_Yes"] = (
            1 if phone_service == "Yes" else 0
        )

        # Scale

        scaled_input = scaler.transform(
            input_df
        )

        prediction = model.predict(
            scaled_input
        )[0]

        probability = model.predict_proba(
            scaled_input
        )[0][1]

        st.subheader("Prediction Result")

        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )

        if prediction == 1:
            st.error(
                "⚠ Customer is likely to churn"
            )
        else:
            st.success(
                "✅ Customer is likely to stay"
            )

        if probability < 0.30:
            st.success("Low Risk")

        elif probability < 0.70:
            st.warning("Medium Risk")

        else:
            st.error("High Risk")
