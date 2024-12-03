import streamlit as st
import pandas as pd

# Load the default dataset
DATASET_PATH = "youth_unemployment_dataset.csv"  # Replace with the path to the dataset
try:
    base_df = pd.read_csv(DATASET_PATH)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

def add_data():
    # Form for adding new data
    with st.form("Add Data Form", clear_on_submit=True):
        # User inputs
        col1, col2 = st.columns(2)
        age = col1.number_input("Age", min_value=16, max_value=100, step=1)
        gender = col2.selectbox("Gender", ["Male", "Female"])

        col3, col4 = st.columns(2)
        education_level = col3.selectbox("Education Level", base_df["education_level"].unique())
        education_mismatch = col4.selectbox("Education Mismatch", [True, False])

        col5, col6 = st.columns(2)
        sector_of_interest = col5.selectbox("Sector of Interest", base_df["sector_of_interest"].unique())
        current_employment_sector = col6.selectbox("Current Employment Sector", base_df["current_employment_sector"].unique())

        col7, col8 = st.columns(2)
        formal_informal = col7.selectbox("Formal or Informal Employment", base_df["formal_informal"].unique())
        region = col8.selectbox("Region", base_df["region"].unique())

        col9, col10 = st.columns(2)
        location_type = col9.selectbox("Location Type", base_df["location_type"].unique())
        monthly_income = col10.number_input("Monthly Income (RWF)", min_value=0.0, step=1000.0)

        # Additional inputs
        unemployment_duration = st.number_input("Unemployment Duration (Months)", min_value=0, step=1)
        digital_skills_level = st.selectbox("Digital Skills Level", base_df["digital_skills_level"].unique())
        technical_skills = st.text_area("Technical Skills (comma-separated)", "e.g., Coding, Data Analysis")
        training_participation = st.selectbox("Training Participation", [True, False])
        program_type = st.selectbox("Program Type", base_df["program_type"].unique())
        household_income = st.number_input("Household Income (RWF)", min_value=0.0, step=1000.0)
        household_size = st.number_input("Household Size", min_value=1, step=1)
        employment_outcome = st.selectbox("Employment Outcome", [True, False])
        intervention_effectiveness = st.selectbox("Intervention Effectiveness", base_df["intervention_effectiveness"].unique())
        employment_duration_post_intervention = st.number_input("Employment Duration Post Intervention (Months)", min_value=0, step=1)
        youth_unemployment_rate = st.number_input("Youth Unemployment Rate (%)", min_value=0.0, step=0.1)
        urban_rural_employment_rate = st.number_input("Urban-Rural Employment Rate (%)", min_value=0.0, step=0.1)

        # Submit button
        btn = st.form_submit_button("Save Data To Dataset")

        # Handle form submission
        if btn:
            if not technical_skills.strip():
                st.warning("Technical skills cannot be empty.")
                return False

            # Add new row
            new_row = {
                "age": age,
                "gender": gender,
                "education_level": education_level,
                "education_mismatch": education_mismatch,
                "sector_of_interest": sector_of_interest,
                "current_employment_sector": current_employment_sector,
                "formal_informal": formal_informal,
                "region": region,
                "location_type": location_type,
                "monthly_income": monthly_income,
                "unemployment_duration": unemployment_duration,
                "digital_skills_level": digital_skills_level,
                "technical_skills": technical_skills.split(","),
                "training_participation": training_participation,
                "program_type": program_type,
                "household_income": household_income,
                "household_size": household_size,
                "employment_outcome": employment_outcome,
                "intervention_effectiveness": intervention_effectiveness,
                "employment_duration_post_intervention": employment_duration_post_intervention,
                "youth_unemployment_rate": youth_unemployment_rate,
                "urban_rural_employment_rate": urban_rural_employment_rate,
            }
            # Append new data to the base dataframe
            updated_df = pd.concat([base_df, pd.DataFrame([new_row])], ignore_index=True)

            # Save to file
            try:
                updated_df.to_csv(DATASET_PATH, index=False)
                st.success("Data added successfully!")
            except Exception as e:
                st.error(f"Error saving dataset: {e}")
                return False

# Run the app
st.title("Youth Data Management")
st.write("Use the form below to add new data to the dataset.")
add_data()
