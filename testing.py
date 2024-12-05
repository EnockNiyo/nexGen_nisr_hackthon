import pandas as pd
import streamlit as st
import plotly.express as px

# File paths
NISR_DATASET_PATH = "nisr_dataset.csv"
YOUTH_DATASET_PATH = "youth_unemployment_dataset.csv"

# Selected columns from NISR dataset based on the project's focus
NISR_SELECTED_COLUMNS = [
    'age3', 'province', 'code_dis', 'gender',
    'employment16', 'UR1', 'LUUR', 'LFPR', 'YUR1',
    'attained', 'education_level', 'B02A', 'B02B',
    'I06A', 'I04', 'hhsize', 'weight2'
]

# Selected columns from Youth dataset based on the project's focus
YOUTH_SELECTED_COLUMNS = [
    'age', 'gender', 'region', 'education_level', 'employment_outcome',
    'monthly_income', 'training_participation'
]


# Function to load and preprocess datasets
@st.cache_data
def load_and_select_columns(file_path, selected_columns, encoding='latin1'):
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        # Keep only the columns that exist in the dataset
        existing_columns = [col for col in selected_columns if col in df.columns]
        if not existing_columns:
            raise ValueError(f"No matching columns found in {file_path}.")
        df = df[existing_columns]
        return df
    except Exception as e:
        st.error(f"Error loading dataset from {file_path}: {e}")
        return None


# Load datasets
nisr_df = load_and_select_columns(NISR_DATASET_PATH, NISR_SELECTED_COLUMNS)
youth_df = load_and_select_columns(YOUTH_DATASET_PATH, YOUTH_SELECTED_COLUMNS)

# Combine datasets
if nisr_df is not None and youth_df is not None:
    # Optionally rename columns for consistency
    nisr_df.rename(columns={'YUR1': 'youth_unemployment_rate'}, inplace=True)

    # Combine datasets (e.g., concatenation or other logic)
    combined_df = pd.concat([youth_df, nisr_df], axis=1)

    # Handle missing values
    combined_df.fillna("Unknown", inplace=True)

    # Display combined dataset
    st.write("### Combined Dataset Preview")
    st.dataframe(combined_df.head())

    # Save combined dataset
    combined_df.to_csv("combined_dataset.csv", index=False)
    st.success("Combined dataset saved as 'combined_dataset.csv'.")

    # --- Visualizations ---
    st.markdown("## Visualizations and Insights")

    # Example 1: Youth Unemployment Rate Distribution - Bar chart
    if 'youth_unemployment_rate' in combined_df.columns:
        st.subheader("Youth Unemployment Rate Distribution")
        unemployment_counts = combined_df['youth_unemployment_rate'].value_counts()
        unemployment_fig = px.bar(
            x=unemployment_counts.index,
            y=unemployment_counts.values,
            labels={'x': 'Youth Unemployment Rate (%)', 'y': 'Count'},
            title="Youth Unemployment Rate Distribution"
        )
        st.plotly_chart(unemployment_fig)

    # Example 2: Work Hours Distribution for Main and Secondary Jobs (C11A and C11B)
    if 'C11A' in combined_df.columns and 'C11B' in combined_df.columns:
        st.subheader("Work Hours Distribution (Main vs Secondary Jobs)")

        # Create a new DataFrame with work hours for main and secondary jobs
        work_hours_df = pd.DataFrame({
            'Work Hours (Main Job)': combined_df['C11A'],
            'Work Hours (Secondary Job)': combined_df['C11B']
        })

        # Melt the DataFrame to make it suitable for a line chart
        work_hours_melted = work_hours_df.melt(var_name='Job Type', value_name='Hours Worked')

        # Plot work hours for main and secondary jobs using a line chart
        work_hours_fig = px.line(
            work_hours_melted,
            x=work_hours_melted.index,
            y='Hours Worked',
            color='Job Type',
            title="Distribution of Work Hours for Main and Secondary Jobs",
            labels={'Hours Worked': 'Number of Hours', 'Job Type': 'Job Type'}
        )
        st.plotly_chart(work_hours_fig)

    # Example 3: Monthly Income vs Job Type (Main vs Secondary Jobs)
    if 'monthly_income' in combined_df.columns:
        st.subheader("Monthly Income Distribution by Job Type (Main vs Secondary Jobs)")

        # Create a DataFrame for monthly income with job types
        income_df = pd.DataFrame({
            'Job Type': ['Main Job'] * len(combined_df) + ['Secondary Job'] * len(combined_df),
            'Monthly Income': list(combined_df['monthly_income']) + list(combined_df['monthly_income'])
        })

        # Plot using a line chart
        income_fig = px.line(
            income_df,
            x='Job Type',
            y='Monthly Income',
            title="Monthly Income Distribution by Job Type",
            labels={'Monthly Income': 'Monthly Income (RWF)', 'Job Type': 'Job Type'}
        )
        st.plotly_chart(income_fig)

    # Example 4: Age Distribution by Gender
    if 'age3' in combined_df.columns and 'gender' in combined_df.columns:
        st.subheader("Age Distribution by Gender")

        # Group the data by age3 (age group) and gender, and count occurrences
        age_gender_distribution = pd.crosstab(combined_df['age3'], combined_df['gender'])

        # Create a stacked bar chart using plotly.express
        age_gender_fig = px.bar(
            age_gender_distribution,
            x=age_gender_distribution.index,
            y=age_gender_distribution.columns,
            title="Age Distribution by Gender",
            labels={'x': 'Age Group', 'y': 'Count'},
            barmode='stack'  # Stacked bar chart to show the gender distribution within each age group
        )
        st.plotly_chart(age_gender_fig)

    # Example 5: Employment Rate by Region
    if 'employment16' in combined_df.columns and 'province' in combined_df.columns:
        st.subheader("Employment Rate by Region")
        employment_region = pd.crosstab(combined_df['province'], combined_df['employment16'])
        employment_region_fig = px.bar(
            employment_region,
            x=employment_region.index,
            y=employment_region.values,
            labels={'x': 'Region', 'y': 'Employment Rate'},
            title="Employment Rate by Region",
            barmode='group'
        )
        st.plotly_chart(employment_region_fig)

    # Example 6: Education Level vs Employment Outcome
    if 'education_level' in combined_df.columns and 'employment_outcome' in combined_df.columns:
        st.subheader("Education Level vs Employment Outcome")

        # Count the occurrences of each combination of education level and employment outcome
        education_employment = pd.crosstab(combined_df['education_level'], combined_df['employment_outcome'])

        # Create a grouped bar chart
        education_employment_fig = px.bar(
            education_employment,
            x=education_employment.index,  # Education level on the x-axis
            y=education_employment.columns,  # Employment outcome counts as y-values
            labels={'x': 'Education Level', 'y': 'Employment Outcome'},
            title="Education Level vs Employment Outcome",
            barmode='group'  # Group bars by education level and employment outcome
        )
        st.plotly_chart(education_employment_fig)

else:
    st.error("One or both datasets could not be loaded.")
