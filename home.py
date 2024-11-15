import datetime
import time

import streamlit as st
import pandas as pd
import plotly.express as px
from pygments.lexers import go
from sklearn.ensemble import RandomForestClassifier
from streamlit_option_menu import option_menu  # Import the option_menu function
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from datetime import datetime
import numpy as np
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv('youth_unemployment_dataset.csv')

# Set up Streamlit layout
st.set_page_config(page_title="Youth Unemployment Analytics Dashboard", page_icon="📊", layout="wide")

# Title
st.markdown(
    "<h1 style='text-align: center; color: #333; margin-top: -45px;'>Youth Unemployment Analytics Dashboard</h1>",
    unsafe_allow_html=True)
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("Filters")
age_range = st.sidebar.slider("Select Age Range", 16, 25, (16, 25))
gender = st.sidebar.selectbox("Select Gender", ["All", "Male", "Female"])
regions = st.sidebar.multiselect("Select Region(s)", options=df['region'].unique(), default=df['region'].unique())
education_levels = st.sidebar.multiselect("Select Education Level(s)", options=df['education_level'].unique(),
                                          default=df['education_level'].unique())

# Apply Filters
filtered_df = df[
    (df['age'] >= age_range[0]) &
    (df['age'] <= age_range[1]) &
    (df['region'].isin(regions)) &
    (df['education_level'].isin(education_levels))
    ]
if gender != "All":
    filtered_df = filtered_df[filtered_df['gender'] == gender]

# Display filtered data summary
st.write("Data Summary:", filtered_df.describe())

# Top metrics
# Custom CSS for card styling and Font Awesome
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

        .metric-card {
            background-color: white;
            border-radius: 0.5rem;
            padding: 1.5rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            height: 100%;
        }

        .metric-card .icon {
            font-size: 24px;
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            margin-bottom: 1rem;
        }

        .metric-card .title {
            font-family: 'Inter', sans-serif;
            font-size: 0.875rem;
            font-weight: 500;
            color: #6B7280;
            margin-bottom: 0.5rem;
        }

        .metric-card .value {
            font-family: 'Inter', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }

        .metric-card .drip {
            position: absolute;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            bottom: -10px;
            right: -10px;
            opacity: 0.6;
            animation: pulse 2s infinite;
        }

        /* Purple Theme */
        .purple .icon {
            background-color: #EDE9FE;
            color: #7C3AED;
        }
        .purple .value {
            color: #7C3AED;
        }
        .purple .drip {
            background-color: #7C3AED;
        }

        /* Blue Theme */
        .blue .icon {
            background-color: #DBEAFE;
            color: #2563EB;
        }
        .blue .value {
            color: #2563EB;
        }
        .blue .drip {
            background-color: #2563EB;
        }

        /* Red Theme */
        .red .icon {
            background-color: #FEE2E2;
            color: #DC2626;
        }
        .red .value {
            color: #DC2626;
        }
        .red .drip {
            background-color: #DC2626;
        }

        /* Green Theme */
        .green .icon {
            background-color: #D1FAE5;
            color: #059669;
        }
        .green .value {
            color: #059669;
        }
        .green .drip {
            background-color: #059669;
        }

        @keyframes pulse {
            0% { transform: scale(0.8); opacity: 0.5; }
            50% { transform: scale(1.2); opacity: 0.8; }
            100% { transform: scale(0.8); opacity: 0.5; }
        }
    </style>
""", unsafe_allow_html=True)


# Model area

# Load the dataset
def preprocess_data(df):
    # Fill missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Encode categorical features
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders


def future_predictions():
    st.write("### Future Employment Outcome Predictions")

    # Load and preprocess data
    df = pd.read_csv('youth_unemployment_dataset.csv')
    df, label_encoders = preprocess_data(df)

    # Define features and target variable
    X = df.drop(columns=['employment_outcome'])
    y = df['employment_outcome']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and display accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")


def forecasting(df, years_to_predict):
    # Calculate probabilities from historical data
    prob_outcome = df['employment_outcome'].value_counts(normalize=True)

    # Calculate conditional probabilities (e.g., based on education level)
    conditional_probs = (
        df.groupby('education_level')['employment_outcome']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # Simulate future predictions
    future_years = list(range(1, years_to_predict + 1))
    predictions = []

    for year in future_years:
        future_data = []
        for _, row in df.iterrows():
            edu_level = row['education_level']
            if edu_level in conditional_probs.index:
                prob = conditional_probs.loc[edu_level]
            else:
                prob = prob_outcome

            # Simulate outcome based on probabilities
            outcome = np.random.choice(prob.index, p=prob.values)
            future_data.append(outcome)

        predictions.append(pd.Series(future_data).value_counts())

    # Aggregate predictions
    prediction_summary = pd.DataFrame(predictions, index=future_years).fillna(0)
    prediction_summary.index.name = "Year"
    forcast1, forecast2 = st.columns(2)
    # Plot Results
    with forcast1:

        st.write("### Future Employment Outcome Predictions")
        st.bar_chart(prediction_summary)
        # New Feature: Unemployment by Education Level
        st.write("### Unemployment by Education Level")
        edu_unemployment = df[df['employment_outcome'] == 0]['education_level'].value_counts()
        fig_unemployment = px.pie(
            values=edu_unemployment.values,
            names=edu_unemployment.index,
            title="Forecast Unemployment Distribution by Education Level"
        )
        st.plotly_chart(fig_unemployment)
    with forecast2:
        # New Feature: Employment Sectors by Education Level
        st.write("### Employment Sectors by Education Level")
        sector_education = df.groupby('education_level')['current_employment_sector'].value_counts().unstack().fillna(0)
        fig_sector_education = px.bar(
            sector_education,
            title="Employment Sectors by Education Level",
            labels={'value': 'Count', 'education_level': 'Education Level'},
            barmode='stack'
        )
        st.plotly_chart(fig_sector_education)

    # New Feature: Current Employment Sector by Education Level
    st.write("### Current Employment Sector by Education Level")
    current_sector = df.groupby('education_level')['current_employment_sector'].value_counts(
        normalize=True).unstack().fillna(0)
    fig_current_sector = px.line(
        current_sector.T,
        title="Current Employment Sector Trends by Education Level",
        labels={'value': 'Proportion', 'current_employment_sector': 'Employment Sector'}
    )
    st.plotly_chart(fig_current_sector)


# Sidebar navigation
def sideBar():
    with st.sidebar:
        selected = st.selectbox("Main Menu", ["Home", "Future Predictions"])
        return selected


# Run the sidebar
selected_option = sideBar()

# Main Content
if selected_option == "Home":
    st.subheader("")


elif selected_option == "Future Predictions":
    st.subheader("Future Predictions")

    # Add a slider to select the number of years to predict
    years_to_predict = st.slider("Select the number of years to predict", 1, 10, 3)

    # Use the provided dataset
    if 'filtered_df' in globals():
        forecasting(filtered_df, years_to_predict)
    else:
        st.error("Please provide a valid dataset in the variable `filtered_df`.")


# def forecast_youth_unemployment():
#     def forecast_youth_unemployment():
#         # Data for Youth Unemployment (assumed structure similar to your previous project)
#         # Group by Year and relevant category (e.g., education level, age group)
#         unemployment_data = df[df['current_employment_sector'] == 'None']  # Adjust condition based on your dataset
#         combined_data_edu = unemployment_data.groupby(['Year', 'Education Level']).size().reset_index(name='Unemployed')
#         historical_data_edu = combined_data_edu.groupby(['Year', 'Education Level'])['Unemployed'].sum().reset_index()
#         historical_data_edu['Yearly Change (%)'] = historical_data_edu.groupby('Education Level')[
#                                                        'Unemployed'].pct_change() * 100
#         avg_change_per_edu = historical_data_edu.groupby('Education Level')['Yearly Change (%)'].mean().reset_index()
#
#         # Forecasting Function
#         def forecast_data(historical_data, avg_change_data, level_column, forecast_years, randomness_factor):
#             future_forecast = pd.DataFrame()
#             for level in avg_change_data[level_column].unique():
#                 last_year = historical_data[historical_data[level_column] == level]['Year'].max()
#                 last_value = historical_data[historical_data[level_column] == level]['Unemployed'].iloc[-1]
#                 avg_change = avg_change_data[avg_change_data[level_column] == level]['Yearly Change (%)'].values[0]
#
#                 future_years = list(range(last_year + 1, last_year + forecast_years + 1))
#                 future_values = []
#
#                 for year in future_years:
#                     random_adjustment = np.random.uniform(-randomness_factor, randomness_factor)
#                     next_value = last_value * (1 + avg_change / 100 + random_adjustment)
#                     future_values.append(next_value)
#                     last_value = next_value
#
#                 future_data = pd.DataFrame({
#                     'Year': future_years,
#                     level_column: [level] * len(future_years),
#                     'Unemployed': future_values
#                 })
#                 future_forecast = pd.concat([future_forecast, future_data])
#
#             return future_forecast
#
#         # Define the number of years for the forecast and a randomness factor
#         forecast_years = st.sidebar.slider('Select number of years to forecast', min_value=1, max_value=11, value=5)
#         randomness_factor = 0.5  # Can adjust based on how much variability you want
#
#         # Forecast Youth Unemployment by Education Level
#         future_forecast_edu = forecast_data(historical_data_edu, avg_change_per_edu, 'Education Level', forecast_years,
#                                             randomness_factor)
#
#         # Combine Historical and Forecasted Data for Education Level
#         combined_forecast_edu = pd.concat(
#             [historical_data_edu[['Year', 'Education Level', 'Unemployed']], future_forecast_edu])
#
#         # Plot Youth Unemployment by Education Level
#         fig1 = px.line(combined_forecast_edu, x='Year', y='Unemployed', color='Education Level',
#                        title=f'Youth Unemployment Forecast by Education Level for Next {forecast_years} Years',
#                        labels={'Unemployed': 'Number of Unemployed Youth', 'Year': 'Years'})
#
#         # Show the chart
#         st.plotly_chart(fig1, use_container_width=True)


# def sideBar():
#     with st.sidebar:
#         selected = option_menu(
#             menu_title="Main Menu",
#             options=["Home", "Youth Unemployment Forecasting"],
#             icons=["house", "chart-line"],
#             menu_icon="cast",
#             default_index=0
#         )
#     if selected == "Youth Unemployment Forecasting":
#         st.subheader(f"Page: {selected}")
#         forecast_youth_unemployment()
#
#
# # Run the application
# sideBar()

# Calculate the metrics
st.markdown("### Key Metrics")
total_users = len(filtered_df)
avg_age = filtered_df['age'].mean()
unemployment_rate = (filtered_df['employment_outcome'] == False).mean() * 100
avg_income = filtered_df['monthly_income'].mean()

# Create columns for the metrics
col1, col2, col3, col4 = st.columns(4)

# Total Youths Card
with col1:
    st.markdown(f"""
        <div class="metric-card purple">
            <div class="icon">
                <i class="fas fa-users"></i>
            </div>
            <div class="title">Total Youths</div>
            <div class="value">{total_users:,}</div>
            <div class="drip"></div>
        </div>
    """, unsafe_allow_html=True)

# Average Age Card
with col2:
    st.markdown(f"""
        <div class="metric-card blue">
            <div class="icon">
                <i class="fas fa-calendar"></i>
            </div>
            <div class="title">Average Age</div>
            <div class="value">{avg_age:.1f}</div>
            <div class="drip"></div>
        </div>
    """, unsafe_allow_html=True)

# Unemployment Rate Card
with col3:
    st.markdown(f"""
        <div class="metric-card red">
            <div class="icon">
                <i class="fas fa-chart-line"></i>
            </div>
            <div class="title">Unemployment Rate</div>
            <div class="value">{unemployment_rate:.1f}%</div>
            <div class="drip"></div>
        </div>
    """, unsafe_allow_html=True)

# Average Monthly Income Card
with col4:
    st.markdown(f"""
        <div class="metric-card green">
            <div class="icon">
                <i class="fas fa-dollar-sign"></i>
            </div>
            <div class="title">Average Monthly Income</div>
            <div class="value">{avg_income:,.0f} RWF</div>
            <div class="drip"></div>
        </div>
    """, unsafe_allow_html=True)

# Visualization Tabs
# Custom CSS for the header and tabs
st.markdown("""
    <style>
    /* Main Dashboard Title */
    .dashboard-header {
        background:rgb(240, 243, 245);
        color: white;
        padding: 1rem 1rem 1rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        # box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }

    .dashboard-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQ0MCIgaGVpZ2h0PSI1MDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnM+PGxpbmVhckdyYWRpZW50IHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjEwMCUiIGlkPSJhIj48c3RvcCBzdG9wLWNvbG9yPSIjZmZmIiBzdG9wLW9wYWNpdHk9Ii4xIiBvZmZzZXQ9IjAlIi8+PHN0b3Agc3RvcC1jb2xvcj0iI2ZmZiIgc3RvcC1vcGFjaXR5PSIuMDUiIG9mZnNldD0iMTAwJSIvPjwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPjxwYXRoIGZpbGw9InVybCgjYSkiIGQ9Ik0wIDBoMTQ0MHY1MDBIMHoiLz48L3N2Zz4=');
        background-size: cover;
        opacity: 0.1;
    }

    .dashboard-title {
        font-size: 2.5rem !important;
        font-weight: 650 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        text-align:justify;
    }

    .dashboard-subtitle {
        font-size: 1.1rem !important;
        opacity: 0.9;
        max-width: 600px;
        line-height: 1.5;
        color:rgb(37, 45, 49);
    }

    /* Custom Tab Styling */
    .stTabs {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px;
        padding: 0 24px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #3498DB, #2980B9) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(52, 152, 219, 0.3);
    }

    .stTabs [role="tablist"] button {
        font-size: 14px;
        font-weight: 600;
    }

    .stTabs [role="tablist"] button:hover {
        background: #e9ecef;
        color: #2C3E50;
    }

    .stTabs [role="tablist"] button[aria-selected="true"]:hover {
        background: linear-gradient(90deg, #3498DB, #2980B9) !important;
        color: white !important;
    }

    /* Tab Content Area */
    .stTabs [data-baseweb="tab-panel"] {
        padding: 1.5rem 0.5rem;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Dashboard Header
st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">Analysis and Visualizations</h1>
        <p class="dashboard-subtitle">Comprehensive insights and data visualization across demographics, education, employment, program impact, and economic indicators.</p>
    </div>
""", unsafe_allow_html=True)

# Create styled tabs
tabs = st.tabs([
    "📊 Demographics Overview",
    "🎓 Education & Skills",
    "💼 Employment Status",
    "📈 Program Impact",
    "💰 Economic Indicators",
    "💼 Additional"
])

# Demographics Overview
with tabs[0]:
    st.markdown('<p class="employment-header">Demographics Overview</p>', unsafe_allow_html=True)
    # st.markdown("### Demographics Overview")
    dem1, dem2 = st.columns(2, gap='small')
    # Aggregate data for Age Distribution by Gender
    age_gender_counts = filtered_df.groupby("age").size().reset_index(name="count")
    with dem1:
        # Age Distribution by Gender using Pie Chart
        st.markdown('<div class="chart-title">Age Distribution</div>', unsafe_allow_html=True)
        age_dist_fig = px.pie(
            age_gender_counts,
            names="age",
            values="count",
            height=300,
            width=370,
            # title="Age Distribution",
            hole=0.3  # Adds a donut hole for better visualization
        )
        st.plotly_chart(age_dist_fig, use_container_width=False)
    with dem2:
        # Aggregate data for Urban vs. Rural Distribution by Region
        st.markdown('<div class="chart-title">Urban vs. Rural Distribution by Region</div>', unsafe_allow_html=True)
        urban_rural_counts = filtered_df.groupby(["region", "location_type"]).size().reset_index(name="count")

        # Urban vs. Rural Distribution by Region using Bar Chart
        urban_rural_fig = px.bar(
            urban_rural_counts,
            x="region",
            y="count",
            height=300,
            width=370,
            color="location_type",
            barmode="group",
            # title="Urban vs. Rural Distribution by Region",
        )
        st.plotly_chart(urban_rural_fig, use_container_width=False)

# Education & Skills
with tabs[1]:
    st.markdown('<p class="employment-header">Education & Skills</p>', unsafe_allow_html=True)
    # st.markdown("### Education & Skills")

    # First Row: Two Columns - Education Level Distribution and Unemployment by Education Level
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-title">Distribution of Education Levels Among Youth</div>',
                    unsafe_allow_html=True)
        # st.markdown("#### Education Level Distribution")
        education_level_fig = px.pie(
            filtered_df,
            height=300,
            width=370,
            names="education_level",
            # title="Distribution of Education Levels Among Youth"
        )
        st.plotly_chart(education_level_fig, use_container_width=False)

    with col2:
        # st.markdown("#### Unemployment by Education Level")
        st.markdown('<div class="chart-title">Unemployment by Education Level</div>', unsafe_allow_html=True)
        edu_employment_fig = px.bar(
            filtered_df,
            x="education_level",
            height=300,
            width=370,
            color="employment_outcome",
            # title="Unemployment by Education Level",
            barmode="stack"
        )
        st.plotly_chart(edu_employment_fig, use_container_width=False)

    # Second Row: Two Columns - Education Mismatch Impact and Digital Skills Level
    col3, col4 = st.columns(2)

    with col3:
        # st.markdown("#### Education Mismatch Impact on Employment")
        st.markdown('<div class="chart-title">Education Mismatch Impact on Employment</div>', unsafe_allow_html=True)
        mismatch_fig = px.pie(
            filtered_df,
            height=300,
            width=370,
            names="education_mismatch",
            # title="Education Mismatch Impact on Employment"
        )
        st.plotly_chart(mismatch_fig, use_container_width=False)

    with col4:
        st.markdown('<div class="chart-title">Digital Skills Level and Employment Outcome</div>',
                    unsafe_allow_html=True)
        # st.markdown("#### Digital Skills Level and Employment Outcome")
        skills_employment_fig = px.bar(
            filtered_df,
            x="digital_skills_level",
            height=300,
            width=370,
            color="employment_outcome",
            # title="Employment Outcome by Digital Skills Level",
            barmode="group"
        )
        st.plotly_chart(skills_employment_fig, use_container_width=False)

    # Third Row: Two Columns - Technical Skills Distribution and Training Participation
    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<div class="chart-title">Technical Skills Distribution</div>', unsafe_allow_html=True)
        # st.markdown("#### Technical Skills Distribution")
        tech_skills_fig = px.histogram(
            filtered_df.explode("technical_skills"),
            x="technical_skills",
            height=300,
            width=370,
            # title="Distribution of Technical Skills"
        )
        st.plotly_chart(tech_skills_fig, use_container_width=False)

    with col6:
        st.markdown('<div class="chart-title">Impact of Training Participation on Employment</div>',
                    unsafe_allow_html=True)
        # st.markdown("#### Training Participation and Employment Outcome")
        training_employment_fig = px.pie(
            filtered_df,
            height=300,
            width=370,
            names="training_participation",
            # title="Impact of Training Participation on Employment"
        )
        st.plotly_chart(training_employment_fig, use_container_width=False)

    # Fourth Row: Two Columns - Current Employment Sector by Education Level (Bar and Sunburst)
    col7, col8 = st.columns(2)

    with col7:
        # st.markdown("#### Current Employment Sector by Education Level (Bar Chart)")
        st.markdown('<div class="chart-title">Current Employment Sector by Education Level</div>',
                    unsafe_allow_html=True)
        sector_education_counts = filtered_df.groupby(
            ["education_level", "current_employment_sector"]).size().reset_index(name="count")
        sector_by_education_bar = px.bar(
            sector_education_counts,
            x="education_level",
            y="count",
            height=300,
            width=370,
            color="current_employment_sector",
            # title="Current Employment Sector by Education Level",
            barmode="group",
            labels={"count": "Number of Users"}
        )
        st.plotly_chart(sector_by_education_bar, use_container_width=False)

    with col8:
        # st.markdown("#### Current Employment Sector by Education Level (Sunburst Chart)")
        st.markdown('<div class="chart-title">Employment Sectors by Education Level</div>', unsafe_allow_html=True)
        sector_by_education_sunburst = px.sunburst(
            sector_education_counts,
            height=300,
            width=370,
            path=["education_level", "current_employment_sector"],
            values="count",
            # title="Employment Sectors by Education Level"
        )
        st.plotly_chart(sector_by_education_sunburst, use_container_width=False)

# Employment Status Tab
with tabs[2]:
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .employment-header {
            font-size: 36px !important;
            font-weight: 600;
            color: #1f77b4;
            padding: 20px 0;
            text-align: center;
            background: #f8f9fa;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        div.stPlotlyChart > div {
            border-radius: 1rem;
            background: white;
            border: 1px solid #edf2f7;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            margin-bottom: 1.5rem;
        }
        
        div.stPlotlyChart > div:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 20px rgba(0,0,0,0.1);
        }
        
        .chart-title {
            font-size: 1.25rem !important;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid #edf2f7;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown('<p class="employment-header">Employment Status Dashboard</p>', unsafe_allow_html=True)

    # First Row: Employment Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-title">Employment Outcome Distribution</div>', unsafe_allow_html=True)

        # Create pie chart
        employment_outcome_fig = px.pie(
            filtered_df,
            names="employment_outcome",
            title="",
            hole=0.4,
            color_discrete_sequence=['#1f77b4', '#ff7f0e']
        )

        employment_outcome_fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hoverinfo='label+percent',
            hoverlabel=dict(bgcolor='white', font_size=14)
        )

        employment_outcome_fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            width=370,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(employment_outcome_fig, use_container_width=False)

    with col2:
        st.markdown('<div class="chart-title">Employment Sector Distribution</div>', unsafe_allow_html=True)

        employment_sector_fig = px.bar(
            filtered_df[filtered_df['employment_outcome'] == True],
            x="current_employment_sector",
            color="formal_informal",
            labels={"current_employment_sector": "Employment Sector",
                    "count": "Number of Youth"},
            barmode="group",
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )

        employment_sector_fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            width=370,
            legend_title="Employment Type",
            xaxis_title="Sector",
            yaxis_title="Number of Youth",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(employment_sector_fig, use_container_width=False)

    # Second Row: Income and Unemployment
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="chart-title">Income Level Distribution by Sector</div>', unsafe_allow_html=True)

        income_sector_fig = px.box(
            filtered_df[filtered_df['employment_outcome'] == True],
            x="current_employment_sector",
            y="monthly_income",
            color="formal_informal",
            labels={
                "monthly_income": "Monthly Income (RWF)",
                "current_employment_sector": "Employment Sector"
            },
            color_discrete_sequence=['#3498db', '#e67e22']
        )

        income_sector_fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            width=370,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(income_sector_fig, use_container_width=False)

    with col4:
        st.markdown('<div class="chart-title">Unemployment Duration Distribution</div>', unsafe_allow_html=True)

        unemployment_duration_fig = px.histogram(
            filtered_df[filtered_df['employment_outcome'] == False],
            x="unemployment_duration",
            nbins=15,
            labels={"unemployment_duration": "Unemployment Duration (Months)"},
            color_discrete_sequence=['#9b59b6']
        )

        unemployment_duration_fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            width=370,
            bargap=0.2,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(unemployment_duration_fig, use_container_width=False)

    # Third Row: Employment Comparison and Analytics
    col5, col6 = st.columns(2)

    with col5:
        card5 = st.container()
        with card5:
            st.markdown('<div class="chart-title">Formal vs Informal Employment by Sector</div>',
                        unsafe_allow_html=True)

            formal_informal_counts = filtered_df[filtered_df['employment_outcome'] == True].groupby(
                ["current_employment_sector", "formal_informal"]
            ).size().reset_index(name="count")

            formal_informal_fig = px.bar(
                formal_informal_counts,
                x="current_employment_sector",
                y="count",
                height=300,
                width=370,
                color="formal_informal",
                title="",
                barmode="stack",
                labels={"count": "Number of Youth", "current_employment_sector": "Employment Sector"},
                color_discrete_sequence=['#27ae60', '#c0392b']
            )

            formal_informal_fig.update_traces(
                hovertemplate='<b>%{x}</b><br>' +
                              'Count: %{y}<br>' +
                              '<extra>%{fullData.name}</extra>',
                hoverlabel=dict(bgcolor='white')
            )

            st.plotly_chart(formal_informal_fig, use_container_width=False)

            st.markdown("""
                    </div>
                </div>
            """, unsafe_allow_html=True)
    with col6:
        st.markdown('<div class="chart-title">Income Analytics by Sector</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # st.markdown('<p class="chart-title">Income Analytics by Sector</p>', unsafe_allow_html=True)
        income_stats = filtered_df[filtered_df['employment_outcome'] == True].groupby("current_employment_sector")[
            "monthly_income"].agg(
            Average_Income="mean",
            Median_Income="median",
            Min_Income="min",
            Max_Income="max"
        ).reset_index()

        income_analytics_fig = px.bar(
            income_stats,
            x="current_employment_sector",
            y="Average_Income",
            height=300,
            width=370,
            error_y=income_stats["Max_Income"] - income_stats["Average_Income"],
            error_y_minus=income_stats["Average_Income"] - income_stats["Min_Income"],
            color="current_employment_sector",
            title="",
            labels={"Average_Income": "Average Monthly Income (RWF)", "current_employment_sector": "Employment Sector"},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        income_analytics_fig.add_scatter(
            x=income_stats["current_employment_sector"],
            y=income_stats["Median_Income"],
            mode="markers",
            marker=dict(color="red", symbol="diamond", size=10),
            name="Median Income"
        )
        income_analytics_fig.update_traces(
            hovertemplate='<b>%{x}</b><br>' +
                          'Average Income: %{y:,.0f} RWF<br>' +
                          '<extra>%{fullData.name}</extra>',
            hoverlabel=dict(bgcolor='white')
        )
        st.plotly_chart(income_analytics_fig, use_container_width=False)

    st.markdown('<div class="stat-table">', unsafe_allow_html=True)
    st.write(income_stats.style.format({
        'Average_Income': '{:,.0f}',
        'Median_Income': '{:,.0f}',
        'Min_Income': '{:,.0f}',
        'Max_Income': '{:,.0f}'
    }))
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Fourth Row: Age Group Analysis
    st.markdown('<div class="chart-title">Income Distribution by Age and Sector</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    # st.markdown('<p class="metric-header">Income Distribution by Age and Sector</p>', unsafe_allow_html=True)
    age_bins = [16, 18, 21, 25]
    age_labels = ["16-18", "19-21", "22-25"]
    filtered_df["age_group"] = pd.cut(filtered_df["age"], bins=age_bins, labels=age_labels, right=True)

    age_income_fig = px.box(
        filtered_df[filtered_df['employment_outcome'] == True],
        x="current_employment_sector",
        y="monthly_income",
        height=400,
        width=800,
        color="age_group",
        title="",
        labels={"monthly_income": "Monthly Income (RWF)", "current_employment_sector": "Employment Sector",
                "age_group": "Age Group"},
        color_discrete_sequence=['#3498db', '#e67e22', '#2ecc71']
    )
    age_income_fig.update_traces(
        hoverlabel=dict(bgcolor='white'),
        hovertemplate='<b>%{x}</b><br>' +
                      'Age Group: %{fullData.name}<br>' +
                      'Median: %{median}<br>' +
                      'Q1: %{q1}<br>' +
                      'Q3: %{q3}<br>' +
                      'Min: %{lowerfence}<br>' +
                      'Max: %{upperfence}<br>' +
                      '<extra></extra>'
    )
    st.plotly_chart(age_income_fig, use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # Fifth Row: Post-Intervention Analysis
    col7, col8 = st.columns(2)

    with col7:
        st.markdown('<div class="chart-title">Employment Duration Trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # st.markdown('<p class="metric-header">Employment Duration Trend</p>', unsafe_allow_html=True)
        post_intervention_data = filtered_df[filtered_df['employment_duration_post_intervention'].notna()]
        duration_trend = post_intervention_data.groupby("employment_duration_post_intervention").size().reset_index(
            name="count")

        employment_duration_fig = px.line(
            duration_trend,
            x="employment_duration_post_intervention",
            y="count",
            height=300,
            width=370,
            title="",
            labels={"employment_duration_post_intervention": "Months Post-Intervention",
                    "count": "Number of Individuals"},
            markers=True,
            line_shape="spline",
            color_discrete_sequence=['#2980b9']
        )
        employment_duration_fig.update_traces(
            marker=dict(size=8),
            hovertemplate='<b>Month %{x}</b><br>' +
                          'Count: %{y}<br>' +
                          '<extra></extra>',
            hoverlabel=dict(bgcolor='white')
        )
        st.plotly_chart(employment_duration_fig, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with col8:
        st.markdown('<div class="chart-title">Income Comparison by Employment Type</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # st.markdown('<p class="metric-header">Income Comparison by Employment Type</p>', unsafe_allow_html=True)
        formal_informal_income = filtered_df[filtered_df['employment_outcome'] == True].groupby(
            ["current_employment_sector", "formal_informal"]
        )["monthly_income"].mean().reset_index()

        formal_informal_income_fig = px.bar(
            formal_informal_income,
            x="current_employment_sector",
            y="monthly_income",
            color="formal_informal",
            height=300,
            width=370,
            title="",
            labels={"monthly_income": "Average Monthly Income (RWF)", "current_employment_sector": "Employment Sector"},
            barmode="group",
            color_discrete_sequence=['#27ae60', '#c0392b']
        )
        formal_informal_income_fig.update_traces(
            hovertemplate='<b>%{x}</b><br>' +
                          'Average Income: %{y:,.0f} RWF<br>' +
                          '<extra>%{fullData.name}</extra>',
            hoverlabel=dict(bgcolor='white')
        )
        st.plotly_chart(formal_informal_income_fig, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

    # Sixth Row: Education and Regional Analysis

with tabs[3]:
    # st.markdown("### Program Impact")

    # First Row: Program Participation and Employment Outcome
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-title">Impact of Program Participation on Employment</div>',
                    unsafe_allow_html=True)
        program_participation_fig = px.pie(
            filtered_df,
            names="training_participation",
            height=300,
            width=370,
            # title="Impact of Program Participation on Employment",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(program_participation_fig, use_container_width=False)

    with col2:
        # st.markdown("#### Employment Outcome by Program Type")
        st.markdown('<div class="chart-title">Employment Outcome by Program Type</div>', unsafe_allow_html=True)
        program_outcome_counts = filtered_df.groupby(["program_type", "employment_outcome"]).size().reset_index(
            name="count")
        employment_by_program_fig = px.bar(
            program_outcome_counts,
            x="program_type",
            y="count",
            height=300,
            width=370,
            color="employment_outcome",
            # title="Employment Outcome by Program Type",
            labels={"count": "Number of Youth", "program_type": "Program Type"},
            barmode="stack"
        )
        st.plotly_chart(employment_by_program_fig, use_container_width=False)

    # Second Row: Effectiveness Features
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="chart-title">Effect of Program Duration on Employment Success Rate</div>',
                    unsafe_allow_html=True)
        # st.markdown("#### Effect of Program Duration on Employment Success Rate")
        sample_data_duration = pd.DataFrame({
            "program_duration": [4, 8, 12, 16, 20, 24],
            "employment_success_rate": [65, 70, 78, 82, 88, 91]
        })
        duration_effect_fig = px.line(
            sample_data_duration,
            x="program_duration",
            y="employment_success_rate",
            height=300,
            width=370,
            # title="Effect of Program Duration on Employment Success Rate",
            labels={"program_duration": "Program Duration (Weeks)",
                    "employment_success_rate": "Employment Success Rate (%)"},
            markers=True
        )
        st.plotly_chart(duration_effect_fig, use_container_width=False)

    with col4:
        # st.markdown("#### Intervention Cost vs. Employment Success Rate")
        st.markdown('<div class="chart-title">Intervention Cost vs. Employment Success Rate</div>',
                    unsafe_allow_html=True)
        sample_data_cost = pd.DataFrame({
            "intervention_cost": [100000, 150000, 200000, 250000, 300000],
            "employment_success_rate": [68, 72, 75, 85, 89]
        })
        cost_effectiveness_fig = px.scatter(
            sample_data_cost,
            x="intervention_cost",
            y="employment_success_rate",
            height=300,
            width=370,
            # title="Intervention Cost vs. Employment Success Rate",
            labels={"intervention_cost": "Intervention Cost (RWF)",
                    "employment_success_rate": "Employment Success Rate (%)"},
            trendline="ols"
        )
        st.plotly_chart(cost_effectiveness_fig, use_container_width=False)

    # Third Row: Job Match and Participant Satisfaction
    col5, col6 = st.columns(2)
    with col5:
        st.markdown('<div class="chart-title">Alignment of Job with Skills Acquired in Intervention</div>',
                    unsafe_allow_html=True)
        # st.markdown("#### Alignment of Job with Skills Acquired")
        job_match_data = pd.DataFrame({
            "Job Match": ["Aligned with Skills", "Not Aligned with Skills"],
            "Count": [180, 45]
        })
        job_match_fig = px.pie(
            job_match_data,
            names="Job Match",
            values="Count",
            height=300,
            width=370,
            # title="Alignment of Job with Skills Acquired in Intervention"
        )
        st.plotly_chart(job_match_fig, use_container_width=False)

    with col6:
        # st.markdown("#### Participant Satisfaction with Intervention Programs")
        st.markdown('<div class="chart-title">Participant Satisfaction with Intervention Programs</div>',
                    unsafe_allow_html=True)
        satisfaction_data = pd.DataFrame({
            "Satisfaction Score": [1, 2, 3, 4, 5],
            "Number of Participants": [20, 35, 85, 120, 200]
        })
        satisfaction_fig = px.bar(
            satisfaction_data,
            x="Satisfaction Score",
            y="Number of Participants",
            height=300,
            width=370,
            # title="Participant Satisfaction with Intervention Programs",
            labels={"Satisfaction Score": "Satisfaction Score (1-5)", "Number of Participants": "Participants"}
        )
        st.plotly_chart(satisfaction_fig, use_container_width=False)

    # Fourth Row: Key Skills Acquired and Follow-Up Support Impact
    col7, col8 = st.columns(2)
    with col7:
        # st.markdown("#### Key Skills Acquired by Program Type")
        st.markdown('<div class="chart-title">Skills Acquired by Program Type</div>', unsafe_allow_html=True)
        skills_data = pd.DataFrame({
            "program_type": ["Digital Skills", "Digital Skills", "Business Development", "Business Development",
                             "Technical Skills"],
            "key_skills_acquired": ["Coding", "Data Analysis", "Project Management", "Entrepreneurship", "Marketing"],
            "count": [60, 45, 75, 30, 90]
        })
        skills_fig = px.sunburst(
            skills_data,
            path=["program_type", "key_skills_acquired"],
            values="count",
            height=300,
            width=370,
            # title="Skills Acquired by Program Type"
        )
        st.plotly_chart(skills_fig, use_container_width=False)

    with col8:
        # st.markdown("#### Follow-Up Support and Employment Retention Rate")
        st.markdown('<div class="chart-title">Impact of Follow-Up Support on 6-Month Employment Retention Rate</div>',
                    unsafe_allow_html=True)
        retention_data = pd.DataFrame({
            "Follow-Up Support": ["Provided", "Not Provided"],
            "employment_retention_rate_6_months": [78, 55]
        })
        retention_fig = px.bar(
            retention_data,
            x="Follow-Up Support",
            y="employment_retention_rate_6_months",
            height=300,
            width=370,
            # title="Impact of Follow-Up Support on 6-Month Employment Retention Rate",
            labels={"employment_retention_rate_6_months": "6-Month Retention Rate (%)",
                    "Follow-Up Support": "Follow-Up Support"}
        )
        st.plotly_chart(retention_fig, use_container_width=False)

    # Fifth Row: Job Placement Rate by Program Type
    st.markdown('<div class="chart-title">Job Placement Rate by Program Type</div>', unsafe_allow_html=True)
    placement_rate_data = pd.DataFrame({
        "program_type": ["Digital Skills", "Business Development", "Technical Skills", "Vocational Training"],
        "job_placement_rate": [85, 75, 80, 78]
    })
    placement_rate_fig = px.bar(
        placement_rate_data,
        x="program_type",
        y="job_placement_rate",
        height=300,
        width=370,
        # title="Job Placement Rate by Program Type",
        labels={"job_placement_rate": "Job Placement Rate (%)", "program_type": "Program Type"}
    )
    st.plotly_chart(placement_rate_fig, use_container_width=False)
# Economic Indicators
with tabs[4]:
    # st.markdown('<p class="employment-header">Economic Indicators</p>', unsafe_allow_html=True)
    # st.markdown("### Economic Indicators")

    # First Row: Youth Unemployment and Household Income
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-title">Youth Unemployment Rate by Region</div>', unsafe_allow_html=True)
        unemployment_region_fig = px.bar(
            filtered_df,
            x="region",
            y="youth_unemployment_rate",
            height=300,
            width=370,
            # title="Youth Unemployment Rate by Region"
        )
        st.plotly_chart(unemployment_region_fig, use_container_width=False)

    with col2:
        st.markdown('<div class="chart-title">Household Income Distribution</div>', unsafe_allow_html=True)
        # st.markdown("#### Household Income Distribution")
        income_distribution_fig = px.histogram(
            filtered_df,
            x="household_income",
            height=300,
            width=370,
            # title="Household Income Distribution",
            nbins=20
        )
        st.plotly_chart(income_distribution_fig, use_container_width=False)

    # Second Row: Employment Rates and Trends
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="chart-title">Urban vs. Rural Employment Rate by Region</div>', unsafe_allow_html=True)
        # st.markdown("#### Urban vs. Rural Employment Rate")
        urban_rural_employment_fig = px.bar(
            filtered_df,
            x="location_type",
            y="urban_rural_employment_rate",
            height=300,
            width=370,
            color="region",
            barmode="group",
            # title="Urban vs. Rural Employment Rate by Region"
        )
        st.plotly_chart(urban_rural_employment_fig, use_container_width=False)

    with col4:
        # st.markdown("#### Regional Employment Rate Trends Over Time")
        st.markdown('<div class="chart-title">Regional Employment Rate Trends Over Time</div>', unsafe_allow_html=True)
        if "year" in filtered_df.columns:
            employment_trend_fig = px.line(
                filtered_df,
                x="year",
                y="region_employment_rate",
                height=300,
                width=370,
                color="region",
                # title="Regional Employment Rate Trends Over Time"
            )
            st.plotly_chart(employment_trend_fig, use_container_width=False)
        else:
            st.markdown('<div class="chart-title">Regional Employment Rate Trends Over Time</div>',
                        unsafe_allow_html=True)
            simulated_data = pd.DataFrame({
                "year": [2018, 2019, 2020, 2021, 2022, 2023],
                "region_employment_rate": [55, 60, 65, 70, 75, 80],
                "region": ["Kigali"] * 6
            })
            employment_trend_fig = px.line(
                simulated_data,
                x="year",
                y="region_employment_rate",
                height=300,
                width=370,
                color="region",
                # title="Regional Employment Rate Trends Over Time"
            )
            st.plotly_chart(employment_trend_fig, use_container_width=False)

    # Third Row: Employment Outcome and Income Level
    col5, col6 = st.columns(2)
    with col5:
        st.markdown('<div class="chart-title">Household Size and Monthly Income by Employment Outcome</div>',
                    unsafe_allow_html=True)
        household_size_outcome_fig = px.box(
            filtered_df,
            x="household_size",
            y="monthly_income",
            height=300,
            width=370,
            color="employment_outcome",
            # title="Household Size and Monthly Income by Employment Outcome"
        )
        st.plotly_chart(household_size_outcome_fig, use_container_width=False)

    with col6:
        st.markdown('<div class="chart-title">Youth Unemployment Rate by Education Level</div>', unsafe_allow_html=True)
        unemployment_edu_level_fig = px.histogram(
            filtered_df,
            x="education_level",
            y="youth_unemployment_rate",
            height=300,
            width=370,
            color="region",
            barmode="stack",
            # title="Youth Unemployment Rate by Education Level"
        )
        st.plotly_chart(unemployment_edu_level_fig, use_container_width=False)

    # Fourth Row: Employment Sector and Digital Skills
    col7, col8 = st.columns(2)
    with col7:
        st.markdown('<div class="chart-title">Income Level by Employment Sector</div>', unsafe_allow_html=True)
        income_sector_fig = px.box(
            filtered_df,
            x="current_employment_sector",
            y="monthly_income",
            height=300,
            width=370,
            color="region",
            title="Income Level by Employment Sector"
        )
        st.plotly_chart(income_sector_fig, use_container_width=False)

    with col8:
        st.markdown('<div class="chart-title">Household Income and Digital Skills Level</div>', unsafe_allow_html=True)
        income_digital_skills_fig = px.sunburst(
            filtered_df,
            height=300,
            width=370,
            path=["household_income", "digital_skills_level"],
            # title="Household Income and Digital Skills Level"
        )
        st.plotly_chart(income_digital_skills_fig, use_container_width=False)

    # Fifth Row: Employment vs. Unemployment and Income Distribution
    col9, col10 = st.columns(2)
    with col9:
        st.markdown('<div class="chart-title">Regional Employment Rate vs. Youth Unemployment Rate</div>',
                    unsafe_allow_html=True)
        employment_vs_unemployment_fig = px.scatter(
            filtered_df,
            x="region_employment_rate",
            y="youth_unemployment_rate",
            height=300,
            width=370,
            color="region",
            # title="Regional Employment Rate vs. Youth Unemployment Rate",
            trendline="ols"
        )
        st.plotly_chart(employment_vs_unemployment_fig, use_container_width=False)

    with col10:
        st.markdown('<div class="chart-title">Monthly Income Distribution by Urban vs. Rural Areas</div>',
                    unsafe_allow_html=True)
        income_distribution_urban_rural_fig = px.violin(
            filtered_df,
            x="location_type",
            y="monthly_income",
            height=300,
            width=370,
            color="location_type",
            box=True,
            points="all",
            # title="Monthly Income Distribution by Urban vs. Rural Areas"
        )
        st.plotly_chart(income_distribution_urban_rural_fig, use_container_width=False)

with tabs[5]:  # Adjust based on the index of this new tab
    # Load the dataset with error handling for encoding issues
    file_path = "nisr_dataset.csv"  # Replace with your actual file path

    # Handling missing data



    @st.cache_data
    def load_data(file_paths):
        try:
            nisr_data = pd.read_csv(file_paths, encoding='latin1')
            return nisr_data
        except UnicodeDecodeError:
            st.error("Error reading file. Please check the file encoding.")
            return None


    # Load data
    dfs = load_data(file_path)

    if dfs is not None:
        # Display dataset header
        # st.write("Dataset Loaded Successfully")
        # st.dataframe(dfs.head())

        # Rename columns if necessary
        # Example: Renaming columns to standardize names
        dfs.rename(columns={'code_dis': 'Region', 'I06A': 'Access_Type', 'I04': 'Energy_Source'}, inplace=True)


        # Visualization 2: Energy Source Distribution (I04)
        st.subheader("Distribution of Energy Sources (Energy_Source)")
        energy_counts = dfs['Energy_Source'].value_counts()
        fig2 = px.bar(x=energy_counts.index, y=energy_counts.values, title="Energy Sources Distribution")
        fig2.update_layout(xaxis_title="Energy Source", yaxis_title="Count")
        st.plotly_chart(fig2)
        #

        # Visualization 6: Cross Analysis - Access Type by Energy Source

        cross_tab = pd.crosstab(df['education_level'], dfs['Energy_Source'])
        fig6 = px.imshow(cross_tab, title="Education Level by Energy Source", aspect="auto",
                         labels=dict(x="Energy Source", y="Access Type", color="Count"))
        st.plotly_chart(fig6)

    else:
        st.warning("Data could not be loaded. Check the file path and encoding.")
