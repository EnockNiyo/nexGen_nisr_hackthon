import datetime
import time

import streamlit as st
import pandas as pd
import plotly.express as px
# from pygments.lexers import go
from sklearn.ensemble import RandomForestClassifier
from streamlit_option_menu import option_menu  # Import the option_menu function
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import classification_report, accuracy_score
# from statsmodels.tsa.arima.model import ARIMA
# Set up Streamlit layout
st.set_page_config(page_title="Youth Unemployment Analytics Dashboard", page_icon="📊", layout="wide")
# Load datasets
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

# Load default dataset
DEFAULT_DATASET_PATH = "youth_unemployment_dataset.csv"

try:
    # Load default dataset
    df = pd.read_csv(DEFAULT_DATASET_PATH)
    st.success("Dataset loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the default dataset: {e}")
    st.stop()

# Title
st.markdown(
    "<h1 style='text-align: center; color: #302; margin-top: -45px;'>Youth Unemployment Analytics Dashboard</h1>",
    unsafe_allow_html=True)
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply custom CSS to style the sidebar
# Import required libraries
import streamlit as st

st.markdown("""
   
    </style>
""", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("Filters")
age_range = st.sidebar.slider("Select Age Range", 16, 30, (16, 30))
gender = st.sidebar.selectbox("Select Gender", ["All", "Male", "Female"])
regions = st.sidebar.multiselect("Select Region(s)", options=df['region'].unique(), default=df['region'].unique())
education_levels = st.sidebar.multiselect("Select Education Level(s)", options=df['education_level'].unique(),
                                          default=df['education_level'].unique())


# Apply Filters
# Define a mapping for regions to provinces
region_to_province = {
    "Northern": "Northern Province",
    "Kigali": "Kigali city",
    "Southern": "Southern Province",
    "Eastern": "Eastern Province",
    "Western": "Western Province"
}

# Add mapping for province before filtering
df['province_mapped'] = df['region'].map(region_to_province)

# Apply filters from Sidebar
filtered_df = df[
    (df['age'] >= age_range[0]) &
    (df['age'] <= age_range[1]) &
    (df['region'].isin(regions)) &
    (df['education_level'].isin(education_levels))
]
if gender != "All":
    filtered_df = filtered_df[filtered_df['gender'] == gender]

# Apply filter to dfs using the mapped province
if dfs is not None and 'province' in dfs.columns:
    filtered_dfs = dfs[dfs['province'].isin(filtered_df['province_mapped'].dropna())]
else:
    filtered_dfs = None




# Title and data summary
st.markdown("#### Data Summary")
st.dataframe(filtered_df.describe())



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
       
        color: rgb(0, 174, 239);
        padding: 1rem 1rem 1rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    
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

    # Demographics columns for layout
    dem1, dem2 = st.columns(2, gap='small')

    # Age Distribution by Age (now supports 16-30)
    age_gender_counts = filtered_df.groupby("age").size().reset_index(name="count")
    age_gender_counts = age_gender_counts.sort_values("age")
    with dem1:
        st.markdown('<div class="chart-title">Age Distribution</div>', unsafe_allow_html=True)
        if age_gender_counts.empty:
            st.info("No data available for the selected age range.")
        else:
            # Use a continuous color scale mapped to age for clarity
            import plotly.colors
            color_scale = px.colors.sequential.Viridis
            color_map = {age: color_scale[i % len(color_scale)] for i, age in enumerate(sorted(age_gender_counts['age'].unique()))}
            age_colors = [color_map[age] for age in age_gender_counts['age']]
            age_dist_fig = px.pie(
                age_gender_counts,
                names="age",
                values="count",
                height=300,
                width=370,
                hole=0.3
            )
            age_dist_fig.update_traces(textinfo='percent+label', marker=dict(colors=age_colors))
            st.plotly_chart(age_dist_fig, use_container_width=False)


with dem2:
    if filtered_dfs is not None and 'Code_UR' in filtered_dfs.columns:
        st.markdown('<div class="chart-title">Urban vs. Rural Distribution by Region</div>', unsafe_allow_html=True)

        # Group filtered data by region and urban/rural
        filtered_dfs['region_mapped'] = filtered_dfs['province'].map({v: k for k, v in region_to_province.items()})
        urban_rural_counts = filtered_dfs.groupby(["region_mapped", "Code_UR"]).size().reset_index(name="count")

        # Create the bar chart
        urban_rural_fig = px.bar(
            urban_rural_counts,
            x="region_mapped",
            y="count",
            color="Code_UR",
            height=300,
            width=370,
            barmode="group",  # Group bars for urban and rural
            labels={'region_mapped': 'Region', 'count': 'Count', 'Code_UR': 'Urban/Rural'}
        )
        st.plotly_chart(urban_rural_fig, use_container_width=False)
    else:
        st.warning("Required columns are missing in the filtered dataset.")


# Handle missing data
dfs = dfs.dropna(subset=['weight2', 'D12A', 'LFS_year'])

# Convert necessary columns to numeric
dfs['LFS_year'] = pd.to_numeric(dfs['LFS_year'], errors='coerce')
dfs['D12A'] = pd.to_numeric(dfs['D12A'], errors='coerce')

# District Trends Visualization
with st.container():
    if filtered_dfs is not None and 'code_dis' in filtered_dfs.columns:
        district_counts = filtered_dfs['code_dis'].value_counts().reset_index()
        district_counts.columns = ['District', 'Count']
        st.markdown('<div class="chart-title">District Trends</div>', unsafe_allow_html=True)
        district_trend_fig = px.bar(
            district_counts,
            x='District',
            y='Count',
            text='Count',
            labels={'District': 'District', 'Count': 'Number of Records'},
            title="District Trends",
            height=300,
            width=990
        )
        district_trend_fig.update_traces(textposition='outside')
        st.plotly_chart(district_trend_fig, use_container_width=False, key='district_trends')

# Prediction Model - RandomForest for Unemployment Classification
if dfs is not None and 'weight2' in dfs.columns and 'D12A' in dfs.columns:
    # Prepare data for prediction
    district_data = dfs.groupby('code_dis').agg({
        'weight2': 'sum',  # Total weight
        'D12A': 'mean'  # Average income
    }).reset_index()

    # Define target (example: high if D12A > median, else low)
    median_income = district_data['D12A'].median()
    district_data['unemployment_level'] = np.where(district_data['D12A'] > median_income, 'Low', 'High')

    # Train model
    X = district_data[['weight2', 'D12A']]
    y = district_data['unemployment_level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    # y_pred = model.predict(X_test)
    # st.write("Classification Report:")
    # st.text(classification_report(y_test, y_pred))
    # st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Visualize results
    district_data['prediction'] = model.predict(X)
    fig = px.bar(
        district_data,
        x='code_dis',
        y='D12A',
        color='prediction',
        title="Predicted Unemployment Levels by District",
        labels={'code_dis': 'District', 'D12A': 'Average Income'}
    )
    st.plotly_chart(fig, key='unemployment_prediction_plot')

else:
    st.warning("Required columns (weight2, D12A) are missing in the dataset.")




# Future Predictions

# Clean and Prepare the Data
if dfs is not None:
    # Drop rows with NaN in required columns
    dfs = dfs.dropna(subset=['LFS_year', 'D12A', 'weight2', 'code_dis'])

    # Convert necessary columns to numeric
    dfs['LFS_year'] = pd.to_numeric(dfs['LFS_year'], errors='coerce')
    dfs['D12A'] = pd.to_numeric(dfs['D12A'], errors='coerce')
    dfs['weight2'] = pd.to_numeric(dfs['weight2'], errors='coerce')

    # Check if the data is still valid after cleaning
    if dfs.empty:
        st.warning("No valid data available for prediction.")
    else:
        # District-level aggregation (weight2 and D12A)
        district_data = dfs.groupby('code_dis').agg({
            'weight2': 'sum',  # Total weight in the district
            'D12A': 'mean'  # Average income in the district
        }).reset_index()

        # Retain LFS_year (use the first value from the group, assuming all rows have the same year)
        district_data['LFS_year'] = dfs.groupby('code_dis')['LFS_year'].first().values

        # Trend Calculation: Calculate the annual growth rate for D12A
        district_data['growth_rate'] = district_data['D12A'].pct_change().fillna(0)

        # Use the historical average growth rate to extrapolate future values
        avg_growth_rate = district_data['growth_rate'].mean()

        # Add a progress bar to select the number of years to forecast
        forecast_years = st.slider('Select Number of Years to Forecast', min_value=1, max_value=10, value=5)
        st.write(f"Forecasting for {forecast_years} years...")

        # Initialize an empty DataFrame to store the forecasted data for all districts
        all_forecasts = []

        # Loop over each district and calculate future predictions
        for district in district_data['code_dis'].unique():
            district_subset = district_data[district_data['code_dis'] == district]

            # Simulate future income for both low and high unemployment levels using the average growth rate and randomness
            np.random.seed(42)  # for reproducibility
            randomness_factor_low = np.random.normal(0, 0.01, size=forecast_years)  # Slight randomness for low unemployment
            randomness_factor_high = np.random.normal(0, 0.02, size=forecast_years)  # Slightly higher randomness for high unemployment

            # Predict future income for both low and high unemployment levels
            future_years = np.arange(district_subset['LFS_year'].max() + 1, district_subset['LFS_year'].max() + forecast_years + 1)

            future_income_low = district_subset['D12A'].iloc[-1] * (1 + avg_growth_rate + randomness_factor_low)
            future_income_high = district_subset['D12A'].iloc[-1] * (1 + avg_growth_rate + randomness_factor_high)

            # Use the same weight for simplicity (could also vary)
            future_weight = district_subset['weight2'].iloc[-1]

            # Create DataFrames for future predictions for both low and high unemployment levels
            future_df_low = pd.DataFrame({
                'LFS_year': future_years,
                'predicted_income': future_income_low,
                'predicted_weight2': future_weight,
                'predicted_unemployment': 'Low',
                'code_dis': district  # Add district info for plotting
            })

            future_df_high = pd.DataFrame({
                'LFS_year': future_years,
                'predicted_income': future_income_high,
                'predicted_weight2': future_weight,
                'predicted_unemployment': 'High',
                'code_dis': district  # Add district info for plotting
            })

            # Calculate percentage change for low and high unemployment levels
            last_value = district_subset['D12A'].iloc[-1]
            future_df_low['percentage_change'] = ((future_df_low['predicted_income'] - last_value) / last_value) * 100
            future_df_high['percentage_change'] = ((future_df_high['predicted_income'] - last_value) / last_value) * 100

            # Combine both DataFrames for the district
            future_df_combined = pd.concat([future_df_low[['LFS_year', 'percentage_change', 'predicted_unemployment', 'code_dis', 'predicted_income']],
                                            future_df_high[['LFS_year', 'percentage_change', 'predicted_unemployment', 'code_dis', 'predicted_income']]])

            # Append the forecasted data for this district
            all_forecasts.append(future_df_combined)

        # Combine all the district forecasts into a single DataFrame
        combined_forecast_df = pd.concat(all_forecasts, ignore_index=True)

        # Visualize the forecast for both high and low unemployment levels by district using a bar chart
        forecast_fig = px.bar(
            combined_forecast_df,
            x='code_dis',  # District code on the x-axis
            y='predicted_income',  # Predicted income on the y-axis
            color='predicted_unemployment',  # Color by unemployment level
            title="Predicted Income in Future Years by District ",
            labels={'code_dis': 'District', 'predicted_income': 'Predicted Income'},
            height=400,
            width=800,
            hover_data={'code_dis': True, 'predicted_income': True, 'predicted_unemployment': True, 'percentage_change': True}
        )

        # Display the plot
        st.plotly_chart(forecast_fig, use_container_width=True)

        # Display forecasted values with percentages
        st.write("Future Forecasts for Low Unemployment:")
        st.write(future_df_low[['LFS_year', 'percentage_change']])

        st.write("Future Forecasts for High Unemployment:")
        st.write(future_df_high[['LFS_year', 'percentage_change']])

else:
    st.warning("Dataset not loaded. Please upload a valid dataset.")




# forecast year based on age,BO8,education_levels, etc

# Clean and Prepare the Data
if df is not None and dfs is not None:
    # Drop rows with NaN in required columns for both dataframes
    df = df.dropna(subset=['age', 'education_level'])
    dfs = dfs.dropna(subset=['B08', 'status1'])

    # Merge df and dfs on the common column (e.g., 'index' or another column)
    merged_data = pd.merge(df, dfs, left_index=True, right_index=True, how='inner')

    # Check if the merge was successful
    if merged_data.empty:
        st.warning("The merged dataset is empty. Please check your input data.")
    else:
        # Convert 'status1' into numerical values for prediction
        status_mapping = {
            'Employed': 1,
            'Unemployed': 0,  # Focus on both 'Employed' and 'Unemployed'
            'Out of labour force': 2  # Optionally handle this status if needed
        }
        merged_data['status1'] = merged_data['status1'].map(status_mapping)

        # Map 'B08' from 'No'/'Yes' to 0/1
        merged_data['B08'] = merged_data['B08'].map({'No': 0, 'Yes': 1})

        # One-hot encode `education_level`
        merged_data = pd.get_dummies(merged_data, columns=['education_level'], drop_first=True)

        # Prepare the data for prediction (we use both employed and unemployed)
        X = merged_data[['age', 'B08'] + [col for col in merged_data.columns if 'education_level' in col]]  # Features
        y = merged_data['status1']  # Target variable

        # Check if there is enough data to train the model
        if len(X) < 2:
            st.warning("Not enough data to split for training. Please ensure your dataset has sufficient data.")
        else:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train a RandomForestClassifier model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)



            # Simulate future data and predict unemployment status
            st.header("Future Predictions")

            # Add a unique key to the slider to avoid duplicate element ID issues
            forecast_years = st.slider('Select Number of Years for Forecasting', min_value=1, max_value=10, value=5,
                                       key="forecast_slider")

            # Simulate aging over the forecasted years (age capped to the range 16 to 30)
            future_data = X.copy()
            future_data['age'] = future_data['age'] + forecast_years  # Increase age by forecast years

            # Cap age to the range 16-30
            future_data['age'] = np.clip(future_data['age'], 16, 30)  # Ensure age is between 16 and 30

            # Predict future unemployment status
            future_predictions = model.predict(future_data)
            future_data['predicted_status1'] = future_predictions

            # Prepare data for bar chart visualization
            future_data['predicted_status1_label'] = future_data['predicted_status1'].map(
                {0: 'Employed', 1: 'Unemployed'})

            # Group by age and predicted status to count the number of people in each category
            status_counts = future_data.groupby(['age', 'predicted_status1_label']).size().reset_index(name='count')

            # Check if both 'Employed' and 'Unemployed' are present in the data
            if status_counts.empty:
                st.warning("No predictions for Unemployed status.")
            else:
                # Plot future predictions using a bar chart
                forecast_fig = px.bar(
                    status_counts,
                    x='age',  # Plot age on the x-axis
                    y='count',  # Count of people in each status (Unemployed)
                    color='predicted_status1_label',  # Color by the unemployment status
                    title="Future Unemployment Status Predictions by Age",
                    labels={'age': 'Age (Years)', 'count': 'Count', 'predicted_status1_label': 'Predicted Status'},
                    height=400,
                    width=800
                )

                st.plotly_chart(forecast_fig, use_container_width=True)

else:
    st.warning("Datasets not loaded. Please upload valid datasets.")

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
    age_bins = [16, 20, 25, 30]
    age_labels = ["16-19", "20-24", "25-30"]
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
                    "employment_success_rate": "Employment Success Rate (%)"}
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
            color="region"
            # title="Regional Employment Rate vs. Youth Unemployment Rate"
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


#theme
# hide_st_style="""
#
# <style>
# #MainMenu {visibility:hidden;}
# footer {visibility:hidden;}
# header {visibility:hidden;}
# </style>
# """