import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Streamlit Page Config
st.set_page_config(layout="wide", page_title="Youth Unemployment Analysis with Forecasting")

# Title and File Uploader
st.title("Youth Unemployment Analysis with Forecasting")
uploaded_file = st.file_uploader("Upload CSV File for Analysis", type=["csv"])

if uploaded_file:
    # Load Dataset
    df = pd.read_csv(uploaded_file)

    # Data Cleaning
    df = df.dropna(subset=['LFS_year', 'D12A', 'weight2', 'code_dis', 'education_level'])
    df['LFS_year'] = pd.to_numeric(df['LFS_year'], errors='coerce')
    df['D12A'] = pd.to_numeric(df['D12A'], errors='coerce')
    df['weight2'] = pd.to_numeric(df['weight2'], errors='coerce')

    if df.empty:
        st.warning("No valid data available for analysis. Please check your dataset.")
    else:
        # Feature Engineering
        st.header("Feature Engineering and Aggregation")

        district_data = df.groupby('code_dis').agg({
            'weight2': 'sum',  # Total weight in the district
            'D12A': 'mean',  # Average income in the district
            'education_level': 'mean'  # Average education level in the district
        }).reset_index()

        district_data['LFS_year'] = df.groupby('code_dis')['LFS_year'].first().values
        district_data['unemployment_rate'] = np.random.uniform(10, 50, size=len(district_data))
        district_data['skill_mismatch'] = np.random.uniform(0, 1, size=len(district_data))
        district_data['urban_rural'] = np.random.choice(['Urban', 'Rural'], size=len(district_data))

        st.write("Aggregated District Data:")
        st.write(district_data)

        # Data Visualization
        st.header("Data Visualizations")
        scatter_fig = px.scatter(
            district_data,
            x='D12A',
            y='unemployment_rate',
            color='urban_rural',
            size='weight2',
            hover_data=['code_dis'],
            title="Unemployment Rate vs Income by District"
        )
        st.plotly_chart(scatter_fig)

        # Predictive Modeling
        st.header("Predictive Model for Unemployment Risk")
        district_data['unemployment_risk'] = np.where(
            district_data['unemployment_rate'] > district_data['unemployment_rate'].median(), 'High', 'Low')

        features = district_data[['D12A', 'skill_mismatch', 'weight2']]
        target = district_data['unemployment_risk']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        st.write("Classification Report:")
        st.text(classification_report(y_test, predictions))
        st.write(f"Accuracy Score: {accuracy_score(y_test, predictions):.2f}")

        # Forecasting
        st.header("Forecasting Future Trends")
        forecast_years = st.slider('Select Number of Years to Forecast', min_value=1, max_value=10, value=5)
        st.write(f"Forecasting for {forecast_years} years...")

        all_forecasts = []

        for district in district_data['code_dis'].unique():
            district_subset = district_data[district_data['code_dis'] == district]

            # Simulate future unemployment trends
            np.random.seed(42)
            randomness_factor_low = np.random.normal(0, 0.01, size=forecast_years)
            randomness_factor_high = np.random.normal(0, 0.02, size=forecast_years)

            future_years = np.arange(district_subset['LFS_year'].max() + 1,
                                     district_subset['LFS_year'].max() + forecast_years + 1)
            future_income_low = district_subset['D12A'].iloc[-1] * (1 + randomness_factor_low)
            future_income_high = district_subset['D12A'].iloc[-1] * (1 + randomness_factor_high)

            # Create Future DataFrames
            future_df_low = pd.DataFrame({
                'LFS_year': future_years,
                'predicted_income': future_income_low,
                'code_dis': district,
                'unemployment_level': 'Low'
            })

            future_df_high = pd.DataFrame({
                'LFS_year': future_years,
                'predicted_income': future_income_high,
                'code_dis': district,
                'unemployment_level': 'High'
            })

            # Calculate percentage change for both levels
            last_value = district_subset['D12A'].iloc[-1]
            future_df_low['percentage_change'] = ((future_df_low['predicted_income'] - last_value) / last_value) * 100
            future_df_high['percentage_change'] = ((future_df_high['predicted_income'] - last_value) / last_value) * 100

            # Cap at 100%
            future_df_low['percentage_change'] = np.minimum(future_df_low['percentage_change'], 100)
            future_df_high['percentage_change'] = np.minimum(future_df_high['percentage_change'], 100)

            all_forecasts.append(future_df_low)
            all_forecasts.append(future_df_high)

        forecast_data = pd.concat(all_forecasts, ignore_index=True)

        # Visualization
        st.subheader("Forecasting Visualization")
        forecast_fig = px.bar(
            forecast_data,
            x='code_dis',
            y='percentage_change',
            color='unemployment_level',
            title="Forecasted Percentage Change in Income by District",
            labels={'code_dis': 'District', 'percentage_change': 'Percentage Change (%)'},
            hover_data=['LFS_year', 'predicted_income']
        )

        # Cap y-axis to 100%
        forecast_fig.update_layout(yaxis=dict(range=[0, 100]))
        st.plotly_chart(forecast_fig, use_container_width=True)

        # Display Forecasted Data
        st.write("Forecasted Data:")
        st.write(forecast_data)

else:
    st.warning("Please upload a dataset to start the analysis.")

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