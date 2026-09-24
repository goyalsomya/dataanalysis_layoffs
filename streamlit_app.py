import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_model import predictor
from data_analysis import generate_eda_stats

# Streamlit Page Config
st.set_page_config(
    page_title="Global Tech Layoffs Analytics & ML Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Styling
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stMetric {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(base_dir, "layoffs1.csv")
    if not os.path.exists(target_path):
        target_path = os.path.join(base_dir, "cleaned_layoffs.csv")
    if not os.path.exists(target_path):
        target_path = os.path.join(base_dir, "layoffs.csv")
    
    df = pd.read_csv(target_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        # Exclude Month 8 of 2026 (August 2026) data - Keep data strictly up to July 31, 2026
        df = df[df['date'] <= '2026-07-31']
        df['year'] = df['date'].dt.year
        df['year_month'] = df['date'].dt.strftime('%Y-%m')
        
    for num_col in ['total_laid_off', 'percentage_laid_off', 'funds_raised']:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce')
            
    df['company'] = df['company'].fillna('Unknown Company').astype(str)
    df['industry'] = df['industry'].fillna('Other').astype(str)
    df['country'] = df['country'].fillna('Unknown Country').astype(str)
    df['stage'] = df['stage'].fillna('Unknown').astype(str)
    return df

df = load_data()

# Ensure ML predictor is trained on existing clean data
if not predictor.is_trained:
    predictor.train(df)

# Header Section
st.title("📊 Global Tech Layoffs Analysis & ML Predictor")
st.markdown("An interactive Python analytics app exploring workforce reductions, Exploratory Data Analysis (EDA), and Machine Learning prediction.")

# Sidebar Filters
st.sidebar.header("🔍 Global Filters")

# Company Filter
top_companies_list = ["All"] + sorted([c for c in df['company'].value_counts().head(50).index if c])
selected_company = st.sidebar.selectbox("Filter by Company", top_companies_list)

# Year Filter
years = ["All"] + sorted([str(int(y)) for y in df['year'].dropna().unique()], reverse=True)
selected_year = st.sidebar.selectbox("Filter by Year", years)

# Industry Filter
industries = ["All"] + sorted([i for i in df['industry'].unique() if i])
selected_industry = st.sidebar.selectbox("Filter by Industry", industries)

# Country Filter
countries = ["All"] + sorted([c for c in df['country'].unique() if c])
selected_country = st.sidebar.selectbox("Filter by Country", countries)

# Stage Filter
stages = ["All"] + sorted([s for s in df['stage'].unique() if s])
selected_stage = st.sidebar.selectbox("Filter by Funding Stage", stages)

# Text Search Filter
search_term = st.sidebar.text_input("Search Company Name", "")

# Apply Filters
filtered_df = df.copy()

if selected_company != "All":
    filtered_df = filtered_df[filtered_df['company'] == selected_company]
if selected_year != "All":
    filtered_df = filtered_df[filtered_df['year'] == int(selected_year)]
if selected_industry != "All":
    filtered_df = filtered_df[filtered_df['industry'] == selected_industry]
if selected_country != "All":
    filtered_df = filtered_df[filtered_df['country'] == selected_country]
if selected_stage != "All":
    filtered_df = filtered_df[filtered_df['stage'] == selected_stage]
if search_term:
    filtered_df = filtered_df[filtered_df['company'].str.contains(search_term, case=False, na=False)]

# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

total_laidoff = int(filtered_df['total_laid_off'].sum())
total_companies = filtered_df['company'].nunique()
total_countries = filtered_df['country'].nunique()
shutdown_count = int((filtered_df['percentage_laid_off'] == 1.0).sum())
avg_layoff = int(filtered_df['total_laid_off'].mean()) if len(filtered_df) > 0 else 0

with col1:
    st.metric("Total Laid Off", f"{total_laidoff:,}")
with col2:
    st.metric("Companies", f"{total_companies:,}")
with col3:
    st.metric("Countries", f"{total_countries}")
with col4:
    st.metric("Shutdowns (100%)", f"{shutdown_count}")
with col5:
    st.metric("Avg Layoff / Event", f"{avg_layoff:,}")

st.markdown("---")

# Analytics Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Trends & Timeline",
    "🏭 Industry & Stage",
    "🏢 Top Companies",
    "🤖 ML Layoffs Predictor",
    "📊 EDA Descriptive Stats",
    "📄 Filtered Data & Export"
])

with tab1:
    st.subheader("Monthly Layoffs Trend")
    trend_df = filtered_df.groupby('year_month')['total_laid_off'].sum().reset_index().sort_values('year_month')
    fig_trend = px.area(
        trend_df,
        x='year_month',
        y='total_laid_off',
        labels={'year_month': 'Month', 'total_laid_off': 'Total Laid Off'},
        title="Monthly Layoffs Volume Over Time (Filtered)",
        color_discrete_sequence=['#58a6ff']
    )
    fig_trend.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    col_ind, col_stg = st.columns(2)
    with col_ind:
        st.subheader("Layoffs by Industry")
        ind_df = filtered_df.groupby('industry')['total_laid_off'].sum().reset_index().sort_values('total_laid_off', ascending=False).head(12)
        fig_ind = px.bar(
            ind_df,
            x='total_laid_off',
            y='industry',
            orientation='h',
            title="Top 12 Impacted Industries",
            color='total_laid_off',
            color_continuous_scale='Reds'
        )
        fig_ind.update_layout(template="plotly_dark", height=450, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_ind, use_container_width=True)
        
    with col_stg:
        st.subheader("Layoffs by Funding Stage")
        stage_df = filtered_df.groupby('stage')['total_laid_off'].sum().reset_index().sort_values('total_laid_off', ascending=False)
        fig_stg = px.pie(
            stage_df,
            names='stage',
            values='total_laid_off',
            title="Distribution by Funding Stage",
            hole=0.4
        )
        fig_stg.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_stg, use_container_width=True)

with tab3:
    st.subheader("Top Companies by Headcount Reduction")
    top_co_df = filtered_df.groupby('company').agg(
        total_laid_off=('total_laid_off', 'sum'),
        industry=('industry', 'first'),
        country=('country', 'first'),
        stage=('stage', 'first')
    ).reset_index().sort_values('total_laid_off', ascending=False).head(15)
    
    fig_co = px.bar(
        top_co_df,
        x='company',
        y='total_laid_off',
        color='industry',
        title="Top 15 Companies by Headcount Cut",
        labels={'company': 'Company', 'total_laid_off': 'Total Laid Off'}
    )
    fig_co.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_co, use_container_width=True)

with tab4:
    st.subheader("🤖 Machine Learning Layoffs Predictor")
    st.markdown("Estimate expected layoff size and risk profile using Random Forest Regression trained on historical data.")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p_industry = st.selectbox("Select Industry", sorted([i for i in df['industry'].unique() if i]))
        p_stage = st.selectbox("Select Funding Stage", sorted([s for s in df['stage'].unique() if s]))
    with col_p2:
        p_country = st.selectbox("Select Country", sorted([c for c in df['country'].unique() if c]))
        p_funds = st.number_input("Funds Raised ($ Millions)", min_value=0.0, value=500.0, step=50.0)
        
    if st.button("🔮 Calculate ML Prediction"):
        pred_res = predictor.predict(p_industry, p_stage, p_country, p_funds)
        st.markdown(f"### Predicted Layoff Headcount: **{pred_res['predicted_layoffs']:,}** employees")
        st.markdown(f"**Risk Level:** `{pred_res['risk_level']}` | **Model R² Confidence:** `{pred_res['model_r2']*100:.1f}%`")

with tab5:
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    eda_data = generate_eda_stats(df)
    
    st.markdown("#### Summary Descriptive Statistics")
    eda_df = pd.DataFrame(eda_data["descriptive_stats"]).T
    st.dataframe(eda_df, use_container_width=True)
    
    st.markdown("#### Pearson Correlation Matrix")
    corr_df = pd.DataFrame(eda_data["correlations"])
    st.dataframe(corr_df, use_container_width=True)

with tab6:
    st.subheader("Filterable Dataset")
    st.dataframe(filtered_df, use_container_width=True)
    
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="layoffs_filtered_analysis.csv",
        mime="text/csv"
    )
