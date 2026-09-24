import os
import io
import json
import pandas as pd
import numpy as np
from typing import Optional
from fastapi import FastAPI, Query, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from ml_model import predictor
from data_analysis import generate_eda_stats

# Initialize FastAPI App
app = FastAPI(
    title="Global Tech Layoffs Analytics & ML Platform",
    description="Python Backend API for Layoff Data Analysis, EDA, & Machine Learning Prediction",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "cleaned_layoffs.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "layoffs1.csv")

df_global = None

def load_dataset():
    global df_global
    target_path = RAW_DATA_PATH if os.path.exists(RAW_DATA_PATH) else CLEANED_DATA_PATH
    if not os.path.exists(target_path):
        target_path = os.path.join(BASE_DIR, "layoffs.csv")

    df = pd.read_csv(target_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        # Exclude Month 8 of 2026 (August 2026) data - Keep data strictly up to July 31, 2026
        df = df[df['date'] <= '2026-07-31']
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['year_month'] = df['date'].dt.strftime('%Y-%m')
    
    for num_col in ['total_laid_off', 'percentage_laid_off', 'funds_raised']:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce')

    df['company'] = df['company'].fillna('Unknown Company').astype(str)
    df['industry'] = df['industry'].fillna('Other').astype(str)
    df['country'] = df['country'].fillna('Unknown Country').astype(str)
    df['stage'] = df['stage'].fillna('Unknown').astype(str)
    
    df_global = df
    print(f"Loaded {len(df_global)} records into API memory.")
    
    # Train ML Predictor Engine strictly on non-null original data
    try:
        predictor.train(df_global)
    except Exception as e:
        print(f"Warning: ML model training deferment: {e}")

def to_clean_json_records(df_slice):
    records = df_slice.to_dict(orient='records')
    for row in records:
        for k, v in row.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                row[k] = None
    return records

@app.on_event("startup")
def startup_event():
    load_dataset()

def filter_dataframe(
    industry: Optional[str] = None,
    country: Optional[str] = None,
    stage: Optional[str] = None,
    year: Optional[int] = None,
    company: Optional[str] = None,
    search: Optional[str] = None,
    min_laidoff: Optional[float] = None
):
    df = df_global.copy()
    if industry and industry != "All":
        df = df[df['industry'].str.lower() == industry.lower()]
    if country and country != "All":
        df = df[df['country'].str.lower() == country.lower()]
    if stage and stage != "All":
        df = df[df['stage'].str.lower() == stage.lower()]
    if year and year != 0:
        df = df[df['year'] == year]
    if company and company != "All":
        df = df[df['company'].str.lower() == company.lower()]
    if search:
        df = df[df['company'].str.contains(search, case=False, na=False)]
    if min_laidoff is not None and min_laidoff > 0:
        df = df[df['total_laid_off'] >= min_laidoff]
    return df

# API Endpoints
@app.get("/api/filters")
def get_filter_options():
    if df_global is None:
        load_dataset()
    industries = sorted([i for i in df_global['industry'].unique() if i and i != 'nan'])
    countries = sorted([c for c in df_global['country'].unique() if c and c != 'nan'])
    stages = sorted([s for s in df_global['stage'].unique() if s and s != 'nan'])
    years = sorted([int(y) for y in df_global['year'].dropna().unique()], reverse=True)
    
    # Top companies for company dropdown filter
    top_companies_list = sorted([c for c in df_global['company'].value_counts().head(50).index if c])
    
    return {
        "industries": ["All"] + industries,
        "countries": ["All"] + countries,
        "stages": ["All"] + stages,
        "years": [0] + years,
        "companies": ["All"] + top_companies_list
    }

@app.get("/api/kpis")
def get_kpis(
    industry: Optional[str] = None,
    country: Optional[str] = None,
    stage: Optional[str] = None,
    year: Optional[int] = None,
    company: Optional[str] = None,
    search: Optional[str] = None
):
    df = filter_dataframe(industry, country, stage, year, company, search)
    
    total_laid_off = float(df['total_laid_off'].sum())
    total_events = len(df)
    total_companies = int(df['company'].nunique())
    total_countries = int(df['country'].nunique())
    total_industries = int(df['industry'].nunique())
    total_funds = float(df['funds_raised'].sum())
    shutdown_count = int((df['percentage_laid_off'] == 1.0).sum())
    avg_layoff_per_event = float(df['total_laid_off'].mean()) if len(df) > 0 else 0
    
    date_min = df['date'].min().strftime('%Y-%m-%d') if not df['date'].empty and pd.notnull(df['date'].min()) else "N/A"
    date_max = df['date'].max().strftime('%Y-%m-%d') if not df['date'].empty and pd.notnull(df['date'].max()) else "N/A"

    return {
        "total_laid_off": total_laid_off,
        "total_events": total_events,
        "total_companies": total_companies,
        "total_countries": total_countries,
        "total_industries": total_industries,
        "total_funds_raised_m": total_funds,
        "shutdown_count": shutdown_count,
        "avg_layoff_per_event": round(avg_layoff_per_event, 1),
        "date_range": {"start": date_min, "end": date_max}
    }

@app.get("/api/trends")
def get_trends(
    industry: Optional[str] = None,
    country: Optional[str] = None,
    stage: Optional[str] = None,
    year: Optional[int] = None,
    company: Optional[str] = None,
    search: Optional[str] = None,
    group_by: str = "month"
):
    df = filter_dataframe(industry, country, stage, year, company, search)
    if group_by == "year":
        grouped = df.groupby('year').agg(
            total_laid_off=('total_laid_off', 'sum'),
            event_count=('company', 'count')
        ).reset_index().rename(columns={'year': 'label'})
    else:
        grouped = df.groupby('year_month').agg(
            total_laid_off=('total_laid_off', 'sum'),
            event_count=('company', 'count')
        ).reset_index().rename(columns={'year_month': 'label'}).sort_values('label')
        
    grouped['total_laid_off'] = grouped['total_laid_off'].fillna(0)
    grouped = grouped.where(pd.notnull(grouped), None)
    return grouped.to_dict(orient='records')

@app.get("/api/by-industry")
def get_by_industry(
    country: Optional[str] = None,
    stage: Optional[str] = None,
    year: Optional[int] = None,
    company: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 15
):
    df = filter_dataframe(country=country, stage=stage, year=year, company=company, search=search)
    grouped = df.groupby('industry').agg(
        total_laid_off=('total_laid_off', 'sum'),
        event_count=('company', 'count')
    ).reset_index().sort_values('total_laid_off', ascending=False).head(limit)
    grouped = grouped.where(pd.notnull(grouped), None)
    return grouped.to_dict(orient='records')

@app.get("/api/by-country")
def get_by_country(
    industry: Optional[str] = None,
    stage: Optional[str] = None,
    year: Optional[int] = None,
    company: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 15
):
    df = filter_dataframe(industry=industry, stage=stage, year=year, company=company, search=search)
    grouped = df.groupby('country').agg(
        total_laid_off=('total_laid_off', 'sum'),
        event_count=('company', 'count')
    ).reset_index().sort_values('total_laid_off', ascending=False).head(limit)
    grouped = grouped.where(pd.notnull(grouped), None)
    return grouped.to_dict(orient='records')

@app.get("/api/by-stage")
def get_by_stage(
    industry: Optional[str] = None,
    country: Optional[str] = None,
    year: Optional[int] = None,
    company: Optional[str] = None,
    search: Optional[str] = None
):
    df = filter_dataframe(industry=industry, country=country, year=year, company=company, search=search)
    grouped = df.groupby('stage').agg(
        total_laid_off=('total_laid_off', 'sum'),
        event_count=('company', 'count')
    ).reset_index().sort_values('total_laid_off', ascending=False)
    grouped = grouped.where(pd.notnull(grouped), None)
    return grouped.to_dict(orient='records')

@app.get("/api/top-companies")
def get_top_companies(
    limit: int = 10,
    industry: Optional[str] = None,
    country: Optional[str] = None,
    stage: Optional[str] = None,
    year: Optional[int] = None,
    company: Optional[str] = None,
    search: Optional[str] = None
):
    df = filter_dataframe(industry=industry, country=country, stage=stage, year=year, company=company, search=search)
    grouped = df.groupby('company').agg(
        total_laid_off=('total_laid_off', 'sum'),
        industry=('industry', 'first'),
        country=('country', 'first'),
        stage=('stage', 'first'),
        event_count=('company', 'count')
    ).reset_index().sort_values('total_laid_off', ascending=False).head(limit)
    grouped = grouped.where(pd.notnull(grouped), None)
    return grouped.to_dict(orient='records')

@app.get("/api/eda")
def get_eda():
    """Returns Exploratory Data Analysis (EDA) summary statistics and correlations."""
    if df_global is None:
        load_dataset()
    return generate_eda_stats(df_global)

@app.get("/api/predict")
def predict_layoffs(
    industry: str = "Retail",
    stage: str = "Post-IPO",
    country: str = "United States",
    funds_raised: float = 500.0,
    year: int = 2026
):
    """Machine Learning Prediction Endpoint."""
    if not predictor.is_trained and df_global is not None:
        predictor.train(df_global)
    return predictor.predict(industry, stage, country, funds_raised, year)

@app.get("/api/insights")
def get_insights():
    df = df_global.copy()
    top_co = df.groupby('company')['total_laid_off'].sum().idxmax()
    top_co_val = int(df.groupby('company')['total_laid_off'].sum().max())
    
    top_ind = df.groupby('industry')['total_laid_off'].sum().idxmax()
    top_ind_val = int(df.groupby('industry')['total_laid_off'].sum().max())
    
    top_country = df.groupby('country')['total_laid_off'].sum().idxmax()
    top_country_val = int(df.groupby('country')['total_laid_off'].sum().max())
    
    shutdowns = int((df['percentage_laid_off'] == 1.0).sum())
    
    peak_month_row = df.groupby('year_month')['total_laid_off'].sum().reset_index().sort_values('total_laid_off', ascending=False).iloc[0]
    peak_month = peak_month_row['year_month']
    peak_month_val = int(peak_month_row['total_laid_off'])

    return {
        "insights": [
            f"🔥 **Highest Single Company Layoffs:** **{top_co}** leads global headcount cuts with **{top_co_val:,}** total employees affected.",
            f"🏭 **Most Impacted Sector:** **{top_ind}** experienced the largest cumulative layoffs (**{top_ind_val:,}** affected staff).",
            f"🌍 **Geographic Hotspot:** **{top_country}** represents the highest volume of tech layoffs globally (**{top_country_val:,}**).",
            f"📅 **Peak Layoff Spike:** The largest single monthly wave occurred in **{peak_month}** with **{peak_month_val:,}** layoffs reported.",
            f"🚨 **Full Company Shutdowns:** **{shutdowns}** tech startups closed operations completely (100% layoff rate)."
        ]
    }

@app.get("/api/layoffs")
def get_layoffs_table(
    page: int = 1,
    page_size: int = 15,
    search: Optional[str] = None,
    industry: Optional[str] = None,
    country: Optional[str] = None,
    stage: Optional[str] = None,
    year: Optional[int] = None,
    company: Optional[str] = None,
    sort_by: str = "total_laid_off",
    sort_order: str = "desc"
):
    df = filter_dataframe(industry, country, stage, year, company, search)
    
    ascending = (sort_order.lower() == "asc")
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending, na_position='last')
    
    total_records = len(df)
    total_pages = (total_records + page_size - 1) // page_size if total_records > 0 else 1
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    sliced = df.iloc[start_idx:end_idx].copy()
    sliced['date_str'] = sliced['date'].dt.strftime('%Y-%m-%d').fillna('N/A')
    records = to_clean_json_records(sliced)
    
    return {
        "page": page,
        "page_size": page_size,
        "total_records": total_records,
        "total_pages": total_pages,
        "data": records
    }

@app.get("/api/export")
def export_csv(
    industry: Optional[str] = None,
    country: Optional[str] = None,
    stage: Optional[str] = None,
    year: Optional[int] = None,
    company: Optional[str] = None,
    search: Optional[str] = None
):
    df = filter_dataframe(industry, country, stage, year, company, search)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    response = Response(content=stream.getvalue(), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=filtered_layoffs_data.csv"
    return response

# Serve static frontend files
static_path = os.path.join(BASE_DIR, "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/", response_class=HTMLResponse)
def root_dashboard():
    index_file = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            return f.read()
    return "<h1>Layoffs Analytics API Running</h1><p>Visit <a href='/docs'>/docs</a> for API Swagger Documentation.</p>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
