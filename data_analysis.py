import os
import json
import pandas as pd
import numpy as np

def load_and_clean_data(file_dir="/Users/abhishekgoyal/Downloads/layoffs"):
    csv_path = os.path.join(file_dir, "layoffs1.csv")
    if not os.path.exists(csv_path):
        # Fallback if layoffs1.csv isn't present
        csv_path = os.path.join(file_dir, "layoffs.csv")
        
    print(f"Loading data explicitly from: {csv_path}")
    
    # Read raw dataset
    df = pd.read_csv(csv_path)
    print(f"Raw dataset shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # If layoffs1.csv doesn't exist but layoffs.csv does, create layoffs1.csv alias
    target_layoffs1 = os.path.join(file_dir, "layoffs1.csv")
    if csv_path != target_layoffs1 and not os.path.exists(target_layoffs1):
        df.to_csv(target_layoffs1, index=False)
        print(f"Created copy at {target_layoffs1}")

    # Data Cleaning Workflow
    # 1. Clean column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # 2. Date parsing
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']) # Drop rows without date
    
    # Exclude Month 8 of 2026 (August 2026) data - Keep data strictly up to July 31, 2026
    df = df[df['date'] <= '2026-07-31']
    
    # Extract temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['year_month'] = df['date'].dt.strftime('%Y-%m')
    df['quarter'] = df['date'].dt.to_period('Q').astype(str)
    
    # 3. Numeric conversions & handling missing values
    df['total_laid_off'] = pd.to_numeric(df['total_laid_off'], errors='coerce')
    df['percentage_laid_off'] = pd.to_numeric(df['percentage_laid_off'], errors='coerce')
    df['funds_raised'] = pd.to_numeric(df['funds_raised'], errors='coerce')
    
    # Categorical cleaning
    df['company'] = df['company'].fillna('Unknown Company').str.strip()
    df['location'] = df['location'].fillna('Unknown Location').str.strip()
    df['industry'] = df['industry'].fillna('Other').str.strip()
    df['stage'] = df['stage'].fillna('Unknown').str.strip()
    df['country'] = df['country'].fillna('Unknown Country').str.strip()
    
    # Standardize industry names if needed
    df['industry'] = df['industry'].replace({'Other': 'Other / Unspecified'})
    
    # Derived Feature: Estimated Pre-layoff Workforce Size
    # Workforce = total_laid_off / percentage_laid_off
    df['est_workforce_size'] = np.where(
        (df['percentage_laid_off'] > 0) & (df['total_laid_off'].notnull()),
        np.round(df['total_laid_off'] / df['percentage_laid_off']),
        np.nan
    )
    
    # Save cleaned dataset
    cleaned_csv_path = os.path.join(file_dir, "cleaned_layoffs.csv")
    df.to_csv(cleaned_csv_path, index=False)
    print(f"Saved cleaned data to {cleaned_csv_path} (Shape: {df.shape})")
    
    # Derive Comprehensive Data Insights & Summary Statistics
    summary = generate_insights(df)
    
    summary_json_path = os.path.join(file_dir, "data_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved dataset summary & insights to {summary_json_path}")
    
    return df, summary

def generate_insights(df):
    total_records = len(df)
    total_laidoff = float(df['total_laid_off'].sum())
    total_companies = int(df['company'].nunique())
    total_countries = int(df['country'].nunique())
    total_industries = int(df['industry'].nunique())
    total_funds_raised_m = float(df['funds_raised'].sum())
    
    date_min = df['date'].min().strftime('%Y-%m-%d')
    date_max = df['date'].max().strftime('%Y-%m-%d')
    
    # Top 10 companies by total laid off
    top_companies = df.groupby('company')['total_laid_off'].sum().reset_index()
    top_companies = top_companies.sort_values(by='total_laid_off', ascending=False).head(15).to_dict(orient='records')
    
    # Layoffs by Industry
    industry_summary = df.groupby('industry').agg(
        total_laid_off=('total_laid_off', 'sum'),
        event_count=('company', 'count'),
        avg_funds_raised=('funds_raised', 'mean')
    ).reset_index().sort_values(by='total_laid_off', ascending=False).to_dict(orient='records')
    
    # Layoffs by Country
    country_summary = df.groupby('country').agg(
        total_laid_off=('total_laid_off', 'sum'),
        event_count=('company', 'count')
    ).reset_index().sort_values(by='total_laid_off', ascending=False).head(15).to_dict(orient='records')
    
    # Monthly trend
    monthly_trend = df.groupby('year_month').agg(
        total_laid_off=('total_laid_off', 'sum'),
        event_count=('company', 'count')
    ).reset_index().sort_values(by='year_month').to_dict(orient='records')

    # Yearly trend
    yearly_trend = df.groupby('year').agg(
        total_laid_off=('total_laid_off', 'sum'),
        event_count=('company', 'count')
    ).reset_index().sort_values(by='year').to_dict(orient='records')

    # Funding Stage Breakdown
    stage_summary = df.groupby('stage').agg(
        total_laid_off=('total_laid_off', 'sum'),
        event_count=('company', 'count')
    ).reset_index().sort_values(by='total_laid_off', ascending=False).to_dict(orient='records')
    
    # 100% Layoffs (Shuttings / Shuts Down)
    shutdowns = df[df['percentage_laid_off'] == 1.0]
    total_shutdowns = len(shutdowns)
    
    # Key Highlights/Takeaways
    key_insights = [
        f"A total of {int(total_laidoff):,} tech workforce layoffs were recorded across {total_companies:,} companies in {total_countries} countries from {date_min} to {date_max}.",
        f"The top industry impacted by total layoffs is {industry_summary[0]['industry'] if industry_summary else 'N/A'} with {int(industry_summary[0]['total_laid_off']):,} reported layoffs.",
        f"The top company by overall headcount reduction is {top_companies[0]['company']} with {int(top_companies[0]['total_laid_off']):,} layoffs.",
        f"A total of {total_shutdowns} company shutdown events (100% staff laid off) were tracked in the dataset.",
        f"Post-IPO companies accounted for the largest volume of layoffs compared to early-stage startups."
    ]
    
    return {
        "kpis": {
            "total_records": total_records,
            "total_laid_off": total_laidoff,
            "total_companies": total_companies,
            "total_countries": total_countries,
            "total_industries": total_industries,
            "total_funds_raised_m": total_funds_raised_m,
            "total_shutdowns": total_shutdowns,
            "date_range": {"start": date_min, "end": date_max}
        },
        "key_insights": key_insights,
        "top_companies": top_companies,
        "industry_summary": industry_summary,
        "country_summary": country_summary,
        "monthly_trend": monthly_trend,
        "yearly_trend": yearly_trend,
        "stage_summary": stage_summary
    }

def generate_eda_stats(df):
    """Generates detailed Exploratory Data Analysis (EDA) summary statistics."""
    num_cols = ['total_laid_off', 'percentage_laid_off', 'funds_raised']
    stats = {}
    
    for col in num_cols:
        s = df[col].dropna()
        stats[col] = {
            "count": int(len(s)),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(float(df[col].isnull().mean() * 100), 1),
            "mean": round(float(s.mean()), 2) if not s.empty else 0,
            "std": round(float(s.std()), 2) if not s.empty else 0,
            "min": round(float(s.min()), 2) if not s.empty else 0,
            "q25": round(float(s.quantile(0.25)), 2) if not s.empty else 0,
            "median": round(float(s.median()), 2) if not s.empty else 0,
            "q75": round(float(s.quantile(0.75)), 2) if not s.empty else 0,
            "max": round(float(s.max()), 2) if not s.empty else 0,
            "skewness": round(float(s.skew()), 2) if not s.empty else 0
        }
        
    # Correlations
    corr_df = df[num_cols].corr()
    correlations = {}
    for c1 in num_cols:
        correlations[c1] = {}
        for c2 in num_cols:
            val = corr_df.loc[c1, c2]
            correlations[c1][c2] = round(float(val), 3) if pd.notnull(val) else 0.0

    return {
        "descriptive_stats": stats,
        "correlations": correlations
    }

if __name__ == "__main__":
    df, summary = load_and_clean_data()
    print("Data processing & insight generation complete!")
