# 📉 Global Tech Layoffs Data Analysis & Machine Learning Platform

An end-to-end Python Data Analytics, Exploratory Data Analysis (EDA), and Machine Learning prediction project built on the global tech workforce layoffs dataset (`layoffs1.csv`).

---

## ✨ Features & Key Capabilities

- **Strict Date Range & Dataset Integrity**: 
  - Excludes Month 8 of 2026 (August 2026) data, covering strictly **March 11, 2020 to July 31, 2026** (**4,543 clean records**).
  - All analysis, metrics, and machine learning models strictly process real clean records without inserting any artificial or synthetic rows.
- **Scikit-Learn Machine Learning Engine (`ml_model.py`)**:
  - Uses `RandomForestRegressor` with log-transformed target variables $\log(1 + \text{total\_laid\_off})$ to predict expected headcount cut size and classify risk tiers (*Low, Moderate, High, Severe Risk*).
- **Sidebar ML Predictor Widget**:
  - Located directly in the **Left Sidebar** for instant interactive risk prediction based on Industry, Stage, Country, and Funding.
- **Layoffs by Companies (Descending Order)**:
  - Visualizes companies sorted strictly in **descending order** of total headcount cuts, with an interactive Company filter dropdown directly in the chart header.
- **Monthly Layoffs Timeline & Year Filter**:
  - Features a dedicated Year selector (`All Years`, `2026`, `2025`, `2024`, `2023`, `2022`, `2021`, `2020`) and Granularity toggle (`Monthly View` / `Yearly Aggregate`) right on the chart card.
- **Column-Level Dataset Log Filters**:
  - The Filtered Dataset Log table features inline **column-level input/dropdown filters** for *Company, Industry, Min Cut Size, Funding Stage, Country,* and *Year*.
- **FastAPI High-Performance Backend (`app.py`)**:
  - Serves REST endpoints for KPIs, monthly/yearly trends, industry breakdown, country distribution, funding stage breakdown, top companies, EDA statistics, ML predictions, and CSV exports.
- **EDA Descriptive Stats**:
  - Live display of Mean, Median, Std Dev, Min, Max, Quantiles, Skewness, Null Ratio, and Pearson Correlations.

---

## 📁 Directory & File Structure

```
layoffs/
├── layoffs1.csv          # Source Dataset (up to July 31, 2026)
├── cleaned_layoffs.csv   # Processed clean dataset
├── data_analysis.py      # Cleaning pipeline & EDA stats generator
├── ml_model.py           # Machine Learning Random Forest Predictor
├── app.py                # FastAPI Backend Web Server
├── streamlit_app.py      # Streamlit Python Interactive App
├── README.md             # Project Documentation
└── static/
    ├── index.html        # Main Web Dashboard Interface
    ├── style.css         # Modern Dark Glassmorphic Design System
    └── app.js            # Client UI Logic & Chart.js Controllers
```

---

## 🚀 Installation & Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pandas numpy scikit-learn plotly streamlit
```

### 2. Preprocess & Clean Dataset
```bash
python3 data_analysis.py
```

### 3. Launch FastAPI Backend & Dashboard
```bash
python3 app.py
```
- Access Web Dashboard: **`http://localhost:8000`**
- Interactive Swagger API Docs: **`http://localhost:8000/docs`**

### 4. Launch Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
- Access Streamlit Dashboard: **`http://localhost:8501`**

---

## 📡 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /` | `GET` | Serves main interactive dashboard HTML |
| `GET /api/filters` | `GET` | Returns lists of available Companies, Years, Industries, Countries, Stages |
| `GET /api/kpis` | `GET` | Calculates summary KPIs (Total Laid Off, Shutdowns, Avg Cut) |
| `GET /api/trends` | `GET` | Monthly and Yearly layoff trends timeline |
| `GET /api/by-industry` | `GET` | Aggregated layoffs by industry sector |
| `GET /api/by-country` | `GET` | Geographic layoffs distribution |
| `GET /api/by-stage` | `GET` | Layoffs grouped by funding stage |
| `GET /api/top-companies` | `GET` | Companies sorted in descending order of total layoffs |
| `GET /api/eda` | `GET` | Descriptive statistics and Pearson correlation matrix |
| `GET /api/predict` | `GET` | Predicts expected layoff size and risk tier using ML model |
| `GET /api/layoffs` | `GET` | Paginated dataset records with column-level filtering |
| `GET /api/export` | `GET` | Exports filtered dataset directly as a CSV file |

---

## 📄 License
This project is open-source and free to use for data analysis, machine learning research, and education.
