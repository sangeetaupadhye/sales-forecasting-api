# Sales Forecasting System — End-to-End Time Series API

## Overview

A production-ready forecasting system that trains **4 models per US state**, automatically selects the best performer, and serves 8-week forecasts via a **REST API**.

---

## Project Structure

```
forecasting_project/
├── data/
│   └── sales_data.csv              # Input dataset (State, Date, Total, Category)
├── utils/
│   └── preprocessing.py            # Data cleaning, feature engineering, train/val split
├── models/
│   ├── forecasters.py              # SARIMA, Prophet, XGBoost, LSTM implementations
│   └── model_selector.py           # Training orchestrator + best-model selection
├── api/
│   └── app.py                      # Flask REST API
├── trained_models/
│   ├── forecast_summary.json       # All forecasts + metrics (auto-generated)
│   └── <State>_best_model.pkl      # Serialized best model per state
├── train.py                        # Entry point: run full training pipeline
└── README.md
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn statsmodels prophet xgboost flask tensorflow
```

### 2. Train All Models
```bash
python train.py
```
This will:
- Load and clean `data/sales_data.csv`
- Engineer features (lags, rolling stats, calendar)
- Train 4 models per state with time-series cross-validation
- Auto-select the best model by RMSE
- Save results to `trained_models/forecast_summary.json`

### 3. Start the API
```bash
python api/app.py
```
API runs at `http://localhost:5000`

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/states` | List all 43 available states |
| GET | `/forecast/<state>` | 8-week forecast using best model |
| GET | `/forecast/<state>?model=XGBoost` | Forecast using a specific model |
| GET | `/models/<state>` | All model metrics for comparison |
| POST | `/retrain` | Reload latest trained results |

### Example Calls
```bash
# Health check
curl http://localhost:5000/health

# Get 8-week forecast for California
curl http://localhost:5000/forecast/California

# Force XGBoost for Texas
curl "http://localhost:5000/forecast/Texas?model=XGBoost"

# Compare all models for Florida
curl http://localhost:5000/models/Florida
```

### Sample Response — `/forecast/California`
```json
{
  "state": "California",
  "best_model": "SARIMA",
  "model_used": "SARIMA",
  "forecast_periods": 8,
  "forecast": [
    {"date": "2023-12-10", "predicted_sales": 841727353.09},
    {"date": "2023-12-17", "predicted_sales": 825985469.12},
    ...
  ],
  "metrics": {"rmse": 61535756.0, "mape": 6.32}
}
```

---

## Models Implemented

### 1. SARIMA (Auto-tuned)
- Grid search over `(p,d,q)` × seasonal `(1,0,1,4)` orders
- Best order selected by AIC
- Handles trend and quarterly seasonality

### 2. Facebook Prophet
- Multiplicative seasonality mode
- Yearly seasonality enabled
- Frequency auto-detected from data gaps

### 3. XGBoost with Lag Features
- Lag features: t-1, t-2, t-3 (last 3 observations)
- Rolling mean/std (3-period window)
- Calendar features: month, quarter, day-of-week, week-of-year, holiday flag
- Iterative one-step-ahead prediction for multi-step forecasting

### 4. LSTM (Deep Learning)
- 2-layer LSTM with dropout regularization
- MinMax scaled inputs
- Sequence length = 4 (adaptively reduced for short series)

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `lag_1, lag_2, lag_3` | Previous 1/2/3 observations per state |
| `rolling_mean_3` | 3-period rolling mean (shift-1 to avoid leakage) |
| `rolling_std_3` | 3-period rolling std |
| `month`, `quarter` | Calendar period features |
| `dayofweek`, `weekofyear` | Sub-monthly temporal features |
| `holiday_flag` | US major holidays binary flag |
| `obs_index` | Monotonic trend proxy per state |

**No data leakage**: all lag/rolling features use `shift(1)` before rolling.

---

## Model Selection Logic

1. Each state's data is split: last 4 observations → validation, rest → training
2. All models are trained on training set and evaluated on validation set
3. Best model = lowest **RMSE** on validation data
4. Best model is used for 8-period future forecasting

---

## Dataset Details

- **Source**: State-level Beverage sales data
- **States**: 43 US states
- **Date range**: Jan 2019 → Dec 2023
- **Frequency**: Approximately monthly/quarterly (mixed)
- **Total records**: 8,084 rows

---

## Design Decisions

- **Mixed date formats** (M/D/YYYY and DD-MM-YYYY) handled via multi-format parsing loop
- **Numeric cleaning**: commas and spaces stripped from the `Total` column
- **Frequency inference**: median gap between dates → maps to weekly/monthly/quarterly
- **Graceful fallbacks**: each model has a fallback if fitting fails
- **JSON serializable output**: all forecasts saved as plain JSON for easy API loading
