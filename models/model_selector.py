"""
Model Selector & Training Orchestrator
- Trains all 4 models per state
- Evaluates on hold-out validation set (RMSE + MAPE)
- Selects best model automatically
- Generates 8-week (2 future period) forecasts
"""

import numpy as np
import pandas as pd
import json, os, pickle, time
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from utils.preprocessing import train_val_split, get_state_series
from models.forecasters import (SARIMAModel, ProphetModel,
                                 XGBoostModel, LSTMModel,
                                 rmse, mape)


FORECAST_PERIODS = 8   # 8 future weekly-equivalent periods ≈ 8 weeks


def train_and_select(state_df: pd.DataFrame, state: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Train all models, evaluate on validation set, pick best by RMSE.
    Returns a dict with: best_model_name, metrics, forecasts, all_forecasts
    """
    train_df, val_df = train_val_split(state_df, val_periods=min(4, max(2, len(state_df)//5)))

    if len(train_df) < 5:
        return None   # Not enough data

    models = {
        'SARIMA':  SARIMAModel(),
        'Prophet': ProphetModel(),
        'XGBoost': XGBoostModel(),
        'LSTM':    LSTMModel(),
    }

    results = {}

    for name, m in models.items():
        t0 = time.time()
        try:
            m.fit(train_df)
            val_preds = m.predict(len(val_df))
            val_actual = val_df['Total'].values

            r = rmse(val_actual, val_preds)
            mp = mape(val_actual, val_preds)
            future_forecast = m.predict(FORECAST_PERIODS)

            results[name] = {
                'rmse':     round(r, 2),
                'mape':     round(mp, 2),
                'forecast': future_forecast.tolist(),
                'model_obj': m,
                'fit_time_s': round(time.time() - t0, 2)
            }
            if verbose:
                print(f"  [{state}] {name:10s} RMSE={r:>15,.0f}  MAPE={mp:>6.2f}%  "
                      f"({results[name]['fit_time_s']}s)")
        except Exception as e:
            if verbose:
                print(f"  [{state}] {name:10s} FAILED: {e}")
            results[name] = None

    # Select best model by RMSE (ignore failed ones)
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        return None

    best_name = min(valid, key=lambda k: valid[k]['rmse'])

    # Build forecast date index
    last_date = state_df['Date'].max()
    gaps = state_df['Date'].sort_values().diff().dt.days.dropna()
    median_gap = gaps.median() if len(gaps) > 0 else 30
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=int(median_gap)),
        periods=FORECAST_PERIODS,
        freq=f'{int(median_gap)}D'
    )

    forecast_df = pd.DataFrame({
        'date':      [d.strftime('%Y-%m-%d') for d in future_dates],
        'predicted_sales': [round(x, 2) for x in valid[best_name]['forecast']]
    })

    metrics_summary = {
        k: {'rmse': v['rmse'], 'mape': v['mape'], 'fit_time_s': v['fit_time_s']}
        for k, v in valid.items()
    }

    return {
        'state':           state,
        'best_model':      best_name,
        'metrics':         metrics_summary,
        'forecast':        forecast_df.to_dict(orient='records'),
        'all_forecasts':   {k: v['forecast'] for k, v in valid.items()},
        'model_obj':       valid[best_name]['model_obj'],
        'forecast_dates':  [d.strftime('%Y-%m-%d') for d in future_dates],
    }


def run_full_training(df: pd.DataFrame,
                      output_dir: str = 'trained_models') -> Dict[str, Any]:
    """Train models for all states, save results."""
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    states = sorted(df['State'].unique())
    print(f"\n{'='*60}")
    print(f"Training {len(states)} states × 4 models")
    print(f"{'='*60}")

    for i, state in enumerate(states, 1):
        print(f"\n[{i}/{len(states)}] {state}")
        state_df = get_state_series(df, state)

        result = train_and_select(state_df, state)
        if result is None:
            print(f"  Skipped: not enough data")
            continue

        # Save model object separately (pickle)
        model_path = os.path.join(output_dir, f'{state.replace(" ", "_")}_best_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(result['model_obj'], f)

        # Store serialisable result
        result_clean = {k: v for k, v in result.items() if k != 'model_obj'}
        all_results[state] = result_clean
        print(f"  ✓ Best model: {result['best_model']}")

    # Save summary JSON
    summary_path = os.path.join(output_dir, 'forecast_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete! Summary saved to {summary_path}")

    # Print best model distribution
    best_counts = {}
    for v in all_results.values():
        bm = v['best_model']
        best_counts[bm] = best_counts.get(bm, 0) + 1
    print("\nBest model distribution across states:")
    for m, c in sorted(best_counts.items(), key=lambda x: -x[1]):
        print(f"  {m:12s}: {c} states")

    return all_results
