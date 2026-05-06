"""
Forecasting Models Module
Implements: SARIMA, Prophet, XGBoost (with lag features), LSTM (deep learning)
Each model exposes: fit(train_df) → model  &  predict(model, n_periods) → np.array
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Evaluation metric
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# 1. SARIMA
class SARIMAModel:
    """Auto-selects ARIMA order via AIC grid search over small parameter space."""

    def __init__(self):
        self.model = None
        self.result = None

    def fit(self, train_df: pd.DataFrame):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        y = train_df['Total'].values.astype(float)

        best_aic, best_cfg, best_res = np.inf, None, None
        # Compact grid to keep runtime reasonable
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 2):
                    try:
                        res = SARIMAX(y, order=(p, d, q),
                                      seasonal_order=(1, 0, 1, 4),
                                      enforce_stationarity=False,
                                      enforce_invertibility=False).fit(disp=False)
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_cfg = (p, d, q)
                            best_res = res
                    except Exception:
                        continue

        if best_res is None:
            # Fallback: simple ARIMA(1,1,1)
            best_res = SARIMAX(y, order=(1, 1, 1),
                               enforce_stationarity=False,
                               enforce_invertibility=False).fit(disp=False)
        self.result = best_res
        self.best_cfg = best_cfg
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        forecast = self.result.forecast(steps=n_periods)
        return np.maximum(forecast, 0)


# 2. Facebook Prophet
class ProphetModel:
    def __init__(self):
        self.model = None
        self.last_date = None
        self.freq = None

    def fit(self, train_df: pd.DataFrame):
        from prophet import Prophet
        pdf = train_df[['Date', 'Total']].rename(columns={'Date': 'ds', 'Total': 'y'})
        pdf = pdf.sort_values('ds').reset_index(drop=True)

        # Infer approximate frequency from median gap
        gaps = pdf['ds'].diff().dt.days.dropna()
        median_gap = gaps.median()
        if median_gap <= 10:
            self.freq = 'W'
        elif median_gap <= 35:
            self.freq = 'MS'   # monthly start
        else:
            self.freq = 'QS'

        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.1
        )
        self.model.fit(pdf)
        self.last_date = pdf['ds'].max()
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        future = self.model.make_future_dataframe(periods=n_periods, freq=self.freq)
        forecast = self.model.predict(future)
        preds = forecast['yhat'].values[-n_periods:]
        return np.maximum(preds, 0)


# 3. XGBoost with Lag Features
FEATURE_COLS = ['lag_1', 'lag_2', 'lag_3',
                'rolling_mean_3', 'rolling_std_3',
                'month', 'quarter', 'dayofweek',
                'weekofyear', 'holiday_flag', 'obs_index']

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.last_window = None   # stores last known feature state for iterative prediction

    def fit(self, train_df: pd.DataFrame):
        import xgboost as xgb
        available = [c for c in FEATURE_COLS if c in train_df.columns]
        X = train_df[available].fillna(0)
        y = train_df['Total'].values

        self.model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        self.model.fit(X, y)
        self.feature_cols = available
        # Store last few rows for iterative forecasting
        self.last_rows = train_df.tail(3).copy()
        self.train_df  = train_df.copy()
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        """Iterative one-step-ahead prediction using previous predictions as lags."""
        history = list(self.train_df['Total'].values)
        last_row = self.last_rows.iloc[-1].copy()
        preds = []

        for i in range(n_periods):
            row = {}
            row['lag_1'] = history[-1]
            row['lag_2'] = history[-2] if len(history) >= 2 else history[-1]
            row['lag_3'] = history[-3] if len(history) >= 3 else history[-1]
            row['rolling_mean_3'] = np.mean(history[-3:])
            row['rolling_std_3']  = np.std(history[-3:]) if len(history) >= 3 else 0
            row['month']       = ((last_row['month'] - 1 + i) % 12) + 1
            row['quarter']     = ((row['month'] - 1) // 3) + 1
            row['dayofweek']   = last_row.get('dayofweek', 0)
            row['weekofyear']  = ((last_row.get('weekofyear', 1) - 1 + i) % 52) + 1
            row['holiday_flag'] = 0
            row['obs_index']   = last_row.get('obs_index', 0) + i + 1

            X_pred = pd.DataFrame([{c: row.get(c, 0) for c in self.feature_cols}])
            p = float(self.model.predict(X_pred)[0])
            p = max(p, 0)
            preds.append(p)
            history.append(p)

        return np.array(preds)

# 4. LSTM (Deep Learning)
class LSTMModel:
    def __init__(self, seq_len: int = 4):
        self.seq_len = seq_len
        self.model   = None
        self.scaler  = None
        self.last_seq = None

    def _build_sequences(self, series: np.ndarray):
        X, y = [], []
        for i in range(self.seq_len, len(series)):
            X.append(series[i - self.seq_len:i])
            y.append(series[i])
        return np.array(X), np.array(y)

    def fit(self, train_df: pd.DataFrame):
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
        tf.get_logger().setLevel('ERROR')

        values = train_df['Total'].values.astype(float).reshape(-1, 1)

        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(values).flatten()

        if len(scaled) < self.seq_len + 2:
            self.seq_len = max(2, len(scaled) // 2)

        X, y = self._build_sequences(scaled)

        if len(X) == 0:
            # Not enough data — store last value
            self.last_seq = scaled[-self.seq_len:]
            return self

        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True,
                                 input_shape=(self.seq_len, 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=50, batch_size=16, verbose=0,
                  validation_split=0.1 if len(X) > 10 else 0.0)

        self.model    = model
        self.last_seq = scaled[-self.seq_len:]
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        if self.model is None:
            # Fallback: repeat last known value
            last_val = self.scaler.inverse_transform([[self.last_seq[-1]]])[0][0]
            return np.full(n_periods, last_val)

        seq = self.last_seq.copy()
        preds_scaled = []
        for _ in range(n_periods):
            inp = seq[-self.seq_len:].reshape(1, self.seq_len, 1)
            p   = float(self.model.predict(inp, verbose=0)[0][0])
            preds_scaled.append(p)
            seq = np.append(seq, p)

        preds = self.scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        return np.maximum(preds, 0)
