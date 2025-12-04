import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb


# ------------------------------
# Dataclass for backtest results
# ------------------------------

@dataclass
class BacktestResult:
    mae_model: float
    mae_seasonal_naive: float
    coverage_80: float
    details: pd.DataFrame


# ------------------------------
# Feature engineering
# ------------------------------

def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects index = datetime (day), column 'arrivals'.
    """
    df = df.copy()
    idx = df.index

    df["dow"] = idx.dayofweek           # 0=Monday
    df["month"] = idx.month
    df["dayofyear"] = idx.dayofyear
    df["weekofyear"] = idx.isocalendar().week.astype(int)

    # simple seasonal dummy
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)

    # NL holidays: placeholder (keep it simple for now)
    # You can later plug in real holidays here
    df["is_holiday"] = 0

    return df


def _add_lag_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 7, 14],
    roll_windows: List[int] = [7, 28],
) -> pd.DataFrame:
    df = df.copy()
    y = df["arrivals"]

    for lag in lags:
        df[f"lag_{lag}"] = y.shift(lag)

    for w in roll_windows:
        df[f"roll_mean_{w}"] = y.shift(1).rolling(w).mean()

    return df


def make_feature_table(
    arrivals_per_day: pd.Series,
    lags: List[int] = [1, 7, 14],
    roll_windows: List[int] = [7, 28],
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Input:
        arrivals_per_day: Series with index = day (datetime), value = arrivals (int)
    Output:
        DataFrame with 'arrivals' + feature columns.
    """
    df = arrivals_per_day.to_frame(name="arrivals").sort_index()
    df = _add_calendar_features(df)
    df = _add_lag_features(df, lags=lags, roll_windows=roll_windows)

    if drop_na:
        df = df.dropna()

    return df





# ------------------------------
# Model training (LightGBM Poisson)
# ------------------------------

def train_lgbm_poisson(
    df_feat: pd.DataFrame,
    target_col: str = "arrivals",
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[lgb.LGBMRegressor, Dict]:
    """
    Trains a LightGBM model with Poisson objective + TimeSeriesSplit.
    Returns (model, cv_stats).
    """
    df = df_feat.copy()
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []

    # simple CV for sanity-check
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(
            objective="poisson",
            metric="mae",
            random_state=random_state,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, y_pred))

    cv_stats = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes))}

    # retrain on all data
    final_model = lgb.LGBMRegressor(
        objective="poisson",
        metric="mae",
        random_state=random_state,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        verbose=-1,
    )
    final_model.fit(X, y)

    return final_model, cv_stats


# ------------------------------
# Predicting (mean + samples)
# ------------------------------

def predict_mean(model: lgb.LGBMRegressor, df_feat: pd.DataFrame) -> pd.Series:
    mu = model.predict(df_feat)
    # Poisson intensity must not be <0
    mu = np.clip(mu, 1e-6, None)
    return pd.Series(mu, index=df_feat.index, name="mu")


def sample_arrivals_from_mu(
    mu_series: pd.Series,
    n_samples: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw samples from Poisson(mu) per day.
    Output shape: (n_samples, n_days)
    """
    rng = rng or np.random.default_rng(42)
    mu = mu_series.values.astype(float)
    n_days = len(mu)
    out = np.zeros((n_samples, n_days), dtype=int)

    for i in range(n_samples):
        out[i, :] = rng.poisson(mu)

    return out


# ------------------------------
# Rolling backtest (horizon = 7/14/28)
# ------------------------------

def rolling_backtest(
    arrivals_per_day: pd.Series,
    horizon: int = 28,
    min_history: int = 180,
    step: int = 7,
):
    y = arrivals_per_day.sort_index()

    preds = []
    naive_preds = []
    reals = []
    cover_flags = []

    for end_idx in range(min_history, len(y) - horizon, step):
        hist = y.iloc[:end_idx]
        future = y.iloc[end_idx:end_idx + horizon]

        # --- ML part ---
        df_feat = make_feature_table(hist, drop_na=True)
        df_train = df_feat.copy()  # train on all historical data
        model, _ = train_lgbm_poisson(df_train)

        # features for future: simple variant with only calendar features
        df_future = future.to_frame(name="arrivals").copy()
        df_future["arrivals"] = np.nan  # target empty
        df_future = _add_calendar_features(df_future)
        # lags simplistic: use last known value(s)
        df_future["lag_1"] = hist.iloc[-1]
        df_future["lag_7"] = hist.iloc[-7] if len(hist) >= 7 else hist.iloc[-1]
        df_future["lag_14"] = hist.iloc[-14] if len(hist) >= 14 else hist.iloc[-1]
        # rolling means also simplistic
        df_future["roll_mean_7"] = hist.iloc[-7:].mean() if len(hist) >= 7 else hist.mean()
        df_future["roll_mean_28"] = hist.iloc[-28:].mean() if len(hist) >= 28 else hist.mean()

        X_future = df_future.drop(columns=["arrivals"])
        mu_fc = predict_mean(model, X_future)

        preds.append(mu_fc.values)
        reals.append(future.values)

        # 80%-interval through sampling
        samples = sample_arrivals_from_mu(mu_fc, n_samples=200)
        q10 = np.quantile(samples, 0.1, axis=0)
        q90 = np.quantile(samples, 0.9, axis=0)
        cover_flags.append((future.values >= q10) & (future.values <= q90))

        # --- seasonal naive: use values from 7 days ago ---
        if len(hist) >= horizon + 7:
            naive_window = hist.iloc[-(horizon + 7):-7]  # length = horizon
            naive_arr = naive_window.values
        else:
            naive_arr = np.full(horizon, hist.mean())

        naive_preds.append(naive_arr)

    preds = np.concatenate(preds)
    reals = np.concatenate(reals)
    naive_preds = np.concatenate(naive_preds)
    cover_flags = np.concatenate(cover_flags)

    mae_model = float(mean_absolute_error(reals, preds))
    mae_naive = float(mean_absolute_error(reals, naive_preds))
    coverage_80 = float(np.mean(cover_flags))

    details = pd.DataFrame({"real": reals, "pred": preds, "naive": naive_preds})

    return BacktestResult(
        mae_model=mae_model,
        mae_seasonal_naive=mae_naive,
        coverage_80=coverage_80,
        details=details,
    )

def forecast_mu_forward(
    arrivals_per_day: pd.Series,
    horizon: int = 180,
) -> pd.Series:
    """
    Train model on full history and make a simple forward forecast
    of mu (expected number of arrivals per day) for the next `horizon` days.
    """
    y = arrivals_per_day.sort_index()
    df_feat = make_feature_table(y, drop_na=True)
    model, _ = train_lgbm_poisson(df_feat)

    last_date = y.index.max()
    future_idx = pd.date_range(last_date + pd.Timedelta(days=1),
                               periods=horizon, freq="D")

    # We build future features "simple but sufficient":
    df_future = pd.DataFrame(index=future_idx)
    df_future["arrivals"] = np.nan
    df_future = _add_calendar_features(df_future)

    # lags and rolling means filled with last known hist stats
    last = y.iloc[-1]
    df_future["lag_1"] = last
    df_future["lag_7"] = y.iloc[-7] if len(y) >= 7 else last
    df_future["lag_14"] = y.iloc[-14] if len(y) >= 14 else last
    df_future["roll_mean_7"] = y.iloc[-7:].mean() if len(y) >= 7 else y.mean()
    df_future["roll_mean_28"] = y.iloc[-28:].mean() if len(y) >= 28 else y.mean()

    X_future = df_future.drop(columns=["arrivals"])
    mu_fc = predict_mean(model, X_future)
    mu_fc.name = "mu"
    return mu_fc

