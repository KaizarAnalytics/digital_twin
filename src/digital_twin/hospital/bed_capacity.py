from __future__ import annotations

from typing import Iterable, Dict, Any, Optional

import numpy as np
import pandas as pd

from digital_twin.hospital.data_prep import (
    arrivals_per_day,
    los_values,
    load_patients,
)
from digital_twin.core.arrivals_ml import (
    make_feature_table,
    train_lgbm_poisson,
    predict_mean,
    forecast_mu_forward
)
from digital_twin.core.mc_simulator import (
    make_arrival_sampler,
    beds_vs_risk_curve,
    beds_vs_risk_from_mu,
)

from digital_twin.core.des_engine import run_single_ward_des
from digital_twin.core.hospital_des import run_hospital_des
from digital_twin.hospital.hospital_model import HospitalConfig


# ----------------- QUICKSCAN ----------------- #

def run_quickscan_pipeline(
    patients: pd.DataFrame,
    service: str,
    bed_grid: Iterable[int],
    days: int = 180,
    n_runs: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Quickscan: Monte Carlo based on historical arrivals + LOS.
    Returns a DataFrame with risk-metrics per bed capacity.

    Columns: ['beds', 'P(max>100%)', 'P(max>95%)', 'p95_max_occ', 'mean_overflow_days']
    """
    rng = np.random.default_rng(seed)

    # arrivals & LOS uit ruwe data
    arr_series = arrivals_per_day(patients, service)["arrivals"]
    arrival_sampler = make_arrival_sampler(arr_series, rng=rng)

    los_arr = los_values(patients, service)
    los_arr = np.asarray(los_arr, dtype=float)
    los_arr = los_arr[~np.isnan(los_arr)]
    los_arr = np.clip(los_arr, 0.5, None)

    def los_sampler(n: int) -> np.ndarray:
        draw = rng.choice(los_arr, size=n, replace=True)
        return np.clip(draw, 0.5, None)

    df_curve = beds_vs_risk_curve(
        bed_values=bed_grid,
        days=days,
        n_runs=n_runs,
        arrival_sampler=arrival_sampler,
        los_sampler=los_sampler,
    )
    return df_curve


# ----------------- POC – SINGLE WARD (ML + MC + DES) ----------------- #

def run_poc_single_ward_pipeline(
    patients: pd.DataFrame,
    service: str,
    horizon_days: int = 180,
    bed_grid: Iterable[int] = (8, 10, 12, 14, 16),
    des_warmup_days: int = 14,
    des_n_rep: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    PoC pipeline for a single ward:
      - tabular ML for arrivals
      - Monte Carlo based on ML forecast
      - DES capacity sweep on the same ward
    Returns a dict with:
      {
        'mu_fc': pd.Series,
        'curve_hist': pd.DataFrame,   # (optional, if you want hist-bootstrap)
        'curve_ml': pd.DataFrame,     # beds_vs_risk_from_mu(...)
        'des_capacity': pd.DataFrame  # run_single_ward_des over bed_grid
      }
    """
    rng = np.random.default_rng(seed)

    # 1) arrivals & LOS from raw data
    arr_series = arrivals_per_day(patients, service)["arrivals"]
    arr_series = arr_series.sort_index()

    los_arr = los_values(patients, service)
    los_arr = np.asarray(los_arr, dtype=float)
    los_arr = los_arr[~np.isnan(los_arr)]
    los_arr = np.clip(los_arr, 0.5, None)

    def los_sampler(n: int) -> np.ndarray:
        draw = rng.choice(los_arr, size=n, replace=True)
        return np.clip(draw, 0.5, None)

    # 2) tabular ML for arrivals
    feat_table = make_feature_table(arr_series)
    model = train_lgbm_poisson(
        feat_table,
        target_col="arrivals",
        seed=seed,
    )

    # forecast for the coming horizon_days
    # assumptions: make_feature_table can build future features,
    # or you use an existing helper; here we do it simply:
    future_idx = pd.date_range(
        start=arr_series.index.max() + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )
    feat_future = make_feature_table(
        pd.Series(index=future_idx, data=np.nan, name="arrivals")
    )
    mu_fc = predict_mean(model, feat_future)
    mu_fc.name = "mu_fc"

    # 3) Monte Carlo based on ML forecast
    df_curve_ml = beds_vs_risk_from_mu(
        mu_series=mu_fc,
        bed_values=bed_grid,
        n_runs=3000,
        processtime_sampler=los_sampler,
    )

    # 4) DES capacity sweep on the same ward
    # We assume here that you simulate one ward with the same LOS and arrivals.
    # DES function: run_single_ward_des(cfg, ward_id, mu_series, los_sampler,...)
    des_rows = []
    from hospital.config_loader import load_hospital_config
    cfg = load_hospital_config("digital_twin/hospital/configs/hospital_2.yml")  # use whatever config you have
    ward_id = "WARD_A"  # or whatever ward you want to simulate

    for b in bed_grid:
        df_b = run_single_ward_des(
            cfg=cfg,
            ward_id=ward_id,
            mu_series=mu_fc,
            processtime_sampler=los_sampler,
            warmup_days=des_warmup_days,
            n_rep=des_n_rep,
            override_capacity=int(b),
        )
        des_rows.append(df_b)

    df_des = pd.concat(des_rows, ignore_index=True)

    return {
        "mu_fc": mu_fc,
        "curve_ml": df_curve_ml,
        "des_capacity": df_des,
    }


# ----------------- FULL – MULTI-WARD HOSPITAL ----------------- #

def run_hospital_multiward_pipeline(
    cfg: HospitalConfig,
    entry_ward: str,
    patients: Optional[pd.DataFrame] = None,
    mu_series: Optional[pd.Series] = None,
    service_for_mu: Optional[str] = None,
    warmup_days: int = 14,
    n_rep: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    

    if patients is None:
        patients = load_patients(cfg.simulation.data_dir)
    if mu_series is None:
        if service_for_mu is None:
            service_for_mu = cfg.simulation.get("service_default")
        arr_series = arrivals_per_day(patients, service_for_mu)["arrivals"]
        arr_series = arr_series.sort_index()

        if arr_series.empty:
            raise ValueError(
                f"No arrivals found for service '{service_for_mu}'."
            )

        mu_series = forecast_mu_forward(
            arrivals_per_day=arr_series,
            horizon=cfg.meta.simulation_horizon_days,
        )
        mu_series.name = "mu_entry"

    df_runs = run_hospital_des(
        cfg=cfg,
        entry_ward=entry_ward,
        mu_series=mu_series,
        patients_df=patients,
        warmup_days=warmup_days,
        n_rep=n_rep,
        seed=seed,
    )

    return {
        "cfg": cfg,
        "mu_series": mu_series,
        "df_runs": df_runs,
    }
