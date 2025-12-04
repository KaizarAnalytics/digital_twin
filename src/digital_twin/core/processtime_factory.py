from __future__ import annotations

import pathlib
from typing import Callable, Optional

import numpy as np
import pandas as pd

from digital_twin.hospital.hospital_model import HospitalConfig, Ward, LOSModel
from digital_twin.hospital.data_prep import processtime_values

# LOS-sampler type: returns process times (in days) for n samples
LOSSampler = Callable[[int], np.ndarray]


def _make_log_normal_sampler(lm: LOSModel, min_processtime: float = 0.5) -> LOSSampler:
    mu = float(lm.params.get("mu", 1.0))
    sigma = float(lm.params.get("sigma", 0.5))

    def sampler(n: int) -> np.ndarray:
        vals = np.random.lognormal(mean=mu, sigma=sigma, size=n)
        return np.clip(vals, min_processtime, None)

    return sampler


def _make_gamma_sampler(lm: LOSModel, min_processtime: float = 0.5) -> LOSSampler:
    shape = float(lm.params.get("shape", 2.0))
    scale = float(lm.params.get("scale", 1.0))

    def sampler(n: int) -> np.ndarray:
        vals = np.random.gamma(shape=shape, scale=scale, size=n)
        return np.clip(vals, min_processtime, None)

    return sampler


def _make_empirical_sampler_from_csv(
    lm: LOSModel,
    base_path: Optional[str | pathlib.Path] = None,
    col: str = "processtime_days",
    min_processtime: float = 0.5,
) -> LOSSampler:
    """
    Expects a CSV with at least one column `col` (default: 'processtime_days'),
    one row per patient stay (in days).
    """
    if lm.source is None:
        raise ValueError("LOSModel.type='empirical_histogram' to 'source' is missing.")

    path = pathlib.Path(base_path) / lm.source if base_path else pathlib.Path(lm.source)

    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {path}. Available: {df.columns.tolist()}")

    vals = df[col].astype(float).to_numpy()
    vals = vals[~np.isnan(vals)]
    vals = np.clip(vals, min_processtime, None)

    if vals.size == 0:
        raise ValueError(f"No valid LOS values found in {path}.")

    def sampler(n: int) -> np.ndarray:
        # simple bootstrap: sample with replacement from historical LOS
        rng = np.random.default_rng()
        draw = rng.choice(vals, size=n, replace=True)
        return np.clip(draw, min_processtime, None)

    return sampler

def _make_from_data_sampler(
    lm: LOSModel,
    patients_df: pd.DataFrame,
    cfg: HospitalConfig,
    min_processtime: float = 0.5,
) -> LOSSampler:
    """
    Builds a LOS sampler from raw patient data using the existing processtime_values().
    Assumes that processtime_values(patients_df, service) returns a 1D array of LOS in days.
    """
    # service name from params, or fallback to config.simulation.service_default
    service_name = lm.params.get("service", cfg.simulation.get("service_default"))
    if service_name is None:
        raise ValueError("LOSModel.type='from_data' but no 'service' in params and no service_default in config.simulation.")

    processtime_arr = processtime_values(patients_df, service_name)
    processtime_arr = np.asarray(processtime_arr, dtype=float)
    processtime_arr = processtime_arr[~np.isnan(processtime_arr)]
    processtime_arr = np.clip(processtime_arr, min_processtime, None)

    if processtime_arr.size == 0:
        raise ValueError(f"No valid LOS values found for service '{service_name}'.")

    def sampler(n: int) -> np.ndarray:
        rng = np.random.default_rng()
        draw = rng.choice(processtime_arr, size=n, replace=True)
        return np.clip(draw, min_processtime, None)

    return sampler



def make_processtime_sampler_for_ward(
    cfg: HospitalConfig,
    ward_id: str,
    base_path: Optional[str | pathlib.Path] = None,
    min_processtime: float = 0.5,
    patients_df: Optional[pd.DataFrame] = None,
) -> LOSSampler:
    """
    Builds a LOS sampler for a Ward from the HospitalConfig.

    Supported processtime_model types:
      - type: "lognormal" (params: mu, sigma)
      - type: "gamma"     (params: shape, scale)
      - type: "empirical_histogram" (source: CSV)
      - type: "from_data" (params: service, uses processtime_values(patients_df, service))
    For type 'from_data', patients_df MUST be provided.
    """
    if ward_id not in cfg.wards:
        raise ValueError(f"Ward '{ward_id}' not found in config.")

    ward: Ward = cfg.wards[ward_id]
    lm: LOSModel = ward.processtime_model
    lm_type = lm.type.lower()

    if lm_type == "lognormal":
        return _make_log_normal_sampler(lm, min_processtime=min_processtime)

    if lm_type == "gamma":
        return _make_gamma_sampler(lm, min_processtime=min_processtime)

    if lm_type in ("empirical_histogram", "empirical"):
        return _make_empirical_sampler_from_csv(
            lm, base_path=base_path, col="processtime_days", min_processtime=min_processtime
        )

    if lm_type == "from_data":
        if patients_df is None:
            raise ValueError("patients_df is required for processtime_model.type='from_data'.")
        return _make_from_data_sampler(lm, patients_df=patients_df, cfg=cfg, min_processtime=min_processtime)

    raise ValueError(f"Unknown processtime_model.type '{lm.type}' for ward '{ward_id}'.")