from __future__ import annotations

from copy import deepcopy
from typing import Dict, Any, Optional, Iterable

import numpy as np
import pandas as pd

from digital_twin.hospital.hospital_model import HospitalConfig
from digital_twin.core.hospital_des import run_hospital_des


def apply_config_overrides(
    cfg: HospitalConfig,
    scenario: Dict[str, Any],
) -> HospitalConfig:
    """
    Makes a deep copy of cfg and applies overrides based on the scenario dict.

    Supports, among others:
      - 'capacity_overrides': {ward_id: new_capacity}
      - (later) policy overrides etc.
    """
    cfg2 = deepcopy(cfg)

    cap_overrides: Dict[str, int] = scenario.get("capacity_overrides", {}) or {}
    for ward_id, new_cap in cap_overrides.items():
        if ward_id in cfg2.wards:
            cfg2.wards[ward_id].capacity = int(new_cap)


    return cfg2


def apply_mu_overrides(
    mu_series: pd.Series,
    scenario: Dict[str, Any],
) -> pd.Series:
    """
    Applies scenario overrides to mu_series.

    Supports:
      - 'arrival_scale': factor by which mu is multiplied.
    """
    mu = mu_series.copy()

    scale = scenario.get("arrival_scale", 1.0)
    if scale is not None:
        mu = mu * float(scale)

    return mu


def run_scenario(
    cfg: HospitalConfig,
    entry_ward: str,
    patients_df: pd.DataFrame,
    mu_base: pd.Series,
    scenario: Dict[str, Any],
    warmup_days: int = 14,
    n_rep: int = 200,
    seed: int = 42,
    min_processtime: float = 0.5,
) -> pd.DataFrame:
    """
    Runs one scenario:
      - cfg + capacity_overrides (via apply_config_overrides)
      - mu_base + arrival_scale (via apply_mu_overrides)
      - optional: processtime_factors -> passed to run_hospital_des
    """
    cfg2 = apply_config_overrides(cfg, scenario)
    mu2 = apply_mu_overrides(mu_base, scenario)

    processtime_factors = scenario.get("processtime_factors") or {}
    

    df_runs = run_hospital_des(
        cfg=cfg2,
        entry_ward=entry_ward,
        mu_series=mu2,
        patients_df=patients_df,
        warmup_days=warmup_days,
        n_rep=n_rep,
        seed=seed,
        min_processtime=min_processtime,
        processtime_factors=processtime_factors,
    )

    scenario_name = scenario.get("name", "scenario")
    df_runs = df_runs.copy()
    df_runs["scenario"] = scenario_name
    return df_runs



def summarize_hospital_runs(
    df_all: pd.DataFrame,
    metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Makes a scenario/ward-summary from a concat of multiple scenario runs.

    Expects df_all to contain columns:
      - 'scenario'
      - 'ward_id'
      - metrics such as: mean_wait, p95_wait, max_wait, mean_occupancy,
                       p95_occupancy, days_over_95pct

    Returns an aggregated table per (scenario, ward_id).
    """
    if metrics is None:
        metrics = [
            "mean_wait",
            "p95_wait",
            "max_wait",
            "mean_occupancy",
            "p95_occupancy",
            "days_over_95pct",
        ]

    metrics = [m for m in metrics if m in df_all.columns]

    grp = (
        df_all
        .groupby(["scenario", "ward_id"], as_index=False)[metrics]
        .mean()
    )
    return grp
