from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import simpy

from digital_twin.hospital.hospital_model import HospitalConfig

# LOS-sampler type: returns process times (in days) for n samples
LOSSampler = Callable[[int], np.ndarray]


@dataclass
class SingleDepartmentRunResult:
    ward_id: str
    beds: int
    rep: int
    mean_wait: float
    p95_wait: float
    max_wait: float
    mean_occupancy: float
    p95_occupancy: float
    days_over_95pct: float


# ----------------- SimPy processes ----------------- #

def order(
    env: simpy.Environment,
    processing_units: simpy.Resource,
    processtime_sampler: LOSSampler,
    stats: Dict[str, List],
    arrival_time: float,
) -> None:
    """
    One order:
      - arrives at arrival_time
      - waits for one processing_unit
      - stays for processing_time
    """
    # wait until arrival_time
    if env.now < arrival_time:
        yield env.timeout(arrival_time - env.now)

    # actual arrival
    real_arrival = env.now

    with processing_units.request() as req:
        yield req
        wait = env.now - real_arrival
        stats["waits"].append(wait)

        # log occupancy after allocation
        stats["occupancy_changes"].append((env.now, processing_units.count))

        processtime = float(processtime_sampler(1)[0])
        yield env.timeout(processtime)

        # log occupancy after departure
        stats["occupancy_changes"].append((env.now, processing_units.count - 1))


def arrival_process(
    env: simpy.Environment,
    beds: simpy.Resource,
    mu_values: np.ndarray,
    processtime_sampler: LOSSampler,
    stats: Dict[str, List],
    rng: np.random.Generator,
) -> None:
    """
    Daily arrival intensity mu_values.
    For day d with intensity mu:
      - sample n ~ Poisson(mu)
      - distribute arrivals uniformly over that day [d, d+1)
    Time unit = days.
    """
    current_day = 0.0
    for mu in mu_values:
        n = rng.poisson(mu)
        for _ in range(n):
            offset = rng.uniform(0.0, 1.0)  # within that day
            arrival_time = current_day + offset
            env.process(order(env, beds, processtime_sampler, stats, arrival_time))
        current_day += 1.0
        yield env.timeout(1.0)  # go to next day

def monitor_occupancy(
    env: simpy.Environment,
    beds: simpy.Resource,
    stats: Dict[str, List],
    max_time: float,
    sample_step: float = 1.0,
) -> None:
    """
    Log occupancy every sample_step.
    """
    while env.now < max_time:
        stats["occupancy_daily"].append((env.now, beds.count))
        yield env.timeout(sample_step)


# ----------------- Main function ----------------- #

def run_single_ward_des(
    cfg: HospitalConfig,
    ward_id: str,
    mu_series: pd.Series,
    processtime_sampler: LOSSampler,
    warmup_days: int = 14,
    n_rep: int = 200,
    seed: int = 42,
    override_capacity: Optional[int] = None,
) -> pd.DataFrame:
    """
    DES for a single ward from HospitalConfig.

    Parameters
    ----------
    cfg : HospitalConfig
        Full hospital configuration.
    ward_id : str
        Which ward (cfg.wards[...] ) we simulate.
    mu_series : pd.Series
        Daily arrival intensity (μ_t) for the simulation horizon.
        Index is sorted; values are λ of Poisson per day.
    processtime_sampler : callable
        Function that returns an array of LOS (in days) for n patients.
    warmup_days : int
        Extra days to let the system reach equilibrium.
    n_rep : int
        Number of independent replications.
    seed : int
        Base seed for the RNG.
    override_capacity : int, optional
        If set, use this capacity instead of cfg.wards[ward_id].capacity.

    Returns
    -------
    pd.DataFrame
        One row per replication with wait time and occupancy metrics.
    """
    if ward_id not in cfg.wards:
        raise ValueError(f"Ward '{ward_id}' not found in config.")

    ward = cfg.wards[ward_id]
    capacity = int(override_capacity) if override_capacity is not None else int(ward.capacity)

    # prepare mu_values
    mu_series = mu_series.sort_index()
    mu_values = mu_series.values.astype(float)
    horizon_days = len(mu_values)

    results: List[SingleDepartmentRunResult] = []

    for rep in range(n_rep):
        rng = np.random.default_rng(seed + rep)
        env = simpy.Environment()
        beds = simpy.Resource(env, capacity=capacity)
        stats: Dict[str, List] = {
            "waits": [],
            "occupancy_daily": [],
            "occupancy_changes": [],
        }

        # warmup: add warmup_days with average intensity
        mean_mu = float(mu_values.mean()) if len(mu_values) > 0 else 0.0
        mu_full = np.concatenate([np.full(warmup_days, mean_mu), mu_values])

        sim_time = warmup_days + horizon_days + 30  # buffer
        env.process(arrival_process(env, beds, mu_full, processtime_sampler, stats, rng))
        env.process(monitor_occupancy(env, beds, stats, max_time=sim_time))

        env.run(until=sim_time)

        # process stats
        waits = np.array(stats["waits"]) if stats["waits"] else np.array([0.0])

        if stats["occupancy_daily"]:
            occ_times, occ_vals = zip(*stats["occupancy_daily"])
            occ_times = np.array(occ_times)
            occ_vals = np.array(occ_vals)
        else:
            occ_times = np.array([0.0])
            occ_vals = np.array([0.0])

        # analyze only after warmup
        mask_steady = occ_times >= warmup_days
        occ_vals_steady = occ_vals[mask_steady] if mask_steady.any() else occ_vals

        if occ_vals_steady.size == 0:
            occ_vals_steady = occ_vals

        days_over_95 = float(
            np.mean(occ_vals_steady > 0.95 * capacity) * len(occ_vals_steady)
        )

        res = SingleDepartmentRunResult(
            ward_id=ward_id,
            beds=capacity,
            rep=rep,
            mean_wait=float(waits.mean()),
            p95_wait=float(np.percentile(waits, 95)),
            max_wait=float(waits.max()),
            mean_occupancy=float(occ_vals_steady.mean()),
            p95_occupancy=float(np.percentile(occ_vals_steady, 95)),
            days_over_95pct=days_over_95,
        )
        results.append(res)

    # to DataFrame
    df = pd.DataFrame([r.__dict__ for r in results])
    return df
