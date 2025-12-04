import numpy as np
import pandas as pd
from digital_twin.core.arrivals_ml import sample_arrivals_from_mu

def make_arrival_sampler(arrivals_per_day: pd.Series, rng=None):
    rng = rng or np.random.default_rng(42)
    vals = arrivals_per_day.values.astype(int)
    def sampler(size=1, scale=1.0):
        draw = rng.choice(vals, size=size, replace=True)
        return np.maximum((draw*scale).round().astype(int), 0)
    return sampler

def simulate_occupancy(
    days: int,
    n_runs: int,
    beds: int,
    arrival_sampler,
    processtime_sampler,
    rng=None,
):
    rng = rng or np.random.default_rng(42)

    arrivals_paths = np.zeros((n_runs, days), dtype=int)
    for run in range(n_runs):
        arrivals_paths[run, :] = [
            int(arrival_sampler(size=1)[0]) for _ in range(days)
        ]

    max_occupancies = np.empty(n_runs, dtype=int)
    overflow_days_counts = np.empty(n_runs, dtype=int)

    for run in range(n_runs):
        processtime_remaining = np.array([], dtype=float)
        overflow_days = 0
        max_occ = 0

        daily_arrivals = arrivals_paths[run, :]

        for d in range(days):
            if processtime_remaining.size > 0:
                processtime_remaining -= 1.0
                processtime_remaining = processtime_remaining[processtime_remaining > 0]

            n_new = daily_arrivals[d]
            if n_new > 0:
                new_processtime = processtime_sampler(n_new)
                new_processtime = np.clip(new_processtime, 0.5, None)
                processtime_remaining = np.concatenate([processtime_remaining, new_processtime])

            occ = processtime_remaining.size
            if occ > max_occ:
                max_occ = occ
            if occ > beds:
                overflow_days += 1

        max_occupancies[run] = max_occ
        overflow_days_counts[run] = overflow_days

    return max_occupancies, overflow_days_counts


def beds_vs_risk_curve(bed_values, days, n_runs, arrival_sampler, processtime_sampler):
    rows = []
    for b in bed_values:
        mo, od = simulate_occupancy(days, n_runs, int(b), arrival_sampler, processtime_sampler)
        rows.append({
            "beds": int(b),
            "P(max>100%)": float(np.mean(mo>int(b))),
            "P(max>95%)": float(np.mean(mo>0.95*int(b))),
            "p95_max_occ": float(np.percentile(mo,95)),
            "mean_overflow_days": float(np.mean(od)),
        })
    return pd.DataFrame(rows)

def simulate_from_mu_series(
    mu_series,
    n_runs: int,
    beds: int,
    processtime_sampler,
    rng=None):
    """
    Monte Carlo simulation where arrivals per day ~ Poisson(mu_t)
    come from an ML forecast (mu_series).
    Returns (max_occupancies, overflow_days_counts).
    """
    
    rng = rng or np.random.default_rng(42)
    mu_series = mu_series.sort_index()
    days = len(mu_series)

    # sample arrivals paths: shape (n_runs, days)
    arrivals_paths = sample_arrivals_from_mu(mu_series, n_samples=n_runs, rng=rng)

    max_occupancies = []
    overflow_days_counts = []

    for run in range(n_runs):
        processtime_remaining = np.array([], dtype=float)
        overflow_days = 0
        max_occ = 0

        daily_arrivals = arrivals_paths[run, :]

        for d in range(days):
            if processtime_remaining.size > 0:
                processtime_remaining -= 1.0
                processtime_remaining = processtime_remaining[processtime_remaining > 0]

            n_new = int(daily_arrivals[d])
            if n_new > 0:
                new_processtime = processtime_sampler(n_new)
                new_processtime = np.clip(new_processtime, 0.5, None)
                processtime_remaining = np.concatenate([processtime_remaining, new_processtime])

            occ = processtime_remaining.size
            max_occ = max(max_occ, occ)
            if occ > beds:
                overflow_days += 1

        max_occupancies.append(max_occ)
        overflow_days_counts.append(overflow_days)

    return np.array(max_occupancies), np.array(overflow_days_counts)



def beds_vs_risk_from_mu(
    mu_series,
    bed_values,
    n_runs,
    processtime_sampler,
    rng=None,
):
    """
    Capacity-squeeze analysis based on ML forecast (mu_series).

    For each bed capacity:
      - simulate n_runs paths
      - calculate P(max>cap), P(max>0.95 cap), p95(max), mean overflow days
    Returns: DataFrame with one row per bed value.
    """

    rng = rng or np.random.default_rng(42)
    mu_series = mu_series.sort_index()
    bed_values = list(bed_values)

    rows = []
    for b in bed_values:
        max_occupancies, overflow_days = simulate_from_mu_series(
            mu_series=mu_series,
            n_runs=n_runs,
            beds=int(b),
            processtime_sampler=processtime_sampler,
            rng=rng,
        )
        rows.append({
            "beds": int(b),
            "P(max>100%)": float(np.mean(max_occupancies > int(b))),
            "P(max>95%)": float(np.mean(max_occupancies > 0.95 * int(b))),
            "p95_max_occ": float(np.percentile(max_occupancies, 95)),
            "mean_overflow_days": float(np.mean(overflow_days)),
        })

    return pd.DataFrame(rows)
