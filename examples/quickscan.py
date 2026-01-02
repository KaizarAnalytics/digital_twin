#!/usr/bin/env python3
"""
Quickscan Example - Capacity Planning Simulation

This script demonstrates the core simulation capabilities:
1. Load hospital configuration and patient data
2. Run Monte Carlo simulations for different capacity levels
3. Find minimum capacity that meets a target SLA

Run from the project root:
    python examples/quickscan.py
"""
import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd

from digital_twin.hospital.config_loader import load_hospital_config
from digital_twin.hospital.data_prep import load_patients, arrivals_per_day, los_values
from digital_twin.core.mc_simulator import simulate_occupancy, make_arrival_sampler
from digital_twin.output.metrics import risk_summary


def main():
    print("=" * 60)
    print("Digital Twin - Capacity Planning Quickscan")
    print("=" * 60)
    print()

    # Configuration
    config_path = src_path / "digital_twin/hospital/configs/hospital_1.yml"
    service = "surgery"
    n_runs = 500  # Keep low for demo; increase to 3000+ for production analysis
    horizon_days = 180
    sla_threshold = 0.05  # 5% overflow probability

    # Load configuration
    print(f"Loading configuration from: {config_path}")
    cfg = load_hospital_config(str(config_path))
    base_capacity = cfg.simulation.get("capacity_default", 36)

    # Resolve data directory (try multiple paths)
    raw_data_dir = cfg.simulation.get("data_dir", "digital_twin/hospital/data")
    candidates = [
        Path(raw_data_dir),
        Path("src") / raw_data_dir,
        src_path / "digital_twin" / "hospital" / "data",
    ]
    data_dir = None
    for candidate in candidates:
        if (candidate / "raw" / "patients.csv").exists():
            data_dir = str(candidate)
            break
    if data_dir is None:
        print(f"Error: Could not find data directory")
        sys.exit(1)

    print(f"Hospital: {cfg.meta.name}")
    print(f"Service: {service}")
    print(f"Base capacity: {base_capacity} beds")
    print()

    # Load patient data
    print("Loading patient data...")
    patients = load_patients(data_dir)
    print(f"  Loaded {len(patients)} patient records")

    # Prepare arrivals and length-of-stay data
    arr = arrivals_per_day(patients, service)["arrivals"]
    los = los_values(patients, service)

    print(f"  Daily arrivals: mean={arr.mean():.2f}, max={arr.max()}")
    print(f"  Length of stay: mean={np.mean(los):.1f} days, max={np.max(los)} days")
    print()

    # Build samplers
    rng = np.random.default_rng(42)
    arrival_sampler = make_arrival_sampler(arr)
    los_sampler = lambda n: rng.choice(los, size=n, replace=True)

    # Run baseline simulation
    print(f"Running Monte Carlo simulation ({n_runs} runs, {horizon_days} days)...")
    max_occ, overflow_days = simulate_occupancy(
        days=horizon_days,
        n_runs=n_runs,
        beds=base_capacity,
        arrival_sampler=arrival_sampler,
        processtime_sampler=los_sampler,
    )

    summary = risk_summary(max_occ, overflow_days, base_capacity)

    print()
    print("Baseline Results:")
    print("-" * 40)
    print(f"  Capacity:              {base_capacity} beds")
    print(f"  P(overflow):           {summary['P(max>100%)']:.2%}")
    print(f"  P(>95% capacity):      {summary['P(max>95%)']:.2%}")
    print(f"  Mean overflow days:    {summary['mean_overflow_days']:.2f}")
    print(f"  Median max occupancy:  {summary['median_max_occ']:.0f}")
    print(f"  95th pctl max occ:     {summary['p95_max_occ']:.0f}")
    print()

    # Capacity squeeze analysis
    print(f"Capacity Squeeze Analysis (target SLA: {sla_threshold:.0%}):")
    print("-" * 40)

    results = []
    for beds in range(10, 26):
        max_occ_b, overflow_b = simulate_occupancy(
            days=horizon_days,
            n_runs=300,  # Keep low for demo; increase to 2000+ for production
            beds=beds,
            arrival_sampler=arrival_sampler,
            processtime_sampler=los_sampler,
        )
        p_overflow = float(np.mean(max_occ_b > beds))
        status = "OK" if p_overflow <= sla_threshold else "RISK"
        results.append({"beds": beds, "p_overflow": p_overflow, "status": status})
        print(f"  {beds:2d} beds: P(overflow) = {p_overflow:6.2%}  [{status}]")

    # Find minimum capacity
    df = pd.DataFrame(results)
    compliant = df[df["p_overflow"] <= sla_threshold]

    print()
    if not compliant.empty:
        min_beds = int(compliant["beds"].min())
        print(f"Minimum capacity for {sla_threshold:.0%} SLA: {min_beds} beds")
    else:
        print(f"No capacity in range meets {sla_threshold:.0%} SLA")

    print()
    print("=" * 60)
    print("Quickscan complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
