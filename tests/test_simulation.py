"""Tests for core simulation functionality."""
import numpy as np
import pytest


class TestMCSimulator:
    """Tests for Monte Carlo simulation."""

    def test_simulate_occupancy_returns_arrays(self):
        """Test that simulate_occupancy returns correct array shapes."""
        from digital_twin.core.mc_simulator import simulate_occupancy

        rng = np.random.default_rng(42)
        arrival_sampler = lambda size: rng.poisson(1.5, size=size)
        los_sampler = lambda n: rng.exponential(5, size=n)

        max_occ, overflow_days = simulate_occupancy(
            days=30,
            n_runs=100,
            beds=20,
            arrival_sampler=arrival_sampler,
            processtime_sampler=los_sampler,
        )

        assert len(max_occ) == 100
        assert len(overflow_days) == 100
        assert all(max_occ >= 0)
        assert all(overflow_days >= 0)

    def test_simulate_occupancy_deterministic_with_seed(self):
        """Test that simulation is reproducible with same seed."""
        from digital_twin.core.mc_simulator import simulate_occupancy

        def run_sim(seed):
            rng = np.random.default_rng(seed)
            arrival_sampler = lambda size: rng.poisson(1.5, size=size)
            los_sampler = lambda n: rng.exponential(5, size=n)
            return simulate_occupancy(
                days=30,
                n_runs=50,
                beds=20,
                arrival_sampler=arrival_sampler,
                processtime_sampler=los_sampler,
            )

        max_occ1, _ = run_sim(42)
        max_occ2, _ = run_sim(42)

        np.testing.assert_array_equal(max_occ1, max_occ2)

    def test_higher_capacity_reduces_overflow(self):
        """Test that higher capacity reduces overflow probability."""
        from digital_twin.core.mc_simulator import simulate_occupancy

        rng = np.random.default_rng(42)
        arrival_sampler = lambda size: rng.poisson(2.0, size=size)
        los_sampler = lambda n: rng.exponential(5, size=n)

        _, overflow_low = simulate_occupancy(
            days=60, n_runs=200, beds=10,
            arrival_sampler=arrival_sampler,
            processtime_sampler=los_sampler,
        )
        _, overflow_high = simulate_occupancy(
            days=60, n_runs=200, beds=30,
            arrival_sampler=arrival_sampler,
            processtime_sampler=los_sampler,
        )

        assert overflow_high.mean() <= overflow_low.mean()


class TestArrivalsSampler:
    """Tests for arrival sampling."""

    def test_make_arrival_sampler(self):
        """Test that arrival sampler produces valid samples."""
        import pandas as pd
        from digital_twin.core.mc_simulator import make_arrival_sampler

        # Create synthetic arrival data
        arrivals = pd.Series([1, 2, 1, 3, 2, 1, 2, 1])
        sampler = make_arrival_sampler(arrivals)

        # Sample multiple times
        samples = sampler(size=100)

        assert len(samples) == 100
        assert all(s >= 0 for s in samples)
        # All samples should be from the original distribution
        assert all(s in arrivals.values for s in samples)


class TestRiskSummary:
    """Tests for risk metrics."""

    def test_risk_summary_structure(self):
        """Test that risk_summary returns expected keys."""
        from digital_twin.output.metrics import risk_summary

        max_occ = np.array([10, 15, 20, 18, 12, 22, 19, 16])
        overflow_days = np.array([0, 0, 2, 1, 0, 3, 1, 0])
        beds = 20

        result = risk_summary(max_occ, overflow_days, beds)

        assert "P(max>100%)" in result
        assert "P(max>95%)" in result
        assert "mean_overflow_days" in result
        assert "p95_max_occ" in result
        assert "median_max_occ" in result

    def test_risk_summary_values(self):
        """Test risk_summary calculations."""
        from digital_twin.output.metrics import risk_summary

        # All below capacity
        max_occ = np.array([10, 10, 10, 10])
        overflow_days = np.array([0, 0, 0, 0])

        result = risk_summary(max_occ, overflow_days, beds=20)

        assert result["P(max>100%)"] == 0.0
        assert result["mean_overflow_days"] == 0.0
