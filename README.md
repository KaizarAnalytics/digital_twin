# Digital Twin - Capacity Planning Simulation Engine

A Monte Carlo simulation engine for capacity planning and risk assessment. Run thousands of stochastic scenarios to quantify uncertainty and find optimal capacity levels.

This open-core version contains the Monte Carlo simulation engine, Discrete Event Simulation (DES), and ML-based arrival forecasting. Optimization algorithms and production infrastructure remain proprietary.

## Key Capabilities

- **Monte Carlo Simulation**: Run thousands of stochastic scenarios to quantify uncertainty and risk
- **Discrete Event Simulation**: SimPy-based patient flow modeling with multi-ward routing
- **ML Arrival Forecasting**: LightGBM-based forecasting of future arrivals with calendar features
- **Config-driven Architecture**: YAML-based hospital configurations for reproducible scenarios
- **Risk Metrics**: Overflow probability, capacity utilization, and SLA compliance analysis

## Example Results

Using the synthetic demo configuration included in this repository:

| Metric | Value |
|--------|-------|
| Simulation horizon | 180 days |
| Monte Carlo replications | 3,000 |
| Capacity | 20 beds |
| P(overflow) | < 1% |

The simulation quantifies the probability of capacity overflow, accounting for stochastic arrivals and variable length-of-stay.

## Repository Structure

```
src/
  digital_twin/
    core/           # Monte Carlo simulation engine
    hospital/       # Data preprocessing, config loading
    output/         # Metrics and visualization
    main_cli/       # Command-line interface
notebooks/          # Demonstration notebook
pyproject.toml      # Project metadata and dependencies
```

## Installation

### From source

```bash
git clone https://github.com/KaizarAnalytics/digital_twin.git
cd digital_twin
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

```bash
# Run a Monte Carlo simulation
digital-twin simulate --beds 20 --runs 5000

# Show help
digital-twin info
```

### Example Script

Run the included quickscan example:

```bash
python examples/quickscan.py
```

### Python API

```python
from digital_twin.hospital.data_prep import load_patients, arrivals_per_day, los_values
from digital_twin.hospital.config_loader import load_hospital_config
from digital_twin.core.mc_simulator import simulate_occupancy, make_arrival_sampler
import numpy as np

# Load config and data
config = load_hospital_config("src/digital_twin/hospital/configs/hospital_1.yml")
patients = load_patients(config.simulation["data_dir"])
arrivals = arrivals_per_day(patients, config.simulation["service_default"])

# Build samplers
rng = np.random.default_rng(42)
arrival_sampler = make_arrival_sampler(arrivals["arrivals"])
los = los_values(patients, config.simulation["service_default"])
los_sampler = lambda n: rng.choice(los, size=n, replace=True)

# Run simulation
max_occ, overflow_days = simulate_occupancy(
    days=180,
    n_runs=3000,
    beds=20,
    arrival_sampler=arrival_sampler,
    processtime_sampler=los_sampler,
)

print(f"Mean max occupancy: {max_occ.mean():.1f}")
print(f"P(overflow): {(max_occ > 20).mean():.2%}")
```

## Notebooks

Interactive demonstrations available in the `notebooks/` directory:

- **01_quickscan_digital_twin.ipynb** - Monte Carlo capacity analysis
- **02_poc_ml_forecast_digital_twin.ipynb** - ML-based arrival forecasting with LightGBM
- **04_generate_capacity_kpis.ipynb** - DES-based scenario analysis and KPI generation

## Configuration

Hospital configurations are defined in YAML files under `src/digital_twin/hospital/configs/`:

```yaml
simulation:
  horizon_days: 180
  n_runs_mc: 3000
  capacity_default: 20

hospital:
  name: "Example Hospital"
  timezone: "UTC"
```

## Testing

Run the test suite:

```bash
pip install -e ".[dev]"
pytest -v
```

## Data Notice

This project includes synthetic example data for demonstration purposes only. All CSVs under `src/digital_twin/hospital/data/raw/` are **fully synthetic** and contain no real patient information.

## License

Source code is proprietary to Kaizar Analytics. Open-core components are provided for research, learning, and experimentation. Commercial deployment requires a license agreement.

## Further Reading

- [Process Capacity Digital Twin](https://kaizar.nl/posts/digital-twin.html) - Case study demonstrating how simulation reveals hidden capacity and operational bottlenecks
- [Building a Digital Clone Protocol (Part I)](https://medium.com/@tobias.beers_63234/building-a-digital-clone-protocol-i-how-organisations-navigate-the-future-using-their-own-0728a4530108) - Conceptual series on how organizations navigate uncertainty using simulation

## About

Developed by [Kaizar](https://kaizar.nl) - simulation engines, forecasting systems, and applied decision intelligence.
