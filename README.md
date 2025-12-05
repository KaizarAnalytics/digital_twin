# Kaizar Digital Twin (Open Core)

A modular, simulation-first digital twin for hospital capacity planning.
Combines **discrete-event simulation**, **Monte Carlo uncertainty modeling**, and **ML-based arrival forecasting** into a cohesive, reproducible framework.

This open-core version contains the full simulation + forecasting engine used by Kaizar.
Integration layers, production APIs, and orchestration pipelines remain proprietary.

---

## Features

* **Synthetic patient-flow generation** & stochastic arrivals
* **Discrete Event Simulation (SimPy)** for bed occupancy, queues, and service pathways
* **Monte Carlo scenario engine** for robustness under uncertainty
* **ML forecasting module** using LightGBM (Poisson regression)
* **Config-driven architecture** (YAML hospital configs, reproducible scenarios)
* **Clear separation of simulation, forecasting, and data prep**
* **Jupyter notebooks for exploration, Kaggle-ready variants included**

---

## Repository Structure

```
src/
  digital_twin/
    hospital/      # Synthetic data, preprocessing, configs, exploration helpers
    core/          # Simulation engine, DES components, Monte Carlo logic
    forecasts/     # ML arrival forecasting (features, models, backtesting)
notebooks/         # Simulation & ML demonstration notebooks
pyproject.toml     # Project metadata + dependencies
requirements.txt
```

The production stack (APIs, monitoring, scheduling, dashboards) is **not included**.

---

## Installation

### From source

```bash
pip install -r requirements.txt
```

Or install as an editable package:

```bash
pip install -e .
```

---

## Usage

### Run a simulation

```python
from digital_twin.core.simulation import simulate_hospital
from digital_twin.hospital.config_loader import load_hospital_config

config = load_hospital_config("digital_twin/hospital/configs/hospital_1.yml").simulation
results = simulate_hospital(config=config, seed=42)
results.head()
```

### Generate synthetic arrivals

```python
from digital_twin.hospital.data_prep import load_patients, arrivals_per_day

patients = load_patients()
arrivals = arrivals_per_day(patients)
```

### Train an ML forecaster

```python
from digital_twin.forecasts.model import train_lgbm_poisson
from digital_twin.forecasts.features import make_feature_table

X, y = make_feature_table(arrivals)
model = train_lgbm_poisson(X, y)
```

---

## Data Usage Notice

This project includes synthetic example data for demonstration purposes only.
The structure and field names are inspired by the jaderz/hospital-beds-management dataset from Kaggle, which is published under the CC0 license:

https://www.kaggle.com/datasets/jaderz/hospital-beds-management

All CSVs under digital_twin/hospital/data/raw/ are fully synthetic and contain no real patient information, no re-identifiable fields, and no protected health data.
---

## License

Source code © Kaizar.
Open-core components are provided for research, learning, and experimentation.
Commercial deployment of proprietary modules is *not permitted*.

---

## Author

Developed by **Kaizar** — simulation engines, forecasting systems, and applied decision intelligence.

---
