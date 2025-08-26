# PropellantCombustionSimulation

Physics-informed propellant combustion and nozzle performance simulator with an optional TensorFlow surrogate for specific impulse. Computes exhaust velocity, thrust, Isp, temperature/pressure profiles, wall heating, and stability spectrum for multiple propellants. Provides a simple CLI, saves plots, and caches surrogate models under `models/`.

## Features

- Finite-rate combustion kinetics modeling
- Equilibrium temperature calculations
- Wall temperature estimation
- Real gas effects consideration
- Flow parameter distributions
- Thrust and specific impulse calculations
- Heat flux analysis
- Combustion stability analysis
- Optional NN surrogate for Isp prediction (TensorFlow + scikit-learn)

## Supported Propellants

- MMH + N2O4
- AF-M3125E
- LOX + Methane
- G.prop

## Dependencies

- NumPy
- Matplotlib
- Pandas
- SciPy
- scikit-learn
- TensorFlow (2.15.x)

## Installation

1. Clone this repository
2. (Recommended) Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the base simulation (analytic model only):
```bash
python src/combustion_simulation.py
```

Use the NN surrogate for specific impulse (trains once and caches under `models/`):
```bash
python src/combustion_simulation.py --use-surrogate
```

Force retraining of the surrogate and overwrite cached artifacts:
```bash
python src/combustion_simulation.py --use-surrogate --retrain
```

This will print a comparison table and save plots for key metrics in the current directory.

### Notes
- In headless environments, Matplotlib will automatically save plots (no display needed). If required, you can explicitly set the backend by adding `matplotlib.use('Agg')` near the top of your script.
- Surrogate artifacts are saved to `models/surrogate.keras` and `models/scaler.pkl` by default; paths can be overridden with `--model-path` and `--scaler-path`.

## Outputs
- Console table of per-fuel metrics: exhaust velocity, thrust, Isp, equilibrium temperature, burn time, wall temperature
- PNG plots saved in the working directory for core metrics
- Optional cached surrogate model and scaler under `models/`

## Repository
- Name: PropellantCombustionSimulation
- GitHub: [PropellantCombustionSimulation](https://github.com/Sarvesh2304/PropellantCombustionSimulation)

