# Combustion Simulation for Rocket Propellants

This project simulates combustion and flow characteristics of various rocket propellants using finite-rate chemistry and real gas effects.

## Features

- Finite-rate combustion kinetics modeling
- Equilibrium temperature calculations
- Wall temperature estimation
- Real gas effects consideration
- Flow parameter distributions
- Thrust and specific impulse calculations
- Heat flux analysis
- Combustion stability analysis

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

This will generate visualizations comparing different propellants and their performance metrics. 

```python
import matplotlib
matplotlib.use('Agg')  # Add this line at the top if running in headless mode

def visualize_results():
    fuels = ["MMH + N2O4", "AF-M3125E", "LOX + Methane", "G.prop"]
    results = [combustion_simulation(fuel) for fuel in fuels]
    
    labels = ["Exhaust Velocity (m/s)", "Thrust (N)", "Specific Impulse (s)", "Equilibrium Temperature (K)", "Burn Time (s)", "Wall Temperature (K)"]
    for i, label in enumerate(labels):
        values = [res[label] for res in results]
        plt.figure(figsize=(8, 4))
        plt.bar(fuels, values, color=['blue', 'green', 'red', 'purple'])
        plt.ylabel(label)
        plt.title(f"Comparison of {label} for Different Fuels")
        plt.tight_layout()
        plt.savefig(f"{label.replace(' ', '_')}.png")
        plt.close()
    
    df = pd.DataFrame(results)[["Fuel"] + labels]
    print(df.to_string(index=False))
``` 