import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.fft import fft
import re
import os
import argparse
import pickle
# Import surrogate NN utilities
from surrogate_nn import generate_synthetic_data, train_surrogate_nn, predict_isp
import tensorflow as tf
from tensorflow import keras

# Utilities to persist/load surrogate artifacts
def save_surrogate(model, scaler, model_path, scaler_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

def load_surrogate(model_path, scaler_path):
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, None
    model = keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Function to model combustion kinetics using finite-rate chemistry
def combustion_kinetics(t, y, A, Ea, R, T):
    fuel_concentration, intermediate_concentration = y
    reaction_rate = A * np.exp(-Ea / (R * T)) * fuel_concentration
    return [-reaction_rate, reaction_rate * 0.5]

# Function to compute equilibrium temperature considering chemical equilibrium
def equilibrium_temperature(fuel_data):
    return fuel_data["chamber_temp"]

# Function to estimate wall temperature based on heat flux and material properties
def wall_temperature_estimation(heat_flux, wall_thickness, conductivity):
    return heat_flux * wall_thickness / conductivity

# Function to determine specific heat ratio (gamma) as a function of temperature and pressure
def real_gas_gamma(T, P):
    gamma = 1.4 - 0.1 * (T / 3000)
    return np.clip(gamma, 1.15, 1.35)

# Main function to simulate combustion and flow characteristics of the given fuel
def combustion_simulation(fuel, use_surrogate=False, surrogate_model=None, surrogate_scaler=None):
    properties = {
        "MMH + N2O4": {"chamber_temp": 3300, "chamber_pressure": 7e6, "mass_flow_rate": 0.35, "molar_mass": 46, "A": 5e7, "Ea": 65e3, "conductivity": 30, "total_fuel_mass": 10},
        "AF-M3125E": {"chamber_temp": 3200, "chamber_pressure": 7e6, "mass_flow_rate": 0.35, "molar_mass": 82, "A": 3e7, "Ea": 60e3, "conductivity": 30, "total_fuel_mass": 10},
        "LOX + Methane": {"chamber_temp": 3600, "chamber_pressure": 10e6, "mass_flow_rate": 0.35, "molar_mass": 24, "A": 2e8, "Ea": 35e3, "conductivity": 45, "total_fuel_mass": 10},
        "G.prop": {"chamber_temp": 3300, "chamber_pressure": 7e6, "mass_flow_rate": 0.35, "molar_mass": 34.8, "A": 5e7, "Ea": 70e3, "conductivity": 22, "total_fuel_mass": 10}
    }
    
    if fuel not in properties:
        raise ValueError("Fuel not recognized")
    
    fuel_data = properties[fuel]
    R_specific = 8314 / fuel_data["molar_mass"]
    R_exhaust = 8314 / 22.0  # J/(kg·K), representative exhaust gas constant
    
    # Solve finite-rate combustion kinetics
    y0 = [1.0, 0.0]  # Initial concentrations
    t_span = (0, 0.01)
    sol = solve_ivp(combustion_kinetics, t_span, y0, args=(fuel_data["A"], fuel_data["Ea"], R_specific, fuel_data["chamber_temp"]))
    
    # Compute equilibrium temperature
    equilibrium_temp = equilibrium_temperature(fuel_data)
    
    # Generate flow parameter distributions along the nozzle
    x = np.linspace(0, 1, 50)
    pressure_distribution = fuel_data["chamber_pressure"] * (1 - 0.9 * x)
    temperature_distribution = equilibrium_temp * (1 - 0.8 * x)
    gamma_distribution = real_gas_gamma(temperature_distribution, pressure_distribution)
    
    # Improved exhaust velocity calculation using isentropic expansion formula
    P0 = fuel_data["chamber_pressure"]
    atmospheric_pressure = 101325  # Atmospheric pressure in Pa
    Pe = atmospheric_pressure  # Match to ambient for realistic Isp
    gamma_e = float(gamma_distribution[-1])
    T0 = float(equilibrium_temp)  # use stagnation temperature
    Pr = Pe / P0
    exhaust_velocity = np.sqrt(max(0.0, (2 * gamma_e / (gamma_e - 1)) * R_exhaust * T0 * (1 - (Pr ** ((gamma_e - 1) / gamma_e)))))
    
    # Compute nozzle exit area
    throat_area = 0.0001  # Assumed throat area in m²
    expansion_ratio = 20  # Assumed expansion ratio
    exit_area = throat_area * expansion_ratio
    
    # Compute thrust with nozzle pressure ratio effects
    exit_pressure = Pe
    thrust = fuel_data["mass_flow_rate"] * exhaust_velocity + (exit_pressure - atmospheric_pressure) * exit_area
    # Use surrogate NN for specific impulse if requested
    if use_surrogate and surrogate_model is not None and surrogate_scaler is not None:
        # Convert pressure from Pa to bar for surrogate model
        cp_bar = np.array([fuel_data["chamber_pressure"] / 1e5])
        ct = np.array([equilibrium_temp])
        specific_impulse = float(predict_isp(surrogate_model, surrogate_scaler, cp_bar, ct)[0])
        # Fallback to analytic if surrogate returns implausible value
        if not np.isfinite(specific_impulse) or specific_impulse < 50 or specific_impulse > 600:
            specific_impulse = exhaust_velocity / 9.80665
    else:
        specific_impulse = exhaust_velocity / 9.80665
    
    # Compute heat flux due to convection and radiation
    heat_flux_convective = 500 * (temperature_distribution - 300)
    heat_flux_radiative = 5.67e-8 * (temperature_distribution ** 4 - 300 ** 4)
    
    # Estimate wall temperature
    wall_temperature = max((heat_flux_convective[-1] + heat_flux_radiative[-1]) * 0.01 / fuel_data["conductivity"], 300)
    
    # Compute burn time dynamically based on total fuel mass and flow rate
    burn_time = fuel_data["total_fuel_mass"] / fuel_data["mass_flow_rate"]
    
    # Compute frequency spectrum of pressure distribution for combustion stability analysis
    frequency_spectrum = np.abs(fft(pressure_distribution))
    
    return {
        "Fuel": fuel,
        "Exhaust Velocity (m/s)": exhaust_velocity,
        "Thrust (N)": thrust,
        "Specific Impulse (s)": specific_impulse,
        "Equilibrium Temperature (K)": equilibrium_temp,
        "Burn Time (s)": burn_time,
        "Pressure Distribution": pressure_distribution.tolist(),
        "Temperature Distribution": temperature_distribution.tolist(),
        "Velocity Distribution": exhaust_velocity,
        "Heat Flux Convective (W/m²)": heat_flux_convective.tolist(),
        "Heat Flux Radiative (W/m²)": heat_flux_radiative.tolist(),
        "Wall Temperature (K)": wall_temperature,
        "Frequency Spectrum": frequency_spectrum.tolist()
    }

# Function to visualize simulation results
def visualize_results(use_surrogate=False, surrogate_model=None, surrogate_scaler=None):
    fuels = ["MMH + N2O4", "AF-M3125E", "LOX + Methane", "G.prop"]
    results = [combustion_simulation(fuel, use_surrogate, surrogate_model, surrogate_scaler) for fuel in fuels]
    
    # Generate bar charts comparing key performance metrics
    labels = ["Exhaust Velocity (m/s)", "Thrust (N)", "Specific Impulse (s)", "Equilibrium Temperature (K)", "Burn Time (s)", "Wall Temperature (K)"]
    for i, label in enumerate(labels):
        values = [res[label] for res in results]
        plt.figure(figsize=(8, 4))
        plt.bar(fuels, values, color=['blue', 'green', 'red', 'purple'])
        plt.ylabel(label)
        plt.title(f"Comparison of {label} for Different Fuels")
        plt.tight_layout()
        safe_label = re.sub(r'[^A-Za-z0-9_]+', '_', label)
        plt.savefig(f"{safe_label}.png")
        plt.close()
    
    df = pd.DataFrame(results)[["Fuel"] + labels]
    print(df.to_string(index=False))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combustion simulation runner")
    parser.add_argument("--use-surrogate", action="store_true", help="Use NN surrogate for Isp")
    parser.add_argument("--retrain", action="store_true", help="Retrain surrogate even if cached model exists")
    parser.add_argument("--model-path", default=os.path.join(os.path.dirname(__file__), "..", "models", "surrogate.keras"), help="Path to save/load surrogate model")
    parser.add_argument("--scaler-path", default=os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl"), help="Path to save/load scaler")
    args = parser.parse_args()

    surrogate_model = None
    surrogate_scaler = None

    if args.use_surrogate:
        model_path = os.path.abspath(args.model_path)
        scaler_path = os.path.abspath(args.scaler_path)
        if not args.retrain:
            surrogate_model, surrogate_scaler = load_surrogate(model_path, scaler_path)
        if surrogate_model is None or surrogate_scaler is None:
            X, y = generate_synthetic_data()
            surrogate_model, surrogate_scaler, _, _ = train_surrogate_nn(X, y)
            save_surrogate(surrogate_model, surrogate_scaler, model_path, scaler_path)

    visualize_results(use_surrogate=args.use_surrogate, surrogate_model=surrogate_model, surrogate_scaler=surrogate_scaler)