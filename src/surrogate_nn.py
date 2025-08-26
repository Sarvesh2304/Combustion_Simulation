import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

def generate_synthetic_data(n_samples=5000, random_seed=0):
    np.random.seed(random_seed)
    # Inputs
    chamber_pressure_bar = np.random.uniform(5.0, 200.0, n_samples)  # bar (wider range)
    chamber_temperature_K = np.random.uniform(1800.0, 3800.0, n_samples)  # Kelvin

    # Constants / assumptions
    g0 = 9.80665
    M_g_per_mol = 22.0  # representative exhaust molar mass (g/mol)
    # Note: numeric g/mol equals numeric kg/kmol (e.g., 22 g/mol -> 22 kg/kmol)
    M_kg_per_kmol = M_g_per_mol
    R_universal = 8314.0
    R_specific = R_universal / M_kg_per_kmol  # J/(kg·K)
    # Match main model: Pe equals ambient (sea level)
    Pe_bar = 1.01325
    # Pressure ratio Pe/P0 depends on chamber pressure
    Pr = Pe_bar / chamber_pressure_bar

    # Gamma varies mildly with temperature; clip to physical range similar to main
    gamma = 1.4 - 0.1 * (chamber_temperature_K / 3000.0)
    gamma = np.clip(gamma, 1.15, 1.35)

    # Isentropic exit velocity approximation with fixed Pe/P0
    # Ve = sqrt( 2*g/(g-1) * R*T0 * (1 - (Pe/P0)^((g-1)/g)) )
    term = (1.0 - np.power(Pr, (gamma - 1.0) / gamma))
    coeff = (2.0 * gamma / (gamma - 1.0))
    Ve = np.sqrt(np.maximum(0.0, coeff * R_specific * chamber_temperature_K * term))

    # Specific impulse in seconds
    isp_true = Ve / g0

    # Add light noise to emulate modeling error
    noise = np.random.normal(0.0, 3.0, size=n_samples)
    isp_noisy = isp_true + noise

    X = np.vstack((chamber_pressure_bar, chamber_temperature_K)).T
    y = isp_noisy
    return X, y

def train_surrogate_nn(X, y, epochs=1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=256, validation_split=0.2, callbacks=[callback], verbose=0)
    return model, scaler, X_test_scaled, y_test

def predict_isp(model, scaler, chamber_pressure, chamber_temperature):
    X_new = np.vstack((chamber_pressure, chamber_temperature)).T
    X_new_scaled = scaler.transform(X_new)
    return model.predict(X_new_scaled).flatten()

def evaluate_and_plot(model, scaler, X_test_scaled, y_test):
    import matplotlib.pyplot as plt
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test MAE: {test_mae:.2f} seconds")
    y_pred = model.predict(X_test_scaled).flatten()
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, label='NN predictions')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal')
    plt.xlabel('True Isp (s)')
    plt.ylabel('Predicted Isp (s)')
    plt.title('NN Surrogate vs. True Values')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    X, y = generate_synthetic_data()
    model, scaler, X_test_scaled, y_test = train_surrogate_nn(X, y)
    evaluate_and_plot(model, scaler, X_test_scaled, y_test)
    # Example prediction
    cp = np.array([10, 15])  # bar
    ct = np.array([2000, 3000])  # K
    pred = predict_isp(model, scaler, cp, ct)
    print(f"Predicted Isp for cp={cp}, ct={ct}: {pred}") 