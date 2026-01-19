from data import load_california_data
from model_training import train_model
from test import split_data, evaluate_model
import matplotlib.pyplot as plt
import random
import requests

# Load California Housing data
X, y = load_california_data()

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Train Baseline Model
baseline_model, baseline_emission = train_model(X_train, y_train, n_estimators=50, max_depth=6)
baseline_r2 = evaluate_model(baseline_model, X_test, y_test)
print("Baseline R² Score:", baseline_r2)
print(f"Baseline CO2 Emission (kg): {baseline_emission:.5f}")

# Train Optimized Model
optimized_model, optimized_emission = train_model(X_train, y_train, n_estimators=20, max_depth=4)
optimized_r2 = evaluate_model(optimized_model, X_test, y_test)
print("Optimized R² Score:", optimized_r2)
print(f"Optimized CO2 Emission (kg): {optimized_emission:.5f}")

#Live API
def get_carbon_intensity(zone="IN"):
    API_KEY = "https://api.electricitymaps.com/v3/carbon-intensity/past?datetime=2026-01-17+08%3A55" 
    url = f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    try:
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()

        if "data" in data:
            return data["data"]["carbonIntensity"], "LIVE"
        else:
            raise Exception("API unavailable")

    except Exception:
        # Fallback simulated realistic values
        simulated = random.randint(300, 700)
        return simulated, "SIMULATED"
    
carbon_intensity, source = get_carbon_intensity()

print(f"Carbon Intensity: {carbon_intensity} gCO2/kWh ({source})") # type: ignore

THRESHOLD = 500

if carbon_intensity < THRESHOLD: # type: ignore
    print("✅ Low-carbon window → Training allowed")
else:
    print("⏸ High-carbon window → Training postponed")

    
    # Plot Graphs
models = ['Baseline RF', 'Optimized RF']

# CO2 Emission
plt.figure()
plt.bar(models, [baseline_emission, optimized_emission], color=['red', 'green'])
plt.xlabel("Model Type")
plt.ylabel("CO₂ Emission (kg)")
plt.title("CO₂ Emission Comparison")
plt.show()

# R² Score
plt.figure()
plt.bar(models, [baseline_r2, optimized_r2], color=['blue', 'orange'])
plt.xlabel("Model Type")
plt.ylabel("R² Score")
plt.title("R² Score Comparison")
plt.show()

