# train_model.py
from sklearn.ensemble import RandomForestRegressor
from codecarbon import EmissionsTracker

def train_model(X_train, y_train, n_estimators=50, max_depth=6):
    """
    Train a Random Forest Regressor and track real CO2 emissions.
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
    
    Returns:
        model: Trained RandomForestRegressor
        co2_emission: CO2 emission in kg (real-time measurement)
    """
    # Start real emissions tracker
    tracker = EmissionsTracker(measure_power_secs=1)  # check every second
    tracker.start()
    
    # Train the model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Stop tracker and get emissions
    co2_emission = tracker.stop()
    
    return model, co2_emission
