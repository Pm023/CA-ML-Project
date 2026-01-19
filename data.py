# data.py
from sklearn.datasets import fetch_california_housing

def load_california_data():
    """
    Load the California Housing dataset.
    
    Returns:
        X: Features as DataFrame
        y: Target as Series
    """
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    return X, y
