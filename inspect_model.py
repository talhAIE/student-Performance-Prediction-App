import joblib
import pandas as pd
import sklearn

try:
    model = joblib.load('student_performance_rf_model.pkl')
    print("Model loaded successfully.")
    
    if hasattr(model, 'feature_names_in_'):
        print("Feature names found:")
        for name in model.feature_names_in_:
            print(f"- {name}")
    else:
        print("Model does not store feature names. Please check input data used for training.")
        
except Exception as e:
    print(f"Error loading model: {e}")
