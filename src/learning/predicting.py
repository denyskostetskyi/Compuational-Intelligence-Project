import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

import constants

def predict_target(X_test: pd.DataFrame):
    model_path = Path(constants.PATH_MODEL_RANDOM_FOREST_CLASSIFIER)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Please, train a model first!")

    model_rfc: RandomForestClassifier = joblib.load(model_path)
    y_pred = model_rfc.predict(X_test)
    mapped_prediction = [constants.TARGET_MAPPING_INV[pred] for pred in y_pred]
    result = X_test.copy()
    result['prediction'] = mapped_prediction
    result.to_csv(constants.PATH_PREDICTION_RESULTS, index=True)
    return mapped_prediction
