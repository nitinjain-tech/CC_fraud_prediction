import pickle
import numpy as np
import logging
import json
from datetime import datetime

# Load artifacts
model = pickle.load(open("artifacts/model.pkl", "rb"))
threshold = pickle.load(open("artifacts/threshold.pkl", "rb"))

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO
    )

def log_prediction(features, prob, prediction):

    log_data = {
        "timestamp": str(datetime.now()),
        "features": features,
        "fraud_probability": float(prob),
        "prediction": int(prediction)
    }

    logging.info(json.dumps(log_data))
    
def predict_fraud(features):

    features = np.array(features).reshape(1, -1)

    prob = model.predict_proba(features)[0][1]

    prediction = 1 if prob > threshold else 0

    log_prediction(features.tolist(), prob, prediction)

    return prob, prediction