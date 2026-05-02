import pandas as pd
import pickle 
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

df = pd.read_csv("../FraudDetectionDataset/Eda_processed.csv")

print('csv read done..')

df = df.drop(columns=['Unnamed: 0'])

X = df.drop("Class",axis=1)
y = df["Class"]

print('data partition done..')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('data split done..')

smote = SMOTE(random_state=42)
X_resampled,y_resampled = smote.fit_resample(X_train, y_train)

print('smote done..')

mlflow.set_tracking_uri("azureml://centralindia.api.azureml.ms/mlflow/v1.0/subscriptions/b28de3df-2d16-4658-a408-e3d20bcd36fe/resourceGroups/AI_ML/providers/Microsoft.MachineLearningServices/workspaces/AI_ML_WS")
mlflow.set_experiment("fraud-detection")

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_resampled, y_resampled)

    y_probs = model.predict_proba(X_test)[:,1]

    from sklearn.metrics import roc_auc_score, average_precision_score

    roc = roc_auc_score(y_test, y_probs)
    pr = average_precision_score(y_test, y_probs)

    # Log params
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)

    # Log metrics
    mlflow.log_metric("roc_auc", roc)
    mlflow.log_metric("pr_auc", pr)

    print('model training done..')

    pickle.dump(model,open("../artifacts/model.pkl","wb"))
    threshold = 0.6
    pickle.dump(threshold,open("../artifacts/threshold.pkl","wb"))
    print('model dumping done..')

    # Log model
    mlflow.log_artifact("../artifacts/model.pkl")   
    mlflow.log_artifact("../artifacts/threshold.pkl")
    print('Completed !')

