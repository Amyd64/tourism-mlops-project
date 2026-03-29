# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))


Xtrain_path = "hf://datasets/Amyd64/tourism-package-prediction/Xtrain.csv"
Xtest_path  = "hf://datasets/Amyd64/tourism-package-prediction/Xtest.csv"
ytrain_path = "hf://datasets/Amyd64/tourism-package-prediction/ytrain.csv"
ytest_path  = "hf://datasets/Amyd64/tourism-package-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest  = pd.read_csv(ytest_path).squeeze()


# Define numeric and categorical features
numeric_features = [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar',
    'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'CityTier', 'Occupation', 'Gender',
    'ProductPitched', 'MaritalStatus', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'Designation'
]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost Classifier
xgb_model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', use_label_encoder=False)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.7, 0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.7, 0.8, 1.0],
    'xgbclassifier__reg_lambda': [0.1, 1, 10]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='f1')
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_f1_score", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test  = best_model.predict(Xtest)
    y_proba_test = best_model.predict_proba(Xtest)[:, 1]

    # Metrics
    train_acc = accuracy_score(ytrain, y_pred_train)
    test_acc  = accuracy_score(ytest, y_pred_test)

    train_f1 = f1_score(ytrain, y_pred_train)
    test_f1  = f1_score(ytest, y_pred_test)

    train_auc = roc_auc_score(ytrain, best_model.predict_proba(Xtrain)[:, 1])
    test_auc  = roc_auc_score(ytest, y_proba_test)

    # Log metrics
    mlflow.log_metrics({
        "train_Accuracy": train_acc,
        "test_Accuracy":  test_acc,
        "train_F1":       train_f1,
        "test_F1":        test_f1,
        "train_AUC":      train_auc,
        "test_AUC":       test_auc
    })

    # Save the model locally
    model_path = "best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id   = "Amyd64/tourism-package-model"
    repo_type = "model"

    # Step 1: Check if the model repo exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{repo_id}' not found. Creating new repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj="best_tourism_package_model_v1.joblib",
        path_in_repo="best_tourism_package_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
