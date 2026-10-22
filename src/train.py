import os
import joblib
import mlflow
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_prep import load_or_generate_data, time_based_split
from src.features import FeatureEngineer, split_X_y
from src.evaluate import optimize_threshold

def train_models(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, float, FeatureEngineer]:
    # ensure no leakage
    fraud_ratio = y_train.mean()
    scale_pos_weight = (1 - fraud_ratio) / fraud_ratio if fraud_ratio > 0 else 1

    models = {
        'LogisticRegression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced_subsample',
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='aucpr'
        )
    }

    best_model = None
    best_f2 = -1.0
    best_thresh = 0.5
    best_name = ""

    mlflow.set_experiment("fraud_detection_benchmark")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_val)[:, 1]

            # optimize threshold for 5% max FPR
            thresh, metrics = optimize_threshold(y_val, y_prob, max_fpr=0.05)

            # log params and metrics
            mlflow.log_param("model_type", name)
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())

            mlflow.log_metrics(metrics)
            mlflow.log_param("optimal_threshold", thresh)

            print(f"{name} Results: PR-AUC: {metrics['pr_auc']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}, F2: {metrics['f2_score']:.4f} at thresh: {thresh:.4f}")

            if metrics['f2_score'] > best_f2:
                best_f2 = metrics['f2_score']
                best_model = model
                best_thresh = thresh
                best_name = name

    print(f"\nBest Model: {best_name} with F2: {best_f2:.4f} at thresh: {best_thresh:.4f}")

    # Save the best model
    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'model': best_model,
        'threshold': best_thresh
    }, 'models/best_model.joblib')

    return best_model, best_thresh

def run_pipeline():
    print("Loading data...")
    df = load_or_generate_data()

    print("Splitting data...")
    df_train, df_val = time_based_split(df)

    print("Engineering features...")
    fe = FeatureEngineer()
    df_train_feat = fe.fit_transform(df_train)
    df_val_feat = fe.transform(df_val)

    # save feature engineer pipeline
    os.makedirs('models', exist_ok=True)
    joblib.dump(fe, 'models/feature_engineer.joblib')

    X_train, y_train = split_X_y(df_train_feat)
    X_val, y_val = split_X_y(df_val_feat)

    best_model, best_thresh = train_models(X_train, y_train, X_val, y_val)
    print("Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()
