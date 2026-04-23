"""
Model training for the Champion XGBoost pipeline on Home Credit data.

Run:
    python train_model.py
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from data_preprocess import CAT_FEATURES, NUM_FEATURES, get_home_credit_data


def build_xgb_pipeline(scale_pos_weight: float) -> Pipeline:
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, NUM_FEATURES),
            ("cat", categorical_transformer, CAT_FEATURES),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="auc",
                    scale_pos_weight=scale_pos_weight,
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_pipeline(name: str, pipeline: Pipeline, X_tr, y_tr, X_te, y_te) -> dict:
    with mlflow.start_run(run_name=name) as run:
        print(f"\n{'=' * 60}")
        print(f"  Training: {name}")
        print(f"{'=' * 60}")

        pipeline.fit(X_tr, y_tr)

        y_pred = pipeline.predict(X_te)
        y_proba = pipeline.predict_proba(X_te)[:, 1]

        f1 = f1_score(y_te, y_pred, zero_division=0)
        auc = roc_auc_score(y_te, y_proba)

        print(f"  F1      : {f1:.4f}")
        print(f"  AUC-ROC : {auc:.4f}")

        mlflow.log_params(
            {
                "model_name": name,
                "dataset": "HomeCredit",
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
            }
        )
        mlflow.log_metrics({"f1": f1, "auc": auc})
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="xgboost_model",
            serialization_format="skops",
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )
        client = MlflowClient()
        model_uri = f"runs:/{run.info.run_id}/xgboost_model"
        registry_result = mlflow.register_model(
            model_uri=model_uri,
            name="CreditRiskModel",
        )
        client.set_model_version_tag(
            name="CreditRiskModel",
            version=registry_result.version,
            key="role",
            value="Champion",
        )

        return {
            "name": name,
            "auc": auc,
            "f1": f1,
            "run_id": run.info.run_id,
            "model": pipeline,
        }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_home_credit_data()

    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio (neg/pos): {imbalance_ratio:.1f}")

    pipeline = build_xgb_pipeline(scale_pos_weight=imbalance_ratio)
    result = evaluate_pipeline(
        "XGBoost_Champion",
        pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    client = MlflowClient()
    client.set_tag(result["run_id"], "Champion", "True")
    print(f"\nChampion model tagged -> Run ID: {result['run_id']}")
    print(f"AUC={result['auc']:.4f} | F1={result['f1']:.4f}")
