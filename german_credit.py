"""
Champion / Challenger evaluation on German Credit data.

Pipeline:
  1. Load and normalize German Credit data
  2. Read the latest PSI drift result from MLflow
  3. Load the Champion XGBoost model from MLflow
  4. Run Champion production predictions on German Credit
  5. If drift is detected, retrain an XGBoost Challenger on German Credit only
  6. Evaluate both models on the same German Credit evaluation split with mlflow.evaluate()
  7. Select the model with the higher AUC

Run:
    python german_credit.py
"""

import re
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


GERMAN_PATH = "data/german_data.csv"
PSI_THRESHOLD = 0.20
RANDOM_STATE = 42

GERMAN_CATEGORICAL_COLS = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

GERMAN_COLUMN_MAP = {
    "checking_status": "checking_status",
    "checking_account_status": "checking_status",
    "duration": "duration",
    "duration_in_month": "duration",
    "credit_history": "credit_history",
    "purpose": "purpose",
    "credit_amount": "credit_amount",
    "savings_status": "savings_status",
    "savings_account_bonds": "savings_status",
    "employment": "employment",
    "installment_commitment": "installment_commitment",
    "installment": "installment_commitment",
    "personal_status": "personal_status",
    "status_n_sex": "personal_status",
    "other_parties": "other_parties",
    "other_debtors_guarantors": "other_parties",
    "residence_since": "residence_since",
    "residence": "residence_since",
    "property_magnitude": "property_magnitude",
    "property": "property_magnitude",
    "age": "age",
    "age_in_years": "age",
    "other_payment_plans": "other_payment_plans",
    "other_installment_plans": "other_payment_plans",
    "housing": "housing",
    "existing_credits": "existing_credits",
    "existing_credits_no": "existing_credits",
    "job": "job",
    "num_dependents": "num_dependents",
    "liability_responsibles": "num_dependents",
    "own_telephone": "own_telephone",
    "telephone": "own_telephone",
    "foreign_worker": "foreign_worker",
    "class": "class",
    "category": "class",
}


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def load_german_credit(path: str = GERMAN_PATH) -> pd.DataFrame:
    df_german = pd.read_csv(path)
    renamed_cols = {
        column: GERMAN_COLUMN_MAP.get(_normalize_column_name(column), _normalize_column_name(column))
        for column in df_german.columns
    }
    df_german = df_german.rename(columns=renamed_cols)

    required_cols = {
        "checking_status",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_status",
        "employment",
        "installment_commitment",
        "personal_status",
        "other_parties",
        "residence_since",
        "property_magnitude",
        "age",
        "other_payment_plans",
        "housing",
        "existing_credits",
        "job",
        "num_dependents",
        "own_telephone",
        "foreign_worker",
        "class",
    }
    missing_cols = sorted(required_cols - set(df_german.columns))
    if missing_cols:
        raise ValueError(f"Missing German Credit columns: {missing_cols}")

    if df_german["class"].isin([1, 2]).all():
        df_german["class"] = (df_german["class"] == 2).astype(int)
    else:
        df_german["class"] = df_german["class"].astype(int)

    for column in GERMAN_CATEGORICAL_COLS:
        df_german[column] = df_german[column].astype(str)

    return df_german


def build_champion_inference_frame(df_german_features: pd.DataFrame) -> pd.DataFrame:
    employment_years = df_german_features["employment"].map(
        {"A71": 0, "A72": 1, "A73": 4, "A74": 7, "A75": 10}
    ).fillna(0)
    gender_map = {"A91": "M", "A92": "F", "A93": "M", "A94": "M", "A95": "F"}
    family_map = {
        "A91": "Separated",
        "A92": "Married",
        "A93": "Single / not married",
        "A94": "Married",
        "A95": "Single / not married",
    }
    housing_map = {
        "A151": "Rented apartment",
        "A152": "House / apartment",
        "A153": "With parents",
    }

    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": df_german_features["credit_amount"]
            / df_german_features["installment_commitment"].clip(lower=1),
            "AMT_CREDIT": df_german_features["credit_amount"],
            "AMT_ANNUITY": df_german_features["credit_amount"]
            / df_german_features["duration"].clip(lower=1),
            "DAYS_EMPLOYED": -(employment_years * 365.0),
            "DAYS_BIRTH": -(df_german_features["age"].clip(lower=1) * 365.0),
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER": df_german_features["personal_status"].map(gender_map).fillna("M"),
            "FLAG_OWN_CAR": "N",
            "FLAG_OWN_REALTY": df_german_features["housing"]
            .map({"A152": "Y", "A151": "N", "A153": "N"})
            .fillna("N"),
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Secondary / secondary special",
            "NAME_FAMILY_STATUS": df_german_features["personal_status"]
            .map(family_map)
            .fillna("Married"),
            "NAME_HOUSING_TYPE": df_german_features["housing"].map(housing_map).fillna(
                "House / apartment"
            ),
        }
    )


def build_challenger_pipeline(
    categorical_cols: list[str], numeric_cols: list[str], scale_pos_weight: float
) -> Pipeline:
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
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    classifier = XGBClassifier(
        use_label_encoder=False,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_model(model_uri: str, evaluation_data: pd.DataFrame, prefix: str):
    result = mlflow.evaluate(
        model=model_uri,
        data=evaluation_data,
        targets="target",
        model_type="classifier",
        evaluator_config={"metric_prefix": prefix},
    )
    return result, result.metrics[f"{prefix}roc_auc"], result.metrics[f"{prefix}f1_score"]


print("[Step 1] Loading German Credit data...")
try:
    df_german = load_german_credit()
    print(f"German Credit loaded: {df_german.shape}")
except FileNotFoundError:
    print(f"ERROR: '{GERMAN_PATH}' not found. Place german_data.csv in data/ and rerun.")
    raise SystemExit(1)
except Exception as exc:
    print(f"ERROR loading German Credit data: {exc}")
    raise SystemExit(1)

X_german = df_german.drop(columns=["class"]).copy()
y_german = df_german["class"].copy()

Xg_train, Xg_test, yg_train, yg_test = train_test_split(
    X_german,
    y_german,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_german,
)
print(f"German Train: {Xg_train.shape}  Test: {Xg_test.shape}")

champion_test_frame = build_champion_inference_frame(Xg_test)
# Both models are evaluated on the same underlying dataset (German Credit),
# but using their respective preprocessing pipelines.
champion_eval_data = champion_test_frame.copy()
champion_eval_data["target"] = yg_test.to_numpy()

challenger_eval_data = Xg_test.copy()
challenger_eval_data["target"] = yg_test.to_numpy()


print("\n[Step 2] Reading latest PSI drift result from MLflow...")
try:
    drift_runs = mlflow.search_runs(
        filter_string="tags.`mlflow.runName` = 'PSI_Drift_Detection'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if drift_runs.empty:
        print("WARNING: No PSI_Drift_Detection run found. Assuming NO drift.")
        any_drift = False
        drifted_features = "none"
    else:
        any_drift = str(drift_runs.iloc[0].get("params.any_drift", "False")).lower() == "true"
        drifted_features = str(drift_runs.iloc[0].get("params.drifted_features", "none"))
        print(f"PSI result -> Drift detected: {'YES' if any_drift else 'NO'}")
        if drifted_features != "none":
            print(f"Drifted features: {drifted_features}")
except Exception as exc:
    print(f"MLflow PSI fetch error: {exc}")
    raise SystemExit(1)


print("\n[Step 3] Loading Champion model from MLflow...")
try:
    champ_runs = mlflow.search_runs(
        filter_string="tags.Champion = 'True'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if champ_runs.empty:
        raise ValueError("No Champion-tagged run found in MLflow. Run train_model.py first.")
    champ_run_id = champ_runs.iloc[0]["run_id"]
    champion_uri = f"runs:/{champ_run_id}/xgboost_model"
    champion_model = mlflow.sklearn.load_model(champion_uri)
    print(f"Champion loaded -> Run ID: {champ_run_id}")
except Exception as exc:
    print(f"MLflow model load error: {exc}")
    raise SystemExit(1)


print("\n[Step 4] Running Champion production predictions on German Credit...")
y_champion_proba = champion_model.predict_proba(champion_test_frame)[:, 1]

all_results = []
selected_name = "XGBoost Champion"
verdict = "NO DRIFT -> Champion stays"

with mlflow.start_run(run_name="German_Credit_Evaluation") as run:
    client = MlflowClient()
    # mlflow.evaluate is used for tracking, but final model selection is based on
    # manually computed metrics to ensure fair comparison across different preprocessing pipelines.
    evaluate_model(champion_uri, champion_eval_data, "champion_")
    champion_auc = roc_auc_score(yg_test, y_champion_proba)
    champion_f1 = f1_score(yg_test, (y_champion_proba > 0.5).astype(int), zero_division=0)
    champion_res = {
        "name": "XGBoost Champion",
        "auc": champion_auc,
        "f1": champion_f1,
        "y_true": yg_test.to_numpy(),
        "y_proba": y_champion_proba,
    }
    all_results.append(champion_res)

    print("\nChampion metrics on German Credit")
    print(f"AUC: {champion_auc:.4f}")
    print(f"F1 : {champion_f1:.4f}")

    if any_drift:
        print(f"\n[Step 5] Drift detected on: {drifted_features}")
        print("Retraining XGBoost Challenger on German Credit only...")

        challenger_categorical_cols = [
            col for col in GERMAN_CATEGORICAL_COLS if col in Xg_train.columns
        ]
        challenger_numeric_cols = [
            col for col in Xg_train.columns if col not in challenger_categorical_cols
        ]
        scale_pos_weight = (yg_train == 0).sum() / max((yg_train == 1).sum(), 1)

        challenger_model = build_challenger_pipeline(
            categorical_cols=challenger_categorical_cols,
            numeric_cols=challenger_numeric_cols,
            scale_pos_weight=scale_pos_weight,
        )
        challenger_model.fit(Xg_train, yg_train)

        mlflow.sklearn.log_model(
            sk_model=challenger_model,
            name="xgboost_challenger",
            serialization_format="skops",
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )
        challenger_uri = f"runs:/{run.info.run_id}/xgboost_challenger"
        registry_result = mlflow.register_model(
            model_uri=challenger_uri,
            name="CreditRiskModel",
        )
        client.set_model_version_tag(
            name="CreditRiskModel",
            version=registry_result.version,
            key="role",
            value="Challenger",
        )
        y_challenger_proba = challenger_model.predict_proba(Xg_test)[:, 1]

        evaluate_model(challenger_uri, challenger_eval_data, "challenger_")
        challenger_auc = roc_auc_score(yg_test, y_challenger_proba)
        challenger_f1 = f1_score(yg_test, (y_challenger_proba > 0.5).astype(int), zero_division=0)
        challenger_res = {
            "name": "XGBoost Challenger",
            "auc": challenger_auc,
            "f1": challenger_f1,
            "y_true": yg_test.to_numpy(),
            "y_proba": y_challenger_proba,
        }
        all_results.append(challenger_res)

        print("\nChallenger metrics on German Credit")
        print(f"AUC: {challenger_auc:.4f}")
        print(f"F1 : {challenger_f1:.4f}")

        if challenger_auc > champion_auc:
            selected_name = challenger_res["name"]
            verdict = "DRIFT DETECTED -> Challenger wins on AUC"
        else:
            selected_name = champion_res["name"]
            verdict = "DRIFT DETECTED -> Champion stays"
    else:
        print(f"\n[Step 5] PSI < {PSI_THRESHOLD} on all features. No retraining triggered.")

    roc_path = "roc_curves.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "tomato"]

    for index, result in enumerate(all_results):
        fpr, tpr, _ = roc_curve(result["y_true"], result["y_proba"])
        ax.plot(
            fpr,
            tpr,
            color=colors[index % len(colors)],
            lw=2,
            label=f"{result['name']} (AUC = {result['auc']:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("AUC-ROC on German Credit")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    mlflow.log_param("evaluation_dataset", "German Credit")
    mlflow.log_param("selected_model", selected_name)
    mlflow.log_param("verdict", verdict)
    mlflow.log_param("any_drift", str(any_drift))
    mlflow.log_param("drifted_features", drifted_features if any_drift else "none")

    mlflow.log_metric("champion_auc", champion_auc)
    mlflow.log_metric("champion_f1", champion_f1)
    mlflow.log_metric("champion_auc_manual", champion_auc)
    mlflow.log_metric("champion_f1_manual", champion_f1)

    if len(all_results) > 1:
        mlflow.log_metric("challenger_auc", all_results[1]["auc"])
        mlflow.log_metric("challenger_f1", all_results[1]["f1"])
        mlflow.log_metric("challenger_auc_manual", all_results[1]["auc"])
        mlflow.log_metric("challenger_f1_manual", all_results[1]["f1"])

    final_result = next(result for result in all_results if result["name"] == selected_name)
    mlflow.log_metric("final_auc", final_result["auc"])
    mlflow.log_metric("final_f1", final_result["f1"])
    mlflow.log_artifact(roc_path)

    print("\nFinal model comparison")
    for result in all_results:
        print(f"{result['name']}: AUC={result['auc']:.4f} | F1={result['f1']:.4f}")
    print(f"Selected model: {selected_name}")
    print(f"Verdict       : {verdict}")
    print(f"MLflow run ID : {run.info.run_id}")
