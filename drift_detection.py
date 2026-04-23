"""
PSI drift detection between Home Credit baseline data and German Credit input data.

Run:
    python drift_detection.py
"""

import warnings

warnings.filterwarnings("ignore")

import mlflow
import numpy as np
import pandas as pd

from data_preprocess import get_home_credit_data


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute PSI between two 1-D distributions.

    For binary features (<= 2 unique values) the positive-class proportion is
    compared directly. For continuous features, percentile-based binning is
    used so the bins are equi-populated on the reference distribution.
    """
    n_unique = len(np.unique(expected))

    if n_unique <= 2:
        p_exp = np.clip(expected.mean(), 1e-4, 1 - 1e-4)
        p_act = np.clip(actual.mean(), 1e-4, 1 - 1e-4)
        exp_pct = np.array([1 - p_exp, p_exp])
        act_pct = np.array([1 - p_act, p_act])
    else:
        breakpoints = np.linspace(0, 100, bins + 1)
        bin_edges = np.unique(np.percentile(expected, breakpoints))
        if len(bin_edges) < 2:
            return 0.0
        exp_pct = np.histogram(expected, bins=bin_edges)[0] / len(expected)
        act_pct = np.histogram(actual, bins=bin_edges)[0] / len(actual)

    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi, 6)


def psi_status(psi: float) -> str:
    if psi >= 0.20:
        return "DRIFT"
    if psi >= 0.10:
        return "WARN"
    return "STABLE"


X_train_hc, _, _, _ = get_home_credit_data()

GERMAN_PATH = "data/german_data.csv"
try:
    df_german = pd.read_csv(GERMAN_PATH)
    if "checking_status" not in df_german.columns:
        df_german.columns = [
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
        ]
    print(f"German Credit loaded: {df_german.shape}")
except Exception as exc:
    print(f"ERROR loading {GERMAN_PATH}: {exc}")
    df_german = pd.DataFrame()


df_baseline = pd.DataFrame(
    {
        "credit_amount": X_train_hc["AMT_CREDIT"],
        "age": X_train_hc["DAYS_BIRTH"].abs() / 365,
        "installment_amount": X_train_hc["AMT_ANNUITY"],
        "income_proxy": X_train_hc["AMT_INCOME_TOTAL"],
        "employment_years": X_train_hc["DAYS_EMPLOYED"].abs() / 365,
    }
)

if not df_german.empty:
    df_incoming = pd.DataFrame(
        {
            "credit_amount": df_german["credit_amount"],
            "age": df_german["age"],
            "installment_amount": df_german["credit_amount"]
            / df_german["duration"].clip(lower=1),
            "income_proxy": df_german["credit_amount"]
            / df_german["installment_commitment"].clip(lower=1),
            "employment_years": df_german["employment"].map(
                lambda x: {"A71": 0, "A72": 1, "A73": 4, "A74": 7, "A75": 10}.get(str(x), 0)
            ),
        }
    )
else:
    df_incoming = df_baseline.copy()

df_baseline.fillna(df_baseline.median(), inplace=True)
df_incoming.fillna(df_incoming.median(), inplace=True)

FEATURES = [
    "credit_amount",
    "age",
    "installment_amount",
    "income_proxy",
    "employment_years",
]

# PSI must be computed on raw distributions. Normalization is removed
# to preserve true distribution shift.


psi_results = []
drifted_feats = []
any_drift = False

with mlflow.start_run(run_name="PSI_Drift_Detection") as run:
    for col in FEATURES:
        psi = compute_psi(df_baseline[col].values, df_incoming[col].values)
        status = psi_status(psi)
        drifted = psi >= 0.20

        psi_results.append(
            {
                "feature": col,
                "psi": psi,
                "status": status,
                "drift": drifted,
            }
        )

        if drifted:
            any_drift = True
            drifted_feats.append(col)

        mlflow.log_metric(f"psi_{col}", psi)

    mlflow.log_param("any_drift", str(any_drift))
    mlflow.log_param("drifted_features", ",".join(drifted_feats) if drifted_feats else "none")
    mlflow.log_param("psi_threshold", 0.20)
    mlflow.log_param("n_features_checked", len(FEATURES))

    print(f"\n{'Feature':<25} {'PSI':>8}   Status")
    print("-" * 45)
    for result in psi_results:
        marker = " <- RETRAIN" if result["drift"] else ""
        print(f"{result['feature']:<25} {result['psi']:>8.4f}   {result['status']}{marker}")

    print(f"\nFeatures checked  : {FEATURES}")
    print(f"Drifted features  : {drifted_feats if drifted_feats else 'None'}")
    print(
        f"Overall drift     : "
        f"{'YES - retraining will be triggered' if any_drift else 'NO - champion stays'}"
    )
    print(f"\nMLflow run ID     : {run.info.run_id}")
