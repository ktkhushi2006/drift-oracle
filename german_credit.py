"""
FILE 4: German Credit — XGBoost vs SVM + PSI-Triggered SVM Retrain + Model Validity
─────────────────────────────────────────────────────────────────────────────────────
Pipeline:
  1. Load + preprocess German Credit data
  2. Load PSI results from File 3  (drift_data.pkl)
  3. Load XGBoost + SVM from File 2 (trained_models.pkl → results list)
  4. Evaluate both models on German Credit (baseline comparison)
  5. If PSI > 0.2 (drift detected):
       → Retrain SVM on German Credit (new/drifted) data
       → Compare predictions: XGBoost vs retrained SVM on same test set
       → Agreement ≥ 90%  →  ✅ OLD MODEL STABLE  (XGBoost remains valid)
       → Agreement <  90%  →  ❌ OLD MODEL INVALID (deploy retrained SVM)
  6. Final comparison table: XGBoost vs SVM (retrained if drift, original if stable)

Run AFTER: 1_data_preprocess.py → 2_train_models.py → 3_drift_detection.py
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────
GERMAN_PATH         = "data/german_data.csv"
PSI_DRIFT_THRESHOLD = 0.2    # PSI > 0.2 triggers SVM retrain
AGREE_THRESHOLD     = 0.90   # ≥90% prediction agreement → stable
# ──────────────────────────────────────────────

# Known categorical columns for German Credit dataset
CATEGORICAL_COLS = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"
]


# ══════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════

def find_best_threshold(y_true, y_proba):
    """Scan thresholds 0.1–0.9, return the one with best F1."""
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_proba >= t).astype(int)
        f1    = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = t
    return round(best_thresh, 2)


def evaluate_fitted_model(name, model, X_tr, y_tr, X_te, y_te, sample_size=None):
    """
    Fit and evaluate a model.
    sample_size: if set, trains on a random subset (used for SVM speed).
    Returns metrics dict including raw y_pred for comparison.
    """
    if sample_size and sample_size < len(X_tr):
        idx  = np.random.RandomState(42).choice(len(X_tr), size=sample_size, replace=False)
        X_tr = X_tr[idx]
        y_tr = y_tr[idx]
        print(f"    (training on {sample_size} samples for speed)")

    model.fit(X_tr, y_tr)
    y_proba = model.predict_proba(X_te)[:, 1]
    thresh  = find_best_threshold(y_te, y_proba)
    y_pred  = (y_proba >= thresh).astype(int)

    auc      = roc_auc_score(y_te, y_proba)
    f1       = f1_score(y_te, y_pred, zero_division=0)
    f1_macro = f1_score(y_te, y_pred, average="macro", zero_division=0)
    accuracy = (y_pred == y_te).mean()

    print(f"\n{'─' * 50}")
    print(f"  Model     : {name}  (threshold={thresh})")
    print(f"{'─' * 50}")
    print(f"  AUC       : {auc:.4f}")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  F1        : {f1:.4f}  (class=1, bad credit)")
    print(f"  F1 Macro  : {f1_macro:.4f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['Good Credit', 'Bad Credit'], zero_division=0)}")

    return {
        "name":     name,
        "model":    model,
        "auc":      auc,
        "f1":       f1,
        "f1_macro": f1_macro,
        "accuracy": accuracy,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
    }


def print_comparison_table(results, title="Model Comparison"):
    """Print a clean side-by-side comparison table."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    df = pd.DataFrame([
        {
            "Model":        r["name"],
            "AUC":          round(r["auc"],      4),
            "Accuracy":     f"{r['accuracy']*100:.1f}%",
            "F1 (Default)": round(r["f1"],       4),
            "F1 Macro":     round(r["f1_macro"], 4),
        }
        for r in results
    ])
    print(df.to_string(index=False))
    best = max(results, key=lambda x: x["auc"])
    print(f"\n  ✅ Best by AUC : {best['name']} (AUC={best['auc']:.4f})")
    return best


def compare_predictions(xgb_preds, svm_preds):
    """
    Compare element-wise predictions between XGBoost and retrained SVM.
    Returns (agreement_ratio, verdict).
    """
    total    = len(xgb_preds)
    agree    = int((xgb_preds == svm_preds).sum())
    disagree = total - agree
    ratio    = agree / total

    print(f"\n{'═' * 60}")
    print("  PREDICTION COMPARISON — XGBoost (Old) vs SVM (Retrained)")
    print(f"{'═' * 60}")
    print(f"  Total test samples   : {total}")
    print(f"  Agreements           : {agree}  ({ratio:.1%})")
    print(f"  Disagreements        : {disagree}  ({1-ratio:.1%})")
    print(f"  Agreement threshold  : {AGREE_THRESHOLD:.0%}")

    if ratio >= AGREE_THRESHOLD:
        verdict = "STABLE"
        print(f"\n  ✅ VERDICT: OLD MODEL IS STABLE")
        print(f"     XGBoost and retrained SVM agree on {ratio:.1%} of predictions.")
        print(f"     The original XGBoost remains valid for production.")
    else:
        verdict = "INVALID"
        print(f"\n  ❌ VERDICT: OLD MODEL IS INVALID")
        print(f"     XGBoost and retrained SVM disagree on {1-ratio:.1%} of predictions.")
        print(f"     The original XGBoost should be retired.")
        print(f"     Deploy the retrained SVM model instead.")
    print(f"{'═' * 60}\n")

    return ratio, verdict


# ══════════════════════════════════════════════
# STEP 8: LOAD + PREPROCESS GERMAN CREDIT DATA
# ══════════════════════════════════════════════
print("=" * 60)
print("STEP 8: German Credit Dataset — Load & Preprocess")
print("=" * 60)

try:
    df_german = pd.read_csv(GERMAN_PATH)

    # Assign column names if CSV has no header row
    if df_german.shape[1] == 21 and "checking_status" not in df_german.columns:
        german_cols = [
            "checking_status", "duration", "credit_history", "purpose", "credit_amount",
            "savings_status", "employment", "installment_commitment", "personal_status",
            "other_parties", "residence_since", "property_magnitude", "age",
            "other_payment_plans", "housing", "existing_credits", "job",
            "num_dependents", "own_telephone", "foreign_worker", "class"
        ]
        df_german.columns = german_cols

    print(f"✅ Loaded German Credit: {df_german.shape}")

    # Find target column (case-insensitive)
    target_col = next((c for c in df_german.columns if c.lower() == "class"), None)
    if target_col is None:
        raise ValueError(f"No 'class' column found. Columns: {df_german.columns.tolist()}")

    # Remap 1→0 (Good), 2→1 (Bad) only if original 1/2 encoding
    if df_german[target_col].isin([1, 2]).all():
        df_german[target_col] = (df_german[target_col] == 2).astype(int)

    print(f"Target distribution:\n{df_german[target_col].value_counts()}\n")

    # OHE
    X_g = df_german.drop(target_col, axis=1).copy()
    y_g = df_german[target_col]

    cat_cols_present = [c for c in CATEGORICAL_COLS if c in X_g.columns]
    if len(cat_cols_present) == 0:
        cat_cols_present = X_g.select_dtypes(include="object").columns.tolist()
        print(f"⚠️  Fallback: dtype-detected categorical cols ({len(cat_cols_present)})")
    else:
        for col in cat_cols_present:
            X_g[col] = X_g[col].astype(str)

    X_g_encoded = pd.get_dummies(X_g, columns=cat_cols_present, drop_first=True).astype(float)
    print(f"Shape before OHE : {X_g.shape}  →  after OHE : {X_g_encoded.shape}")

    if X_g_encoded.shape[1] == 0:
        raise ValueError(
            "OHE produced 0 columns. "
            "Check that categorical column names match your CSV headers."
        )

    # Train-test split on German Credit
    Xg_train, Xg_test, yg_train, yg_test = train_test_split(
        X_g_encoded, y_g, test_size=0.2, random_state=42, stratify=y_g
    )

    # Scale
    scaler_g     = StandardScaler()
    Xg_train_sc  = scaler_g.fit_transform(Xg_train)
    Xg_test_sc   = scaler_g.transform(Xg_test)
    yg_train_arr = yg_train.values
    yg_test_arr  = yg_test.values

    neg_g = (yg_train_arr == 0).sum()
    pos_g = (yg_train_arr == 1).sum()

    print(f"Train size       : {Xg_train_sc.shape}")
    print(f"Test size        : {Xg_test_sc.shape}")
    print(f"Imbalance ratio  : {neg_g / pos_g:.1f}:1\n")

except FileNotFoundError:
    print(f"⚠️  File not found: '{GERMAN_PATH}'")
    print("   Place german_data.csv inside the data/ folder and rerun.")
    raise SystemExit(1)

except ValueError as e:
    print(f"⚠️  Data error: {e}")
    raise SystemExit(1)


# ══════════════════════════════════════════════
# STEP 9: LOAD PSI RESULTS (File 3 output)
# ══════════════════════════════════════════════
print("=" * 60)
print("STEP 9: Loading PSI Results from File 3")
print("=" * 60)

try:
    with open("drift_data.pkl", "rb") as f:
        drift_data = pickle.load(f)

    psi_results      = drift_data["psi_results"]
    any_drift        = drift_data["any_drift"]
    drifted_features = drift_data["drifted_features"]

    print(f"\n  {'Feature':<22} {'PSI':>8}  Status")
    print("  " + "─" * 42)
    for r in psi_results:
        print(f"  {r['feature']:<22} {r['psi']:>8.4f}  {r['status']}")

    print(f"\n  PSI Guide: < 0.1 = Stable | 0.1–0.2 = Warning | > 0.2 = Drift")
    print(f"\n  Drifted features (PSI > {PSI_DRIFT_THRESHOLD}) : {drifted_features}")
    print(f"  Overall drift flag : "
          f"{'⚠️  YES — SVM retrain will trigger' if any_drift else '✅  NO — no retrain needed'}\n")

except FileNotFoundError:
    print("⚠️  drift_data.pkl not found. Run 3_drift_detection.py first.")
    raise SystemExit(1)


# ══════════════════════════════════════════════
# STEP 10: LOAD TRAINED MODELS (File 2 output)
# ══════════════════════════════════════════════
print("=" * 60)
print("STEP 10: Loading Trained Models from File 2")
print("=" * 60)

try:
    with open("trained_models.pkl", "rb") as f:
        model_data = pickle.load(f)

    # Extract XGBoost and SVM from the results list
    results_list = model_data["results"]

    xgb_model = next((r["model"] for r in results_list if "XGBoost" in r["name"]), None)
    svm_model = next((r["model"] for r in results_list if "SVM"     in r["name"]), None)

    if xgb_model is None:
        raise KeyError("XGBoost model not found in trained_models.pkl.")
    if svm_model is None:
        raise KeyError("SVM model not found in trained_models.pkl.")

    print(f"✅ Loaded XGBoost : {xgb_model.__class__.__name__}")
    print(f"✅ Loaded SVM     : {svm_model.__class__.__name__}\n")

except FileNotFoundError:
    print("⚠️  trained_models.pkl not found. Run 2_train_models.py first.")
    raise SystemExit(1)

except KeyError as e:
    print(f"⚠️  Model loading error: {e}")
    raise SystemExit(1)


# ══════════════════════════════════════════════
# STEP 11: BASELINE — XGBoost vs SVM on German Credit
# ══════════════════════════════════════════════
print("=" * 60)
print("STEP 11: Baseline Evaluation — XGBoost vs SVM on German Credit")
print("=" * 60)

imbalance_g = neg_g / pos_g

print("\n--- XGBoost (retrained on German Credit) ---")
xgb_g = XGBClassifier(
    n_estimators     = 300,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = imbalance_g,
    use_label_encoder= False,
    eval_metric      = "auc",
    random_state     = 42,
    n_jobs           = -1,
)
xgb_g_res = evaluate_fitted_model(
    "XGBoost (German Credit)", xgb_g,
    Xg_train_sc, yg_train_arr,
    Xg_test_sc,  yg_test_arr
)

print("\n--- SVM (retrained on German Credit) ---")
svm_g = SVC(
    kernel       = "rbf",
    C            = 1.0,
    class_weight = "balanced",
    probability  = True,
    random_state = 42,
)
svm_g_res = evaluate_fitted_model(
    "SVM (German Credit)", svm_g,
    Xg_train_sc, yg_train_arr,
    Xg_test_sc,  yg_test_arr,
    sample_size  = 20000          # SVM subsampling for speed
)

baseline_results = [xgb_g_res, svm_g_res]
print_comparison_table(baseline_results, title="Baseline: XGBoost vs SVM — German Credit")


# ══════════════════════════════════════════════
# STEP 12: DRIFT CHECK → CONDITIONAL SVM RETRAIN
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 12: Drift Check → Conditional SVM Retrain")
print("=" * 60)

verdict       = None
ratio         = None
final_results = []

if not any_drift:
    # ─────────────────────────────────────────
    # NO DRIFT — use baseline results as final
    # ─────────────────────────────────────────
    print("\n✅ No drift detected (all PSI ≤ 0.2)")
    print("   SVM retraining is NOT required.")
    print("   Both models remain valid. Using baseline results as final.\n")

    final_results = baseline_results
    verdict       = "STABLE (no drift)"
    ratio         = 1.0

else:
    # ─────────────────────────────────────────
    # DRIFT DETECTED — retrain SVM on German Credit
    # ─────────────────────────────────────────
    print(f"\n⚠️  Drift detected on features : {drifted_features}")
    print(f"   PSI > {PSI_DRIFT_THRESHOLD} — retraining SVM on German Credit (drifted) data.\n")

    print("=" * 60)
    print("STEP 13: Retraining SVM on Drifted German Credit Data")
    print("=" * 60)

    n_svm    = min(20000, len(Xg_train_sc))
    svm_idx  = np.random.RandomState(42).choice(len(Xg_train_sc), size=n_svm, replace=False)

    print(f"\n  Retraining SVM on {n_svm} samples...")
    svm_retrained = SVC(
        kernel       = "rbf",
        C            = 1.0,
        class_weight = "balanced",
        probability  = True,
        random_state = 42,
    )
    svm_retrained.fit(Xg_train_sc[svm_idx], yg_train_arr[svm_idx])
    print("  ✅ SVM retrained successfully.\n")

    # Evaluate retrained SVM on test set
    print("--- Retrained SVM Evaluation ---")
    svm_retrained_res = evaluate_fitted_model(
        "SVM (Retrained on Drift)", svm_retrained,
        Xg_train_sc, yg_train_arr,    # needed for signature; already fitted above
        Xg_test_sc,  yg_test_arr,
    )
    # Override model with already-fitted one (avoid double fit)
    y_proba_svm = svm_retrained.predict_proba(Xg_test_sc)[:, 1]
    thresh_svm  = find_best_threshold(yg_test_arr, y_proba_svm)
    y_pred_svm  = (y_proba_svm >= thresh_svm).astype(int)

    auc_svm      = roc_auc_score(yg_test_arr, y_proba_svm)
    f1_svm       = f1_score(yg_test_arr, y_pred_svm, zero_division=0)
    f1_macro_svm = f1_score(yg_test_arr, y_pred_svm, average="macro", zero_division=0)
    acc_svm      = (y_pred_svm == yg_test_arr).mean()

    svm_retrained_res = {
        "name":     "SVM (Retrained on Drift)",
        "model":    svm_retrained,
        "auc":      auc_svm,
        "f1":       f1_svm,
        "f1_macro": f1_macro_svm,
        "accuracy": acc_svm,
        "y_pred":   y_pred_svm,
        "y_proba":  y_proba_svm,
    }

    print(f"\n{'─' * 50}")
    print(f"  Model     : SVM (Retrained on Drift)  (threshold={thresh_svm})")
    print(f"{'─' * 50}")
    print(f"  AUC       : {auc_svm:.4f}")
    print(f"  Accuracy  : {acc_svm:.4f}  ({acc_svm*100:.1f}%)")
    print(f"  F1        : {f1_svm:.4f}  (class=1, bad credit)")
    print(f"  F1 Macro  : {f1_macro_svm:.4f}")
    print(f"\n{classification_report(yg_test_arr, y_pred_svm, target_names=['Good Credit','Bad Credit'], zero_division=0)}")

    # XGBoost predictions on same test set (already fitted above)
    y_proba_xgb = xgb_g.predict_proba(Xg_test_sc)[:, 1]
    thresh_xgb  = find_best_threshold(yg_test_arr, y_proba_xgb)
    y_pred_xgb  = (y_proba_xgb >= thresh_xgb).astype(int)

    xgb_g_res["y_pred"] = y_pred_xgb   # update with tuned threshold preds

    # ── Prediction Comparison & Validity ─────
    print("=" * 60)
    print("STEP 14: Prediction Comparison → Model Validity Decision")
    print("=" * 60)

    ratio, verdict = compare_predictions(xgb_g_res["y_pred"], svm_retrained_res["y_pred"])

    final_results = [xgb_g_res, svm_retrained_res]

    # ── Validity Summary ─────────────────────
    print("=" * 60)
    print("STEP 15: Model Validity Summary")
    print("=" * 60)
    print(f"""
  DECISION LOGIC:
  ┌───────────────────────────────────────────────────────┐
  │  PSI > 0.2            → Data drift confirmed          │
  │  SVM retrained        → Adapts to new distribution    │
  │                                                       │
  │  Compare XGBoost vs SVM predictions on German Credit: │
  │                                                       │
  │  Agreement ≥ {AGREE_THRESHOLD:.0%}    → ✅ OLD MODEL STABLE           │
  │                  Both models agree → XGBoost stays   │
  │                                                       │
  │  Agreement <  {AGREE_THRESHOLD:.0%}    → ❌ OLD MODEL INVALID          │
  │                  Models diverge → XGBoost must retire │
  │                  Deploy retrained SVM to production   │
  └───────────────────────────────────────────────────────┘

  RESULT:
    Drifted features : {drifted_features}
    Agreement ratio  : {ratio:.1%}
    Verdict          : {verdict}
    Action           : {"✅ Keep XGBoost in production." if verdict == "STABLE" else "❌ Retire XGBoost. Deploy retrained SVM."}
""")


# ══════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ══════════════════════════════════════════════
print_comparison_table(
    final_results,
    title="FINAL: XGBoost vs SVM — German Credit" + (" (Post-Drift Retrain)" if any_drift else "")
)

# ── Save final results ────────────────────────
with open("german_credit_results.pkl", "wb") as f:
    pickle.dump({
        "final_results":    final_results,
        "baseline_results": baseline_results,
        "verdict":          verdict,
        "agree_ratio":      ratio,
        "any_drift":        any_drift,
    }, f)

print("\n✅ Saved german_credit_results.pkl")
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
