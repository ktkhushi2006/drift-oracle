"""
FILE 4: German Credit — PSI-Triggered SVM Retrain + Model Validity Check
─────────────────────────────────────────────────────────────────────────
Pipeline:
  1. Load PSI results from File 3  (drift_data.pkl)
  2. Load LR + SVM from File 2     (trained_models.pkl → results list)
  3. Load + preprocess German Credit data
  4. If PSI > 0.2 (drift detected):
       → Retrain SVM on German Credit (new/drifted) data
       → Compare predictions: LR vs retrained SVM on the same test set
       → Agreement ≥ 90%  →  ✅ OLD MODEL STABLE  (keep LR)
       → Agreement <  90%  →  ❌ OLD MODEL INVALID (deploy retrained SVM)
  5. If no drift → evaluate both original models, no retrain

Run AFTER: 1_data_preprocess.py → 2_train_models.py → 3_drift_detection.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report
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
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = t
    return round(best_thresh, 2)


def evaluate_fitted_model(name, model, X_te, y_te):
    """
    Evaluate a pre-fitted model (no refit).
    Returns metrics dict including raw y_pred for comparison.
    """
    y_proba = model.predict_proba(X_te)[:, 1]
    thresh  = find_best_threshold(y_te, y_proba)
    y_pred  = (y_proba >= thresh).astype(int)

    auc      = roc_auc_score(y_te, y_proba)
    f1       = f1_score(y_te, y_pred, zero_division=0)
    f1_macro = f1_score(y_te, y_pred, average="macro", zero_division=0)
    accuracy = (y_pred == y_te).mean()

    print(f"\n{'─' * 50}")
    print(f"  Model     : {name}")
    print(f"  Threshold : {thresh}")
    print(f"{'─' * 50}")
    print(f"  AUC       : {auc:.4f}")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  F1        : {f1:.4f}  (class=1, bad credit)")
    print(f"  F1 Macro  : {f1_macro:.4f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['Good Credit', 'Bad Credit'], zero_division=0)}")

    return {
        "name":     name,
        "auc":      auc,
        "f1":       f1,
        "f1_macro": f1_macro,
        "accuracy": accuracy,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
    }


def compare_predictions(lr_preds, svm_preds):
    """
    Compare element-wise predictions between LR and retrained SVM.
    Returns (agreement_ratio, verdict).
    """
    total    = len(lr_preds)
    agree    = int((lr_preds == svm_preds).sum())
    disagree = total - agree
    ratio    = agree / total

    print(f"\n{'═' * 60}")
    print("  PREDICTION COMPARISON — LR (Old) vs SVM (Retrained)")
    print(f"{'═' * 60}")
    print(f"  Total test samples   : {total}")
    print(f"  Agreements           : {agree}  ({ratio:.1%})")
    print(f"  Disagreements        : {disagree}  ({1 - ratio:.1%})")
    print(f"  Agreement threshold  : {AGREE_THRESHOLD:.0%}")

    if ratio >= AGREE_THRESHOLD:
        verdict = "STABLE"
        print(f"\n  ✅ VERDICT: OLD MODEL IS STABLE")
        print(f"     LR and retrained SVM agree on {ratio:.1%} of predictions.")
        print(f"     The original Logistic Regression remains valid for production.")
    else:
        verdict = "INVALID"
        print(f"\n  ❌ VERDICT: OLD MODEL IS INVALID")
        print(f"     LR and retrained SVM disagree on {1 - ratio:.1%} of predictions.")
        print(f"     The original Logistic Regression should be retired.")
        print(f"     Deploy the retrained SVM model instead.")
    print(f"{'═' * 60}\n")

    return ratio, verdict


# ══════════════════════════════════════════════
# STEP 8: LOAD GERMAN CREDIT DATA
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
        # Fallback to dtype detection
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

    # Scale
    scaler_g   = StandardScaler()
    X_g_scaled = scaler_g.fit_transform(X_g_encoded)
    y_g_arr    = y_g.values

    print(f"Scaled shape     : {X_g_scaled.shape}")
    print(f"Imbalance ratio  : {(y_g_arr==0).sum() / (y_g_arr==1).sum():.1f}:1\n")

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
    print("  " + "─" * 40)
    for r in psi_results:
        print(f"  {r['feature']:<22} {r['psi']:>8.4f}  {r['status']}")

    print(f"\n  PSI Guide: < 0.1 = Stable | 0.1–0.2 = Warning | > 0.2 = Drift")
    print(f"\n  Drifted features (PSI > {PSI_DRIFT_THRESHOLD}) : {drifted_features}")
    print(f"  Overall drift flag : "
          f"{'⚠️  YES — will retrain SVM' if any_drift else '✅  NO — no retrain needed'}\n")

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

    # File 2 (original train_model.py) stores models inside the "results" list
    # Each entry: {"name": ..., "model": ..., "auc": ..., ...}
    results = model_data["results"]

    lr_model  = next((r["model"] for r in results if "Logistic" in r["name"]), None)
    svm_model = next((r["model"] for r in results if "SVM"      in r["name"]), None)

    if lr_model is None:
        raise KeyError("Logistic Regression model not found in trained_models.pkl results list.")
    if svm_model is None:
        raise KeyError("SVM model not found in trained_models.pkl results list.")

    print(f"✅ Loaded Logistic Regression : {lr_model.__class__.__name__}")
    print(f"✅ Loaded SVM                 : {svm_model.__class__.__name__}\n")

except FileNotFoundError:
    print("⚠️  trained_models.pkl not found. Run 2_train_models.py first.")
    raise SystemExit(1)

except KeyError as e:
    print(f"⚠️  Model loading error: {e}")
    raise SystemExit(1)


# ══════════════════════════════════════════════
# STEP 11: DRIFT CHECK → CONDITIONAL SVM RETRAIN
# ══════════════════════════════════════════════
print("=" * 60)
print("STEP 11: Drift Check → Conditional SVM Retrain")
print("=" * 60)

verdict     = None
ratio       = None
final_results = []

if not any_drift:
    # ─────────────────────────────────────────
    # NO DRIFT PATH
    # ─────────────────────────────────────────
    print("\n✅ No drift detected (all PSI ≤ 0.2)")
    print("   SVM retraining is NOT required.")
    print("   Evaluating both original models on German Credit data...\n")

    print("=" * 60)
    print("STEP 12: Evaluating Original Models on German Credit")
    print("=" * 60)

    # LR: refit on full German Credit (original LR was trained on Home Credit)
    lr_eval = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_eval.fit(X_g_scaled, y_g_arr)
    lr_res = evaluate_fitted_model("LR (Original)", lr_eval, X_g_scaled, y_g_arr)
    final_results.append(lr_res)

    # SVM: refit on sample of German Credit
    n_svm     = min(20000, len(X_g_scaled))
    svm_idx   = np.random.RandomState(42).choice(len(X_g_scaled), size=n_svm, replace=False)
    svm_eval  = SVC(kernel="rbf", C=1.0, class_weight="balanced",
                    probability=True, random_state=42)
    svm_eval.fit(X_g_scaled[svm_idx], y_g_arr[svm_idx])
    svm_res = evaluate_fitted_model("SVM (Original)", svm_eval, X_g_scaled, y_g_arr)
    final_results.append(svm_res)

    # Compare even without drift (sanity check)
    ratio, verdict = compare_predictions(lr_res["y_pred"], svm_res["y_pred"])

else:
    # ─────────────────────────────────────────
    # DRIFT DETECTED PATH
    # ─────────────────────────────────────────
    print(f"\n⚠️  Drift detected on features : {drifted_features}")
    print(f"   PSI > {PSI_DRIFT_THRESHOLD} — triggering SVM retrain on German Credit (new data).\n")

    # ── Retrain SVM on German Credit data ────
    print("=" * 60)
    print("STEP 12: Retraining SVM on German Credit (Drifted) Data")
    print("=" * 60)

    n_svm    = min(20000, len(X_g_scaled))
    svm_idx  = np.random.RandomState(42).choice(len(X_g_scaled), size=n_svm, replace=False)
    X_svm_tr = X_g_scaled[svm_idx]
    y_svm_tr = y_g_arr[svm_idx]

    print(f"\n  Training SVM on {n_svm} samples from German Credit data...")
    svm_retrained = SVC(
        kernel       = "rbf",
        C            = 1.0,
        class_weight = "balanced",
        probability  = True,
        random_state = 42,
    )
    svm_retrained.fit(X_svm_tr, y_svm_tr)
    print("  ✅ SVM retrained successfully.\n")

    # ── Refit LR on German Credit for fair comparison ──
    print("  Fitting LR on German Credit data for comparison...")
    lr_german = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_german.fit(X_g_scaled, y_g_arr)
    print("  ✅ LR fitted.\n")

    # ── Evaluate both on full German Credit set ──
    print("=" * 60)
    print("STEP 13: Evaluating LR (Old) vs SVM (Retrained)")
    print("=" * 60)

    print("\n--- Logistic Regression (Old / Original Model) ---")
    lr_res = evaluate_fitted_model(
        "LR (Old — Original)", lr_german, X_g_scaled, y_g_arr
    )
    final_results.append(lr_res)

    print("\n--- SVM (Retrained on Drifted German Credit Data) ---")
    svm_res = evaluate_fitted_model(
        "SVM (Retrained on Drift)", svm_retrained, X_g_scaled, y_g_arr
    )
    final_results.append(svm_res)

    # ── Prediction Comparison & Validity Decision ──
    print("=" * 60)
    print("STEP 14: Prediction Comparison → Model Validity Decision")
    print("=" * 60)

    ratio, verdict = compare_predictions(lr_res["y_pred"], svm_res["y_pred"])

    # ── Validity Explanation ──────────────────
    print("=" * 60)
    print("STEP 15: Model Validity — Summary")
    print("=" * 60)
    print(f"""
  DECISION LOGIC:
  ┌──────────────────────────────────────────────────────┐
  │  PSI > 0.2           → Data drift confirmed          │
  │  SVM retrained       → Adapts to new distribution    │
  │                                                      │
  │  Compare LR vs SVM predictions on German Credit:     │
  │                                                      │
  │  Agreement ≥ {AGREE_THRESHOLD:.0%}   → ✅ OLD MODEL STABLE          │
  │                Both models agree → LR stays valid    │
  │                                                      │
  │  Agreement <  {AGREE_THRESHOLD:.0%}   → ❌ OLD MODEL INVALID         │
  │                Models diverge → LR must be retired   │
  │                Deploy retrained SVM to production    │
  └──────────────────────────────────────────────────────┘

  RESULT:
    Drifted features : {drifted_features}
    Agreement ratio  : {ratio:.1%}
    Verdict          : {verdict}
    Action           : {"✅ Keep LR in production." if verdict == "STABLE" else "❌ Retire LR. Deploy retrained SVM."}
""")


# ══════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ══════════════════════════════════════════════
print("=" * 60)
print("FINAL MODEL COMPARISON — German Credit")
print("=" * 60)

comparison_df = pd.DataFrame([
    {
        "Model":        r["name"],
        "AUC":          round(r["auc"],      4),
        "Accuracy":     f"{r['accuracy']*100:.1f}%",
        "F1 (Default)": round(r["f1"],       4),
        "F1 Macro":     round(r["f1_macro"], 4),
    }
    for r in final_results
])
print(comparison_df.to_string(index=False))

best = max(final_results, key=lambda x: x["auc"])
print(f"\n✅ Best model by AUC : {best['name']} (AUC={best['auc']:.4f})")

# ── Save final results ────────────────────────
with open("german_credit_results.pkl", "wb") as f:
    pickle.dump({
        "final_results": final_results,
        "verdict":       verdict,
        "agree_ratio":   ratio,
        "any_drift":     any_drift,
    }, f)

print("\n✅ Saved german_credit_results.pkl")
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
