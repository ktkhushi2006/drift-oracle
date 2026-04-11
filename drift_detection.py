"""
FILE 3: PSI Drift Detection
Simulates an economic shock and measures feature distribution shift
using Population Stability Index (PSI).

PSI < 0.1   → Stable   (no significant shift)
PSI 0.1-0.2 → Warning  (moderate shift)
PSI > 0.2   → Drift    (significant shift — triggers SVM retrain in File 4)

Run AFTER: 1_data_preprocess.py
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Load raw (pre-OHE) dataframe and feature list from Step 1 ──
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

df           = data["df"]
NUM_FEATURES = data["NUM_FEATURES"]
# ───────────────────────────────────────────────────────────────


def compute_psi(expected, actual, bins=10):
    """
    Population Stability Index between two distributions.
    Uses percentile-based binning from the expected (training) distribution.

    PSI < 0.1   → Stable
    PSI 0.1-0.2 → Warning
    PSI > 0.2   → Drift → triggers SVM retrain in File 4
    """
    breakpoints   = np.linspace(0, 100, bins + 1)
    bin_edges     = np.percentile(expected, breakpoints)

    expected_perc = np.histogram(expected, bins=bin_edges)[0] / len(expected)
    actual_perc   = np.histogram(actual,   bins=bin_edges)[0] / len(actual)

    # Replace zeros to avoid log(0)
    expected_perc = np.where(expected_perc == 0, 1e-4, expected_perc)
    actual_perc   = np.where(actual_perc   == 0, 1e-4, actual_perc)

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(psi)


def psi_status(psi):
    if psi > 0.2:  return "🔴 DRIFT"
    if psi > 0.1:  return "🟡 WARN"
    return "🟢 STABLE"


print("=" * 60)
print("STEP 7: PSI Drift Detection (Simulated Economic Shock)")
print("=" * 60)

# ── Simulate Economic Shock ───────────────────
drift_df = df[NUM_FEATURES].copy()
drift_df["AMT_INCOME_TOTAL"] *= 0.7   # income drops 30%
drift_df["AMT_CREDIT"]       *= 1.3   # loan amounts increase 30%
drift_df["AMT_ANNUITY"]      *= 1.2   # EMI increases 20%
# DAYS_EMPLOYED and DAYS_BIRTH unchanged → should show STABLE

print("\nSimulated economic shock applied:")
print("  AMT_INCOME_TOTAL × 0.7  (income drops 30%)")
print("  AMT_CREDIT       × 1.3  (loans increase 30%)")
print("  AMT_ANNUITY      × 1.2  (EMI increases 20%)")
print("  DAYS_EMPLOYED    —      (unchanged)")
print("  DAYS_BIRTH       —      (unchanged)\n")

# ── Compute PSI Per Feature ───────────────────
print(f"  {'Feature':<22} {'PSI':>8}  Status")
print("  " + "─" * 42)

psi_results = []
any_drift   = False

for col in NUM_FEATURES:
    psi    = compute_psi(df[col].values, drift_df[col].values)
    status = psi_status(psi)
    print(f"  {col:<22} {psi:>8.4f}  {status}")
    psi_results.append({
        "feature": col,
        "psi":     round(psi, 4),
        "status":  status,
        "drift":   psi > 0.2,
    })
    if psi > 0.2:
        any_drift = True

print("\n  PSI Guide: < 0.1 = Stable | 0.1–0.2 = Warning | > 0.2 = Drift")

# ── Summary ───────────────────────────────────
drifted_features = [r["feature"] for r in psi_results if r["drift"]]

print(f"\n{'─' * 60}")
print(f"  Total features checked  : {len(NUM_FEATURES)}")
print(f"  Drifted features (>0.2) : {len(drifted_features)}")
if drifted_features:
    print(f"  Drifted                 : {drifted_features}")
print(f"  Overall drift flag      : "
      f"{'⚠️  YES — SVM retrain will trigger in File 4' if any_drift else '✅  NO — models remain stable'}")
print(f"{'─' * 60}\n")

# ── Save ──────────────────────────────────────
psi_df = pd.DataFrame(psi_results)
psi_df.to_csv("psi_metrics.csv", index=False)

with open("drift_data.pkl", "wb") as f:
    pickle.dump({
        "psi_results":      psi_results,
        "psi_df":           psi_df,
        "drift_df":         drift_df,            # shifted feature values (new data)
        "original_df":      df[NUM_FEATURES],    # original feature values
        "any_drift":        any_drift,
        "drifted_features": drifted_features,
    }, f)

print("✅ Saved psi_metrics.csv")
print("✅ Saved drift_data.pkl — run 4_german_credit.py next")
print("\n" + "=" * 60)
print("DRIFT DETECTION COMPLETE")
print("=" * 60)