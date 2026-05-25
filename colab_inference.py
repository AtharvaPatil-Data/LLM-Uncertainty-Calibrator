# =============================================================================
# LLM-Uncertainty-Calibrator: Uncertainty Calibration & Conformal Prediction
# =============================================================================
# Run this on Google Colab with T4 GPU (Runtime > Change runtime type > T4 GPU)
#
# Install dependencies first:
#   !pip install transformers torch datasets scikit-learn -q
#
# This script:
#   1. Loads a financial text classifier (DistilBERT)
#   2. Evaluates on financial risk classification task
#   3. Extracts raw probabilities (uncalibrated)
#   4. Applies Temperature Scaling, Platt Scaling, Conformal Prediction
#   5. Measures Expected Calibration Error (ECE) before/after
#   6. Saves results.json for Streamlit dashboard
# =============================================================================

import json
import torch
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Device setup ──────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"  # We'll use this as base classifier
NUM_CLASSES = 3  # low_risk, medium_risk, high_risk
OUTPUT_FILE = "results.json"
CONF_ALPHA = 0.1  # 90% coverage for conformal prediction

# ── Financial Text Dataset ───────────────────────────────────────────────────
# Synthetic financial statements with risk labels
FINANCIAL_TEXTS = [
    # Low Risk
    ("Company maintains strong cash reserves and consistent dividend payments.", "low_risk"),
    ("Revenue growth of 5% year-over-year with stable profit margins.", "low_risk"),
    ("Investment-grade credit rating with low debt-to-equity ratio.", "low_risk"),
    ("Diversified revenue streams across multiple geographic markets.", "low_risk"),
    ("Conservative capital allocation strategy with minimal leverage.", "low_risk"),
    ("Strong balance sheet with ample liquidity for operations.", "low_risk"),
    ("Steady customer base with high retention rates.", "low_risk"),
    ("Regulated utility with predictable cash flows.", "low_risk"),
    ("Blue-chip stock with decades of profitability.", "low_risk"),
    ("AAA-rated government bonds with sovereign backing.", "low_risk"),
    ("Established brand with dominant market position.", "low_risk"),
    ("Consistent earnings growth over the past 10 years.", "low_risk"),
    ("Low volatility in share price over extended period.", "low_risk"),
    ("Conservative accounting practices with transparent reporting.", "low_risk"),
    ("Minimal exposure to cyclical industries.", "low_risk"),
    
    # Medium Risk
    ("Moderate debt levels with coverage ratio of 2.5x.", "medium_risk"),
    ("Revenue growth of 15% but declining profit margins.", "medium_risk"),
    ("Expansion into new markets showing mixed results.", "medium_risk"),
    ("Recent management changes impacting strategic direction.", "medium_risk"),
    ("Industry facing regulatory headwinds but company adapting.", "medium_risk"),
    ("Seasonal business with concentrated revenue periods.", "medium_risk"),
    ("Moderate competition in core markets affecting pricing power.", "medium_risk"),
    ("Technology upgrade cycle requiring significant capital expenditure.", "medium_risk"),
    ("Emerging market exposure with currency fluctuation risk.", "medium_risk"),
    ("Product recall impacting short-term profitability.", "medium_risk"),
    ("Acquisition integration ongoing with uncertain outcomes.", "medium_risk"),
    ("Commodity price exposure partially hedged.", "medium_risk"),
    ("Customer concentration with top 3 clients representing 40% of revenue.", "medium_risk"),
    ("Patent expiration approaching for key product line.", "medium_risk"),
    ("Mid-cap company with moderate volatility.", "medium_risk"),
    
    # High Risk
    ("Negative cash flow for three consecutive quarters.", "high_risk"),
    ("Debt-to-equity ratio exceeds 3.0 with covenant violations.", "high_risk"),
    ("Revenue decline of 25% year-over-year amid market share loss.", "high_risk"),
    ("Regulatory investigation into accounting practices.", "high_risk"),
    ("Credit rating downgraded to junk status.", "high_risk"),
    ("Going concern opinion issued by auditors.", "high_risk"),
    ("Major customer bankruptcy representing 30% of revenue.", "high_risk"),
    ("Pending litigation with material adverse outcome likely.", "high_risk"),
    ("Technology disruption rendering core product obsolete.", "high_risk"),
    ("Management turnover with three CFOs in two years.", "high_risk"),
    ("Failed product launch consuming significant capital.", "high_risk"),
    ("Liquidity crisis with inability to meet short-term obligations.", "high_risk"),
    ("Shareholder activist campaign targeting board composition.", "high_risk"),
    ("Cybersecurity breach exposing customer financial data.", "high_risk"),
    ("Speculative biotech with no approved products.", "high_risk"),
]

# Expand dataset with variations
texts, labels = zip(*FINANCIAL_TEXTS)
texts = list(texts) * 4  # Replicate 4x for more data
labels = list(labels) * 4

LABEL_MAP = {"low_risk": 0, "medium_risk": 1, "high_risk": 2}
label_ids = [LABEL_MAP[l] for l in labels]

print(f"[INFO] Dataset: {len(texts)} samples, {NUM_CLASSES} classes")

# ── Model Setup ───────────────────────────────────────────────────────────────

class FinancialRiskClassifier:
    """Wrapper for text classification with logit extraction."""
    
    def __init__(self, model_name: str, num_labels: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(DEVICE)
        self.model.eval()
    
    def predict_with_logits(self, texts: list) -> tuple:
        """Returns logits, probabilities, and predictions."""
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()
        
        # Convert logits to probabilities via softmax
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        
        return logits, probs, preds

# ── Calibration Methods ───────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, preds: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    Measures the difference between confidence and accuracy.
    """
    confidences = np.max(probs, axis=1)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if in_bin.sum() > 0:
            bin_acc = (preds[in_bin] == labels[in_bin]).mean()
            bin_conf = confidences[in_bin].mean()
            bin_size = in_bin.sum()
            ece += (bin_size / len(labels)) * abs(bin_acc - bin_conf)
    
    return ece


def temperature_scaling(logits_val: np.ndarray, labels_val: np.ndarray) -> float:
    """
    Find optimal temperature T by minimizing NLL on validation set.
    Calibrated probs = softmax(logits / T)
    """
    from scipy.optimize import minimize
    
    def nll(T):
        scaled_logits = logits_val / T
        scaled_probs = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)
        # Negative log-likelihood
        return -np.log(scaled_probs[np.arange(len(labels_val)), labels_val] + 1e-12).mean()
    
    result = minimize(nll, x0=1.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')
    optimal_T = result.x[0]
    print(f"[INFO] Temperature Scaling: Optimal T = {optimal_T:.4f}")
    return optimal_T


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply learned temperature to logits."""
    scaled_logits = logits / temperature
    probs = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)
    return probs


def platt_scaling(logits_val: np.ndarray, labels_val: np.ndarray, logits_test: np.ndarray) -> np.ndarray:
    """
    Platt Scaling: Train logistic regression on validation logits.
    """
    # Use max logit as feature (or all logits for multiclass)
    clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    clf.fit(logits_val, labels_val)
    
    calibrated_probs = clf.predict_proba(logits_test)
    print(f"[INFO] Platt Scaling: Logistic regression fitted")
    return calibrated_probs


def conformal_prediction(probs_cal: np.ndarray, labels_cal: np.ndarray, probs_test: np.ndarray, alpha: float):
    """
    Conformal Prediction: Compute prediction sets with (1-alpha) coverage.
    Returns list of prediction sets (one per test sample).
    """
    # Compute non-conformity scores on calibration set
    scores_cal = 1 - probs_cal[np.arange(len(labels_cal)), labels_cal]
    
    # Quantile for (1-alpha) coverage
    n_cal = len(scores_cal)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    threshold = np.quantile(scores_cal, q_level)
    
    # Build prediction sets for test samples
    prediction_sets = []
    for prob in probs_test:
        # Include class i if 1 - prob[i] <= threshold
        pred_set = [i for i in range(len(prob)) if 1 - prob[i] <= threshold]
        if len(pred_set) == 0:  # Edge case: include most confident
            pred_set = [np.argmax(prob)]
        prediction_sets.append(pred_set)
    
    print(f"[INFO] Conformal Prediction: Coverage target = {1-alpha:.1%}, Threshold = {threshold:.4f}")
    return prediction_sets, threshold


# ── Main Experiment ───────────────────────────────────────────────────────────

def run_experiment():
    # Split data: train (not used for our pretrained model), val (calibration), test
    indices = np.arange(len(texts))
    idx_train, idx_temp = train_test_split(indices, test_size=0.5, random_state=42, stratify=label_ids)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42, stratify=[label_ids[i] for i in idx_temp])
    
    texts_val = [texts[i] for i in idx_val]
    labels_val = np.array([label_ids[i] for i in idx_val])
    
    texts_test = [texts[i] for i in idx_test]
    labels_test = np.array([label_ids[i] for i in idx_test])
    
    print(f"[INFO] Val set: {len(texts_val)} | Test set: {len(texts_test)}")
    
    # Load model
    print("[INFO] Loading model...")
    classifier = FinancialRiskClassifier(MODEL_NAME, NUM_CLASSES)
    
    # Get predictions on val and test
    print("[INFO] Running inference on validation set...")
    logits_val, probs_val, preds_val = classifier.predict_with_logits(texts_val)
    
    print("[INFO] Running inference on test set...")
    logits_test, probs_test, preds_test = classifier.predict_with_logits(texts_test)
    
    # Baseline accuracy and ECE
    acc_test = accuracy_score(labels_test, preds_test)
    ece_uncalibrated = compute_ece(probs_test, preds_test, labels_test)
    
    print(f"\n[BASELINE] Test Accuracy: {acc_test:.4f} ({acc_test*100:.2f}%)")
    print(f"[BASELINE] Uncalibrated ECE: {ece_uncalibrated:.4f}")
    
    # ── Temperature Scaling ────────────────────────────────────────────────────
    print("\n[CALIBRATION] Applying Temperature Scaling...")
    optimal_temp = temperature_scaling(logits_val, labels_val)
    probs_temp = apply_temperature_scaling(logits_test, optimal_temp)
    preds_temp = np.argmax(probs_temp, axis=1)
    ece_temp = compute_ece(probs_temp, preds_temp, labels_test)
    
    print(f"[TEMP SCALING] ECE after calibration: {ece_temp:.4f}")
    print(f"[TEMP SCALING] ECE reduction: {ece_uncalibrated - ece_temp:.4f}")
    
    # ── Platt Scaling ──────────────────────────────────────────────────────────
    print("\n[CALIBRATION] Applying Platt Scaling...")
    probs_platt = platt_scaling(logits_val, labels_val, logits_test)
    preds_platt = np.argmax(probs_platt, axis=1)
    ece_platt = compute_ece(probs_platt, preds_platt, labels_test)
    
    print(f"[PLATT SCALING] ECE after calibration: {ece_platt:.4f}")
    print(f"[PLATT SCALING] ECE reduction: {ece_uncalibrated - ece_platt:.4f}")
    
    # ── Conformal Prediction ───────────────────────────────────────────────────
    print("\n[CALIBRATION] Applying Conformal Prediction...")
    pred_sets, cp_threshold = conformal_prediction(probs_val, labels_val, probs_test, CONF_ALPHA)
    
    # Compute coverage (% of test samples where true label is in prediction set)
    coverage = np.mean([labels_test[i] in pred_sets[i] for i in range(len(labels_test))])
    avg_set_size = np.mean([len(s) for s in pred_sets])
    
    print(f"[CONFORMAL] Empirical coverage: {coverage:.4f} (target: {1-CONF_ALPHA:.2f})")
    print(f"[CONFORMAL] Average prediction set size: {avg_set_size:.2f}")
    
    # ── Save Results ───────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "model": MODEL_NAME,
            "num_classes": NUM_CLASSES,
            "val_size": len(texts_val),
            "test_size": len(texts_test),
            "conf_alpha": CONF_ALPHA,
            "timestamp": datetime.now().isoformat() + "Z",
        },
        "baseline": {
            "accuracy": float(acc_test),
            "ece": float(ece_uncalibrated),
        },
        "temperature_scaling": {
            "optimal_temperature": float(optimal_temp),
            "ece": float(ece_temp),
            "ece_reduction": float(ece_uncalibrated - ece_temp),
        },
        "platt_scaling": {
            "ece": float(ece_platt),
            "ece_reduction": float(ece_uncalibrated - ece_platt),
        },
        "conformal_prediction": {
            "coverage": float(coverage),
            "target_coverage": 1 - CONF_ALPHA,
            "avg_set_size": float(avg_set_size),
            "threshold": float(cp_threshold),
        },
        "test_samples": [
            {
                "text": texts_test[i],
                "true_label": int(labels_test[i]),
                "pred_uncalibrated": int(preds_test[i]),
                "probs_uncalibrated": probs_test[i].tolist(),
                "probs_temp_scaled": probs_temp[i].tolist(),
                "probs_platt_scaled": probs_platt[i].tolist(),
                "conformal_pred_set": pred_sets[i],
            }
            for i in range(len(texts_test))
        ],
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[INFO] Results saved to {OUTPUT_FILE}")
    print("[INFO] Download this file and place it in dashboard/data/results.json")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
    
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"  Baseline ECE:      {results['baseline']['ece']:.4f}")
    print(f"  Temp Scaling ECE:  {results['temperature_scaling']['ece']:.4f}  (↓ {results['temperature_scaling']['ece_reduction']:.4f})")
    print(f"  Platt Scaling ECE: {results['platt_scaling']['ece']:.4f}  (↓ {results['platt_scaling']['ece_reduction']:.4f})")
    print(f"  Conformal Coverage: {results['conformal_prediction']['coverage']:.2%} (target: {results['conformal_prediction']['target_coverage']:.0%})")
    print("=" * 60)
