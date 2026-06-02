# =============================================================================
# LLM-Uncertainty-Calibrator: Uncertainty Calibration & Conformal Prediction
# =============================================================================
# Run this on Google Colab with T4 GPU (Runtime > Change runtime type > T4 GPU)
#
# Install dependencies first (run in a Colab cell):
#   !pip install -q datasets scikit-learn scipy
#
#   IMPORTANT: do NOT use `-U` / `--upgrade` and do NOT reinstall torch,
#   torchvision, or transformers. Colab already ships working, CUDA-matched
#   builds; upgrading them breaks the GPU stack (torchvision::nms errors).
#   transformers is pre-installed on Colab. If it is somehow missing, add it
#   WITHOUT upgrading torch:  !pip install -q transformers
#
#   (Optional) For the Financial PhraseBank specifically:
#       !pip install -q "datasets==2.21.0"
#   Otherwise the script falls back to a Parquet-native dataset automatically.
#
# This script:
#   1. Loads FinBERT — a transformer ALREADY trained on financial text
#   2. Evaluates it on the Financial PhraseBank (a real, peer-reviewed benchmark)
#   3. Extracts raw probabilities (uncalibrated)
#   4. Applies Temperature Scaling, Platt Scaling, Conformal Prediction
#   5. Measures Expected Calibration Error (ECE) before/after
#   6. Saves results.json for the Streamlit dashboard
#
# WHY FinBERT + Financial PhraseBank?
#   A randomly-initialised classifier produces ~uniform (1/3, 1/3, 1/3) outputs,
#   which makes calibration meaningless. FinBERT is fine-tuned on financial
#   sentiment, so its confidence scores are real — and therefore worth calibrating.
# =============================================================================

import json
import torch
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# ── Device setup ──────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "ProsusAI/finbert"          # Pre-trained financial sentiment model
DATASET_CONFIG = "sentences_50agree"     # PhraseBank subset (>=50% annotator agreement)
OUTPUT_FILE = "results.json"
CONF_ALPHA = 0.1                          # 90% target coverage for conformal prediction
MAX_SAMPLES = 600                         # Subset for fast, robust evaluation
BATCH_SIZE = 32
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ── Load Dataset ──────────────────────────────────────────────────────────────
# Primary: Financial PhraseBank (the benchmark FinBERT was designed for).
# Newer `datasets` versions (v3/v4) dropped script-based datasets, so PhraseBank
# may not load. We therefore fall back to a Parquet-native financial sentiment
# dataset that always loads cleanly. Either way the calibration story holds.

def load_financial_sentiment():
    """Return (texts, labels_int, dataset_label_names). Tries multiple sources."""
    # ---- Source 1: Financial PhraseBank (best match for FinBERT) ----
    for repo in ("financial_phrasebank", "takala/financial_phrasebank"):
        try:
            ds = load_dataset(repo, DATASET_CONFIG, trust_remote_code=True)["train"]
            names = ds.features["label"].names
            print(f"[INFO] Loaded Financial PhraseBank from '{repo}'")
            return list(ds["sentence"]), np.array(ds["label"]), names
        except Exception as e:
            print(f"[WARN] PhraseBank via '{repo}' unavailable: {str(e)[:110]}")

    # ---- Source 2: Twitter Financial News Sentiment (Parquet, always loads) ----
    try:
        ds = load_dataset("zeroshot/twitter-financial-news-sentiment")["train"]
        text_col = "text" if "text" in ds.column_names else ds.column_names[0]
        # This dataset's labels: 0=Bearish, 1=Bullish, 2=Neutral
        names = ["Bearish", "Bullish", "Neutral"]
        print("[INFO] Loaded Twitter Financial News Sentiment (Parquet fallback)")
        return list(ds[text_col]), np.array(ds["label"]), names
    except Exception as e:
        print(f"[WARN] Twitter sentiment unavailable: {str(e)[:110]}")

    raise RuntimeError("Could not load any financial sentiment dataset.")

texts_all, labels_all, dataset_label_names = load_financial_sentiment()
print(f"[INFO] Dataset labels: {dataset_label_names}")

# Subsample (stratified) for speed if dataset is large
if len(texts_all) > MAX_SAMPLES:
    idx_sub, _ = train_test_split(
        np.arange(len(texts_all)),
        train_size=MAX_SAMPLES,
        random_state=RANDOM_SEED,
        stratify=labels_all,
    )
    texts_all = [texts_all[i] for i in idx_sub]
    labels_all = labels_all[idx_sub]

print(f"[INFO] Using {len(texts_all)} samples")

# ── Load Model ────────────────────────────────────────────────────────────────
print("[INFO] Loading FinBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# FinBERT's own label order, e.g. {0: 'positive', 1: 'negative', 2: 'neutral'}
model_id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
model_label_order = [model_id2label[i] for i in range(len(model_id2label))]
print(f"[INFO] Model labels: {model_label_order}")

NUM_CLASSES = len(model_label_order)

# ── CRITICAL: Align dataset labels to the model's output index space ──────────
# Different datasets name/order classes differently (e.g. PhraseBank uses
# negative/neutral/positive; the Twitter set uses Bearish/Bullish/Neutral).
# We canonicalise each dataset label to FinBERT's vocabulary, then remap true
# labels into the model's index space so column i of the model's probabilities
# corresponds to the same class as true label i. Getting this wrong is the
# classic bug that makes a working model look random.
SENTIMENT_SYNONYMS = {
    "bearish": "negative", "bullish": "positive",
    "negative": "negative", "positive": "positive", "neutral": "neutral",
}

def canon(name):
    return SENTIMENT_SYNONYMS.get(name.lower(), name.lower())

dataset_to_model = {}
for ds_idx, name in enumerate(dataset_label_names):
    target = canon(name)
    if target not in model_label_order:
        raise ValueError(
            f"Dataset label '{name}' -> '{target}' not in model labels {model_label_order}"
        )
    dataset_to_model[ds_idx] = model_label_order.index(target)

labels_aligned = np.array([dataset_to_model[int(l)] for l in labels_all])
print(f"[INFO] Label alignment (dataset_idx -> model_idx): {dataset_to_model}")

# Human-readable class names in the MODEL's index order (used by the dashboard)
CLASS_NAMES = [name.capitalize() for name in model_label_order]

# ── Inference: extract logits ─────────────────────────────────────────────────

def predict_logits(texts):
    """Run batched inference and return raw logits (N x num_classes)."""
    all_logits = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()
        all_logits.append(logits)
    return np.concatenate(all_logits, axis=0)

def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

# ── Calibration Methods ───────────────────────────────────────────────────────

def compute_ece(probs, preds, labels, n_bins=10):
    """Expected Calibration Error: gap between confidence and accuracy."""
    confidences = probs.max(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() > 0:
            acc = (preds[in_bin] == labels[in_bin]).mean()
            conf = confidences[in_bin].mean()
            ece += (in_bin.sum() / len(labels)) * abs(acc - conf)
    return ece

def temperature_scaling(logits_val, labels_val):
    """Find T minimising NLL on the validation set."""
    from scipy.optimize import minimize_scalar

    def nll(T):
        p = softmax(logits_val / T)
        return -np.log(p[np.arange(len(labels_val)), labels_val] + 1e-12).mean()

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    T = float(res.x)
    print(f"[INFO] Temperature Scaling: optimal T = {T:.4f}")
    return T

def platt_scaling(logits_val, labels_val, logits_test):
    """Multinomial logistic regression on logits (multiclass Platt).

    Note: newer scikit-learn (>=1.7) removed the `multi_class` argument;
    multinomial behaviour is the default for multiclass problems with lbfgs.
    """
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(logits_val, labels_val)
    print("[INFO] Platt Scaling: logistic regression fitted")
    return clf.predict_proba(logits_test)

def conformal_prediction(probs_cal, labels_cal, probs_test, alpha):
    """Split-conformal prediction sets with (1-alpha) marginal coverage."""
    scores = 1 - probs_cal[np.arange(len(labels_cal)), labels_cal]
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    threshold = float(np.quantile(scores, q_level))
    pred_sets = []
    for p in probs_test:
        s = [i for i in range(len(p)) if (1 - p[i]) <= threshold]
        if not s:
            s = [int(np.argmax(p))]
        pred_sets.append(s)
    print(f"[INFO] Conformal: target={1-alpha:.0%}, threshold={threshold:.4f}")
    return pred_sets, threshold

# ── Main Experiment ───────────────────────────────────────────────────────────

def run_experiment():
    # Split into calibration (val) and test halves, stratified
    idx = np.arange(len(texts_all))
    idx_val, idx_test = train_test_split(
        idx, test_size=0.5, random_state=RANDOM_SEED, stratify=labels_aligned
    )

    texts_val = [texts_all[i] for i in idx_val]
    texts_test = [texts_all[i] for i in idx_test]
    labels_val = labels_aligned[idx_val]
    labels_test = labels_aligned[idx_test]

    print(f"[INFO] Calibration set: {len(texts_val)} | Test set: {len(texts_test)}")

    print("[INFO] Running inference (calibration set)...")
    logits_val = predict_logits(texts_val)
    print("[INFO] Running inference (test set)...")
    logits_test = predict_logits(texts_test)

    probs_test = softmax(logits_test)
    probs_val = softmax(logits_val)
    preds_test = probs_test.argmax(axis=1)

    acc = accuracy_score(labels_test, preds_test)
    ece_base = compute_ece(probs_test, preds_test, labels_test)
    print(f"\n[BASELINE] Test accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"[BASELINE] Uncalibrated ECE: {ece_base:.4f}")

    # Temperature Scaling
    print("\n[CALIBRATION] Temperature Scaling...")
    T = temperature_scaling(logits_val, labels_val)
    probs_temp = softmax(logits_test / T)
    ece_temp = compute_ece(probs_temp, probs_temp.argmax(1), labels_test)
    print(f"[TEMP] ECE: {ece_temp:.4f} (reduction: {ece_base - ece_temp:+.4f})")

    # Platt Scaling
    print("\n[CALIBRATION] Platt Scaling...")
    probs_platt = platt_scaling(logits_val, labels_val, logits_test)
    ece_platt = compute_ece(probs_platt, probs_platt.argmax(1), labels_test)
    print(f"[PLATT] ECE: {ece_platt:.4f} (reduction: {ece_base - ece_platt:+.4f})")

    # Conformal Prediction
    print("\n[CALIBRATION] Conformal Prediction...")
    pred_sets, cp_thr = conformal_prediction(probs_val, labels_val, probs_test, CONF_ALPHA)
    coverage = float(np.mean([labels_test[i] in pred_sets[i] for i in range(len(labels_test))]))
    avg_set = float(np.mean([len(s) for s in pred_sets]))
    print(f"[CONFORMAL] Coverage: {coverage:.4f} | Avg set size: {avg_set:.2f}")

    results = {
        "metadata": {
            "model": MODEL_NAME,
            "dataset": f"financial_phrasebank/{DATASET_CONFIG}",
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "val_size": int(len(texts_val)),
            "test_size": int(len(texts_test)),
            "conf_alpha": CONF_ALPHA,
            "timestamp": datetime.now().isoformat() + "Z",
        },
        "baseline": {"accuracy": float(acc), "ece": float(ece_base)},
        "temperature_scaling": {
            "optimal_temperature": float(T),
            "ece": float(ece_temp),
            "ece_reduction": float(ece_base - ece_temp),
        },
        "platt_scaling": {
            "ece": float(ece_platt),
            "ece_reduction": float(ece_base - ece_platt),
        },
        "conformal_prediction": {
            "coverage": coverage,
            "target_coverage": 1 - CONF_ALPHA,
            "avg_set_size": avg_set,
            "threshold": cp_thr,
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
    print(f"\n[INFO] Saved {OUTPUT_FILE}")
    print("[INFO] Download it and place it in dashboard/data/results.json")
    return results


if __name__ == "__main__":
    r = run_experiment()
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"  Model:              {r['metadata']['model']}")
    print(f"  Dataset:            {r['metadata']['dataset']}")
    print(f"  Classes:            {r['metadata']['class_names']}")
    print(f"  Test accuracy:      {r['baseline']['accuracy']:.4f}")
    print(f"  Baseline ECE:       {r['baseline']['ece']:.4f}")
    print(f"  Temp Scaling ECE:   {r['temperature_scaling']['ece']:.4f}  (T={r['temperature_scaling']['optimal_temperature']:.3f})")
    print(f"  Platt Scaling ECE:  {r['platt_scaling']['ece']:.4f}")
    print(f"  Conformal Coverage: {r['conformal_prediction']['coverage']:.2%} (target {r['conformal_prediction']['target_coverage']:.0%})")
    print(f"  Avg Set Size:       {r['conformal_prediction']['avg_set_size']:.2f}")
    print("=" * 60)
