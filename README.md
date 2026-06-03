# 🎯 LLM-Uncertainty-Calibrator

> **Statistical Calibration & Conformal Prediction for a Financial Language Model**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_Cloud-FF4B4B.svg)](https://llm-uncertainty-calibrator-fclovm2shodudx8cl5k6st.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57+-FF4B4B.svg)](https://streamlit.io)
[![Model](https://img.shields.io/badge/Model-FinBERT-2ea44f.svg)](https://huggingface.co/ProsusAI/finbert)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**▶️ Try the live dashboard:** https://llm-uncertainty-calibrator-fclovm2shodudx8cl5k6st.streamlit.app/

---

## 🔍 Overview

**LLM-Uncertainty-Calibrator** measures and corrects the **miscalibration** of a real financial language model. When a model says it is "95% confident," it should be right about 95% of the time but neural networks are often overconfident, which is dangerous in financial decision-making where confidence drives downstream risk decisions.

This project uses **FinBERT** (`ProsusAI/finbert`), a transformer already fine-tuned on financial text, and evaluates it on a **financial sentiment benchmark**. By default it uses the **Twitter Financial News Sentiment** dataset (Parquet-native, loads on any environment); it will automatically prefer the **Financial PhraseBank** if that dataset is available. We then apply three calibration techniques and measure the improvement with **Expected Calibration Error (ECE)**.

**Task:** Financial sentiment classification (negative / neutral / positive) a core financial NLP task whose calibrated confidence feeds directly into risk assessment.

> **Why a pre-trained model on a real benchmark?** An untrained classifier outputs near-uniform probabilities (≈1/3 per class), which makes "calibration" meaningless. FinBERT produces genuine, varied confidence scores, so calibrating them is a real, defensible exercise.

---

## ⚠️ The Problem: Overconfident Models

- A model predicting "90% confidence" may only be correct 70% of the time.
- Overconfidence tends to grow with model size and complexity.
- Miscalibrated probabilities mislead any risk system built on top of them.
- **Real-world impact:** an overconfident financial assistant exposes an institution to liability and regulatory risk.

---

## ✅ Three Calibration Methods

### 1️⃣ Temperature Scaling
Learns a single scalar `T` and rescales logits before softmax: `softmax(logits / T)`. Minimises validation negative log-likelihood. Simple, fast, and usually effective.

### 2️⃣ Platt Scaling
Fits a multinomial logistic regression on validation logits. More flexible than temperature scaling, but can **overfit** on small validation sets — a useful failure mode to observe.

### 3️⃣ Conformal Prediction
Produces **prediction sets** with a distribution-free coverage guarantee: the true label lies in the set with probability ≥ (1−α). Output looks like *"the label is in {neutral, positive} at 90% confidence."*

### Metric: Expected Calibration Error (ECE)
Weighted average gap between confidence and accuracy across bins. `ECE = Σ |acc(bin) − conf(bin)| × (|bin| / N)`. Lower is better; 0 = perfect calibration.

---

## 📊 Key Results

> Numbers below are from a run on the Twitter Financial News Sentiment set (300 held-out samples); re-running `colab_inference.py` regenerates them and the dashboard renders them live.

| Metric | Baseline | Temperature Scaling | Platt Scaling |
|--------|----------|---------------------|---------------|
| **Test Accuracy** | 73.0% | — | — |
| **ECE** | 0.0948 | **0.0414** (↓56%) | 0.0682 (↓28%) |
| **Conformal Coverage** | — | — | **91.7%** (target 90%) |
| **Avg Prediction Set Size** | — | — | **1.54** classes |

**Temperature found:** T = 1.28 (T > 1 confirms the model was overconfident).

### What to look for
- ✅ **Confidence spans a wide range** (≈0.5–1.0), not a flat band around 0.33 proof the model is actually classifying.
- ✅ **Temperature Scaling reduces ECE** the headline calibration result.
- ✅ **Platt Scaling may or may not help** on small validation sets it can overfit and *increase* ECE, which is an honest, instructive finding.
- ✅ **Conformal coverage lands near the 90% target** with **small** prediction sets (mostly 1–2 classes), i.e. the guarantee is non-trivial.

---

## 🚀 Quick Start

### 1. Run calibration on Google Colab

1. Open [Google Colab](https://colab.research.google.com) → **Runtime → Change runtime type → T4 GPU**
2. Upload `colab_inference.py`
3. Install dependencies:
   ```python
   !pip install "transformers>=4.40" torch "datasets>=2.18,<3.0" scikit-learn scipy -q
   ```
4. Run:
   ```python
   !python colab_inference.py
   ```
5. Download results:
   ```python
   from google.colab import files
   files.download("results.json")
   ```

**Runtime:** ~2–5 minutes on a T4.

### 2. Launch the dashboard locally

```bash
git clone https://github.com/AtharvaPatil-Data/LLM-Uncertainty-Calibrator.git
cd LLM-Uncertainty-Calibrator
pip install streamlit plotly pandas numpy matplotlib
# place the downloaded results.json:
cp /path/to/results.json dashboard/data/
streamlit run dashboard/app.py
```

Dashboard opens at `http://localhost:8501`.

---

## 🧪 Methodology

### Dataset Financial Sentiment
The script evaluates on a financial sentiment benchmark with three classes. It first tries the **Financial PhraseBank** (sentences from financial news, labelled negative / neutral / positive), and if that dataset cannot be loaded in the current environment it automatically falls back to the **Twitter Financial News Sentiment** dataset (financial-news headlines labelled Bearish / Bullish / Neutral, which map to negative / positive / neutral). Both are genuine financial-text benchmarks, so FinBERT produces meaningful, varied confidence scores exactly what makes calibration worth studying. The results shown here were produced on the Twitter Financial News Sentiment set (≈73% accuracy on 300 held-out samples).

### Label Alignment (important detail)
FinBERT emits classes in **its** order (`positive, negative, neutral`) while a dataset may store them in a **different** order or under different names (PhraseBank: `negative, neutral, positive`; Twitter: `Bearish, Bullish, Neutral`). The script canonicalises each dataset label to FinBERT's vocabulary (e.g. Bearish→negative, Bullish→positive) and remaps the dataset's labels into the model's index space, so probability column *i* always corresponds to true-label *i*. Getting this wrong is the classic bug that makes a working model look random.

### Pipeline
1. Run FinBERT over each sentence, extract raw **logits**.
2. Convert to uncalibrated probabilities via softmax.
3. Learn calibration parameters on the **calibration half** (Temperature `T`, Platt regression, conformal threshold).
4. Apply to the **test half**.
5. Report accuracy, ECE (before/after), conformal coverage and set sizes.

**Split:** 50% calibration / 50% test, stratified.

### Conformal Prediction
1. Non-conformity score on calibration set: `s(x,y) = 1 − P(y | x)`.
2. Threshold `q` at the `⌈(n+1)(1−α)⌉/n` quantile of scores.
3. Include class *i* in the test prediction set if `1 − P(i | x) ≤ q`.

With α = 0.1 this targets **90%** marginal coverage.

---

## 📸 Dashboard

The Streamlit dashboard includes:
- **Reliability diagram** confidence vs accuracy for all three methods.
- **Confidence distribution** violin plots showing the spread of max-probabilities.
- **ECE comparison** bar chart + method table.
- **Conformal prediction** set-size distribution and worked examples.
- **Sample explorer** per-sentence probabilities across all methods.

*(Add screenshots to `screenshots/` and reference them here.)*

---

## 🎓 PhD Research Connection

Directly supports the **Central Bank of Ireland PhD Programme** — *"Framework for Interpretable and Behavioural Risk Assessment of Intelligent Language Models"*:

- **Decision-Making Under Uncertainty** calibration turns overconfident scores into trustworthy probabilities for downstream risk decisions.
- **Risk Assessment in Financial Services** post-hoc calibration corrects miscalibration without retraining.
- **Statistical Rigour & Formal Guarantees** conformal prediction gives distribution-free coverage guarantees.
- **Interpretability** reliability diagrams and prediction sets make uncertainty explicit and auditable.

---

## 🔧 Technical Details

- **Model:** `ProsusAI/finbert` (BERT-base fine-tuned on financial text, 3-class)
- **Dataset:** Twitter Financial News Sentiment (default) or `financial_phrasebank` (`sentences_50agree`) when available
- **Calibration:** Temperature Scaling (1 param), Platt Scaling (multinomial logistic), Conformal Prediction (non-parametric)
- **Metrics:** Accuracy, ECE (10-bin), conformal coverage, average set size
- **Compute:** Colab T4 GPU for inference; Streamlit (CPU) for the dashboard

---

## 📚 References

1. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** *On Calibration of Modern Neural Networks.* ICML. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
2. **Platt, J. (1999).** *Probabilistic Outputs for Support Vector Machines.*
3. **Vovk, V., Gammerman, A., & Shafer, G. (2005).** *Algorithmic Learning in a Random World.* Springer.
4. **Angelopoulos, A. N., & Bates, S. (2021).** *A Gentle Introduction to Conformal Prediction.* [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
5. **Malo, P. et al. (2014).** *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts* (Financial PhraseBank).
6. **Araci, D. (2019).** *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* [arXiv:1908.10063](https://arxiv.org/abs/1908.10063)

---

## 📂 Project Structure

```
LLM-Uncertainty-Calibrator/
├── colab_inference.py          # FinBERT + Financial PhraseBank calibration (run on Colab)
├── dashboard/
│   ├── app.py                  # Streamlit dashboard UI
│   ├── plots.py                # Plotly visualizations
│   └── data/
│       └── results.json        # Results from colab_inference.py
├── screenshots/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 👤 Author

**Atharva Patil** MSc Computing (Data Analytics), Dublin City University
PhD Applicant, Central Bank of Ireland PhD Programme
GitHub: [github.com/AtharvaPatil-Data](https://github.com/AtharvaPatil-Data)

---

## 🙏 Acknowledgments

- **Supervisors:** Dr. Lili Zhang (DCU), Prof. Tomás Ward (DCU), Prof. Robert Whelan (TCD)
- **Funding:** Central Bank of Ireland + Insight Centre for Data Analytics

---

## 📄 License

MIT License.
