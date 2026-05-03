# Can FDA Drug Approvals Predict Stock Price Movements?

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-FF6600?logo=xgboost&logoColor=white)
![FinBERT](https://img.shields.io/badge/FinBERT-ProsusAI%2Ffinbert-22C55E?logo=huggingface&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-F59E0B)

**Author:** Hari Vykuntapu  
**Program:** MS Artificial Intelligence, Southwest Baptist University, United States  
**Contact:** hpvykuntapu@gmail.com  
**GitHub:** [github.com/HariVykuntapu/fda-stock-prediction](https://github.com/HariVykuntapu/fda-stock-prediction)  
**Live App:** https://fda-stock-prediction.streamlit.app

---

## 📄 Preprint

Published on Zenodo: https://doi.org/10.5281/zenodo.20011255

**Citation:** Vykuntapu, H. P. (2026). Regulatory Sentiment as a Predictor of Pharmaceutical Stock Movements: A FinBERT + XGBoost Analysis of FDA Drug Approvals. Zenodo. https://doi.org/10.5281/zenodo.20011255

---

## The Question

Every FDA drug approval hits the news in the same format. Yet some move the stock 10% and others barely register. The announcement text is nearly identical. What varies is the regulatory structure — what kind of drug, what novelty classification, when it happened.

Most stock prediction work optimizes RSI and MACD. I wanted to know if the regulatory metadata of an FDA decision carries a signal that technical indicators miss entirely — and whether structured domain knowledge or NLP sentiment does more of the predictive work.

**Short answer from the data:** both matter, they're not redundant, and XGBoost hits AUROC 0.5725 on the held-out test set. The FinBERT sentiment feature came out on top in SHAP — which I didn't expect.

---

## The RSS Formula (My Original Contribution)

I built the **Regulatory Sentiment Score (RSS)** to encode three dimensions of FDA regulatory context into a single number — without touching any text:

```
RSS = (approval_type_weight × 0.45) + (drug_novelty_score × 0.35) + (market_timing_factor × 0.20)
```

| Component | Description | Values |
|-----------|-------------|--------|
| `approval_type_weight` | NDA/BLA vs generic classification | NDA=1.0, BLA=0.8, ANDA=0.3 |
| `drug_novelty_score` | First-in-class vs follow-on vs generic | 1.0 / 0.5 / 0.0 |
| `market_timing_factor` | Day-of-week × month-end × Q4 effects | 0.0 – 1.0 |

RSS doesn't touch the announcement text. It's pure regulatory structure — which is the whole point. The test: does this metadata add predictive value beyond what FinBERT can extract from the same event?

---

## Dataset

- **FDA approvals:** OpenFDA API — 2018–2023, 12 major pharmaceutical companies
- **Stock prices:** Yahoo Finance via `yfinance` — daily adjusted close
- **Companies:** PFE, MRNA, JNJ, AZN, MRK, BMY, LLY, ABBV, GILD, AMGN, BIIB, REGN
- **Target:** Binary — stock UP or DOWN 7 calendar days after approval date

---

## Models & Results

| Model | Accuracy | Precision | Recall | F1 | AUROC |
|-------|----------|-----------|--------|----|-------|
| Logistic Regression | 53.3% | 54.6% | 75.0% | 63.2% | 0.4431 |
| Random Forest | 50.0% | 52.6% | 62.5% | 57.1% | 0.5525 |
| **XGBoost** ✓ | **53.3%** | **56.3%** | **56.3%** | **56.3%** | **0.5725** |

**Best model:** XGBoost (AUROC 0.5725) — 296 samples, 236 train / 60 test, 10 features

---

## Project Structure

```
project2-fda-stock-prediction/
├── data/
│   ├── raw/                    # FDA approvals CSV, stock prices CSV
│   └── processed/              # Feature-engineered dataset
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_results_visuals.ipynb
├── models/                     # Saved .pkl models + metadata JSON
├── outputs/
│   ├── eda/                    # EDA plots
│   └── results/                # ROC curves, SHAP, confusion matrix
├── streamlit_app/
│   └── app.py                  # 4-tab interactive research app
├── paper/
│   └── arxiv_draft.md          # Full paper draft (arXiv target)
├── requirements.txt
├── LICENSE                     # MIT
└── README.md
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/HariVykuntapu/fda-stock-prediction.git
cd fda-stock-prediction

# Install
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/01_data_collection.ipynb

# Launch app
streamlit run streamlit_app/app.py
```

---

## What the Data Actually Showed

I built this expecting structured regulatory metadata to dominate NLP sentiment. That's not quite what happened.

**SHAP ranking on XGBoost:** FinBERT compound sentiment ranked first — above RSS. Approval type weight ranked second. RSS composite ranked third, above its own individual components. Which means: the weighted interaction of my formula adds something that the parts don't, but FinBERT is picking up tonal variation in the text that I can't fully account for given how formulaic FDA announcement language is.

**FinBERT vs. VADER:** Weakly correlated (r ≈ 0.3–0.5), meaning they're measuring different things. VADER tends to skew positive on approval language — words like "accepted" and "approved" read as positive in general English. FinBERT is more calibrated. Finance-domain pre-training earns its keep even on out-of-domain regulatory text.

**Overall AUROC 0.5725:** Honest. Not tradeable, but meaningfully above random on a noisy 7-day target with 296 samples. The signal is real; the dataset is small.

Full pipeline: OpenFDA API → yfinance → EDA → RSS + FinBERT features → XGBoost + SHAP → Streamlit app → arXiv draft.

---

## Contact

Hari Vykuntapu  
hpvykuntapu@gmail.com  
[linkedin.com/in/harivykuntapu](https://www.linkedin.com/in/harivykuntapu)  
[github.com/HariVykuntapu](https://github.com/HariVykuntapu)
