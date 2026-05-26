# 🫀 Kardia — Multimodal Early Heart Disease Prediction

> **AUC: 0.9733 | F1: 0.9231 | Accuracy: 91.3%**
> Multimodal deep learning system combining clinical tabular data + 12-lead ECG signals for early heart disease detection with full explainability.

---

## 📌 Project Overview

**Kardia** is an end-to-end multimodal ML pipeline that fuses:

- **Tabular clinical data** — UCI Cleveland Heart Disease dataset (303 patients, 18 features after engineering)
- **ECG signals** — PTB-XL dataset (12-lead, 10-second recordings at 100 Hz)

The system produces three simultaneous outputs:

1. **Binary prediction** — Heart disease present or absent
2. **Risk score** — 0–100% continuous risk probability
3. **Severity classification** — None / Mild / Moderate / Severe

All predictions are fully explained via **SHAP** (tabular features) and **GradCAM** (ECG signals).

---

## 🏆 Results

| Model | Test AUC | F1 Score | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.8514 | 0.7778 | 73.9% |
| Random Forest | 0.8448 | 0.7692 | 73.9% |
| XGBoost | 0.8324 | 0.7347 | 71.7% |
| MLP (tabular only) | 0.8438 | 0.7692 | 73.9% |
| ResNet1D (ECG only) | 0.9205 | 0.7273 | 80.0% |
| **Multimodal Fusion v2** | **0.9733** | **0.9231** | **91.3%** |

---

## 🗂️ Project Structure

```
heart-disease-prediction/
├── data/
│   ├── tabular/
│   │   ├── heart_uci.csv                  ← raw UCI Cleveland data
│   │   └── processed/                     ← train/val/test splits (CSV)
│   └── ptbxl/
│       ├── ptbxl_database.csv             ← PTB-XL metadata
│       ├── scp_statements.csv             ← SCP label definitions
│       ├── signals/                       ← raw ECG .dat/.hea files
│       └── processed/
│           ├── ecg_signals.npy            ← (200, 12, 1000) float32
│           └── ecg_labels.npy             ← (200,) binary labels
├── src/
│   ├── download_data.py                   ← tabular data download
│   ├── load_ptbxl.py                      ← ECG download + preprocessing
│   ├── preprocess.py                      ← feature engineering + SMOTE
│   ├── baseline_models.py                 ← LR + RF + XGBoost baselines
│   ├── mlp_model.py                       ← tabular MLP architecture
│   ├── train_mlp.py                       ← MLP training loop
│   ├── resnet1d.py                        ← ResNet1D ECG encoder
│   ├── train_resnet1d.py                  ← ECG encoder training
│   ├── fusion_model.py                    ← attention-gated fusion architecture
│   ├── train_fusion.py                    ← fusion v1 (end-to-end)
│   ├── train_fusion_v2.py                 ← fusion v2 (feature-level) ← BEST
│   └── explainability.py                  ← SHAP + GradCAM
├── models/
│   ├── best_mlp.pt                        ← trained tabular MLP
│   ├── best_resnet1d.pt                   ← trained ECG encoder
│   ├── best_fusion_v2.pt                  ← final multimodal model
│   ├── scaler.pkl                         ← fitted StandardScaler
│   └── imputer.pkl                        ← fitted KNNImputer
├── reports/                               ← training curves, ROC, confusion matrices
├── explainability/                        ← SHAP plots + GradCAM heatmaps
├── notebooks/
│   └── 01_EDA_tabular.ipynb               ← exploratory data analysis
└── requirements.txt
```

---

## ⚙️ Architecture

### Full Pipeline

```
Clinical Data (18 features)            ECG Signal (12 leads × 1000 samples)
        ↓                                              ↓
  KNN Imputation                            Butterworth Bandpass Filter
  Feature Engineering                       Z-score Normalisation per Lead
  SMOTE-NC Balancing                                  ↓
  StandardScaler                            ResNet1D Encoder
        ↓                               (Stem → Stage1 → Stage2 → Stage3
  Tabular Embeddings                         → GlobalAvgPool → 256-dim)
   (18 → 128 → 64)                                    ↓
        ↓                                      Frozen Embedding
        └──────────────┬───────────────────────────────┘
                       ↓
             Feature Concatenation
              (64 tab + 256 ECG = 274-dim fused vector)
                       ↓
                 Fusion MLP Head
              274 → 256 → 128 → 64
                       ↓
        ┌──────────────┼──────────────┐
     Binary          Risk          Severity
  (Sigmoid →      (0 – 100       (Softmax →
   Yes/No)         score)        4 classes)
```

### Key Design Choices

- **Feature-level fusion** over end-to-end — more stable and effective on small datasets
- **Label-matched ECG pairing** — ECG embeddings are matched by label when tabular and ECG patients don't share the same individuals
- **Frozen ResNet1D** as a feature extractor — pretrained ECG representations are preserved and not overwritten during fusion training
- **SMOTE-NC** balancing applied to training set only — prevents data leakage into validation and test sets
- **Three output heads** on a single shared backbone — binary, risk score, and severity in one forward pass

---

## 🔍 Explainability

### SHAP — Global Feature Importance
Top predictors globally: `sex` > `thalach` > `thal` > `ca` > `cp`

Engineered features `hr_reserve` and `cp_sex` ranked in the global SHAP top 10, confirming that feature engineering added genuine predictive value beyond the original 13 UCI features.

### Per-patient SHAP Waterfall
Each patient receives a local explanation showing:
- Top 10 features driving the prediction (red = increases risk, green = decreases risk)
- Raw feature values alongside SHAP impact magnitudes
- Risk score, severity class, and confidence percentage

### GradCAM — ECG Temporal Attention
12-lead activation heatmaps highlighting which time segments of the ECG signal drove the ResNet1D classification decision.

---

## 🚀 Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/balajisabk32-cmd/heart-disease-prediction.git
cd heart-disease-prediction

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download data

```bash
python src/download_data.py       # UCI Cleveland tabular data
python src/load_ptbxl.py          # PTB-XL ECG (downloads ~200 records from PhysioNet)
```

### 3. Preprocess

```bash
python src/preprocess.py          # imputation, feature engineering, SMOTE, splits
```

### 4. Train all models (in order)

```bash
python src/baseline_models.py     # Logistic Regression, Random Forest, XGBoost
python src/train_mlp.py           # tabular MLP with 3 output heads
python src/train_resnet1d.py      # ECG encoder (ResNet1D)
python src/train_fusion_v2.py     # final multimodal fusion model ← BEST
```

### 5. Run explainability

```bash
python src/explainability.py      # SHAP global + per-patient + GradCAM ECG maps
```

---

## 📦 Requirements

```
torch
scikit-learn
xgboost
imbalanced-learn
shap
lime
captum
wfdb
neurokit2
scipy
pandas
numpy
matplotlib
seaborn
ucimlrepo
jupyter
ipykernel
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📊 Datasets

| Dataset | Source | Records | Used |
|---|---|---|---|
| UCI Heart Disease (Cleveland) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) | 303 | All 303 |
| PTB-XL 12-lead ECG | [PhysioNet v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) | 21,799 | 200 (balanced) |

Both datasets are **free and publicly available** with no login required.

---

## 🧪 Feature Engineering

Five new features were derived from the original 13 UCI clinical variables:

| Feature | Formula | Clinical Meaning |
|---|---|---|
| `hr_reserve` | `thalach - (220 - age)` | How close max HR is to age-predicted maximum |
| `chol_age_ratio` | `chol / age` | Cholesterol burden relative to age |
| `cp_sex` | `cp × sex` | Interaction between chest pain type and sex |
| `thalach_oldpeak` | `thalach × oldpeak` | Combined HR and ST depression signal |
| `age_bucket` | `pd.cut(age, [0,40,55,65,120])` | Age group: young / mid / senior / elderly |

---

## 🧠 Key Learnings

1. **Multimodal fusion on small datasets** requires feature-level fusion — end-to-end joint training fails when datasets don't share the same patients
2. **ECG signals alone** (AUC 0.9205) outperform all tabular baselines (best: AUC 0.8514) on this benchmark
3. **Label-matched ECG pairing** is an effective bridge when tabular and imaging cohorts don't overlap
4. **Feature engineering** (`hr_reserve`, `cp_sex`) ranked in global SHAP top 10, validating their predictive contribution
5. **SMOTE-NC** handles mixed categorical and continuous features correctly, unlike standard SMOTE which assumes continuous data throughout

---

## 📈 Training Details

| Component | Config |
|---|---|
| ResNet1D optimizer | AdamW, LR=5e-4, weight decay=1e-4 |
| ResNet1D scheduler | CosineAnnealingLR, T_max=80 |
| ResNet1D batch size | 16 |
| Fusion optimizer | AdamW, LR=1e-3 → 1e-4 |
| Fusion loss | BCELoss + 0.3 × CrossEntropyLoss (severity) |
| Early stopping | patience=25 on val AUC |
| Device | CPU (Intel/AMD) — no GPU required |

---

## 👨‍💻 Author

**Balaji** — AI/ML Engineer
GitHub: [@balajisabk32-cmd](https://github.com/balajisabk32-cmd)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🔖 Citation

If you use this project in your research or work, please cite:

```
@misc{kardia2026,
  author = {Balaji},
  title  = {Kardia: Multimodal Early Heart Disease Prediction},
  year   = {2026},
  url    = {https://github.com/balajisabk32-cmd/heart-disease-prediction}
}
```
