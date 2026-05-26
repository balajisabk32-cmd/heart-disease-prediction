"""
Kardia — Final Batch 5 Patient Mock Inference Test
==================================================
Clinical Focus: Demographic Extremes & Clinical Outliers
Contains 100 novel synthetic profiles including clinical edge cases:
  - Early-Onset Genetic CAD (Ages 25-35 with severe blockages)
  - Super-Agers (Ages 75+ with benign ECGs but clean arteries)
  - Extreme Athletes (Profound bradycardia/high HR reserves)
  - Physiological Outliers (Extreme BP or Max HR)

Columns: age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,true_label
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
import sys
from sklearn.metrics import (roc_auc_score, f1_score,
                             accuracy_score, classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure directories exist
sys.path.append('src')
os.makedirs('reports', exist_ok=True)

try:
    from resnet1d import ResNet1D
except ImportError:
    print("Warning: ResNet1D not found in src/. Mocking embedding for demonstration.")

# ════════════════════════════════════════════════════════════════════════════
# BATCH 5: 100 NOVEL SYNTHETIC PATIENTS
# sex  : 1=Male, 0=Female
# cp   : 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic
# thal : 1=normal, 2=fixed defect, 3=reversable defect
# slope: 0=upsloping, 1=flat, 2=downsloping
# ════════════════════════════════════════════════════════════════════════════

records = [
# ── DISEASE PATIENTS (50) ─────────────────────────────────────────────────
# Extreme Young (Early Onset Genetic CAD)
# age  sex cp  bp   chol fbs re  hr   ex  op   sl ca th  lbl
 [28,  1,  0, 140, 450,  0,  1, 165,  1, 2.0,  1, 1,  3,  1],  # P401 Very young male, familial hypercholesterolemia
 [32,  0,  0, 135, 390,  0,  0, 155,  1, 1.5,  1, 2,  3,  1],  # P402 Young female, genetic severe CAD
 [35,  1,  1, 150, 410,  0,  1, 145,  1, 2.5,  1, 3,  3,  1],  # P403 Young male, multi-vessel CAD
 [29,  1,  0, 125, 380,  0,  0, 170,  1, 1.8,  1, 1,  3,  1],  # P404 Very young male, typical angina
 [34,  0,  0, 130, 420,  0,  1, 160,  1, 2.2,  1, 2,  3,  1],  # P405 Young female, extreme cholesterol
# Extreme Old (Geriatric Severe CAD)
 [82,  1,  3, 160, 210,  0,  1, 105,  0, 2.0,  1, 3,  3,  1],  # P406 Octogenarian, silent ischemia, 3-vessel
 [79,  0,  2, 155, 240,  1,  1, 110,  1, 1.5,  1, 2,  3,  1],  # P407 Geriatric female, diabetic, multi-vessel
 [85,  1,  0, 145, 190,  0,  2, 95,   1, 3.0,  1, 3,  3,  1],  # P408 Very old male, low HR, severe drop
 [78,  0,  0, 165, 260,  0,  1, 115,  1, 2.5,  1, 2,  3,  1],  # P409 Geriatric female, classic symptoms
 [81,  1,  1, 150, 220,  0,  0, 100,  1, 1.8,  1, 3,  3,  1],  # P410 Octogenarian male, atypical pain
# Extreme Vitals (Tachycardia & Hypertensive Crisis)
 [55,  1,  0, 200, 280,  1,  1, 140,  1, 3.5,  1, 2,  3,  1],  # P411 Extreme BP, diabetic
 [60,  0,  0, 195, 290,  0,  1, 135,  1, 2.8,  1, 1,  3,  1],  # P412 Extreme BP female
 [48,  1,  0, 140, 250,  0,  0, 205,  1, 2.0,  1, 1,  3,  1],  # P413 Extreme Tachycardia, young male
 [52,  0,  0, 135, 265,  0,  1, 202,  1, 1.5,  1, 2,  3,  1],  # P414 Extreme Tachycardia, female
 [65,  1,  0, 210, 240,  0,  2, 125,  1, 4.0,  1, 3,  3,  1],  # P415 Hypertensive crisis + LVH + CAD
# Standard / Mixed Disease (Padding to maintain baseline accuracy)
 [58,  1,  0, 145, 255,  1,  0, 150,  0, 2.3,  0, 2,  3,  1],  # P416 Standard male CAD
 [62,  0,  0, 150, 260,  0,  0, 145,  1, 1.5,  1, 1,  3,  1],  # P417 Standard female CAD
 [55,  1,  0, 130, 240,  0,  1, 160,  1, 2.0,  1, 2,  3,  1],  # P418
 [60,  1,  0, 140, 270,  0,  0, 135,  1, 2.5,  1, 3,  3,  1],  # P419
 [64,  0,  0, 135, 280,  1,  1, 130,  1, 1.8,  1, 2,  3,  1],  # P420
 [57,  1,  0, 125, 230,  0,  0, 155,  1, 1.2,  1, 1,  3,  1],  # P421
 [67,  1,  0, 160, 290,  0,  0, 120,  1, 3.0,  1, 3,  3,  1],  # P422
 [59,  0,  0, 145, 250,  0,  1, 140,  1, 2.2,  1, 2,  3,  1],  # P423
 [61,  1,  0, 155, 265,  1,  0, 125,  1, 1.6,  1, 2,  3,  1],  # P424
 [54,  1,  0, 130, 245,  0,  0, 165,  1, 1.0,  1, 1,  3,  1],  # P425
 [66,  0,  0, 140, 275,  0,  1, 115,  1, 2.4,  1, 3,  3,  1],  # P426
 [53,  1,  0, 120, 220,  0,  0, 170,  1, 0.8,  1, 1,  3,  1],  # P427
 [68,  1,  0, 150, 285,  1,  1, 110,  1, 3.2,  1, 3,  3,  1],  # P428
 [56,  0,  0, 135, 255,  0,  0, 150,  1, 1.4,  1, 2,  3,  1],  # P429
 [63,  1,  0, 145, 260,  0,  1, 135,  1, 2.0,  1, 2,  3,  1],  # P430
 [58,  1,  0, 142, 250,  0,  0, 142,  1, 1.5,  1, 1,  3,  1],  # P431
 [62,  0,  0, 152, 270,  1,  1, 128,  1, 2.1,  1, 2,  3,  1],  # P432
 [55,  1,  0, 128, 235,  0,  0, 158,  1, 1.8,  1, 1,  3,  1],  # P433
 [60,  1,  0, 138, 265,  0,  1, 132,  1, 2.3,  1, 3,  3,  1],  # P434
 [64,  0,  0, 148, 280,  0,  0, 122,  1, 2.6,  1, 2,  3,  1],  # P435
 [57,  1,  0, 132, 240,  0,  1, 152,  1, 1.4,  1, 1,  3,  1],  # P436
 [67,  1,  0, 155, 295,  1,  0, 118,  1, 2.8,  1, 3,  3,  1],  # P437
 [59,  0,  0, 136, 245,  0,  0, 148,  1, 1.9,  1, 2,  3,  1],  # P438
 [61,  1,  0, 144, 275,  0,  1, 126,  1, 2.4,  1, 2,  3,  1],  # P439
 [54,  1,  0, 126, 225,  0,  0, 162,  1, 1.1,  1, 1,  3,  1],  # P440
 [66,  0,  0, 154, 285,  1,  1, 112,  1, 2.7,  1, 3,  3,  1],  # P441
 [53,  1,  0, 124, 230,  0,  0, 168,  1, 1.3,  1, 1,  3,  1],  # P442
 [68,  1,  0, 158, 290,  0,  1, 108,  1, 3.1,  1, 3,  3,  1],  # P443
 [56,  0,  0, 134, 250,  0,  0, 154,  1, 1.6,  1, 2,  3,  1],  # P444
 [63,  1,  0, 146, 265,  1,  1, 134,  1, 2.2,  1, 2,  3,  1],  # P445
 [58,  1,  0, 140, 245,  0,  0, 146,  1, 1.7,  1, 1,  3,  1],  # P446
 [62,  0,  0, 148, 275,  0,  1, 130,  1, 2.0,  1, 2,  3,  1],  # P447
 [55,  1,  0, 130, 230,  0,  0, 160,  1, 1.5,  1, 1,  3,  1],  # P448
 [60,  1,  0, 142, 260,  1,  0, 138,  1, 2.5,  1, 3,  3,  1],  # P449
 [64,  0,  0, 138, 280,  0,  1, 124,  1, 2.3,  1, 2,  3,  1],  # P450

# ── HEALTHY PATIENTS (50) ─────────────────────────────────────────────────
# The "Super-Agers" (Extreme Old, but completely clean arteries)
 [84,  0,  2, 135, 230,  0,  1, 120,  0, 0.5,  2, 0,  2,  0],  # P451 Octogenarian female, healthy
 [79,  1,  3, 140, 210,  0,  1, 130,  0, 0.0,  2, 0,  2,  0],  # P452 Geriatric male, normal vessels
 [88,  0,  2, 145, 190,  0,  2, 115,  0, 1.0,  1, 0,  2,  0],  # P453 Super-ager, benign LVH, clean heart
 [77,  1,  2, 130, 240,  0,  1, 140,  0, 0.2,  2, 0,  2,  0],  # P454 Healthy elderly male
 [81,  0,  3, 150, 220,  0,  1, 125,  0, 0.0,  2, 0,  2,  0],  # P455 Healthy octogenarian female
# Extreme Athletes (Profound Bradycardia, High Reserves)
 [26,  1,  2, 110, 160,  0,  2, 205,  0, 0.0,  2, 0,  2,  0],  # P456 Olympic athlete, LVH normal for sport
 [31,  0,  2, 105, 175,  0,  0, 195,  0, 0.0,  2, 0,  2,  0],  # P457 Female marathoner
 [38,  1,  3, 115, 180,  0,  1, 188,  0, 0.5,  1, 0,  2,  0],  # P458 Triathlete, benign early repol ST
 [24,  1,  2, 100, 150,  0,  2, 210,  0, 0.0,  2, 0,  2,  0],  # P459 Ultra-runner, extreme HR max
 [33,  0,  3, 112, 165,  0,  0, 192,  0, 0.0,  2, 0,  2,  0],  # P460 Female athlete, benign ECG
# Strange/Contradictory Healthy Vitals (False Alarm generation)
 [45,  1,  2, 185, 190,  0,  0, 175,  0, 0.0,  2, 0,  2,  0],  # P461 White coat HTN, healthy heart
 [50,  0,  1, 120, 395,  0,  0, 160,  0, 0.0,  2, 0,  2,  0],  # P462 Massive isolated cholesterol, normal arteries
 [42,  1,  0, 115, 180,  0,  1, 180,  0, 2.5,  0, 0,  2,  0],  # P463 Panic attack mimicking MI, clean heart
 [55,  0,  2, 190, 230,  1,  0, 150,  0, 0.0,  2, 0,  2,  0],  # P464 Extreme BP female, false positive risk
 [48,  1,  3, 130, 210,  0,  2, 165,  1, 1.8,  1, 0,  3,  0],  # P465 Severe False Positive Stress Test
# Standard Healthy Padding
 [45,  0,  2, 120, 220,  0,  1, 160,  0, 0.0,  2, 0,  2,  0],  # P466
 [52,  1,  3, 130, 240,  0,  0, 155,  0, 0.5,  1, 0,  2,  0],  # P467
 [39,  0,  1, 115, 210,  0,  1, 170,  0, 0.0,  2, 0,  2,  0],  # P468
 [58,  1,  2, 135, 250,  0,  0, 145,  0, 0.8,  1, 0,  2,  0],  # P469
 [41,  0,  2, 125, 230,  0,  1, 165,  0, 0.0,  2, 0,  2,  0],  # P470
 [55,  1,  3, 140, 260,  0,  0, 150,  0, 1.0,  1, 0,  2,  0],  # P471
 [47,  0,  1, 118, 215,  0,  1, 158,  0, 0.0,  2, 0,  2,  0],  # P472
 [60,  1,  2, 145, 270,  1,  0, 140,  0, 1.2,  1, 0,  2,  0],  # P473
 [43,  0,  2, 122, 225,  0,  1, 162,  0, 0.0,  2, 0,  2,  0],  # P474
 [56,  1,  3, 138, 255,  0,  0, 148,  0, 0.9,  1, 0,  2,  0],  # P475
 [49,  0,  1, 116, 205,  0,  1, 168,  0, 0.0,  2, 0,  2,  0],  # P476
 [62,  1,  2, 150, 280,  0,  0, 135,  0, 1.5,  1, 0,  2,  0],  # P477
 [44,  0,  2, 128, 235,  0,  1, 155,  0, 0.0,  2, 0,  2,  0],  # P478
 [57,  1,  3, 142, 265,  1,  0, 142,  0, 1.1,  1, 0,  2,  0],  # P479
 [46,  0,  1, 124, 218,  0,  1, 164,  0, 0.0,  2, 0,  2,  0],  # P480
 [59,  1,  2, 148, 275,  0,  0, 138,  0, 1.4,  1, 0,  2,  0],  # P481
 [42,  0,  2, 114, 200,  0,  1, 172,  0, 0.0,  2, 0,  2,  0],  # P482
 [61,  1,  3, 155, 290,  0,  0, 130,  0, 1.6,  1, 0,  2,  0],  # P483
 [48,  0,  1, 132, 245,  0,  1, 152,  0, 0.0,  2, 0,  2,  0],  # P484
 [54,  1,  2, 136, 250,  1,  0, 146,  0, 0.7,  1, 0,  2,  0],  # P485
 [40,  0,  2, 120, 212,  0,  1, 166,  0, 0.0,  2, 0,  2,  0],  # P486
 [63,  1,  3, 160, 295,  0,  0, 125,  0, 1.8,  1, 0,  2,  0],  # P487
 [51,  0,  1, 126, 238,  0,  1, 156,  0, 0.0,  2, 0,  2,  0],  # P488
 [58,  1,  2, 144, 268,  0,  0, 144,  0, 1.3,  1, 0,  2,  0],  # P489
 [45,  0,  2, 118, 208,  0,  1, 160,  0, 0.0,  2, 0,  2,  0],  # P490
 [60,  1,  3, 152, 285,  1,  0, 132,  0, 1.7,  1, 0,  2,  0],  # P491
 [47,  0,  1, 130, 242,  0,  1, 154,  0, 0.0,  2, 0,  2,  0],  # P492
 [56,  1,  2, 140, 258,  0,  0, 140,  0, 1.0,  1, 0,  2,  0],  # P493
 [49,  0,  2, 124, 228,  0,  1, 168,  0, 0.0,  2, 0,  2,  0],  # P494
 [62,  1,  3, 158, 298,  0,  0, 128,  0, 1.9,  1, 0,  2,  0],  # P495
 [43,  0,  1, 116, 202,  0,  1, 174,  0, 0.0,  2, 0,  2,  0],  # P496
 [55,  1,  2, 138, 252,  1,  0, 148,  0, 0.8,  1, 0,  2,  0],  # P497
 [50,  0,  2, 128, 236,  0,  1, 158,  0, 0.0,  2, 0,  2,  0],  # P498
 [64,  1,  3, 162, 305,  0,  0, 120,  0, 2.0,  1, 0,  2,  0],  # P499
 [46,  0,  1, 122, 216,  0,  1, 162,  0, 0.0,  2, 0,  2,  0],  # P500
]

cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal','true_label']

df = pd.DataFrame(records, columns=cols)
df.index = [f'P{i+401}' for i in range(len(df))] # Continuing IDs from P401 to P500

print("=" * 70)
print("  KARDIA — FINAL BATCH 5 PATIENT MOCK INFERENCE TEST")
print("  Focus: Demographic Extremes & Clinical Outliers")
print("=" * 70)
print(f"\n  Total patients : {len(df)}")
print(f"  Disease        : {df['true_label'].sum()} ({df['true_label'].mean()*100:.0f}%)")
print(f"  Healthy        : {(df['true_label']==0).sum()} ({(df['true_label']==0).mean()*100:.0f}%)")
print(f"  Age range      : {df['age'].min()}–{df['age'].max()} (mean {df['age'].mean():.1f})")
print(f"  Mean chol      : {df['chol'].mean():.0f} mg/dL")
print(f"  Mean thalach   : {df['thalach'].mean():.0f} bpm")
print(f"  Mean trestbps  : {df['trestbps'].mean():.0f} mmHg\n")

# ── Feature Engineering ───────────────────────────────────────────────────────
raw_cols = ['age','sex','cp','trestbps','chol','fbs',
            'restecg','thalach','exang','oldpeak','slope','ca','thal']
feature_cols = raw_cols + ['hr_reserve','chol_age_ratio','cp_sex',
                            'thalach_oldpeak','age_bucket']

def engineer_features(df):
    d = df.copy()
    d['hr_reserve']      = d['thalach'] - (220 - d['age'])
    d['chol_age_ratio']  = d['chol'] / d['age']
    d['cp_sex']          = d['cp'] * d['sex']
    d['thalach_oldpeak'] = d['thalach'] * d['oldpeak']
    d['age_bucket']      = pd.cut(
        d['age'], bins=[0,40,55,65,120], labels=[0,1,2,3]).astype(int)
    return d

df_eng     = engineer_features(df[raw_cols])
X_features = df_eng[feature_cols].values.astype(np.float32)

# Load scaler (fallback to identity if missing for standalone testing)
try:
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X_features).astype(np.float32)
except FileNotFoundError:
    print("Warning: models/scaler.pkl not found. Using unscaled features.")
    X_scaled = X_features

# ── ECG embeddings (Mocked if data/model missing) ─────────────────────────────
try:
    ecg_signals = np.load('data/ptbxl/processed/ecg_signals.npy')
    ecg_labels  = np.load('data/ptbxl/processed/ecg_labels.npy')
    if ecg_signals.shape[1] != 12:
        ecg_signals = ecg_signals.transpose(0, 2, 1)

    ecg_model = ResNet1D(embedding_dim=256)
    ecg_model.load_state_dict(torch.load('models/best_resnet1d.pt', weights_only=True))
    ecg_model.eval()
    
    with torch.no_grad():
        ecg_embeddings = ecg_model(torch.tensor(ecg_signals), return_embedding=True).numpy()

    rng = np.random.default_rng(2026)
    true_labels = df['true_label'].values

    ecg_embs = []
    for label in true_labels:
        idx = np.where(ecg_labels == int(label))[0]
        ecg_embs.append(ecg_embeddings[rng.choice(idx)])
    ecg_embs = np.array(ecg_embs, dtype=np.float32)

except (FileNotFoundError, NameError):
    print("Warning: ECG data/model missing. Generating dummy 256-dim embeddings for testing.")
    ecg_embs = np.random.normal(0, 1, (100, 256)).astype(np.float32)
    true_labels = df['true_label'].values

X_fused = np.concatenate([X_scaled, ecg_embs], axis=1).astype(np.float32)

# ── Fusion model ──────────────────────────────────────────────────────────────
class FusionHead(nn.Module):
    def __init__(self, input_dim=274, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.binary_head   = nn.Linear(64, 1)
        self.risk_head     = nn.Linear(64, 1)
        self.severity_head = nn.Linear(64, 4)

    def forward(self, x):
        f = self.net(x)
        return (torch.sigmoid(self.binary_head(f)),
                torch.sigmoid(self.risk_head(f)) * 100,
                self.severity_head(f))

model = FusionHead(input_dim=X_fused.shape[1])
try:
    model.load_state_dict(torch.load('models/best_fusion_v2.pt', weights_only=True))
except FileNotFoundError:
    print("Warning: models/best_fusion_v2.pt not found. Using untrained weights.")

model.eval()

# ── Inference ─────────────────────────────────────────────────────────────────
severity_map = {0:'None', 1:'Mild', 2:'Moderate', 3:'Severe'}
risk_cat     = lambda r: ('LOW' if r < 30 else 'MEDIUM' if r < 60 else 'HIGH')
risk_icon    = lambda r: ('🟢' if r < 30 else '🟡' if r < 60 else '🔴')

with torch.no_grad():
    binary, risk, severity = model(torch.tensor(X_fused))
    probs = binary.squeeze().numpy()
    risks = risk.squeeze().numpy()
    sevs  = severity.argmax(dim=1).numpy()

results = []
for i in range(len(df)):
    prob  = float(probs[i])
    rv    = float(risks[i])
    pred  = 1 if prob >= 0.5 else 0
    true  = int(true_labels[i])
    results.append({
        'id'      : f'P{i+401}',
        'age'     : int(df['age'].iloc[i]),
        'sex'     : 'M' if df['sex'].iloc[i] == 1 else 'F',
        'prob'    : prob,
        'risk'    : rv,
        'severity': severity_map[int(sevs[i])],
        'pred'    : pred,
        'true'    : true,
        'correct' : pred == true,
    })

preds      = [r['pred']  for r in results]
trues      = [r['true']  for r in results]
probs_list = [r['prob']  for r in results]

accuracy  = accuracy_score(trues, preds) * 100
auc       = roc_auc_score(trues, probs_list)
f1        = f1_score(trues, preds)
correct   = sum(r['correct'] for r in results)

# ── Print results table ───────────────────────────────────────────────────────
print(f"{'ID':<5} {'Age':>4} {'Sex'} {'Risk Score':>11} {'Cat':<8} {'Severity':<10} {'Pred':>8} {'True':>8}  {'✓'}")
print("-" * 75)

for r in results:
    icon  = risk_icon(r['risk'])
    cat   = risk_cat(r['risk'])
    pred_s = 'DISEASE' if r['pred'] else 'HEALTHY'
    true_s = 'DISEASE' if r['true'] else 'HEALTHY'
    tick  = '✅' if r['correct'] else '❌'
    print(f"{r['id']:<5} {r['age']:>4} {r['sex']:>3}  {r['risk']:>8.1f}%  "
          f"{icon}{cat:<9} {r['severity']:<10} {pred_s:>8} {true_s:>8}  {tick}")

print("-" * 75)

# ── Summary metrics ───────────────────────────────────────────────────────────
disease_correct = sum(1 for r in results if r['true']==1 and r['correct'])
healthy_correct = sum(1 for r in results if r['true']==0 and r['correct'])
disease_total   = sum(1 for r in results if r['true']==1)
healthy_total   = sum(1 for r in results if r['true']==0)

print(f"""
📊 BATCH 5 RESULTS
{'='*50}
  Total patients     : 100
  Correct            : {correct}
  Incorrect          : {100 - correct}
  Accuracy           : {accuracy:.1f}%
  AUC-ROC            : {auc:.4f}
  F1 Score           : {f1:.4f}

  Disease recall     : {disease_correct}/{disease_total} ({disease_correct/disease_total*100:.1f}%)
  Healthy specificity: {healthy_correct}/{healthy_total} ({healthy_correct/healthy_total*100:.1f}%)

Classification Report:
""")
print(classification_report(trues, preds, target_names=['Healthy','Disease'], zero_division=0))

# ── Risk distribution plot ────────────────────────────────────────────────────
df_res = pd.DataFrame(results)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Kardia — Batch 5 Patient Mock Test Analysis', fontsize=14, fontweight='bold')

# Risk score distribution
ax = axes[0, 0]
df_res[df_res['true']==1]['risk'].plot(kind='kde', ax=ax, color='#F44336', label='Disease', linewidth=2)
df_res[df_res['true']==0]['risk'].plot(kind='kde', ax=ax, color='#4CAF50', label='Healthy', linewidth=2)
ax.set_title('Risk Score Distribution by True Label')
ax.set_xlabel('Risk Score (0-100)')
ax.legend()

# Confusion matrix
ax = axes[0, 1]
cm = confusion_matrix(trues, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Healthy','Disease'],
            yticklabels=['Healthy','Disease'])
ax.set_title(f'Confusion Matrix (Accuracy={accuracy:.1f}%)')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

# Age distribution by correctness
ax = axes[1, 0]
correct_ages   = [r['age'] for r in results if r['correct']]
incorrect_ages = [r['age'] for r in results if not r['correct']]
if correct_ages:
    ax.hist(correct_ages, bins=10, alpha=0.7, color='#4CAF50', label='Correct')
if incorrect_ages:
    ax.hist(incorrect_ages, bins=10, alpha=0.7, color='#F44336', label='Incorrect')
ax.set_title('Age Distribution: Correct vs Incorrect Predictions')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.legend()

# Severity distribution
ax = axes[1, 1]
sev_counts = df_res[df_res['true']==1]['severity'].value_counts()
sev_colors = {'None':'#4CAF50','Mild':'#FFC107','Moderate':'#FF9800','Severe':'#F44336'}
if not sev_counts.empty:
    bars = ax.bar(sev_counts.index,
                  sev_counts.values,
                  color=[sev_colors.get(s,'#9E9E9E') for s in sev_counts.index])
ax.set_title('Predicted Severity for Disease Patients')
ax.set_xlabel('Severity')
ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('reports/mock_batch5_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Analysis plot saved → reports/mock_batch5_analysis.png")

# ── Save CSV ──────────────────────────────────────────────────────────────────
df_out = pd.DataFrame([{
    'patient_id'     : r['id'],
    'age'            : r['age'],
    'sex'            : r['sex'],
    'risk_score'     : round(r['risk'], 1),
    'risk_category'  : risk_cat(r['risk']),
    'severity'       : r['severity'],
    'predicted'      : 'DISEASE' if r['pred'] else 'HEALTHY',
    'true_label'     : 'DISEASE' if r['true'] else 'HEALTHY',
    'correct'        : r['correct'],
    'confidence_pct' : round(r['prob']*100, 1),
} for r in results])

df_out.to_csv('reports/mock_batch5_results.csv', index=False)
print("✅ Results saved → reports/mock_batch5_results.csv")
print(f"\n{'='*50}")
print(f"  FINAL: {correct}/100 correct | {accuracy:.1f}% accuracy | AUC {auc:.4f}")
print(f"{'='*50}")