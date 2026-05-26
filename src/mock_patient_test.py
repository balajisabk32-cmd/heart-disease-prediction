"""
Kardia — 100 Patient Mock Inference Test
========================================
Patient data distributions based on real UCI Cleveland Heart Disease statistics:
  - Mean age: 54.4 (range 29-77)
  - Mean trestbps: 132 mmHg (range 94-200)
  - Mean chol: 247 mg/dL (range 126-564)
  - Mean thalach: 150 bpm (range 71-202)
  - Mean oldpeak: 1.04 (range 0-6.2)
  - Disease rate: 54.5% (165/303)
  - Female disease rate: 72%, Male disease rate: 42%

Source: UCI Cleveland Heart Disease Dataset (303 patients, 1988)
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

sys.path.append('src')
from resnet1d import ResNet1D

os.makedirs('reports', exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# 100 PATIENTS — based on real UCI Cleveland clinical distributions
# Columns: age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,
#          oldpeak,slope,ca,thal,true_label
#
# sex  : 1=Male, 0=Female
# cp   : 0=typical angina,1=atypical,2=non-anginal,3=asymptomatic
# thal : 1=normal,2=fixed defect,3=reversable defect
# slope: 0=upsloping,1=flat,2=downsloping
# ════════════════════════════════════════════════════════════════════════════

records = [
# ── DISEASE PATIENTS (55) ─────────────────────────────────────────────────
# age  sex cp  bp   chol fbs re  hr   ex  op   sl ca th  lbl
 [63,  1,  0, 145, 233,  1,  0, 150,  0, 2.3,  0, 0,  1,  1],  # P1  Classic angina, diabetic
 [67,  1,  0, 160, 286,  0,  0, 108,  1, 1.5,  1, 3,  3,  1],  # P2  HTN, 3-vessel CAD
 [67,  1,  0, 120, 229,  0,  0, 129,  1, 2.6,  1, 2,  3,  1],  # P3  Reversible defect
 [62,  1,  0, 140, 268,  0,  0, 160,  0, 3.6,  0, 2,  3,  1],  # P4  ST depression
 [60,  1,  0, 130, 253,  0,  1, 144,  1, 1.4,  2, 1,  3,  1],  # P5  Flat slope
 [63,  0,  0, 150, 407,  0,  0, 154,  0, 4.0,  1, 3,  3,  1],  # P6  Very high chol
 [59,  1,  0, 134, 204,  0,  1, 162,  0, 0.8,  2, 2,  3,  1],  # P7  Multi-vessel
 [53,  1,  0, 123, 282,  0,  1, 195,  1, 1.0,  1, 0,  3,  1],  # P8  Exertional
 [57,  1,  0, 150, 276,  0,  0, 112,  1, 0.6,  1, 1,  1,  1],  # P9  Low HR
 [56,  1,  2, 130, 256,  1,  0, 142,  1, 0.6,  1, 1,  1,  1],  # P10 Diabetic
 [65,  1,  0, 110, 248,  0,  0, 158,  0, 0.6,  2, 2,  1,  1],  # P11 Elderly male
 [65,  0,  0, 150, 225,  0,  0, 114,  0, 2.6,  0, 0,  3,  1],  # P12 Elderly female
 [55,  1,  0, 160, 289,  0,  0, 145,  1, 0.8,  1, 1,  3,  1],  # P13 HTN angina
 [46,  1,  0, 120, 249,  0,  0, 144,  0, 0.8,  2, 0,  3,  1],  # P14 Young CAD
 [54,  1,  0, 122, 286,  0,  0, 116,  1, 3.2,  1, 2,  3,  1],  # P15 Severe ischemia
 [71,  0,  0, 112, 149,  0,  1, 125,  0, 1.6,  1, 0,  3,  1],  # P16 Elderly female
 [43,  1,  0, 132, 247,  1,  0, 143,  1, 0.1,  1, 0,  3,  1],  # P17 Young diabetic
 [34,  1,  0, 118, 182,  0,  0, 174,  0, 0.0,  2, 0,  2,  1],  # P18 Very young MI
 [57,  0,  0, 140, 241,  0,  1, 123,  1, 0.2,  1, 0,  3,  1],  # P19 Female angina
 [52,  1,  0, 128, 255,  0,  1, 161,  1, 0.0,  2, 1,  3,  1],  # P20 Exertional
 [51,  1,  0, 140, 261,  0,  0, 186,  1, 0.0,  2, 0,  3,  1],  # P21 High HR
 [45,  1,  0, 115, 260,  0,  0, 185,  0, 0.0,  2, 0,  3,  1],  # P22 Young angina
 [53,  0,  0, 138, 234,  0,  0, 160,  0, 0.0,  2, 0,  3,  1],  # P23 Female
 [58,  1,  0, 128, 216,  0,  0, 131,  1, 2.2,  1, 3,  3,  1],  # P24 3-vessel
 [46,  1,  0, 120, 249,  0,  0, 144,  0, 0.8,  2, 0,  3,  1],  # P25 Fixed defect
 [48,  1,  0, 122, 222,  0,  0, 186,  0, 0.0,  2, 0,  2,  1],  # P26 Young reversible
 [60,  1,  0, 145, 282,  0,  0, 142,  1, 2.8,  1, 2,  3,  1],  # P27 HTN
 [64,  1,  0, 128, 263,  0,  1, 105,  1, 0.2,  1, 1,  3,  1],  # P28 Elderly
 [66,  1,  0, 120, 302,  0,  0, 151,  0, 0.4,  1, 0,  3,  1],  # P29 HTN
 [59,  1,  0, 138, 271,  0,  0, 182,  0, 0.0,  2, 0,  3,  1],  # P30 Asymptomatic CAD
 [44,  1,  0, 130, 233,  0,  1, 179,  1, 0.4,  2, 0,  3,  1],  # P31 Young angina
 [61,  1,  0, 148, 203,  0,  1, 161,  0, 0.0,  2, 1,  3,  1],  # P32 Multi-vessel
 [58,  1,  0, 114, 318,  0,  2, 140,  0, 4.4,  0, 3,  1,  1],  # P33 High chol CAD
 [70,  1,  0, 130, 322,  0,  0, 109,  0, 2.4,  1, 3,  3,  1],  # P34 Elderly severe
 [62,  0,  0, 138, 294,  1,  1, 106,  0, 1.9,  1, 3,  3,  1],  # P35 Female diabetic
 [57,  1,  0, 110, 335,  0,  1, 143,  1, 3.0,  1, 1,  3,  1],  # P36 Angina+HTN
 [74,  0,  0, 120, 269,  0,  0, 121,  1, 0.2,  2, 1,  3,  1],  # P37 Very elderly F
 [63,  1,  0, 130, 254,  0,  0, 147,  0, 1.4,  1, 1,  3,  1],  # P38 Elderly angina
 [55,  1,  0, 132, 353,  0,  1, 132,  1, 1.2,  1, 1,  3,  1],  # P39 High chol
 [68,  1,  0, 144, 193,  1,  1, 141,  0, 3.4,  1, 2,  3,  1],  # P40 Diabetic severe
 [57,  1,  0, 154, 232,  0,  0, 164,  0, 0.0,  2, 1,  3,  1],  # P41 HTN angina
 [76,  0,  0, 140, 197,  0,  2, 116,  0, 1.1,  1, 0,  3,  1],  # P42 Very elderly F
 [69,  1,  0, 160, 234,  1,  0, 131,  0, 0.1,  1, 1,  3,  1],  # P43 Elderly diabetic
 [63,  0,  0, 135, 252,  0,  0, 172,  0, 0.0,  2, 0,  3,  1],  # P44 Female typical
 [42,  1,  0, 136, 315,  0,  1, 125,  1, 1.8,  1, 0,  1,  1],  # P45 Young angina
 [67,  0,  0, 106, 223,  0,  1, 142,  0, 0.3,  2, 2,  3,  1],  # P46 Female elderly
 [60,  1,  0, 117, 230,  1,  1, 160,  1, 1.4,  2, 2,  3,  1],  # P47 Diabetic angina
 [56,  1,  0, 150, 213,  1,  0, 125,  1, 1.0,  1, 2,  3,  1],  # P48 Diabetic multi
 [58,  1,  0, 146, 218,  0,  0, 105,  0, 2.0,  1, 1,  3,  1],  # P49 Low HR
 [56,  1,  0, 130, 167,  0,  0, 114,  0, 0.0,  2, 1,  3,  1],  # P50 Low chol CAD
 [56,  0,  0, 200, 288,  1,  0, 133,  1, 4.0,  0, 2,  3,  1],  # P51 Very HTN female
 [67,  1,  0, 100, 299,  0,  0, 125,  1, 0.9,  1, 2,  3,  1],  # P52 Low BP CAD
 [62,  1,  0, 120, 267,  0,  1, 99,   1, 1.8,  1, 2,  3,  1],  # P53 Very low HR
 [47,  1,  0, 112, 204,  0,  1, 143,  0, 0.1,  2, 0,  3,  1],  # P54 Young CAD
 [52,  1,  0, 112, 230,  0,  1, 160,  0, 0.0,  2, 1,  2,  1],  # P55 Fixed defect

# ── HEALTHY PATIENTS (45) ─────────────────────────────────────────────────
 [29,  1,  1, 130, 204,  0,  0, 202,  0, 0.0,  2, 0,  2,  0],  # P56 Young healthy M
 [37,  1,  2, 130, 250,  0,  1, 187,  0, 3.5,  0, 0,  2,  0],  # P57 Athletic young
 [41,  0,  1, 130, 204,  0,  0, 172,  0, 1.4,  2, 0,  2,  0],  # P58 Young female
 [56,  1,  1, 120, 236,  0,  1, 178,  0, 0.8,  2, 0,  2,  0],  # P59 Normal ECG
 [57,  0,  0, 120, 354,  0,  1, 163,  1, 0.6,  2, 0,  2,  0],  # P60 High chol F
 [44,  1,  1, 120, 263,  0,  1, 173,  0, 0.0,  2, 0,  3,  0],  # P61 Mild atypical
 [52,  1,  2, 172, 199,  1,  1, 162,  0, 0.5,  2, 0,  3,  0],  # P62 High BP healthy
 [54,  1,  2, 125, 273,  0,  0, 152,  0, 0.5,  0, 1,  2,  0],  # P63 Non-anginal
 [35,  1,  0, 120, 198,  0,  1, 130,  1, 1.6,  1, 0,  3,  0],  # P64 Young normal
 [51,  1,  3, 125, 213,  0,  0, 125,  1, 1.4,  2, 1,  2,  0],  # P65 Asymptomatic
 [45,  0,  1, 112, 160,  0,  1, 138,  0, 0.0,  1, 0,  3,  0],  # P66 Young female
 [58,  0,  2, 120, 340,  0,  1, 172,  0, 0.0,  2, 0,  2,  0],  # P67 Female high chol
 [50,  0,  2, 120, 219,  0,  1, 158,  0, 1.6,  1, 0,  2,  0],  # P68 Female healthy
 [66,  0,  3, 150, 226,  0,  1, 114,  0, 2.6,  0, 0,  2,  0],  # P69 Elderly female
 [43,  1,  0, 150, 247,  0,  1, 171,  0, 1.5,  2, 0,  2,  0],  # P70 Normal slope
 [69,  0,  3, 140, 239,  0,  1, 151,  0, 1.8,  2, 2,  2,  0],  # P71 Elderly female
 [59,  1,  0, 135, 234,  0,  1, 161,  0, 0.5,  1, 0,  3,  0],  # P72 Normal thal
 [37,  1,  2, 130, 250,  0,  1, 187,  0, 3.5,  0, 0,  2,  0],  # P73 Low risk
 [40,  1,  3, 140, 199,  0,  1, 178,  1, 1.4,  2, 0,  3,  0],  # P74 Asymptomatic
 [41,  0,  1, 105, 198,  0,  1, 168,  0, 0.0,  2, 1,  2,  0],  # P75 Female non-ang
 [65,  0,  2, 140, 417,  1,  0, 157,  0, 0.8,  2, 1,  2,  0],  # P76 Elderly female
 [48,  0,  2, 130, 275,  0,  1, 139,  0, 0.2,  2, 0,  2,  0],  # P77 Female healthy
 [49,  1,  1, 130, 266,  0,  1, 171,  0, 0.6,  2, 0,  2,  0],  # P78 Normal
 [54,  1,  2, 150, 195,  0,  1, 122,  0, 0.6,  1, 0,  3,  0],  # P79 Normal thal
 [54,  0,  2, 135, 304,  1,  1, 170,  0, 0.0,  2, 0,  2,  0],  # P80 Female normal
 [58,  0,  2, 130, 197,  0,  1, 131,  0, 0.6,  1, 0,  2,  0],  # P81 Female mild
 [59,  1,  2, 150, 212,  1,  1, 157,  0, 1.6,  2, 0,  2,  0],  # P82 Diabetic healthy
 [51,  0,  2, 130, 256,  0,  0, 149,  0, 0.5,  2, 0,  2,  0],  # P83 Female normal
 [39,  1,  2, 94,  199,  0,  1, 179,  0, 0.0,  2, 0,  2,  0],  # P84 Young healthy
 [45,  1,  0, 110, 264,  0,  1, 132,  0, 1.2,  1, 0,  3,  0],  # P85 Low BP
 [68,  1,  2, 180, 274,  1,  0, 150,  1, 1.6,  1, 0,  3,  0],  # P86 HTN diabetic ok
 [57,  1,  2, 128, 229,  0,  0, 150,  0, 0.4,  1, 1,  3,  0],  # P87 Normal elderly
 [57,  0,  0, 140, 241,  0,  1, 123,  1, 0.2,  1, 0,  3,  0],  # P88 Female normal
 [38,  1,  2, 138, 175,  0,  1, 173,  0, 0.0,  2, 0,  2,  0],  # P89 Young healthy
 [62,  0,  2, 124, 209,  0,  1, 163,  0, 0.0,  2, 0,  2,  0],  # P90 Female normal
 [47,  1,  2, 108, 243,  0,  1, 152,  0, 0.0,  2, 0,  2,  0],  # P91 Normal ECG
 [55,  0,  1, 132, 342,  0,  1, 166,  0, 1.2,  2, 0,  2,  0],  # P92 Female healthy
 [35,  0,  0, 138, 183,  0,  1, 182,  0, 1.4,  2, 0,  2,  0],  # P93 Young female
 [46,  0,  2, 142, 177,  0,  0, 160,  1, 1.4,  0, 0,  2,  0],  # P94 Female healthy
 [50,  0,  2, 120, 244,  0,  1, 162,  0, 1.1,  2, 0,  2,  0],  # P95 Female normal
 [44,  0,  2, 108, 141,  0,  1, 175,  0, 0.6,  2, 0,  2,  0],  # P96 Female low chol
 [48,  0,  2, 120, 254,  0,  1, 154,  0, 0.7,  2, 0,  3,  0],  # P97 Female normal
 [54,  0,  2, 108, 267,  0,  0, 167,  0, 0.0,  2, 0,  2,  0],  # P98 Female healthy
 [64,  0,  2, 130, 303,  0,  1, 122,  0, 2.0,  1, 2,  2,  0],  # P99 Elderly female
 [57,  1,  2, 124, 261,  0,  1, 141,  0, 0.3,  2, 0,  3,  0],  # P100 Normal male
]

cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal','true_label']

df = pd.DataFrame(records, columns=cols)
df.index = [f'P{i+1}' for i in range(len(df))]

print("=" * 70)
print("  KARDIA — 100 PATIENT MOCK INFERENCE TEST")
print("  Based on real UCI Cleveland Heart Disease statistics")
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

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X_features).astype(np.float32)


# ── ECG embeddings ────────────────────────────────────────────────────────────
ecg_signals = np.load('data/ptbxl/processed/ecg_signals.npy')
ecg_labels  = np.load('data/ptbxl/processed/ecg_labels.npy')
if ecg_signals.shape[1] != 12:
    ecg_signals = ecg_signals.transpose(0, 2, 1)

ecg_model = ResNet1D(embedding_dim=256)
ecg_model.load_state_dict(
    torch.load('models/best_resnet1d.pt', weights_only=True))
ecg_model.eval()
for p in ecg_model.parameters():
    p.requires_grad = False

with torch.no_grad():
    ecg_embeddings = ecg_model(
        torch.tensor(ecg_signals), return_embedding=True).numpy()

rng = np.random.default_rng(2026)
true_labels = df['true_label'].values

ecg_embs = []
for label in true_labels:
    idx = np.where(ecg_labels == int(label))[0]
    ecg_embs.append(ecg_embeddings[rng.choice(idx)])
ecg_embs = np.array(ecg_embs, dtype=np.float32)

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
model.load_state_dict(
    torch.load('models/best_fusion_v2.pt', weights_only=True))
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
        'id'      : f'P{i+1}',
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
📊 100-PATIENT TEST RESULTS
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
print(classification_report(trues, preds,
      target_names=['Healthy','Disease']))

print(f"""
Comparison vs real test set:
  Real test AUC      : 0.9733
  Mock 100-pt AUC    : {auc:.4f}
  Real test Accuracy : 91.3%
  Mock 100-pt Acc    : {accuracy:.1f}%
""")


# ── Threshold sensitivity ─────────────────────────────────────────────────────
print("📈 THRESHOLD SENSITIVITY ANALYSIS")
print(f"{'Threshold':>10} {'Accuracy':>10} {'Recall':>10} {'Specificity':>13} {'F1':>8}")
print("-" * 55)
for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    pt   = [1 if p >= thresh else 0 for p in probs_list]
    ac   = accuracy_score(trues, pt) * 100
    rec  = sum(p==1 and t==1 for p,t in zip(pt,trues)) / disease_total * 100
    spec = sum(p==0 and t==0 for p,t in zip(pt,trues)) / healthy_total * 100
    f1t  = f1_score(trues, pt, zero_division=0)
    print(f"{thresh:>10.2f} {ac:>9.1f}% {rec:>9.1f}% {spec:>12.1f}% {f1t:>8.4f}")


# ── Risk distribution plot ────────────────────────────────────────────────────
df_res = pd.DataFrame(results)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Kardia — 100 Patient Mock Test Analysis', fontsize=14, fontweight='bold')

# Risk score distribution
ax = axes[0, 0]
df_res[df_res['true']==1]['risk'].plot(kind='kde', ax=ax,
    color='#F44336', label='Disease', linewidth=2)
df_res[df_res['true']==0]['risk'].plot(kind='kde', ax=ax,
    color='#4CAF50', label='Healthy', linewidth=2)
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
ax.hist(correct_ages,   bins=10, alpha=0.7, color='#4CAF50', label='Correct')
ax.hist(incorrect_ages, bins=10, alpha=0.7, color='#F44336', label='Incorrect')
ax.set_title('Age Distribution: Correct vs Incorrect Predictions')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.legend()

# Severity distribution
ax = axes[1, 1]
sev_counts = df_res[df_res['true']==1]['severity'].value_counts()
sev_colors = {'None':'#4CAF50','Mild':'#FFC107','Moderate':'#FF9800','Severe':'#F44336'}
bars = ax.bar(sev_counts.index,
              sev_counts.values,
              color=[sev_colors.get(s,'#9E9E9E') for s in sev_counts.index])
ax.set_title('Predicted Severity for Disease Patients')
ax.set_xlabel('Severity')
ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('reports/mock_100_patient_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Analysis plot saved → reports/mock_100_patient_analysis.png")


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

df_out.to_csv('reports/mock_100_patient_results.csv', index=False)
print("✅ Results saved → reports/mock_100_patient_results.csv")
print(f"\n{'='*50}")
print(f"  FINAL: {correct}/100 correct | {accuracy:.1f}% accuracy | AUC {auc:.4f}")
print(f"{'='*50}")