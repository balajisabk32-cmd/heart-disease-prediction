import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
import pickle

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv('data/tabular/heart_uci.csv')
print(f"Loaded: {df.shape}")

# ── 2. KNN Imputation (handles missing values) ────────────────────────────────
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print(f"Missing after imputation: {df_imputed.isnull().sum().sum()}")

# ── 3. Feature Engineering ────────────────────────────────────────────────────
# HR Reserve: difference between max HR achieved and age-predicted max HR
df_imputed['hr_reserve']     = df_imputed['thalach'] - (220 - df_imputed['age'])

# Chol-Age ratio: cholesterol relative to age
df_imputed['chol_age_ratio'] = df_imputed['chol'] / df_imputed['age']

# Interaction: chest pain type × sex
df_imputed['cp_sex']         = df_imputed['cp'] * df_imputed['sex']

# Interaction: max HR × ST depression
df_imputed['thalach_oldpeak']= df_imputed['thalach'] * df_imputed['oldpeak']

# Age bucket: young=0, mid=1, senior=2, elderly=3
df_imputed['age_bucket']     = pd.cut(
    df_imputed['age'],
    bins=[0, 40, 55, 65, 120],
    labels=[0, 1, 2, 3]
).astype(int)

print(f"Features after engineering: {df_imputed.shape[1]}")
print(f"New features: hr_reserve, chol_age_ratio, cp_sex, thalach_oldpeak, age_bucket")

# ── 4. Split features and target ─────────────────────────────────────────────
X = df_imputed.drop(columns=['target'])
y = df_imputed['target'].astype(int)

print(f"\nClass distribution before SMOTE:")
print(y.value_counts())

# ── 5. Train / Val / Test split (70 / 15 / 15) ───────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nSplit sizes:")
print(f"  Train : {X_train.shape[0]} samples")
print(f"  Val   : {X_val.shape[0]} samples")
print(f"  Test  : {X_test.shape[0]} samples")

# ── 6. SMOTE on training set only ────────────────────────────────────────────
# Categorical feature indices (sex, cp, fbs, restecg, exang, slope, ca, thal,
# cp_sex, age_bucket) — SMOTENC handles mixed types
cat_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal',
            'cp_sex','age_bucket']
cat_idx  = [X_train.columns.get_loc(c) for c in cat_cols]

smote = SMOTENC(categorical_features=cat_idx, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"\nClass distribution after SMOTE (train only):")
print(pd.Series(y_train_bal).value_counts())

# ── 7. Scale features ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# ── 8. Save everything ───────────────────────────────────────────────────────
os.makedirs('data/tabular/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save splits as CSVs
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
    'data/tabular/processed/X_train.csv', index=False)
pd.DataFrame(X_val_scaled,   columns=X.columns).to_csv(
    'data/tabular/processed/X_val.csv',   index=False)
pd.DataFrame(X_test_scaled,  columns=X.columns).to_csv(
    'data/tabular/processed/X_test.csv',  index=False)

pd.Series(y_train_bal).to_csv('data/tabular/processed/y_train.csv', index=False)
pd.Series(y_val).to_csv(      'data/tabular/processed/y_val.csv',   index=False)
pd.Series(y_test).to_csv(     'data/tabular/processed/y_test.csv',  index=False)

# Save scaler and imputer for later inference
with open('models/scaler.pkl',  'wb') as f: pickle.dump(scaler,  f)
with open('models/imputer.pkl', 'wb') as f: pickle.dump(imputer, f)

print("\n✅ All files saved to data/tabular/processed/")
print("✅ Scaler and imputer saved to models/")
print(f"\nFinal training shape : {X_train_scaled.shape}")
print(f"Final val shape      : {X_val_scaled.shape}")
print(f"Final test shape     : {X_test_scaled.shape}")
print(f"Feature names        : {list(X.columns)}")