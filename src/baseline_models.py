import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (accuracy_score, roc_auc_score,
                                     f1_score, classification_report,
                                     confusion_matrix, RocCurveDisplay)
from xgboost                 import XGBClassifier

# ── 1. Load processed data ────────────────────────────────────────────────────
X_train = pd.read_csv('data/tabular/processed/X_train.csv')
X_val   = pd.read_csv('data/tabular/processed/X_val.csv')
X_test  = pd.read_csv('data/tabular/processed/X_test.csv')
y_train = pd.read_csv('data/tabular/processed/y_train.csv').values.ravel()
y_val   = pd.read_csv('data/tabular/processed/y_val.csv').values.ravel()
y_test  = pd.read_csv('data/tabular/processed/y_test.csv').values.ravel()

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

os.makedirs('reports', exist_ok=True)
os.makedirs('models',  exist_ok=True)

# ── 2. Define models ──────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=200,
                                                   random_state=42, n_jobs=-1),
    'XGBoost'            : XGBClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=4, random_state=42,
                                          eval_metric='logloss',
                                          verbosity=0)
}

# ── 3. Train, evaluate, store results ────────────────────────────────────────
results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"  Training: {name}")
    print(f"{'='*50}")

    model.fit(X_train, y_train)

    # Validation metrics
    val_preds  = model.predict(X_val)
    val_probs  = model.predict_proba(X_val)[:, 1]
    val_auc    = roc_auc_score(y_val, val_probs)
    val_acc    = accuracy_score(y_val, val_preds)
    val_f1     = f1_score(y_val, val_preds)

    # Test metrics
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]
    test_auc   = roc_auc_score(y_test, test_probs)
    test_acc   = accuracy_score(y_test, test_preds)
    test_f1    = f1_score(y_test, test_preds)

    results[name] = {
        'val_auc' : val_auc,  'val_acc' : val_acc,  'val_f1' : val_f1,
        'test_auc': test_auc, 'test_acc': test_acc, 'test_f1': test_f1,
        'model'   : model,
        'test_preds': test_preds,
        'test_probs': test_probs
    }

    print(f"  Val  → AUC: {val_auc:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
    print(f"  Test → AUC: {test_auc:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    print(f"\n  Classification Report (Test):")
    print(classification_report(y_test, test_preds,
                                target_names=['No Disease','Disease']))

    # Save model
    with open(f'models/{name.lower().replace(" ","_")}.pkl', 'wb') as f:
        pickle.dump(model, f)

# ── 4. Summary table ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  BASELINE RESULTS SUMMARY")
print(f"{'='*60}")
summary = pd.DataFrame({
    name: {
        'Val AUC' : f"{r['val_auc']:.4f}",
        'Val Acc' : f"{r['val_acc']:.4f}",
        'Val F1'  : f"{r['val_f1']:.4f}",
        'Test AUC': f"{r['test_auc']:.4f}",
        'Test Acc': f"{r['test_acc']:.4f}",
        'Test F1' : f"{r['test_f1']:.4f}",
    }
    for name, r in results.items()
}).T
print(summary.to_string())
summary.to_csv('reports/baseline_results.csv')

# ── 5. ROC Curves ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for name, r in results.items():
    RocCurveDisplay.from_predictions(
        y_test, r['test_probs'], name=f"{name} (AUC={r['test_auc']:.3f})", ax=ax)
ax.plot([0,1],[0,1],'k--', label='Random')
ax.set_title('ROC Curves — Baseline Models')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('reports/roc_curves_baseline.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ ROC curve saved")

# ── 6. Confusion matrices ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r['test_preds'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap='Blues', cbar=False,
                xticklabels=['No Disease','Disease'],
                yticklabels=['No Disease','Disease'])
    ax.set_title(f"{name}\nAUC={r['test_auc']:.3f}")
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('reports/confusion_matrices_baseline.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrices saved")

# ── 7. Feature importance (XGBoost) ──────────────────────────────────────────
xgb_model = results['XGBoost']['model']
feat_imp   = pd.Series(xgb_model.feature_importances_,
                        index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='bar', color='steelblue')
plt.title('XGBoost Feature Importance')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('reports/xgboost_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Feature importance saved")
print(f"\nTop 5 features:\n{feat_imp.head()}")

print("\n✅ Phase 4 complete — all models trained and saved to models/")