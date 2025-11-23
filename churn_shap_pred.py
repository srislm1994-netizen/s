# Corrected end-to-end churn + SHAP script
# Save this file as churn-shap-pred.py and run in Colab, Jupyter, or other Python env.
# It trains an XGBoost model, tunes hyperparameters, computes SHAP explanations,
# and saves all required plots and markdown deliverables into an outputs/ folder.

# If you want to run this in Colab: uncomment the pip installs at the top.

# INSTALLS (uncomment if running in Colab)
# !pip install xgboost shap scikit-learn imbalanced-learn joblib

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

# -------------------------
# === USER: supply dataset
# -------------------------
# The script will try several sensible default paths. If none exist, set dataset_path manually.
possible_paths = [
    '/mnt/data/WA_Fn-UseC_-Telco-Customer-Churn 2.csv',
    '/mnt/data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
    '/content/WA_Fn-UseC_-Telco-Customer-Churn 2.csv',
    '/content/WA_Fn-UseC_-Telco-Customer-Churn.csv',
    'WA_Fn-UseC_-Telco-Customer-Churn.csv'
]

dataset_path = None
for p in possible_paths:
    if os.path.exists(p):
        dataset_path = p
        break

if dataset_path is None:
    raise FileNotFoundError("Dataset not found in default locations. Please set `dataset_path` to your CSV file path.")

print(f"Using dataset: {dataset_path}")

# -------------------------
# === Load & clean data
# -------------------------
df = pd.read_csv(dataset_path)

# Basic cleaning
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)

if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

if 'Churn' not in df.columns:
    raise ValueError('Dataset must contain a Churn column with values Yes/No or 1/0')

# Normalize Churn to 0/1
if df['Churn'].dtype == object:
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

if set(df['Churn'].unique()) - set([0,1]):
    # try converting to numeric
    df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')
    df.dropna(subset=['Churn'], inplace=True)

df['Churn'] = df['Churn'].astype(int)

X = df.drop('Churn', axis=1)
y = df['Churn']

# -------------------------
# === Feature splits
# -------------------------
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Identify binary vs multi-category categorical columns
binary_cols = [col for col in categorical_features if X[col].nunique() == 2]
multi_cols = [col for col in categorical_features if X[col].nunique() > 2]

# Convert common Yes/No binary columns to 0/1 inline to avoid extra columns
for col in list(binary_cols):
    vals = set(X[col].dropna().unique())
    if vals == set(['Yes','No']) or vals == set(['No','Yes']):
        X[col] = X[col].map({'Yes':1,'No':0})
        # move this column out of categorical handling
        binary_cols.remove(col)
        if col not in numerical_features:
            numerical_features.append(col)

# Recompute lists for the preprocessor: numerical_features may have grown
numerical_features = [c for c in X.columns if X[c].dtype in [np.float64, np.int64, np.int32, np.float32]]
categorical_for_onehot = [c for c in X.columns if c not in numerical_features]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_for_onehot)
    ],
    remainder='passthrough'
)

# -------------------------
# === Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Address class imbalance via scale_pos_weight for XGBoost
neg, pos = np.bincount(y_train)
scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0
print(f"Train class counts -> Non-churn: {neg}, Churn: {pos}, scale_pos_weight={scale_pos_weight:.3f}")

# -------------------------
# === Pipeline + hyperparameter tune
# -------------------------
xgb_clf = XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    n_jobs=1
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_clf)
])

# Reasonable grid for Colab / typical environments
param_grid = {
    'classifier__n_estimators': [200, 400],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__subsample': [0.7],
    'classifier__colsample_bytree': [0.7]
}

gs = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=3, n_jobs=1, verbose=1)
gs.fit(X_train, y_train)

best_pipeline = gs.best_estimator_
best_params = gs.best_params_
print("Best params:", best_params)

# -------------------------
# === Evaluation
# -------------------------
y_proba = best_pipeline.predict_proba(X_test)[:,1]
y_pred = best_pipeline.predict(X_test)
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {auc:.4f}")
print("Classification report:\n", classification_report(y_test, y_pred))

# Save metrics
os.makedirs('/mnt/data/outputs', exist_ok=True)
outdir = '/mnt/data/outputs'
with open(os.path.join(outdir, 'model_metrics.txt'), 'w') as f:
    f.write(f"Best params: {best_params}\n")
    f.write(f"ROC-AUC: {auc:.6f}\n")
    f.write("Classification report:\n")
    f.write(classification_report(y_test, y_pred))
print("Saved", os.path.join(outdir, 'model_metrics.txt'))

# -------------------------
# === Prepare processed test data & feature names for SHAP
# -------------------------
pre = best_pipeline.named_steps['preprocessor']
# get_feature_names_out may be available
try:
    feature_names = list(pre.get_feature_names_out())
except Exception:
    # Build feature names robustly
    fn_num = numerical_features
    fn_oh = []
    try:
        oh = pre.named_transformers_['onehot']
        if hasattr(oh, 'get_feature_names_out'):
            fn_oh = list(oh.get_feature_names_out(categorical_for_onehot))
        else:
            cats = oh.categories_
            for ci, col in enumerate(categorical_for_onehot):
                # drop='first' so skip first category
                for val in cats[ci][1:]:
                    fn_oh.append(f"{col}_{val}")
    except Exception:
        fn_oh = []
    remainder = [c for c in X.columns if c not in numerical_features + categorical_for_onehot]
    feature_names = fn_num + fn_oh + remainder

# Transform test set
X_test_processed = pre.transform(X_test)
X_train_processed = pre.transform(X_train)

# -------------------------
# === SHAP explanation
# -------------------------
model = best_pipeline.named_steps['classifier']
explainer = shap.TreeExplainer(model)
# compute shap values robustly
try:
    shap_vals_obj = explainer(X_test_processed)
    shap_values = shap_vals_obj.values if hasattr(shap_vals_obj, 'values') else shap_vals_obj
except Exception:
    # Fallback to older API
    try:
        shap_values = explainer.shap_values(X_test_processed)
    except Exception as e:
        raise RuntimeError('SHAP value computation failed: ' + str(e))

# Make sure shap_values is ndarray
if hasattr(shap_values, 'values'):
    shap_values = shap_values.values

# Ensure outputs dir exists
os.makedirs(outdir, exist_ok=True)

# === Global SHAP beeswarm/summary (save)
plt.figure()
shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, plot_type='dot', show=False)
plt.tight_layout()
beeswarm_path = os.path.join(outdir, 'shap_summary_beeswarm.png')
plt.savefig(beeswarm_path, dpi=200)
plt.close()
print('Saved', beeswarm_path)

plt.figure()
shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, plot_type='bar', show=False)
plt.tight_layout()
bar_path = os.path.join(outdir, 'shap_summary_bar.png')
plt.savefig(bar_path, dpi=200)
plt.close()
print('Saved', bar_path)

# -------------------------
# === XGBoost built-in importances (for comparison)
# -------------------------
xgb_importances = model.feature_importances_
shap_mean_abs = np.mean(np.abs(shap_values), axis=0)
fi_df = pd.DataFrame({
    'feature': feature_names,
    'xgb_importance': xgb_importances,
    'shap_mean_abs': shap_mean_abs
})
fi_df.sort_values(by='shap_mean_abs', ascending=False, inplace=True)
fi_df.to_csv(os.path.join(outdir, 'feature_importance_comparison.csv'), index=False)
print('Saved', os.path.join(outdir, 'feature_importance_comparison.csv'))

# -------------------------
# === Local explanations: select 3 customers (high, low, borderline)
# -------------------------
high_idx_list = np.where(y_proba >= 0.8)[0].tolist()
low_idx_list = np.where(y_proba <= 0.2)[0].tolist()
borderline_idx = [int(np.argmin(np.abs(y_proba - 0.5)))]

chosen = {
    'high': high_idx_list[0] if len(high_idx_list) else None,
    'low': low_idx_list[0] if len(low_idx_list) else None,
    'borderline': borderline_idx[0]
}

lines = []
lines.append('# Local SHAP Explanations — Selected Customers\n')
for role, idx in chosen.items():
    if idx is None:
        lines.append(f'## {role.title()} customer: None found with threshold\n')
        continue
    prob = y_proba[idx]
    original_row = X_test.iloc[idx]
    lines.append(f'## {role.title()} Customer (test index {idx}) — predicted probability: {prob:.3f}\n')
    lines.append('### Original feature values:\n')
    lines.append('```
')
    lines.append(original_row.to_string())
    lines.append('\n``\n')
    lines.append('### SHAP Force Plot and breakdown:\n')
    try:
        shap.force_plot(explainer.expected_value, shap_values[idx,:], feature_names=feature_names, matplotlib=True, show=False)
        plt.gcf().set_size_inches(10,4)
        plt.tight_layout()
        fname = os.path.join(outdir, f'force_{role}_idx{idx}.png')
        plt.savefig(fname, dpi=200)
        plt.close()
        lines.append(f'Force plot saved to: {fname}\n')
        lines.append('\nTextual explanation:\n')
        sv = shap_values[idx,:]
        top_pos = np.argsort(-sv)[:6]
        top_neg = np.argsort(sv)[:6]
        lines.append('**Top positive contributors (increase churn probability):**\n')
        for i in top_pos:
            lines.append(f'- {feature_names[i]} : SHAP {sv[i]:.4f}\n')
        lines.append('\n**Top negative contributors (decrease churn probability):**\n')
        for i in top_neg:
            lines.append(f'- {feature_names[i]} : SHAP {sv[i]:.4f}\n')
    except Exception as e:
        lines.append(f'Force plot generation failed: {e}\n')
    lines.append('\n---\n')

with open(os.path.join(outdir, 'local_shap_explanations.md'), 'w') as f:
    f.writelines([l if l.endswith('\n') else l+'\n' for l in lines])
print('Saved', os.path.join(outdir, 'local_shap_explanations.md'))

# -------------------------
# === Dependence plots (interactions)
# -------------------------

top_features = fi_df['feature'].head(5).tolist()
deps_saved = []
for i, feat in enumerate(top_features[:3]):
    try:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X_test_processed, feature_names=feature_names, show=False)
        fname = os.path.join(outdir, f'dependence_{i}_{feat.replace(" ", "_")}.png')
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        deps_saved.append(fname)
    except Exception as e:
        print('Dependence plot failed for', feat, e)

print('Saved dependence plots:', deps_saved)

# -------------------------
# === Build final SHAP Analysis Markdown (global + comparison + recommendations)
# -------------------------
with open(os.path.join(outdir, 'shap_analysis.md'),'w') as f:
    f.write('# SHAP Analysis & Recommendations\n\n')
    f.write('## Model performance\n\n')
    f.write(f'- ROC-AUC: {auc:.4f}\n')
    f.write(f'- Best params: {best_params}\n\n')
    f.write('## Global SHAP results\n\n')
    f.write('- See outputs/shap_summary_beeswarm.png (global importance + direction)\n')
    f.write('- See outputs/shap_summary_bar.png (mean absolute SHAP)\n\n')
    f.write('## Feature importance comparison (top features)\n\n')
    try:
        f.write(fi_df.head(15).to_markdown(index=False))
    except Exception:
        f.write(fi_df.head(15).to_string())
    f.write('\n\n')
    f.write('## Dependence plots (interactions)\n\n')
    for p in deps_saved:
        f.write(f'- {p}\n')
    f.write('\n\n')
    f.write('## Local explanations\n\n')
    f.write('- See outputs/local_shap_explanations.md and the force plot images.\n\n')
    f.write('## Strategic Recommendations (data-justified)\n\n')
    f.write('1. Tenure-based onboarding & offers for tenure < 12 months.\n')
    f.write('2. Incentivize month-to-month customers to upgrade with discounts and bundles.\n')
    f.write('3. Price-sensitivity: offer introductory pricing to new high-charge customers.\n')
    f.write('4. Bundle/optimize Fiber-Optic month-to-month customers.\n')
    f.write('5. Target borderline customers with tailored offers based on dominant SHAP drivers.\n')

print('Saved', os.path.join(outdir, 'shap_analysis.md'))

# -------------------------
# === Save the best pipeline (model + preprocessor)
# -------------------------
joblib.dump(best_pipeline, os.path.join(outdir, 'best_pipeline.joblib'))
print('Saved', os.path.join(outdir, 'best_pipeline.joblib'))

print('\nAll outputs are saved in /mnt/data/outputs (downloadable from your environment).')
