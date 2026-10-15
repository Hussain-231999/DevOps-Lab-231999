import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. DATA INGESTION VIA CHUNK-SAMPLING (1.16M RECORDS)
# -------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
file_name = ROOT_DIR / "data" / "flights.csv"
SAMPLE_FRAC = 0.20

use_cols = [
    'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE',
    'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
    'SCHEDULED_DEPARTURE', 'SCHEDULED_TIME', 'DISTANCE',
    'DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'CANCELLED', 'DIVERTED'
]

chunks = []
for chunk in pd.read_csv(
    file_name,
    usecols=use_cols,
    dtype={'ORIGIN_AIRPORT': str, 'DESTINATION_AIRPORT': str},
    chunksize=500_000
):
    chunks.append(chunk.sample(frac=SAMPLE_FRAC, random_state=42))

df = pd.concat(chunks, ignore_index=True)
del chunks

print(f"\nLoaded {file_name} successfully. Total Records: {df.shape[0]}")

# -------------------------------------------------------------------------
# 2. FILTERING & TARGET CONSTRUCTION
# -------------------------------------------------------------------------
df_clean = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()
df_clean['IS_DELAYED'] = (df_clean['ARRIVAL_DELAY'] >= 15).astype(int)

# Pre-Departure Feature Engineering
if 'SCHEDULED_DEPARTURE' in df_clean.columns:
    df_clean['SCHEDULED_DEPARTURE_HOUR'] = df_clean['SCHEDULED_DEPARTURE'] // 100
    df_clean['IS_PEAK_HOURS'] = df_clean['SCHEDULED_DEPARTURE_HOUR'].isin([6, 7, 8, 16, 17, 18, 19, 20]).astype(int)
    df_clean['IS_RED_EYE'] = df_clean['SCHEDULED_DEPARTURE_HOUR'].isin([23, 0, 1, 2, 3, 4]).astype(int)

df_clean['SCHEDULED_SPEED'] = df_clean['DISTANCE'] / np.maximum(df_clean['SCHEDULED_TIME'], 1)
df_clean['ROUTE'] = df_clean['ORIGIN_AIRPORT'] + "_" + df_clean['DESTINATION_AIRPORT']

# Drop raw identifiers and non-predictive post-flight metrics
drop_cols = [
    'ARRIVAL_DELAY', 'CANCELLED', 'DIVERTED', 'SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL'
]
df_clean = df_clean.drop(columns=[col for col in drop_cols if col in df_clean.columns])

# -------------------------------------------------------------------------
# 3. SPLITTING EXPERIMENTAL PIPELINES (PATH B VS. PATH A)
# -------------------------------------------------------------------------
X_path_b = df_clean.drop(columns=['IS_DELAYED', 'DEPARTURE_DELAY'])
y_path_b = df_clean['IS_DELAYED']

X_path_a = df_clean.drop(columns=['IS_DELAYED'])
y_path_a = df_clean['IS_DELAYED']

def run_experiment(X, y, experiment_name="Path B (Pre-Departure)"):
    print(f"\n=======================================================")
    print(f" RUNNING EXPERIMENT: {experiment_name}")
    print(f"=======================================================")
    
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), 
                          ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=20, sparse_output=True))]), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    X_fit, X_val, y_fit, y_val = train_test_split(X_train_prep, y_train, test_size=0.10, random_state=42, stratify=y_train)

    scale_pos = (y_fit == 0).sum() / (y_fit == 1).sum()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "XGBoost Optimized": XGBClassifier(
            eval_metric='auc', random_state=42, tree_method='hist',
            n_estimators=300, max_depth=7, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos, n_jobs=-1
        )
    }

    def best_f1_threshold(y_true, probs):
        p, r, t = precision_recall_curve(y_true, probs)
        f1 = 2 * p * r / (p + r + 1e-12)
        return t[np.argmax(f1[:-1])]

    results = []
    plot_data = {}
    thresholds = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_fit, y_fit)
        
        thr = best_f1_threshold(y_val, model.predict_proba(X_val)[:, 1])
        thresholds[name] = thr
        trained_models[name] = {
            'model': model,
            'threshold': float(thr),
        }
        
        test_probs = model.predict_proba(X_test_prep)[:, 1]
        test_preds = (test_probs >= thr).astype(int)
        plot_data[name] = test_probs
        
        cm = confusion_matrix(y_test, test_preds)
        print(f"\n[{experiment_name}] {name} Confusion Matrix (Threshold = {thr:.3f}):")
        print(cm)
        
        results.append({
            'Experiment': experiment_name,
            'Model': name,
            'Threshold': round(thr, 3),
            'Accuracy': round(accuracy_score(y_test, test_preds), 4),
            'Precision': round(precision_score(y_test, test_preds), 4),
            'Recall': round(recall_score(y_test, test_preds), 4),
            'F1-Score': round(f1_score(y_test, test_preds), 4),
            'ROC-AUC': round(roc_auc_score(y_test, test_probs), 4),
            'PR-AUC': round(average_precision_score(y_test, test_probs), 4)
        })

    # PLOTTING ROC AND PRECISION-RECALL CURVES FOR EXPERIMENT
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5))

    for name, test_probs in plot_data.items():
        thr = thresholds[name]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        roc_auc = roc_auc_score(y_test, test_probs)
        line, = ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_test, test_probs)
        pr_auc = average_precision_score(y_test, test_probs)
        ax_pr.plot(recall, precision, color=line.get_color(), label=f'{name} (PR-AUC = {pr_auc:.3f})')
        
        # Mark operating point (threshold)
        preds = (test_probs >= thr).astype(int)
        tp = ((preds == 1) & (y_test == 1)).sum()
        fp = ((preds == 1) & (y_test == 0)).sum()
        op_tpr = tp / (y_test == 1).sum()
        op_fpr = fp / (y_test == 0).sum()
        op_prec = tp / max(tp + fp, 1)
        
        ax_roc.scatter(op_fpr, op_tpr, color=line.get_color(), marker='o', s=60, zorder=5)
        ax_pr.scatter(op_tpr, op_prec, color=line.get_color(), marker='o', s=60, zorder=5)

    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax_roc.set_title(f'ROC Curve - {experiment_name}')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend()
    ax_roc.grid(True)

    ax_pr.axhline(y_test.mean(), color='k', linestyle=':', label='Random Classifier')
    ax_pr.set_title(f'Precision-Recall Curve - {experiment_name}')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend()
    ax_pr.grid(True)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(results), {
        'preprocessor': preprocessor,
        'models': trained_models,
        'feature_columns': X.columns.tolist(),
        'categorical_columns': cat_cols,
        'numeric_columns': num_cols,
    }

# Execute core strict model (Path B) followed by target leakage comparison (Path A)
results_b, artifact_b = run_experiment(
    X_path_b, y_path_b, "Path B: Strict Pre-Departure"
)
results_a, artifact_a = run_experiment(
    X_path_a, y_path_a, "Path A: Target Leakage Baseline"
)

combined_results = pd.concat([results_b, results_a], ignore_index=True)
print("\n=======================================================")
print(" FINAL METHODOLOGICAL COMPARISON TABLE")
print("=======================================================")
print(combined_results.to_string(index=False))

artifact = {
    'artifact_version': 2,
    'experiments': {
        'Path B: Strict Pre-Departure': artifact_b,
        'Path A: Target Leakage Baseline': artifact_a,
    },
    'metrics': combined_results.to_dict(orient='records'),
}

model_path = ROOT_DIR / "models" / "flight_delay_models.pkl"
with model_path.open("wb") as file:
    pickle.dump(artifact, file)
print(f"\nSaved trained model artifact to {model_path}")