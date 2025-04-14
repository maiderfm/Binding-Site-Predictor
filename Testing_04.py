import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, auc, roc_curve
)
from sklearn.preprocessing import StandardScaler

model_data = joblib.load('random_forest_model.pkl')

if isinstance(model_data, tuple) and len(model_data) >= 2:
    random_forest_model = model_data[0]
    optimal_threshold = model_data[1]
    if len(model_data) >= 3:
        scaler = model_data[2]
    else:
        scaler = StandardScaler()  
else:
    random_forest_model = model_data
    optimal_threshold = 0.5  
    scaler = StandardScaler()

print(f"Optimal threshold: {optimal_threshold}")


df = pd.read_csv('test_data_transformed.csv')

X = df.drop(columns=["Binding_Site", "Protein", "Position", "Residue", "Chain"])
y = df["Binding_Site"]

X_scaled = scaler.transform(X)  


def evaluate_model(model, X_scaled, y, model_name, threshold=0.5):
    """Evaluate the model with a given threshold."""

    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
    auc = roc_auc_score(y, y_pred_proba)
    
    print(f"\n--- {model_name} Performance (threshold={threshold:.3f}) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score: {f2:.4f}")  
    print(f"AUC-ROC: {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Binding', 'Binding'],
                yticklabels=['Non-Binding', 'Binding'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name} (threshold={threshold:.3f})')
    plt.savefig(f'{model_name.replace(" ", "_")}_confusion_matrix_threshold_{threshold:.3f}.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'auc': auc,
        'y_pred_proba': y_pred_proba
    }

def perform_cross_validation(model, X, y, model_name, threshold=0.5, cv=5):
    """Perform stratified k-fold cross-validation and return average classification metrics for a given model."""

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = {
        'accuracy': [], 'precision': [], 'recall': [],
        'f1': [], 'f2': [], 'auc': []
    }

    for train_idx, test_idx in cv_strategy.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['precision'].append(precision_score(y_test, y_pred))
        scores['recall'].append(recall_score(y_test, y_pred))
        scores['f1'].append(f1_score(y_test, y_pred))

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
        scores['f2'].append(f2)

        scores['auc'].append(roc_auc_score(y_test, y_proba))

    print(f"\n--- {model_name} Cross-Validation (k={cv}, threshold={threshold:.3f}) ---")
    for metric in scores:
        arr = np.array(scores[metric])
        print(f"{metric.capitalize()}: {arr.mean():.4f} (±{arr.std():.4f})")

    return {f'{k}_mean': np.mean(v) for k, v in scores.items()}

# Evaluate the model on the test set with the default threshold (0.5)
rf_results_default = evaluate_model(random_forest_model, X_scaled, y, "Random Forest", threshold=0.5)

# Evaluate the model on the test set with the optimal threshold
rf_results_optimal = evaluate_model(random_forest_model, X_scaled, y, "Random Forest", threshold=optimal_threshold)

X_scaled_np = np.array(X_scaled)
y_np = np.array(y)

# Perform cross-validation on model with the optimal threshold
rf_cv_results = perform_cross_validation(random_forest_model, X_scaled_np, y_np, "Random Forest", threshold=optimal_threshold)

metrics = ['accuracy', 'precision', 'recall', 'f1', 'f2', 'auc']
rf_default_scores = [rf_results_default[metric] for metric in metrics]
rf_optimal_scores = [rf_results_optimal[metric] for metric in metrics]

plt.figure(figsize=(12, 8))
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, rf_default_scores, width, label=f'Default Threshold (0.5)')
plt.bar(x + width/2, rf_optimal_scores, width, label=f'Optimal Threshold ({optimal_threshold:.3f})')

plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Model Performance Comparison: Default vs. Optimal Threshold')
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.legend()
plt.savefig('threshold_comparison.png')
plt.close()

# Feature importance analysis 
if hasattr(random_forest_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': random_forest_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))  # Show top 20 features
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nTop 20 Important Features:")
    print(feature_importance.head(20))


plt.figure(figsize=(10, 8))
precision, recall, thresholds = precision_recall_curve(y, random_forest_model.predict_proba(X_scaled)[:, 1])

f2_scores = []
for p, r in zip(precision, recall):
    if p + r > 0:
        f2_scores.append((5 * p * r) / (4 * p + r))  
    else:
        f2_scores.append(0)

optimal_idx = np.argmax(f2_scores[:-1])  
current_optimal_threshold = thresholds[optimal_idx]

plt.plot(recall, precision, marker='.', label=f'Random Forest (AUC = {auc(recall, precision):.3f})')
plt.scatter(recall[optimal_idx], precision[optimal_idx], marker='o', color='red',
           label=f'Test Optimal Threshold = {current_optimal_threshold:.3f}')
plt.scatter([recall[np.where(thresholds >= optimal_threshold)[0][-1]] if np.any(thresholds >= optimal_threshold) else recall[0]], 
            [precision[np.where(thresholds >= optimal_threshold)[0][-1]] if np.any(thresholds >= optimal_threshold) else precision[0]], 
            marker='s', color='green',
            label=f'Training Optimal Threshold = {optimal_threshold:.3f}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Thresholds')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('precision_recall_curve_test.png')
plt.close()

print(f"\nTraining Optimal Threshold: {optimal_threshold:.3f}")
print(f"Test Set Optimal Threshold: {current_optimal_threshold:.3f}")

# ROC CURVES FOR THRESHOLDS 0.4 and 0.5
y_pred_proba = rf_results_default['y_pred_proba']  

fpr_05, tpr_05, _ = roc_curve(y, y_pred_proba)
roc_auc_05 = auc(fpr_05, tpr_05)

threshold_04 = 0.4
threshold_05 = 0.5

plt.figure(figsize=(10, 8))
plt.plot(fpr_05, tpr_05, label=f'Random Forest (AUC = {roc_auc_05:.4f})', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)

def get_coords_for_thresh(thresh, y_true, y_scores):
    """Return false positive and true positive ranges at a given threshold."""
    y_pred_thresh = (y_scores >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred_thresh)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    return fpr, tpr

fpr_04_point, tpr_04_point = get_coords_for_thresh(0.4, y, y_pred_proba)
fpr_05_point, tpr_05_point = get_coords_for_thresh(0.5, y, y_pred_proba)

plt.scatter(fpr_04_point, tpr_04_point, color='blue', label='Threshold = 0.4', s=80, marker='o')
plt.scatter(fpr_05_point, tpr_05_point, color='green', label='Threshold = 0.5', s=80, marker='s')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest at Thresholds 0.4 and 0.5')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('rf_roc_curve_thresholds_0.4_0.5.png')
plt.close()
