import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# LOADING MODEL AND THRESHOLD
model_data = joblib.load('random_forest_model.pkl')
# Unpack the model, threshold, and scaler
if isinstance(model_data, tuple) and len(model_data) >= 2:
    random_forest_model = model_data[0]
    optimal_threshold = model_data[1]
    if len(model_data) >= 3:
        scaler = model_data[2]
    else:
        scaler = StandardScaler()  # Create a new scaler if one wasn't saved
else:
    random_forest_model = model_data
    optimal_threshold = 0.5  # Use default threshold if none was saved
    scaler = StandardScaler()

print(f"Model type: {type(random_forest_model)}")
print(f"Optimal threshold: {optimal_threshold}")

# LOADING AND PROCESSING THE TESTING SET
df = pd.read_csv('test_data_transformed.csv')

# Define features and target
X = df.drop(columns=["Binding_Site", "Protein", "Position", "Residue", "Chain"])
y = df["Binding_Site"]

# Scaling - use the scaler from the loaded model
X_scaled = scaler.transform(X)  # Use transform, not fit_transform for test data

# PERFORMANCE METRICS
# Function to evaluate a model with threshold
def evaluate_model(model, X_scaled, y, model_name, threshold=0.5):
    # Make predictions with probability
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Apply the threshold for binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
    auc = roc_auc_score(y, y_pred_proba)
    
    # Print results
    print(f"\n--- {model_name} Performance (threshold={threshold:.3f}) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score: {f2:.4f}")  # Added F2 score
    print(f"AUC-ROC: {auc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Plot confusion matrix
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
        'auc': auc
    }

# Cross-validation function
def perform_cross_validation(model, X, y, model_name, threshold=0.5, cv=5):
    # Define cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Custom scorer for threshold
    def custom_f2_scorer(estimator, X, y):
        y_proba = estimator.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
        return f2
    
    # Calculate cross-validation scores
    cv_accuracy = []
    cv_precision = []
    cv_recall = []
    cv_f1 = []
    cv_f2 = []
    cv_auc = []
    
    # Manual cross-validation to apply custom threshold
    for train_idx, test_idx in cv_strategy.split(X, y):
        X_cv_train, X_cv_test = X[train_idx], X[test_idx]
        y_cv_train, y_cv_test = y[train_idx], y[test_idx]
        
        # Fit on this fold's training data
        model.fit(X_cv_train, y_cv_train)
        
        # Predict on this fold's test data
        y_cv_proba = model.predict_proba(X_cv_test)[:, 1]
        y_cv_pred = (y_cv_proba >= threshold).astype(int)
        
        # Calculate metrics
        cv_accuracy.append(accuracy_score(y_cv_test, y_cv_pred))
        cv_precision.append(precision_score(y_cv_test, y_cv_pred))
        cv_recall.append(recall_score(y_cv_test, y_cv_pred))
        cv_f1.append(f1_score(y_cv_test, y_cv_pred))
        
        # Calculate F2 score
        precision = precision_score(y_cv_test, y_cv_pred)
        recall = recall_score(y_cv_test, y_cv_pred)
        f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
        cv_f2.append(f2)
        
        cv_auc.append(roc_auc_score(y_cv_test, y_cv_proba))
    
    # Convert to numpy arrays for statistics
    cv_accuracy = np.array(cv_accuracy)
    cv_precision = np.array(cv_precision)
    cv_recall = np.array(cv_recall)
    cv_f1 = np.array(cv_f1)
    cv_f2 = np.array(cv_f2)
    cv_auc = np.array(cv_auc)
    
    print(f"\n--- {model_name} Cross-Validation Results (k={cv}, threshold={threshold:.3f}) ---")
    print(f"Accuracy: {cv_accuracy.mean():.4f} (±{cv_accuracy.std():.4f})")
    print(f"Precision: {cv_precision.mean():.4f} (±{cv_precision.std():.4f})")
    print(f"Recall: {cv_recall.mean():.4f} (±{cv_recall.std():.4f})")
    print(f"F1 Score: {cv_f1.mean():.4f} (±{cv_f1.std():.4f})")
    print(f"F2 Score: {cv_f2.mean():.4f} (±{cv_f2.std():.4f})")
    print(f"AUC-ROC: {cv_auc.mean():.4f} (±{cv_auc.std():.4f})")
    
    return {
        'accuracy_mean': cv_accuracy.mean(),
        'precision_mean': cv_precision.mean(),
        'recall_mean': cv_recall.mean(),
        'f1_mean': cv_f1.mean(),
        'f2_mean': cv_f2.mean(),
        'auc_mean': cv_auc.mean()
    }

# Evaluate model on the test set with default threshold (0.5)
rf_results_default = evaluate_model(random_forest_model, X_scaled, y, "Random Forest", threshold=0.5)

# Evaluate model on the test set with optimal threshold
rf_results_optimal = evaluate_model(random_forest_model, X_scaled, y, "Random Forest", threshold=optimal_threshold)

# Convert to numpy array for cross-validation
X_scaled_np = np.array(X_scaled)
y_np = np.array(y)

# Perform cross-validation on model with optimal threshold
rf_cv_results = perform_cross_validation(random_forest_model, X_scaled_np, y_np, "Random Forest", threshold=optimal_threshold)

# Compare models visually
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

# Feature importance analysis for Random Forest
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

# Optional: Add Precision-Recall curve to visualize threshold effect
from sklearn.metrics import precision_recall_curve, auc

plt.figure(figsize=(10, 8))
precision, recall, thresholds = precision_recall_curve(y, random_forest_model.predict_proba(X_scaled)[:, 1])

# Calculate F2 scores for each threshold
f2_scores = []
for p, r in zip(precision, recall):
    if p + r > 0:
        f2_scores.append((5 * p * r) / (4 * p + r))  # F2 score formula
    else:
        f2_scores.append(0)

# Find optimal threshold from this dataset
optimal_idx = np.argmax(f2_scores[:-1])  # Exclude the last element as it might not have a matching threshold
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
