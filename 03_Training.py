import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import gc

# Simple F2 calculation
def calculate_f2(precision, recall):
    if precision + recall > 0:
        return (5 * precision * recall) / (4 * precision + recall)
    return 0

print("Loading data...")
# Load just a sample of the data for testing
train_df = pd.read_csv('train_data_transformed.csv')

# Split into features and target
X = train_df.drop(columns=["Binding_Site", "Protein", "Position", "Residue", "Chain"])
y = train_df["Binding_Site"]

# Free memory
del train_df
gc.collect()

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Free memory
del X_scaled, X
gc.collect()

# Define lightweight model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_leaf=2,
    class_weight={0: 1, 1: 15},  # Handle imbalance with class_weight instead of SMOTE
    random_state=42,
    verbose=1,
    n_jobs=1
)

# Train model (without SMOTE for now)
print("Training model...")
model.fit(X_train, y_train)

# Free memory
del X_train, y_train
gc.collect()

# Test different thresholds
print("Testing thresholds...")
y_pred_proba = model.predict_proba(X_test)[:, 1]

thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
results = {}

for threshold in thresholds_to_test:
    print(f"Testing threshold: {threshold}")
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = calculate_f2(precision, recall)
    
    results[threshold] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2
    }
    
    print(f"Threshold: {threshold:.2f}, F2: {f2:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Find optimal threshold
f2_scores = [result['f2'] for threshold, result in results.items()]
thresholds = list(results.keys())
optimal_threshold = thresholds[np.argmax(f2_scores)]

print(f"\nOptimal threshold: {optimal_threshold:.2f} (F2: {results[optimal_threshold]['f2']:.4f})")

# Show optimal threshold results
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
print(f"\nClassification report with optimal threshold ({optimal_threshold:.2f}):")
print(classification_report(y_test, y_pred_optimal))

# Save the trained model
print("Saving model...")
joblib.dump((model, optimal_threshold, scaler), 'random_forest_model.pkl')

print("Model, scaler and optimal threshold saved successfully.")

print("""
NEXT STEPS:
1. If this script runs successfully, use the optimal threshold with your full training script
2. For the final model, train on all data with SMOTE and the optimal threshold
3. Save the model with the optimal threshold for testing
""")
