import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import gc

def calculate_f2(precision, recall):
    if precision + recall > 0:
        return (5 * precision * recall) / (4 * precision + recall)
    return 0

print("Loading data...")
train_df = pd.read_csv('train_data_transformed.csv')

# Split into features (X) and target (y)
X = train_df.drop(columns=["Binding_Site", "Protein", "Position", "Residue", "Chain"])
y = train_df["Binding_Site"]

del train_df
gc.collect()

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

del X_scaled, X
gc.collect()

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_leaf=2,
    class_weight={0: 1, 1: 15},  
    random_state=42,
    verbose=1,
    n_jobs=1
)

print("Training model...")
model.fit(X_train, y_train)

del X_train, y_train
gc.collect()

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

f2_scores = [result['f2'] for threshold, result in results.items()]
thresholds = list(results.keys())
optimal_threshold = thresholds[np.argmax(f2_scores)]

print(f"\nOptimal threshold: {optimal_threshold:.2f} (F2: {results[optimal_threshold]['f2']:.4f})")

y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
print(f"\nClassification report with optimal threshold ({optimal_threshold:.2f}):")
print(classification_report(y_test, y_pred_optimal))

print("Saving model...")
joblib.dump((model, optimal_threshold, scaler), 'random_forest_model.pkl')

print("Model, scaler and optimal threshold saved successfully.")