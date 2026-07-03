import pandas as pd
import numpy as np

# Import train-test splitting and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import the 3 core models from the base paper
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

print("==================================================")
print("   DISEASE PREDICTION MACHINE LEARNING PIPELINE   ")
print("==================================================\n")

# ----------------------------------------------------
# 1. LOAD THE BALANCED DATASET
# ----------------------------------------------------
try:
    # Read the balanced file
    data = pd.read_csv("balanced_disease_dataset.csv")
    print(f"✔ Successfully loaded balanced dataset with shape: {data.shape}")
except FileNotFoundError:
    print("❌ Error: 'balanced_disease_dataset.csv' not found in this folder!")
    print("Please ensure your script and the CSV file are in the same folder.")
    exit()

# ----------------------------------------------------
# 2. SPLIT FEATURES (X) AND TARGET CLASSES (Y)
# ----------------------------------------------------
# Features: All the 132 symptom columns (everything except the last column)
X = data.iloc[:, :-1] 

# Target: The 'Disease' column
y = data.iloc[:, -1]

# Split into Training (80%) and Testing (20%) sets as specified in the paper
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"✔ Splitting complete: Training on {len(X_train)} rows, Testing on {len(X_test)} rows.\n")

# ----------------------------------------------------
# 3. INITIALIZE THE 3 MODELS
# ----------------------------------------------------
models = {
    "Naïve Bayes (Gaussian)": GaussianNB(),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Dictionary to hold the final accuracies for comparison
accuracy_results = {}

# ----------------------------------------------------
# 4. TRAIN AND EVALUATE EACH MODEL
# ----------------------------------------------------
for model_name, model_object in models.items():
    print(f"--- Training {model_name} ---")
    
    # Train the algorithm
    model_object.fit(X_train, y_train)
    
    # Predict outcomes on the 20% validation test data
    predictions = model_object.predict(X_test)
    
    # Calculate performance accuracy metric
    acc = accuracy_score(y_test, predictions)
    accuracy_results[model_name] = acc
    
    print(f"👉 {model_name} Accuracy: {acc * 100:.2f}%")
    print("Classification Report details:")
    print(classification_report(y_test, predictions))
    print("-" * 50)

# ----------------------------------------------------
# 5. COMPARATIVE SUMMARY OUTPUT
# ----------------------------------------------------
print("\n==================================================")
print("         FINAL MODEL PERFORMANCE COMPARISON       ")
print("==================================================")
for model_name, score in accuracy_results.items():
    print(f"  • {model_name:<25}: {score * 100:.2f}%")
print("==================================================")
