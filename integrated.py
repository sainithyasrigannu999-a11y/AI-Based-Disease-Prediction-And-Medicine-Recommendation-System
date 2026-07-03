import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- THIS BLOCKS THE RED WARNINGS FROM PRINTING ---
import warnings
warnings.filterwarnings("ignore")
# --------------------------------------------------

print("=================================================================")
print("  STEP 2: COUPLING DISEASE PREDICTION WITH DRUG RECOMMENDATION ")
print("=================================================================\n")

# 1. LOAD THE DATASETS
try:
    disease_df = pd.read_csv("balanced_disease_dataset.csv")
    drug_df = pd.read_csv("balanced_drug_dataset.csv")
    print("✔ Both balanced data repositories successfully loaded!")
except FileNotFoundError:
    print("❌ Error: Missing your CSV dataset files in this folder!")
    exit()

# 2. TRAIN THE PRIMARY DISEASE PREDICTION ENGINE
X_disease = disease_df.iloc[:, :-1]
y_disease = disease_df.iloc[:, -1]

disease_model = RandomForestClassifier(n_estimators=100, random_state=42)
disease_model.fit(X_disease, y_disease)
print("✔ Disease Classifier successfully trained.")

# 3. TRAIN THE MEDICATION RECOMMENDATION MODEL
drug_df_encoded = drug_df.copy()
drug_df_encoded['Gender'] = drug_df_encoded['Gender'].map({'Male': 0, 'Female': 1})

disease_mapping = {name: i for i, name in enumerate(drug_df_encoded['Disease'].unique())}
drug_df_encoded['Disease_Code'] = drug_df_encoded['Disease'].map(disease_mapping)

X_drug = drug_df_encoded[['Disease_Code', 'Age', 'Gender']]
y_drug = drug_df_encoded['Drug']

drug_model = RandomForestClassifier(n_estimators=100, random_state=42)
drug_model.fit(X_drug, y_drug)
print("✔ Medication Recommendation system successfully trained.\n")

# 4. SIMULATE A PATIENT INTERFACE (TEST RUN)
print("--- RUNNING PATIENT DIAGNOSTIC SIMULATION ---")

patient_age = 45
patient_gender = "Female"

sample_symptoms = np.zeros(132)
sample_symptoms[101] = 1 
sample_symptoms[102] = 1
sample_symptoms[104] = 1

predicted_disease = disease_model.predict([sample_symptoms])[0]
print(f"\n[Patient Symptoms Input] -> Processing Symptoms...")
print(f"🩺 Predicted Diagnosis: {predicted_disease}")

gender_code = 1 if patient_gender == "Female" else 0
mapped_disease_name = "Diabetes" if "Diabetes" in predicted_disease else predicted_disease
disease_code = disease_mapping.get(mapped_disease_name, 0)

recommended_drug = drug_model.predict([[disease_code, patient_age, gender_code]])[0]

print(f"👤 Patient Demographics: {patient_age} year old {patient_gender}")
print(f"💊 Recommended Medication Plan: {recommended_drug}")
print("=================================================================")
