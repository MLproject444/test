import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 5000

ages = np.random.randint(18, 90, size=n_samples)
bmis = np.round(np.random.uniform(18.5, 48.0, size=n_samples), 1)
bps = np.random.randint(85, 210, size=n_samples)
cholesterols = np.random.randint(120, 380, size=n_samples)
glucoses = np.random.randint(65, 300, size=n_samples)

symptoms = np.random.choice(
    ['none', 'chest_pain', 'shortness_breath', 'frequent_urination', 'fatigue', 'dizziness'],
    size=n_samples,
    p=[0.35, 0.15, 0.15, 0.15, 0.10, 0.10]
)

symptom_disease_map = {
    'chest_pain': 'Coronary Artery Disease',
    'shortness_breath': 'Cardiopulmonary Disease',
    'frequent_urination': 'Type 2 Diabetes',
    'fatigue': 'Metabolic Syndrome / Anemia',
    'dizziness': 'Hypertension / Arrhythmia',
    'none': 'General Cardiovascular Risk'
}

symptom_weights = {
    'chest_pain': 35,
    'shortness_breath': 25,
    'frequent_urination': 30,
    'fatigue': 15,
    'dizziness': 20,
    'none': 0
}

data = []

for age, bmi, bp, chol, glu, sym in zip(ages, bmis, bps, cholesterols, glucoses, symptoms):
    logit = -8.5 + (0.04 * age) + (0.08 * bmi) + (0.02 * bp) + (0.01 * chol) + (0.025 * glu)
    base_prob = 1 / (1 + np.exp(-logit))
    
    weight = symptom_weights[sym]
    risk_pct = round(min(base_prob * 100 + weight, 99.9), 1)
    
    # Categorize Risk Category
    if risk_pct < 35.0:
        risk_category = 'Low Risk'
    elif risk_pct <= 65.0:
        risk_category = 'Moderate Risk'
    else:
        risk_category = 'High Risk'
        
    predicted_disease = symptom_disease_map[sym]

    data.append({
        'Age': age,
        'BMI': bmi,
        'Systolic_BP': bp,
        'Cholesterol': chol,
        'Fasting_Glucose': glu,
        'Primary_Symptom': sym,
        'Calculated_Risk_Pct': risk_pct,
        'Risk_Category': risk_category,
        'Predicted_Disease': predicted_disease
    })

# 3. Create DataFrame & Save to CSV
df = pd.DataFrame(data)
df.to_csv('medical_risk_dataset.csv', index=False)

print(f"Successfully generated 'medical_risk_dataset.csv' with {len(df)} patient records.")
print("\nDataset Preview:")
print(df.head())