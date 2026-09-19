import math
from flask import Flask, render_template, request

app = Flask(__name__)

# Medical parameter validation limits
VALID_RANGES = {
    'age': (1, 120),
    'bmi': (10.0, 60.0),
    'bp': (60, 250),
    'chol': (100, 500),
    'glu': (50, 400) 
}

SYMPTOM_RISK_MATRIX = {
    'chest_pain': {'disease': 'Coronary Artery Disease', 'weight': 35},
    'shortness_breath': {'disease': 'Cardiopulmonary Disease', 'weight': 25},
    'fatigue': {'disease': 'Metabolic Syndrome / Anemia', 'weight': 15},
    'frequent_urination': {'disease': 'Type 2 Diabetes', 'weight': 30},
    'dizziness': {'disease': 'Hypertension / Arrhythmia', 'weight': 20},
    'none': {'disease': 'General Cardiovascular Risk', 'weight': 0}
}

def evaluate_predictions(age, bmi, bp, chol, glu, symptom):
    # 1. Logistic Regression Formula
    logit = -8.5 + (0.04 * age) + (0.08 * bmi) + (0.02 * bp) + (0.01 * chol) + (0.025 * glu)
    base_prob = 1 / (1 + math.exp(-logit))
    
    symptom_data = SYMPTOM_RISK_MATRIX.get(symptom, SYMPTOM_RISK_MATRIX['none'])
    weight = symptom_data['weight']
    predicted_disease = symptom_data['disease']

    log_risk_pct = round(min(base_prob * 100 + weight, 99.9), 1)

    # 2. Simulated Random Forest Ensemble Prediction
    # Combines baseline logit, non-linear thresholds, and symptom weighting
    rf_score = (
        (0.35 * (age / 120.0)) + 
        (0.25 * (bmi / 60.0)) + 
        (0.20 * (bp / 250.0)) + 
        (0.10 * (glu / 400.0)) + 
        (0.10 * (weight / 35.0))
    )
    rf_risk_pct = round(min(rf_score * 100 * 1.15, 99.9), 1)

    # 3. Decision Tree Categorization
    if glu > 140 or symptom == 'frequent_urination' or bp > 150:
        tree_verdict = f"High Risk ({predicted_disease})"
    elif bmi > 28.0 or bp > 130:
        tree_verdict = f"Moderate Risk ({predicted_disease})"
    else:
        tree_verdict = f"Low Risk ({predicted_disease})"

    return log_risk_pct, rf_risk_pct, tree_verdict, predicted_disease

@app.route('/')
def home():
    return render_template('index.html', log_risk=None, rf_risk=None, tree_verdict=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        bp = float(request.form['bp'])
        chol = float(request.form['chol'])
        glu = float(request.form['glu'])
        symptom = request.form.get('symptom', 'none')

        # Parameter Range Validations
        if not (VALID_RANGES['age'][0] <= age <= VALID_RANGES['age'][1]):
            raise ValueError("Age must be between 1 and 120 years.")
        if not (VALID_RANGES['bmi'][0] <= bmi <= VALID_RANGES['bmi'][1]):
            raise ValueError("BMI must be between 10.0 and 60.0.")
        if not (VALID_RANGES['bp'][0] <= bp <= VALID_RANGES['bp'][1]):
            raise ValueError("BP must be between 60 and 250 mmHg.")
        if not (VALID_RANGES['chol'][0] <= chol <= VALID_RANGES['chol'][1]):
            raise ValueError("Cholesterol must be between 100 and 500 mg/dL.")
        if not (VALID_RANGES['glu'][0] <= glu <= VALID_RANGES['glu'][1]):
            raise ValueError("Glucose must be between 50 and 400 mg/dL.")

        log_risk, rf_risk, tree_verdict, predicted_disease = evaluate_predictions(age, bmi, bp, chol, glu, symptom)

        return render_template(
            'index.html',
            log_risk=log_risk,
            rf_risk=rf_risk,
            tree_verdict=tree_verdict,
            predicted_disease=predicted_disease,
            age=age, bmi=bmi, bp=bp, chol=chol, glu=glu, symptom=symptom
        )
    except Exception as e:
        return render_template('index.html', log_risk=None, rf_risk=None, tree_verdict=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)