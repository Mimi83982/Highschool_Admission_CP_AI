from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# 1. Load the Admission/Classification Models
print("Loading Admission Models...")
mlp_model = joblib.load('admission_mlp_model.pkl')
scaler = joblib.load('admission_scaler.pkl')

# 2. Load the Forecasting/Prediction Models
print("Loading Forecasting Models...")
forecaster = joblib.load('grade_forecaster.pkl')
forecaster_scaler = joblib.load('forecaster_scaler.pkl')

# Define the columns the admission model expects
SUBJECTS = ['MTK', 'B_Indo', 'B_Inggris', 'IPA', 'IPS']
FEATURE_COLUMNS = [f'{subj}_Sem{sem}' for subj in SUBJECTS for sem in range(1, 7)]


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "SMA Admission AI Backend is running!"})


@app.route('/process_csv', methods=['POST'])
def process_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be a CSV format"}), 400

    try:
        df = pd.read_csv(file)

        # Predict Eligibility
        X_new = df[FEATURE_COLUMNS]
        X_scaled = scaler.transform(X_new)
        predictions = mlp_model.predict(X_scaled)

        # Ranking logic
        df['rata_rata_akhir'] = X_new.mean(axis=1).round(2)
        df['status_eligible'] = predictions

        # Sort by Eligibility (1 first), then by GPA (Highest first)
        df_sorted = df.sort_values(by=['status_eligible', 'rata_rata_akhir'], ascending=[False, False])

        results = df_sorted.to_dict(orient='records')

        return jsonify({
            "status": "success",
            "message": "Data successfully processed and ranked",
            "data": results
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process data: {str(e)}"}), 500


@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    try:
        data = request.json
        # Expecting a list of 6 semester averages [Sem1, Sem2, Sem3, Sem4, Sem5, Sem6]
        sem_avgs = data['semesters']

        # Prepare input for the Forecaster model
        X_input = np.array(sem_avgs).reshape(1, -1)
        X_scaled = forecaster_scaler.transform(X_input)

        # Run prediction
        prediction = forecaster.predict(X_scaled)

        # CLAMP FUNCTION: Force the AI's math to stay between 0 and 100
        def clamp(val):
            return round(min(100.0, max(0.0, float(val))), 2)

        return jsonify({
            "sem7": clamp(prediction[0][0]),
            "sem8": clamp(prediction[0][1]),
            "sem9": clamp(prediction[0][2]),
            "sem10": clamp(prediction[0][3]),
            "sem11": clamp(prediction[0][4]),
            "sem12": clamp(prediction[0][5])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)