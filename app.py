from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('learning_disability_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    reading_age = float(request.form['reading_age'])
    math_score = float(request.form['math_score'])

    # Prepare input for prediction
    user_input = pd.DataFrame([[reading_age, math_score]], columns=['Reading Age Assessment', 'Math Score'])
    user_input_scaled = scaler.transform(user_input)

    # Make prediction
    prediction = model.predict(user_input_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)

    return render_template('result.html', prediction=predicted_label[0])

if __name__ == '__main__':
    app.run(debug=True)