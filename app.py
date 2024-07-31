from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

# Load the preprocessed dataset
data = pd.read_csv('data/heart_preprocessed.csv')

# Define features and target
X = data.drop(columns=['target'])
y = data['target']

# Train the model (use the best model from the previous day)
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save the model
joblib.dump(model, 'model.pkl')

# Load the model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Output prediction
    output = prediction[0]
    
    return render_template('index.html', prediction_text='Heart Disease Prediction: {}'.format('Positive' if output == 1 else 'Negative'))

if __name__ == "__main__":
    app.run(debug=True)
