import numpy as np
import pickle

class HealthModel:
    def __init__(self):
        with open('heart_disease_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, features):
        return self.model.predict(features)

    def get_health_tips(self, prediction):
        if prediction == 1:
            return [
                "Follow a heart-healthy diet.",
                "Engage in regular physical activity.",
                "Avoid smoking and limit alcohol consumption.",
                "Maintain a healthy weight."
            ]
        else:
            return [
                "Continue following a healthy lifestyle to maintain good heart health.",
                "Regularly monitor your blood pressure and cholesterol levels.",
                "Stay active and eat a balanced diet.",
                "Avoid stress and practice relaxation techniques."
            ]
