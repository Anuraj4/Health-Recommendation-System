import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('data/heart_preprocessed.csv')

# Assuming your dataset has 'target' as the label column and others are features
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to a file
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as heart_disease_model.pkl")
