# main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('data/heart.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Basic preprocessing
# Assuming 'target' is the column to predict
# If the target column has a different name, replace 'target' with the actual column name
target_column = 'target'  # Replace with the correct column name if different
if target_column not in data.columns:
    raise ValueError(f"'{target_column}' column not found in dataset")

X = data.drop(target_column, axis=1)
y = data[target_column]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importances (optional)
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)


print("\nFeature Importances:")
print(importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()
