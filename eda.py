# eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
data = pd.read_csv('data/heart.csv')

# Display basic info and check for missing values
print("Basic Information:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

# Handling missing values (if any)
# For simplicity, let's fill missing numerical values with the median and categorical with the mode
for column in data.columns:
    if data[column].isnull().sum() > 0:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].median(), inplace=True)

# Encode categorical variables
# For this dataset, let's assume 'sex' and 'cp' are categorical
categorical_features = ['sex', 'cp']
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le

# Scale numerical features
# Assuming 'age', 'trestbps', 'chol', 'thalach', 'oldpeak' are numerical features to be scaled
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Save the preprocessed data
data.to_csv('data/heart_preprocessed.csv', index=False)

# Exploratory Data Analysis (EDA)

# Distribution of features
plt.figure(figsize=(20, 15))
for i, column in enumerate(data.columns, 1):
    plt.subplot(5, 4, i)
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Pairplot
sns.pairplot(data, diag_kind='kde', hue='target')
plt.title('Pairplot of Features')
plt.show()

# Analyzing target variable distribution
plt.figure(figsize=(6, 6))
sns.countplot(x='target', data=data)
plt.title('Distribution of Target Variable')
plt.show()

# Additional insights
# For example, let's look at the relationship between chest pain type and target
plt.figure(figsize=(8, 6))
sns.countplot(x='cp', hue='target', data=data)
plt.title('Chest Pain Type vs Target')
plt.show()

print("EDA complete. Preprocessed data saved as 'heart_preprocessed.csv'.")
