# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, mean_squared_error, 
                             r2_score, recall_score, classification_report, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings("ignore")
sns.set()
plt.style.use("ggplot")
%matplotlib inline

# Loading the Dataset
df = pd.read_csv('/kaggle/input/diabetes-data-set/diabetes.csv')

# Initial Data Exploration
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Check for missing values
print("\nMissing values in each column:")
print(df.isna().sum())

# Check for duplicated rows
duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicated rows: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"Duplicates removed. New shape: {df.shape}")

# Basic Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Class distribution of the target variable
print("\nTarget class distribution (counts):")
print(df['Outcome'].value_counts())

print("\nTarget class distribution (%):")
print(df['Outcome'].value_counts() * 100 / len(df))

# Visualizing the class imbalance
sns.countplot(x='Outcome', data=df)
plt.title('Class Distribution')
plt.show()

# Splitting Features and Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Handling class imbalance using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Train and Evaluate models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.show()
