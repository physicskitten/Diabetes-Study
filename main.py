# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")
sns.set()
plt.style.use("ggplot")

# Load Dataset
df = pd.read_csv('/kaggle/input/diabetes-data-set/diabetes.csv')

# Initial Exploration
def explore_data(df):
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Shape:", df.shape)
    print("\nMissing values in each column:")
    print(df.isna().sum())
    print(f"\nNumber of duplicated rows: {df.duplicated().sum()}")
    print("\nStatistical Summary:")
    print(df.describe())
    print("\nTarget class distribution (counts):")
    print(df['Outcome'].value_counts())
    print("\nTarget class distribution (%):")
    print(df['Outcome'].value_counts() * 100 / len(df))
    sns.countplot(x='Outcome', data=df)
    plt.title('Class Distribution')
    plt.show()

explore_data(df)

# Drop duplicates if any
df.drop_duplicates(inplace=True)

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

# Define models with some parameter tuning
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Hyperparameter tuning grid (optional)
param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7]},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Decision Tree': {'max_depth': [3, 5, 7]},
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [5, 10]},
    'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
}

# Function to train and evaluate models
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    grid = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1) if name in param_grids else None
    if grid:
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return acc, f1

# Store results for comparison
results = {}

# Train and Evaluate all models
for name, model in models.items():
    acc, f1 = evaluate_model(name, model, X_train_scaled, X_test_scaled, y_train, y_test)
    results[name] = {'Accuracy': acc, 'F1 Score': f1}

# Visualizing model performance
results_df = pd.DataFrame(results).T
results_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.show()
