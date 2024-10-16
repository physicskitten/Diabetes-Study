# Import Libraries
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, classification_report, roc_curve, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
import warnings

# Ignore warnings to keep output clean
warnings.filterwarnings("ignore")

# Seaborn and Matplotlib styles
sns.set(style="whitegrid")  # Seaborn's whitegrid theme for consistency
plt.style.use("ggplot")  # Matplotlib ggplot style for better plot visuals

# Set up argument parsing
parser = argparse.ArgumentParser(description='Diabetes Prediction Model')
parser.add_argument('--data', type=str, default='diabetes.csv', help='Path to the dataset')
parser.add_argument('--test_size', type=float, default=0.3, help='Test set size (0 to 1)')
args = parser.parse_args()

# Load Dataset
df = pd.read_csv(args.data)  # Read dataset into DataFrame

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
    
    # Visualize target class distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(x='Outcome', data=df, palette='Set2')
    plt.title('Target Class Distribution')
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.show()

    # Crosstab for Pregnancies vs Outcome
    print("\nCrosstab of Pregnancies and Outcome:")
    print(pd.crosstab(df.Pregnancies, df.Outcome))

explore_data(df)

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Splitting Features and Target
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']  # Target

# Handling class imbalance using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Split data into train and test sets (using the test size from arguments)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=args.test_size, random_state=42, stratify=y_resampled)

# Feature Scaling (standardize features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Hyperparameter grids for tuning models
param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7]},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Decision Tree': {'max_depth': [3, 5, 7]},
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [5, 10]},
    'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
}

# Train, evaluate, and visualize model performance
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    grid = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1) if name in param_grids else None
    best_model = grid.best_estimator_ if grid else model
    if grid: 
        grid.fit(X_train, y_train)
    else: 
        best_model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n{name} Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Test set predictions
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{name} Test Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    # ROC Curve and AUC
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return acc, f1

# Model performance comparison
results = {}
plt.figure(figsize=(10, 6))
for name, model in models.items():
    acc, f1 = evaluate_model(name, model, X_train_scaled, X_test_scaled, y_train, y_test)
    results[name] = {'Accuracy': acc, 'F1 Score': f1}

# Plot ROC curves
plt.plot([0, 1], [0, 1], 'k--')  # Reference diagonal line
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Visualize model performance comparison
results_df = pd.DataFrame(results).T  # Convert results to DataFrame
results_df.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'salmon'])
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.show()

# Save results to CSV
results_df.to_csv('model_results.csv', index=True)
print("Model results saved to 'model_results.csv'.")
