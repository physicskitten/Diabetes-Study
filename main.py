# Import Libraries
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
sns.set(style="whitegrid")  # Use Seaborn's whitegrid theme for plots
plt.style.use("ggplot")  # Use ggplot style for consistent plot visuals

# Load Dataset
df = pd.read_csv('/kaggle/input/diabetes-data-set/diabetes.csv')  # Read dataset into DataFrame

# Initial Exploration
def explore_data(df):
    print("First 5 rows of the dataset:")
    print(df.head())  # Show first 5 rows to get a feel of the data
    print("\nDataset Shape:", df.shape)  # Show dataset dimensions (rows, columns)
    print("\nMissing values in each column:")
    print(df.isna().sum())  # Count missing values per column
    print(f"\nNumber of duplicated rows: {df.duplicated().sum()}")  # Check for duplicate rows
    print("\nStatistical Summary:")
    print(df.describe())  # Get basic stats (mean, std, etc.) for numeric columns
    
    # Visualizing target class distribution
    plt.figure(figsize=(10,5))
    sns.countplot(x='Outcome', data=df, palette='Set2')  # Show class imbalance (0/1 count of Outcome)
    plt.title('Target Class Distribution')
    plt.show()

    # Correlation Heatmap to see how features are related
    plt.figure(figsize=(15,10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')  # Heatmap for feature correlations
    plt.title('Feature Correlation Heatmap')
    plt.show()

    # Crosstab for Pregnancies vs Outcome (how often pregnant women have diabetes)
    print("\nCrosstab of Pregnancies and Outcome:")
    print(pd.crosstab(df.Pregnancies, df.Outcome))  # Cross-tab of pregnancies and diabetes outcome

explore_data(df)

# Drop any duplicate rows in the dataset
df.drop_duplicates(inplace=True)

# Splitting Features and Target
X = df.drop('Outcome', axis=1)  # Features (all columns except Outcome)
y = df['Outcome']  # Target variable (Outcome column)

# Handling class imbalance using RandomOverSampler (over-sample the minority class)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)  # Resample data to balance the classes

# Split data into train and test sets (30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

# Feature Scaling (standardize features to mean=0 and variance=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit scaler on training data and transform
X_test_scaled = scaler.transform(X_test)  # Transform test data using the same scaler

# Define models with hyperparameters for tuning (some basic models)
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Define grid search parameters for tuning models (optional, if needed)
param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7]},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Decision Tree': {'max_depth': [3, 5, 7]},
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [5, 10]},
    'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
}

# Function to train, evaluate, and visualize the model performance
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    # If hyperparameters are defined for the model, use GridSearchCV to find the best ones
    grid = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1) if name in param_grids else None
    if grid:
        grid.fit(X_train, y_train)  # Train model with cross-validation and hyperparameter tuning
        best_model = grid.best_estimator_  # Use the best model found by GridSearchCV
    else:
        best_model = model  # If no grid search, just train the model as-is
        best_model.fit(X_train, y_train)

    # Perform cross-validation to check model's accuracy across different data splits
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n{name} Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)  # Calculate accuracy
    f1 = f1_score(y_test, y_pred)  # Calculate F1 score
    print(f"\n{name} Test Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))  # Detailed performance metrics
    
    # Plot ROC Curve and calculate AUC
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)  # Calculate Area Under Curve (AUC)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')  # Plot ROC curve for the model
    
    # Plot Confusion Matrix to visualize prediction results
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return acc, f1

# Store results for comparison across models
results = {}

# Loop through each model and evaluate it
plt.figure(figsize=(10, 6))
for name, model in models.items():
    acc, f1 = evaluate_model(name, model, X_train_scaled, X_test_scaled, y_train, y_test)
    results[name] = {'Accuracy': acc, 'F1 Score': f1}  # Save accuracy and F1 score for later comparison

# Plot ROC Curves for all models
plt.plot([0, 1], [0, 1], 'k--')  # Add diagonal line for reference
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Visualizing Model Performance Comparison (accuracy and F1 score)
results_df = pd.DataFrame(results).T  # Convert results to DataFrame for easy plotting
results_df.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'salmon'])
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.show()
