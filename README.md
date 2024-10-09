# Overview
In this project, we explore a dataset containing health-related features to predict whether a person has diabetes. Multiple models such as Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Trees, Random Forests, and Gradient Boosting are trained and evaluated.

## The pipeline involves:
- Data exploration and visualization
- Handling class imbalance using `RandomOverSampler`
- Model training and evaluation
- Hyperparameter tuning using `GridSearchCV`
- Performance comparison of models

## Dataset
The dataset used is a diabetes dataset, which can be found [here](#). It contains the following features:

- **Pregnancies**: Number of pregnancies
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: A function which scores likelihood of diabetes based on family history
- **Age**: Age of the patient
- **Outcome**: The target variable (1 for diabetic, 0 for non-diabetic)

# Model Evaluation
Six machine learning models are evaluated:

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Support Vector Machine (SVM)**
4. **Decision Tree**
5. **Random Forest**
6. **Gradient Boosting**

The models are tuned using `GridSearchCV`, and performance is evaluated based on:

- **Accuracy**: The proportion of correctly classified instances.
- **F1 Score**: A balance between precision and recall.

Additionally, confusion matrices are plotted for each model.

## Results
The results are compared using a bar chart of model performance based on accuracy and F1 score. The confusion matrix for each model is also displayed to visualize the performance on both classes (diabetic and non-diabetic).
