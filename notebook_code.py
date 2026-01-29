# 1. Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Set visual style
sns.set(style='whitegrid')
pd.set_option('display.max_columns', None)

# %% CELL

# 2. Load Data
# Ensure the csv file is in the same directory
try:
    df = pd.read_csv('student_performance.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'student_performance.csv' not found.")

df.head()

# %% CELL

# Check dataset info
print(df.info())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum().sum())

# %% CELL

# Statistical description
df.describe()

# %% CELL

# Target Variable Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['G3'], bins=20, kde=True, color='blue')
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('Grade (0-20)')
plt.show()

# %% CELL

# Correlation Matrix
plt.figure(figsize=(20, 15))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# %% CELL

# Impact of Alcohol Consumption on Grades
# Dalc: Workday alcohol, Walc: Weekend alcohol

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(x='Dalc', y='G3', data=df, ax=axes[0], palette='viridis')
axes[0].set_title('Workday Alcohol Consumption vs G3')

sns.boxplot(x='Walc', y='G3', data=df, ax=axes[1], palette='viridis')
axes[1].set_title('Weekend Alcohol Consumption vs G3')

plt.show()

# %% CELL

# Create new aggregate features

# 1. Total Alcohol Consumption
df['Total_Alc'] = df['Dalc'] + df['Walc']

# 2. Parent Average Education
df['Parent_Edu'] = (df['Medu'] + df['Fedu']) / 2

# 3. Family Support + School Support (Total Support)
# Convert yes/no to 1/0 for addition
df['schoolsup_bin'] = df['schoolsup'].map({'yes': 1, 'no': 0})
df['famsup_bin'] = df['famsup'].map({'yes': 1, 'no': 0})
df['Total_Support'] = df['schoolsup_bin'] + df['famsup_bin']

# Drop intermediate columns if desired (keeping them for now)

# 4. Target for Classification Task
# Define Pass/Fail. Usually >= 10 is Pass in the Portuguese system.
df['passed'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

print("New Features Added: Total_Alc, Parent_Edu, Total_Support, passed")
df[['Total_Alc', 'Parent_Edu', 'Total_Support', 'passed']].head()

# %% CELL

# Encoding Categorical Variables
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"Categorical Columns: {list(categorical_cols)}")

# We will use Label Encoding for binary/ordinal and One-Hot (get_dummies) for nominal
# For simplicity in this demo, we use get_dummies for all nominal, or LabelEncoder for binary.
le = LabelEncoder()

# Binary columns
binary_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# Nominal columns -> One Hot Encoding
nominal_cols = ['Mjob', 'Fjob', 'reason', 'guardian']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

print("Encoding Complete. Shape:", df.shape)

# %% CELL

# Check class distribution
sns.countplot(x='passed', data=df)
plt.title("Class Distribution (Pass vs Fail)")
plt.show()

pass_count = df['passed'].value_counts()
print("Class Counts:\n", pass_count)

# If imbalance is severe, we will use SMOTE during the model training phase.

# %% CELL

# Define Features (X) and Target (y)
# Target for Regression
y_reg = df['G3']
# Target for Classification
y_class = df['passed']

# 1. Early Prediction Features (No G1, G2)
X_early = df.drop(['G1', 'G2', 'G3', 'passed', 'schoolsup_bin', 'famsup_bin'], axis=1) # Dropping derivatives of G3/G2/G1 if any (none exist except passed)

# 2. Full Features (With G1, G2)
X_full = df.drop(['G3', 'passed', 'schoolsup_bin', 'famsup_bin'], axis=1)

print("X_early shape:", X_early.shape)
print("X_full shape:", X_full.shape)

# %% CELL

# Split Data (Using Early Set for primary demonstration)
X_train, X_test, y_train, y_test = train_test_split(X_early, y_reg, test_size=0.2, random_state=42)

# Normalization (Tree models don't strictly need it, but good for Neural Nets or KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Train/Test Split Complete.")

# %% CELL

# Initialize Models
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
lr_reg = LinearRegression()

# Train Random Forest
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

# Train XGBoost
xgb_reg.fit(X_train, y_train)
y_pred_xgb = xgb_reg.predict(X_test)

# Train Linear Regression (Baseline)
lr_reg.fit(X_train_scaled, y_train)
y_pred_lr = lr_reg.predict(X_test_scaled)

# %% CELL

def evaluate_regression(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {model_name} ---")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}\n")
    return r2

evaluate_regression(y_test, y_pred_rf, "Random Forest (Early)")
evaluate_regression(y_test, y_pred_xgb, "XGBoost (Early)")
evaluate_regression(y_test, y_pred_lr, "Linear Regression (Early)")

# %% CELL

# Results Visualization: Actual vs Predicted (Random Forest)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual G3')
plt.ylabel('Predicted G3')
plt.title('Actual vs Predicted Grades (Random Forest - Early Prediction)')
plt.show()

# %% CELL

# Split Data for Classification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_early, y_class, test_size=0.2, random_state=42)

# Apply SMOTE (Synthetic Minority Over-sampling Technique)
# This balances the training data by creating synthetic examples of the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_c, y_train_c)

print("Original Class Distribution:", y_train_c.value_counts().to_dict())
print("Resampled Class Distribution:", y_train_resampled.value_counts().to_dict())

# Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_resampled, y_train_resampled)

# Predict
y_pred_class = rf_clf.predict(X_test_c)

# Evaluation
print("\nClassification Report:\n", classification_report(y_test_c, y_pred_class))
print("Confusion Matrix:\n", confusion_matrix(y_test_c, y_pred_class))

# Plot Confusion Matrix
sns.heatmap(confusion_matrix(y_test_c, y_pred_class), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Pass/Fail)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %% CELL

feature_importances = pd.DataFrame(rf_reg.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances.importance, y=feature_importances.index)
plt.title('Feature Importance (Early Prediction Model)')
plt.show()

# %% CELL

joblib.dump(rf_reg, 'student_performance_rf_model.pkl')
print("Model saved as 'student_performance_rf_model.pkl'")

# %% CELL

