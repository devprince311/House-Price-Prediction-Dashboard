
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import joblib

print("Loading King County House Sales dataset...")
try:
    df_live = pd.read_csv('kc_house_data.csv')
except FileNotFoundError:
    print("Error: kc_house_data.csv not found.")
    exit(1)

features = [
    'sqft_living', 'bedrooms', 'bathrooms', 'floors', 'waterfront',
    'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built'
]

# Basic cleaning
df_live = df_live[features + ['price']].dropna()
X_live = df_live[features]
y_live = df_live['price']

print(f"Training on {len(df_live)} records...")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_live, y_live, test_size=0.2, random_state=42)

# Build Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=0.1))
])

# Fit Model
print("Fitting model...")
pipeline.fit(X_train, y_train)

# Evaluate
r2_train = pipeline.score(X_train, y_train)
r2_test = pipeline.score(X_test, y_test)
print(f"R² on Training Data: {r2_train:.4f}")
print(f"R² on Test Data: {r2_test:.4f}")

# Save the Model
print("Saving model to ridge_poly_model.pkl...")
joblib.dump(pipeline, "ridge_poly_model.pkl")
print("Done.")
