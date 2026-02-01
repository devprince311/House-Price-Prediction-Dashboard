# =========================================================
# 🏡 KING COUNTY HOUSE PRICE PREDICTION PROJECT
# ---------------------------------------------------------
# Predict housing prices for real-estate investment analysis
# Includes full EDA, modeling, and business insights
# =========================================================

# 1️⃣ Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import joblib

sns.set(style="whitegrid")

# 2️⃣ Load Dataset
print("Loading King County House Sales dataset...")
df_live = pd.read_csv('kc_house_data.csv')

print(f"✅ Loaded {len(df_live)} housing records")
print(f"🏠 Price range: ${df_live['price'].min():,.0f} - ${df_live['price'].max():,.0f}")
print(f"📊 Average price: ${df_live['price'].mean():,.0f}")
print(f"🧱 Dataset columns: {list(df_live.columns)}")

# 3️⃣ Feature Selection and Cleaning
features = [
    'sqft_living', 'bedrooms', 'bathrooms', 'floors', 'waterfront',
    'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built'
]

df_live = df_live[features + ['price']].dropna()
X_live = df_live[features]
y_live = df_live['price']

print(f"📉 After preprocessing: {len(df_live)} records with {len(features)} features")

# 4️⃣ Data Visualization (EDA)
# ----------------------------------

# Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df_live['price'], bins=40, kde=True, color='royalblue')
plt.title("Distribution of House Prices")
plt.xlabel("Price ($)")
plt.ylabel("Count")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df_live.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# Pairplot of Key Features
key_features = ['price', 'sqft_living', 'grade', 'bathrooms', 'view', 'sqft_above']
sns.pairplot(df_live[key_features], diag_kind='kde')
plt.suptitle("Pairwise Relationships Between Key Variables", y=1.02)
plt.show()

# Boxplots for Categorical Features
plt.figure(figsize=(8,5))
sns.boxplot(x='waterfront', y='price', data=df_live)
plt.title("Price Distribution: Waterfront vs Non-Waterfront")
plt.xlabel("Waterfront (0 = No, 1 = Yes)")
plt.ylabel("Price ($)")
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='grade', y='price', data=df_live)
plt.title("Price Distribution by Property Grade")
plt.xlabel("Grade")
plt.ylabel("Price ($)")
plt.show()

# Scatter + Regression
plt.figure(figsize=(8,5))
sns.regplot(x='sqft_living', y='price', data=df_live, scatter_kws={'alpha':0.6})
plt.title("Living Area vs Price (Regression Line)")
plt.xlabel("Living Area (sqft)")
plt.ylabel("Price ($)")
plt.show()

# 5️⃣ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_live, y_live, test_size=0.2, random_state=42)

# 6️⃣ Build Pipeline (Scaling → Polynomial → Ridge)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=0.1))
])

# 7️⃣ Fit Model
pipeline.fit(X_train, y_train)

# 8️⃣ Evaluate Performance
r2_train = pipeline.score(X_train, y_train)
r2_test = pipeline.score(X_test, y_test)
print(f"🎯 R² on Training Data: {r2_train:.4f}")
print(f"🧪 R² on Test Data: {r2_test:.4f}")

# 9️⃣ Save the Model
joblib.dump(pipeline, "ridge_poly_model.pkl")

# 🔟 Model Performance Visualizations
# Actual vs Predicted
y_pred = pipeline.predict(X_test)
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices ($)")
plt.ylabel("Predicted Prices ($)")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=40, kde=True, color='purple')
plt.title("Distribution of Model Residuals (Errors)")
plt.xlabel("Residual (Actual - Predicted)")
plt.show()

# 🔢 Feature Importance (Top Polynomial Coefficients)
feature_names = pipeline.named_steps['poly'].get_feature_names_out(features)
coefs = pipeline.named_steps['ridge'].coef_

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values('AbsCoef', ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title("Top 10 Most Influential Features (Polynomial Ridge)")
plt.show()

# 🔍 Example Prediction (Business Case)
example = np.array([[2500, 3, 2.5, 2, 0, 2, 4, 8, 2200, 300, 2010]])
predicted_price = pipeline.predict(example)
print(f"🏡 Predicted price for example house: ${predicted_price[0]:,.2f}")

# Feature Impact Simulation
print("\nFeature Impact Analysis:")
base_features = [2500, 3, 2.5, 2, 0, 2, 4, 8, 2200, 300, 2010]

# Waterfront
wf_features = base_features.copy(); wf_features[4] = 1
wf_price = pipeline.predict(np.array([wf_features]))
print(f"🌊 With waterfront: ${wf_price[0]:,.2f} (+${wf_price[0]-predicted_price[0]:,.2f})")

# High Grade
hg_features = base_features.copy(); hg_features[7] = 11
hg_price = pipeline.predict(np.array([hg_features]))
print(f"🏗️ With high grade (11): ${hg_price[0]:,.2f} (+${hg_price[0]-predicted_price[0]:,.2f})")
