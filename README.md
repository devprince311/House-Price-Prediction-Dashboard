# 🏠 House Price Prediction Dashboard

## 📌 Overview

This project focuses on analyzing and predicting house prices using real-world real estate data. It was developed as part of my internship to apply **data analytics, machine learning, and visualization techniques** to uncover key factors influencing housing prices and present insights through an interactive dashboard.

The goal was not only to build predictive models, but also to **extract actionable insights** that explain *why* prices vary across properties.


## 📊 Dataset

* **Size:** 1,000+ real estate listings
* **Features include:**

  * Living area
  * Number of bedrooms and bathrooms
  * Property grade
  * Waterfront and view indicators
  * Location-related attributes

The dataset was cleaned and preprocessed to handle missing values, outliers, and inconsistent entries.

---

## 🔍 Methodology

### 1. Data Cleaning & Preprocessing

* Removed duplicates and invalid records
* Handled missing values
* Scaled and transformed numerical features
* Encoded categorical variables where required

### 2. Exploratory Data Analysis (EDA)

* Studied price distributions and correlations
* Identified key relationships between features and house prices
* Visualized trends using charts and summary statistics

### 3. Feature Engineering

* Created meaningful derived features
* Analyzed diminishing returns for bedroom count
* Evaluated the impact of premium features such as waterfront and scenic views

### 4. Model Building & Evaluation

* Trained and compared multiple regression models
* Performance metrics used:

  * **RMSE (Root Mean Squared Error)**
  * **R² Score**
* Selected models based on predictive accuracy and interpretability

---

## 📈 Key Insights

* **Living area** and **property grade** are the strongest drivers of house prices
* Adding bedrooms increases price only up to **3–4 bedrooms**, after which returns diminish
* Properties with **waterfront** or **view** features consistently command higher prices
* Structural and qualitative features often outweigh sheer size alone

---

## 🧠 Dashboard

An interactive dashboard was built to:

* Explore price trends dynamically
* Compare feature impacts visually
* Support data-driven decision-making

*(Dashboard screenshots or deployment link can be added here)*

---

## 🛠️ Tools & Technologies

* **Python** (Pandas, NumPy, Scikit-learn)
* **Data Visualization:** Matplotlib/Streamlit 
* **Machine Learning:** Regression models
* **Evaluation Metrics:** RMSE, R²

---

## 🚀 Results

The project demonstrates how combining **EDA, feature engineering, and regression modeling** can deliver both accurate predictions and meaningful business insights for the real estate domain.

---

## 📁 Project Structure

```
├── data/
│   └── housing_data.csv
├── notebooks/
│   └── analysis.ipynb
├── dashboard/
│   └── app.py
├── README.md
```

---

## 📌 Future Improvements

* Incorporate location-based geospatial analysis
* Try advanced models (XGBoost, Random Forest)
* Deploy dashboard publicly
* Add time-based price trend analysis

---

## 👤 Author

**Dev Prince Thachil**
Computer Engineering | Data & Product Analytics

---
