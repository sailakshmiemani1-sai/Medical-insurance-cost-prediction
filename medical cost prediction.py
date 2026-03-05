# ============================
# Medical Insurance Prediction
# ============================

# 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 📂 Load Dataset
data = pd.read_csv("D:insurance.csv")
print(data.head())

# 🧹 Preprocessing
data_encoded = pd.get_dummies(data, drop_first=True)
X = data_encoded.drop("charges", axis=1)
y = data_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📊 Exploratory Data Analysis
plt.figure(figsize=(8,5))
sns.histplot(data["charges"], bins=30, kde=True)
plt.title("Distribution of Insurance Charges")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="smoker", y="charges", data=data)
plt.title("Charges vs Smoking Status")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(data_encoded.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 🤖 Model Training: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Linear Regression Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# 🤖 Model Training: Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2 Score:", r2_score(y_test, y_pred_rf))

# 📈 Model Comparison
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [mean_absolute_error(y_test, y_pred_lr), mean_absolute_error(y_test, y_pred_rf)],
    "RMSE": [np.sqrt(mean_squared_error(y_test, y_pred_lr)), np.sqrt(mean_squared_error(y_test, y_pred_rf))],
    "R2 Score": [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_rf)]
})
print(results)

# 💾 Save Best Model
joblib.dump(rf, "insurance_model.pkl")

# ============================
# 🌐 Streamlit App (app.py)
# ============================
# Save the following in a separate file app.py and run: streamlit run app.py

"""
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("insurance_model.pkl")

st.title("Medical Insurance Cost Prediction")

age = st.number_input("Age", min_value=18, max_value=100)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)
children = st.number_input("Number of Children", min_value=0, max_value=10)
smoker = st.selectbox("Smoker", ["yes", "no"])
sex = st.selectbox("Sex", ["male", "female"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prepare input data
input_data = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "sex_male": [1 if sex=="male" else 0],
    "smoker_yes": [1 if smoker=="yes" else 0],
    "region_northwest": [1 if region=="northwest" else 0],
    "region_southeast": [1 if region=="southeast" else 0],
    "region_southwest": [1 if region=="southwest" else 0]
})

# Predict
prediction = model.predict(input_data)[0]
st.write("Predicted Insurance Cost: $", round(prediction, 2))
"""
