import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ---------------------------
# 1. Generate Synthetic Dataset
# ---------------------------
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'turbidity': np.random.uniform(1, 10, n),
    'pH': np.random.uniform(6, 8, n),
    'bacteria': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    'rainfall': np.random.uniform(0, 200, n),
    'cases_last_week': np.random.randint(0, 30, n),
    'season': np.random.choice(['summer','monsoon','winter'], n)
})

# Encode season (consistent dummy columns)
data = pd.get_dummies(data, columns=['season'])
# Ensure all season columns are present
for col in ["season_summer", "season_monsoon", "season_winter"]:
    if col not in data.columns:
        data[col] = 0

# Drop first to avoid dummy variable trap (optional)
# OR keep all and let model decide → here we keep all for safety
X = data.drop('outbreak', axis=1, errors="ignore")

# Target variable
data['outbreak'] = ((data['bacteria'] == 1) &
                    (data['rainfall'] > 80) &
                    (data['cases_last_week'] > 10)).astype(int)

X = data.drop('outbreak', axis=1)
y = data['outbreak']

# Save training columns for later use
training_columns = X.columns.tolist()

# ---------------------------
# 2. Train Model
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# ---------------------------
# 3. Streamlit App UI
# ---------------------------
st.title("🌍 Smart Health Monitoring - Outbreak Prediction")
st.write("Prototype ML model using **XGBoost** to predict water-borne disease outbreak risk.")

st.sidebar.header("Input Parameters")

# Sidebar inputs
turbidity = st.sidebar.slider("Water Turbidity", 1.0, 10.0, 5.0)
pH = st.sidebar.slider("Water pH", 6.0, 8.0, 7.0)
bacteria = st.sidebar.selectbox("Bacteria Presence", [0, 1])
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 200.0, 50.0)
cases_last_week = st.sidebar.slider("Reported Cases Last Week", 0, 30, 5)
season = st.sidebar.selectbox("Season", ["summer", "monsoon", "winter"])

# ---------------------------
# 4. Prepare Input Data
# ---------------------------
# Encode season dynamically
season_dict = {"season_summer": 0, "season_monsoon": 0, "season_winter": 0}
season_dict[f"season_{season}"] = 1

input_data = pd.DataFrame([{
    "turbidity": turbidity,
    "pH": pH,
    "bacteria": bacteria,
    "rainfall": rainfall,
    "cases_last_week": cases_last_week,
    **season_dict
}])

# Align columns with training data
input_data = input_data.reindex(columns=training_columns, fill_value=0)

# ---------------------------
# 5. Prediction
# ---------------------------
if st.sidebar.button("Predict Outbreak Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🔮 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Outbreak (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Outbreak (Probability: {probability:.2f})")

# ---------------------------
# 6. Model Evaluation & Plots
# ---------------------------
st.subheader("📊 Model Performance on Test Data")

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
st.json(report)

cm = confusion_matrix(y_test, y_pred)

st.write("### Confusion Matrix")
st.write(cm)

# Feature Importance Plot
st.write("### Feature Importance")
fig, ax = plt.subplots()
xgb.plot_importance(model, ax=ax)
st.pyplot(fig)
