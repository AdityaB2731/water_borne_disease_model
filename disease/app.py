import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ---------------------------
# Generate Synthetic Dataset and Train Model
# ---------------------------
@st.cache_data
def generate_and_train_model():
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame({
        'turbidity': np.random.uniform(1, 10, n),
        'pH': np.random.uniform(6, 8, n),
        'bacteria': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'rainfall': np.random.uniform(0, 200, n),
        'cases_last_week': np.random.randint(0, 30, n),
        'fever': np.random.choice([0, 1], n),
        'diarrhea': np.random.choice([0, 1], n),
        'abdominal_pain': np.random.choice([0, 1], n),
        'season': np.random.choice(['summer', 'monsoon', 'winter'], n)
    })

    data = pd.get_dummies(data, columns=['season'])

    def assign_disease(row):
        if row['bacteria'] == 1 and row['turbidity'] > 7 and row['fever'] and row['diarrhea']:
            return 'cholera'
        elif row['bacteria'] == 1 and row['fever'] and row['abdominal_pain']:
            return 'typhoid'
        elif row['bacteria'] == 1 and row['diarrhea'] and row['rainfall'] > 100:
            return 'giardiasis'
        else:
            return 'none'

    data['disease'] = data.apply(assign_disease, axis=1)

    X = data.drop('disease', axis=1)
    y = data['disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and training column order
    joblib.dump(model, 'disease_predictor.pkl')
    training_columns = X_train.columns.tolist()
    joblib.dump(training_columns, 'training_columns.pkl')

    return model, X_test, y_test

# Train the model
model, X_test, y_test = generate_and_train_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Water-Borne Disease Predictor Prototype")

st.sidebar.header("Input Environment and Symptoms")

turbidity = st.sidebar.slider("Water Turbidity (1-10 NTU)", 1.0, 10.0, 5.0)
pH = st.sidebar.slider("Water pH (6-8)", 6.0, 8.0, 7.0)
bacteria = st.sidebar.selectbox("Bacteria Presence", [0, 1])
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 200.0, 50.0)
cases_last_week = st.sidebar.slider("Reported Cases Last Week", 0, 30, 5)
fever = st.sidebar.selectbox("Fever", [0, 1])
diarrhea = st.sidebar.selectbox("Diarrhea", [0, 1])
abdominal_pain = st.sidebar.selectbox("Abdominal Pain", [0, 1])
season = st.sidebar.selectbox("Season", ["summer", "monsoon", "winter"])

season_dict = {"season_summer": 0, "season_monsoon": 0, "season_winter": 0}
season_dict[f"season_{season}"] = 1

input_data = pd.DataFrame([{
    "turbidity": turbidity,
    "pH": pH,
    "bacteria": bacteria,
    "rainfall": rainfall,
    "cases_last_week": cases_last_week,
    "fever": fever,
    "diarrhea": diarrhea,
    "abdominal_pain": abdominal_pain,
    **season_dict
}])

if st.sidebar.button("Predict Disease"):
    # Load the correct training column order
    training_columns = joblib.load('training_columns.pkl')

    # Reindex input data to match training columns
    input_data = input_data.reindex(columns=training_columns, fill_value=0)

    prediction = model.predict(input_data)[0]
    st.subheader("Predicted Water-Borne Disease")
    if prediction == 'none':
        st.success("No significant disease detected.")
    else:
        st.warning(f"Possible Disease: {prediction}")

# Model Performance
st.subheader("Model Performance on Test Data")
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
st.json(report)

cm = classification_report(y_test, y_pred, output_dict=False)
st.write("### Sample Confusion Matrix")
st.text(cm)
