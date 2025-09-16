import requests
import json
import pandas as pd
import joblib

# Load document text
with open('dummy_input.txt', 'r') as file:
    document_text = file.read()

# Define Groq API key and endpoint
API_KEY = 'YOUR_GROQ_API_KEY'
API_URL = 'https://api.groq.ai/v1/parse'

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# Request payload
payload = {
    "prompt": """
You are an intelligent parser. Extract the following structured information from the text:
- turbidity (float)
- pH (float)
- bacteria (0 or 1)
- rainfall (float, in mm)
- cases_last_week (int)
- season (summer, monsoon, winter)

Return only a JSON object.

Example input text:
\"\"\"
The turbidity of the sample is measured at 5.8 NTU.
pH level recorded: 7.1
Bacteria presence detected: Yes
Rainfall in the past week: 130 mm
Reported disease cases last week: 18
Current season: monsoon
\"\"\"
""",
    "document": document_text
}

# Send request to Groq API
response = requests.post(API_URL, headers=headers, json=payload)
parsed_json = response.json()

# Example parsed_json (simulate response if needed)
# parsed_json = {
#     "turbidity": 5.8,
#     "pH": 7.1,
#     "bacteria": 1,
#     "rainfall": 130.0,
#     "cases_last_week": 18,
#     "season": "monsoon"
# }

# Prepare data for ML model
season_dict = {"season_summer": 0, "season_monsoon": 0, "season_winter": 0}
season_dict[f"season_{parsed_json['season']}"] = 1

input_data = pd.DataFrame([{
    "turbidity": parsed_json['turbidity'],
    "pH": parsed_json['pH'],
    "bacteria": parsed_json['bacteria'],
    "rainfall": parsed_json['rainfall'],
    "cases_last_week": parsed_json['cases_last_week'],
    **season_dict
}])

training_columns = ['turbidity', 'pH', 'bacteria', 'rainfall', 'cases_last_week',
                    'season_summer', 'season_monsoon', 'season_winter']
input_data = input_data.reindex(columns=training_columns, fill_value=0)

# Load the saved XGBoost model
model = joblib.load('xgb_model.pkl')

# Make prediction
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

print(f"Predicted Outbreak Risk: {'High' if prediction == 1 else 'Low'} (Probability: {probability:.2f})")
