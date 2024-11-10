import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the data
df = pd.read_csv('genetic_diseases_dataset.csv')  # Use the actual path of the file

# Clean and preprocess data
def preprocess_data(df):
    df = df.fillna('Unknown')
    encoders = {}
    categorical_columns = ['Disease Name', 'Gene(s) Involved', 'Inheritance Pattern',
                         'Symptoms', 'Severity Level', 'Risk Assessment']

    for column in categorical_columns:
        encoders[column] = LabelEncoder()
        df[column + '_encoded'] = encoders[column].fit_transform(df[column])

    return df, encoders

# Preprocess the data
processed_df, encoders = preprocess_data(df)

# Train model
features = ['Disease Name_encoded', 'Gene(s) Involved_encoded', 'Inheritance Pattern_encoded',
           'Symptoms_encoded', 'Severity Level_encoded']
X = processed_df[features]
y = processed_df['Risk Assessment_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Streamlit main interface
st.title("Genetic Disease Information System")

def get_disease_info(disease_input):
    if disease_input in df['Disease Name'].values:
        disease_info = df[df['Disease Name'] == disease_input].iloc[0]

        st.write("### Disease Information")
        st.write(f"Disease: {disease_info['Disease Name']}")
        st.write(f"Genes Involved: {disease_info['Gene(s) Involved']}")
        st.write(f"Inheritance Pattern: {disease_info['Inheritance Pattern']}")
        st.write(f"Symptoms: {disease_info['Symptoms']}")
        st.write(f"Severity Level: {disease_info['Severity Level']}")
        st.write(f"Risk Assessment: {disease_info['Risk Assessment']}")
        st.write(f"Treatment Options: {disease_info['Treatment Options']}")

    else:
        st.write("Disease not found in database.")

def check_symptoms(symptoms_input):
    symptoms_list = [s.strip().lower() for s in symptoms_input.split(',')]
    matching_diseases = []

    for _, row in df.iterrows():
        disease_symptoms = str(row['Symptoms']).lower()
        matches = sum(1 for symptom in symptoms_list if symptom in disease_symptoms)
        if matches > 0:
            matching_diseases.append({
                'disease': row['Disease Name'],
                'matches': matches,
                'symptoms': row['Symptoms'],
                'severity': row['Severity Level'],
                'risk': row['Risk Assessment']
            })

    if matching_diseases:
        matching_diseases.sort(key=lambda x: x['matches'], reverse=True)
        st.write("### Potential Matching Diseases")
        for match in matching_diseases:
            st.write(f"Disease: {match['disease']}")
            st.write(f"Matching Symptoms Count: {match['matches']}")
            st.write(f"Disease Symptoms: {match['symptoms']}")
            st.write(f"Severity Level: {match['severity']}")
            st.write(f"Risk Assessment: {match['risk']}")
            st.write("---")
    else:
        st.write("No matching diseases found for the given symptoms.")

# Interface
option = st.selectbox("Choose an option", ["Search by Disease", "Check Symptoms", "Exit"])

if option == "Search by Disease":
    disease_input = st.selectbox("Select a disease", df['Disease Name'].unique())
    get_disease_info(disease_input)

elif option == "Check Symptoms":
    symptoms_input = st.text_input("Enter symptoms (separate multiple symptoms with commas):")
    if st.button("Check"):
        check_symptoms(symptoms_input)

else:
    st.write("Thank you for using the system!")
