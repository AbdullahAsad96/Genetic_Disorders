import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the data
df = pd.read_csv('/content/genetic_diseases_dataset.csv')

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

def get_disease_info():
    # Display available diseases
    print("\nAvailable Diseases:")
    for i, disease in enumerate(df['Disease Name'].unique(), 1):
        print(f"{i}. {disease}")

    # Get user input for disease
    disease_input = input("\nEnter disease name from the list above: ")

    if disease_input in df['Disease Name'].values:
        disease_info = df[df['Disease Name'] == disease_input].iloc[0]

        print("\nDisease Information:")
        print(f"Disease: {disease_info['Disease Name']}")
        print(f"Genes Involved: {disease_info['Gene(s) Involved']}")
        print(f"Inheritance Pattern: {disease_info['Inheritance Pattern']}")
        print(f"Symptoms: {disease_info['Symptoms']}")
        print(f"Severity Level: {disease_info['Severity Level']}")
        print(f"Risk Assessment: {disease_info['Risk Assessment']}")
        print(f"Treatment Options: {disease_info['Treatment Options']}")

        return disease_info
    else:
        print("Disease not found in database.")
        return None

def check_symptoms():
    print("\nEnter symptoms (separate multiple symptoms with commas):")
    symptoms_input = input().strip().lower()
    symptoms_list = [s.strip() for s in symptoms_input.split(',')]

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
        # Sort by number of matching symptoms
        matching_diseases.sort(key=lambda x: x['matches'], reverse=True)

        print("\nPotential Matching Diseases:")
        print("----------------------------")
        for match in matching_diseases:
            print(f"\nDisease: {match['disease']}")
            print(f"Matching Symptoms Count: {match['matches']}")
            print(f"Disease Symptoms: {match['symptoms']}")
            print(f"Severity Level: {match['severity']}")
            print(f"Risk Assessment: {match['risk']}")
    else:
        print("\nNo matching diseases found for the given symptoms.")

def main():
    while True:
        print("\n=== Genetic Disease Information System ===")
        print("1. Search by Disease")
        print("2. Check Symptoms")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ")

        if choice == '1':
            get_disease_info()
        elif choice == '2':
            check_symptoms()
        elif choice == '3':
            print("Thank you for using the system. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if _name_ == "_main_":
    main()
