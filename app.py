import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import chardet
import plotly
import plotly.express as px

st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

with tab1:
    # Input fields
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Gender", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Convert categorical inputs to numerical
    sex = 1 if sex == "Female" else 0
    chest_pain_dict = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "Yes" else 0
    rest_ecg_dict = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(rest_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope_dict = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Create a DataFrame using numerical encodings
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain_dict],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [rest_ecg_dict],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope_dict]
    })

    algoNames = ["Logistic Regression", "Support Vector Machine", "Random Forest", "XG Boost Classifier", "BEACONX_VWDMEC"]
    modelNames = ["LogisticR .pickle", "SVM.pickle", "RFC1.pickle", "XGB.pickle", "LogisticR .pickle"]

    # Function to make predictions
    def predict_heart_disease(data):
        predictions = []
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for modelName in modelNames:
            model_path = os.path.join(base_dir, modelName)
            model = pickle.load(open(model_path, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions

    # Submit button
    if st.button("Submit"):
        st.subheader("---------- RESULTS ----------")
        st.markdown("----------------------------")

        result = predict_heart_disease(input_data)

        for i in range(len(result)):
            st.subheader(algoNames[i])
            if result[i][0] == 0:
                st.write("No Heart Disease detected")
            else:
                st.write("Heart Disease detected")
            st.markdown("------------------------")


with tab2:
    st.title("Bulk Prediction")

    st.subheader("Follow these instructions before uploading the CSV file:")
    st.info("""
        1. The CSV file should not contain NAN values\n
        2. The CSV file should not contain the target column\n
        3. The CSV file should have the following columns in order:\n\n
            - Age: age of the patient (in years) \n
            - Sex: Gender of the patient [0:male, 1:female]\n
            - Chest Pain Type: Type of chest pain experienced ["3: Typical Angina", "0: Atypical Angina", "1: Non-Anginal Pain", "2: Asymptomatic"]\n
            - Resting Blood Pressure (in mm Hg)\n
            - Cholesterol (in mg/dl)\n
            - Fasting Blood Sugar > 120 mg/dl [1: true; 0: false]\n
            - Resting ECG results [0: normal; 1: having ST-T wave abnormality; 2: showing probable or definite left ventricular hypertrophy by Estes' criteria]\n
            - Maximum Heart Rate Achieved [Numeric value achieved between 60 and 220]
            - Exercise Angina: Exercise induced Angina [1: yes, 0: no]\n
            - Oldpeak: oldpeak = ST (Numeric value achieved in depression)\n
            - ST Slope: Slope of the peak exercise ST segment [0: upsloping; 1: flat; 2: downsloping]\n\n"""
    )

    # --- Sidebar uploader ---
    st.sidebar.header("Bulk Prediction")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    # --- Configuration ---
    OUTPUT_FILE_NAME = "test_output.csv" 

    if uploaded_file is not None:
        
        # 1. Clear previous session state and output file if re-uploading
        if 'prediction_results' in st.session_state:
            del st.session_state['prediction_results']
        if os.path.exists(OUTPUT_FILE_NAME):
            os.remove(OUTPUT_FILE_NAME)

        # --- Detect encoding ---
        raw_data = uploaded_file.read()
        encoding = chardet.detect(raw_data)['encoding']
        uploaded_file.seek(0)  # reset file pointer for reading again

        # --- Read CSV safely (auto-detect delimiter) ---
        try:
            input_data = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine='python')
        except Exception:
            uploaded_file.seek(0)
            input_data = pd.read_csv(uploaded_file, encoding=encoding)

        st.write(f"✅ File loaded with {len(input_data)} rows and {len(input_data.columns)} columns.")

        # --- Data Cleaning and Preparation ---
        input_data = input_data.loc[:, ~input_data.columns.str.contains('^Unnamed')]
        input_data.columns = input_data.columns.str.strip()

        for col in input_data.columns:
            temp_series = input_data[col].astype(str).str.strip()
            input_data[col] = pd.to_numeric(temp_series, errors='coerce') 

        input_data = input_data.fillna(0)
        
        # --- Check for minimum rows needed ---
        if len(input_data) == 0:
            st.error("The uploaded file contains no data rows after cleaning. Please check the file.")
            st.stop()

        # --- Prediction Logic ---
        model = pickle.load(open("RFC1.pickle", 'rb')) 

        # Add Prediction column initialized as numeric NaN to avoid string type-casting error
        df_results = input_data.copy()
        df_results["Prediction"] = np.nan

        # Make predictions
        for i in range(len(df_results)):
            arr = df_results.iloc[i, :-1].values
            df_results.loc[i, 'Prediction'] = model.predict([arr])[0]

        # Save the final results to Streamlit Session State and a temporary file
        st.session_state['prediction_results'] = df_results
        df_results.to_csv(OUTPUT_FILE_NAME, index=False)
        st.success(f"✅ Predictions made successfully for **{len(df_results)} rows**! Download the results below.")

    # --- Display and Download Section ---
    if 'prediction_results' in st.session_state:
        st.subheader("Prediction Results")
        
        # Display the DataFrame from session state
        st.dataframe(st.session_state['prediction_results'])

        # Download button using the file path
        with open(OUTPUT_FILE_NAME, "rb") as f:
            st.download_button(
                "⬇️ Download Results", 
                f, 
                file_name="Bulk_Prediction_Results.csv", 
                mime="text/csv"
            )


with tab3:
    st.subheader("🔍 Model Performance Overview")
    st.write("""
             Coronary artery disease (CAD), a leading form of heart disease, occurs when arterial blockages reduce
blood flow to the heart, often causing heart attacks and other serious conditions. Early detection is crucial for effective management and treatment.
             

    Machine learning models can help doctors and researchers identify patients who are at high risk 
    of heart disease based on various clinical parameters.  
    Below is a comparison of different models and how well they performed on the dataset.
    """)

    data = {
        'XG Boost': 0.89,
        'Random Forest Classifier': 0.87,
        'Support Vector Machine': 0.85,
        'Logistic Regression': 0.86
    }
    
    models = list(data.keys())
    accuracy = list(data.values())
    df = pd.DataFrame({'Models': models, 'Accuracy': accuracy})
    
    fig = px.bar(df, x='Models', y='Accuracy',
                 title='Model Accuracy Comparison',
                 text='Accuracy',
                 color='Models',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_range=[0, 1])  
    
    # Modern Streamlit container scaling configuration
    st.plotly_chart(fig, width='stretch')
