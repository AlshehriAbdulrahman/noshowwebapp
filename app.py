import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Load the saved models
encoder_file = "ordinal_encoder.joblib"
scaler_file = "min_max_scaler.joblib"
joblib_file = "best_random_forest_model_s.joblib"

loaded_encoder = joblib.load(encoder_file)
loaded_scaler = joblib.load(scaler_file)
loaded_model = joblib.load(joblib_file)

# Define the prediction function
def predict_from_csv(file_path):
    df = pd.read_csv(file_path)

    # Convert date columns to datetime
    df['appt_slot_date'] = pd.to_datetime(df['appt_slot_date'])
    df['appt_date'] = pd.to_datetime(df['appt_date'])

    # Calculate the days_difference
    df['days_difference'] = (df['appt_slot_date'] - df['appt_date']).dt.days

    input_columns = ['facilitytype', 'appt_type', 'service_name', 'facility_name', 'directorate_name', 'year', 'region', 'days_difference']
    test_data = df[input_columns]
    test_data = test_data.astype(str)

    test_data_encoded = loaded_encoder.transform(test_data)
    test_data_scaled = loaded_scaler.transform(test_data_encoded)

    predictions = loaded_model.predict(test_data_scaled)

    df['Prediction'] = predictions

    df['Prediction'] = np.where(predictions == 0, "No Show", "Show")
    print(df['Prediction'].value_counts())

    output_file = "predictions.csv"
    df.to_csv(output_file, index=False)

    return output_file

# Define the Gradio interface
interface = gr.Interface(
    fn=predict_from_csv,
    inputs=gr.File(label='Upload CSV file', type='filepath', file_count='single'),
    outputs=gr.File(label='Download CSV file with predictions'),
    title="Appointment Status Prediction in Saudi Arabian Health Care System",
    description="Upload a CSV file with the required columns to get predictions."
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch(debug=True)
