from helper_functions import pivot
from model_training import encoding, automate_forecasting
import pandas as pd
import pickle
import os
from helper_functions import parse_date, abc, xyz, categorize_medicine, match_and_replace_medicine_name, mno
from medicine_dets import medicine_to_include_lower, mappings

def process_data(channel, processed_df_1):
    pickle_filename = f"{channel}_processed_data.pkl"

    # Check if the pickle file for the channel exists
    if os.path.exists(pickle_filename):
        with open(pickle_filename, "rb") as file:
            encoded_df = pickle.load(file)
    else:
        # Load the processed DataFrame from the pickle file
        if os.path.exists("processed_data.pkl"):
            with open("processed_data.pkl", "rb") as file:
                processed_df_1 = pickle.load(file)
        else:
            raise FileNotFoundError("Processed data file not found.")

        print("********************************************************************************")
        print("Processing channel:", channel)
        data = processed_df_1[["Date", "Clinic Name", "Channel", "Prescrption json"]]
        data = data[data["Channel"].str.contains(channel, case=False, na=False)]
        print("Data shape:", data.shape)
        print("Extracting Json Data")
        medication_df = abc(data)
        print(medication_df.shape)
        print("Parsing Data")
        medication_df["Date"] = medication_df["Date"].apply(parse_date)
        medication_df["Date"] = pd.to_datetime(medication_df["Date"], format="%Y:%m:%d")
        medication_df["Medicine Name"] = medication_df["Medicine Name"].str.lower()
        print("Converting medicines")
        filtered_df = xyz(medication_df)
        print(filtered_df.shape)
        print("Standardizing them")
        filtered_df["category"] = filtered_df["Medicine Name"].apply(
            lambda row: categorize_medicine(row, medicines_to_include=medicine_to_include_lower)
        )
        print("Matching them")
        filtered_df["Medicine Name"] = filtered_df["Medicine Name"].apply(
            lambda name: match_and_replace_medicine_name(name, medicine_to_include_lower)
        )
        print("Final Mapping")
        filtered_df["Medicine Name"] = filtered_df["Medicine Name"].replace(mappings)
        print("Taking 2023 data")
        combined_df = mno(filtered_df)
        print("Combined DataFrame shape:", combined_df.shape)

        print("Converting to month format")
        training_df = pivot(combined_df)
        print("Encoding")
        encoded_df = encoding(training_df)

        # Save the processed data to a pickle file
        with open(pickle_filename, "wb") as file:
            pickle.dump(encoded_df, file)

    clinics = encoded_df["Clinic Name"].unique()
    print("Clinic Len", len(clinics))
    return encoded_df, clinics


def train_model(combined_df, selected_clinics, channel):
    print("selected clinins")
    print(selected_clinics)
    print("Filtered DataFrame shape:", combined_df.shape)
    combined_df = combined_df[combined_df["Clinic Name"].isin(
        selected_clinics)]
    print("Filtered DataFrame shape after clinic selection:", combined_df.shape)

    # Automating
    print("Model Training")
    result = automate_forecasting(combined_df, start_index=0, end_index=30)
    result_file = f"{channel}_Prediction1.csv"
    result.to_csv(result_file)
    return result_file
