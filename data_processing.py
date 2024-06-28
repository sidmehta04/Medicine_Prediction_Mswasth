import pandas as pd
import pickle
import os
from helper_functions import parse_date, abc, xyz, categorize_medicine, match_and_replace_medicine_name, mno, pivot
from model_training import encoding, automate_forecasting
from medicine_dets import medicine_to_include_lower, mappings

def process_and_predict(channel):
    # Load the processed DataFrame from the pickle file
    with open("processed_data.pkl", "rb") as file:
        processed_df_1 = pickle.load(file)

    print("********************************************************************************")
    print("Processing channel:", channel)
    data = processed_df_1[["Date", "Clinic Name", "Channel", "Prescrption json"]]
    data = data[data["Channel"].str.contains(channel, case=False, na=False)]
    print("Data shape:", data.shape)
    print("Extracting Json Data")
    medication_df=abc(data)
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
    combined_df=mno(filtered_df)
    print("Combined DataFrame shape:", combined_df.shape)
    l1=combined_df["Clinic Name"].unique()
    print("Clinic Len",len(l1))
    print("Converting to month format")
    training_df=pivot(combined_df)
    print("Encoding")
    encodeed_df=encoding(training_df)
    #automating
    print("Model Training")
    result=automate_forecasting(encodeed_df,start_index=1,end_index=30)
    return result.to_csv(f"{channel}_Prediction1.csv")
