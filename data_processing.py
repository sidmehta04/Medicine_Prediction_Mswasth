import pandas as pd
import pickle
import os
from helper_functions import parse_date, abc, xyz, categorize_medicine, match_and_replace_medicine_name, mno, pivot
from model_training import encoding, automate_forecasting
from medicine_dets import medicine_to_include_lower, mappings
import streamlit as st

def process_and_predict(channel, processed_df_1):
    st.write("********************************************************************************")
    st.write("Processing channel:", channel)
    
    pickle_file = f"{channel}_processed_data.pkl"
    
    if os.path.exists(pickle_file):
        st.write("Loading data from pickle file.")
        with open(pickle_file, 'rb') as file:
            encodeed_df = pickle.load(file)
    else:
        st.write("Processing data.")
        data = processed_df_1[["Date", "Clinic Name", "Channel", "Prescrption json"]]
        data = data[data["Channel"].str.contains(channel, case=False, na=False)]
        st.write("Data shape:", data.shape)
        st.write("Extracting Json Data")
        medication_df = abc(data)
        st.write(medication_df.shape)
        st.write("Parsing Data")
        medication_df["Date"] = medication_df["Date"].apply(parse_date)
        medication_df["Date"] = pd.to_datetime(medication_df["Date"], format="%Y:%m:%d")
        medication_df["Medicine Name"] = medication_df["Medicine Name"].str.lower()
        st.write("Converting medicines")
        filtered_df = xyz(medication_df)
        st.write(filtered_df.shape)
        st.write("Standardizing them")
        filtered_df["category"] = filtered_df["Medicine Name"].apply(
            lambda row: categorize_medicine(row, medicines_to_include=medicine_to_include_lower)
        )
        st.write("Matching them")
        filtered_df["Medicine Name"] = filtered_df["Medicine Name"].apply(
            lambda name: match_and_replace_medicine_name(name, medicine_to_include_lower)
        )
        st.write("Final Mapping")
        filtered_df["Medicine Name"] = filtered_df["Medicine Name"].replace(mappings)
        st.write("Taking 2023 data")
        combined_df = mno(filtered_df)
        st.write("Combined DataFrame shape:", combined_df.shape)
        l1 = combined_df["Clinic Name"].unique()
        st.write("Clinic Len", len(l1))
        st.write("Converting to month format")
        training_df = pivot(combined_df)
        st.write("Encoding")
        encodeed_df = encoding(training_df)
        
        with open(pickle_file, 'wb') as file:
            st.write("Saving processed data to pickle file.")
            pickle.dump(encodeed_df, file)
    
    # automating
    st.write("Model Training")
    result = automate_forecasting(encodeed_df, start_index=1, end_index=3)
    result_csv = f"{channel}_Prediction1.csv"
    result.to_csv(result_csv)
    return result_csv
