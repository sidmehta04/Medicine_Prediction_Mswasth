import pandas as pd
import zipfile
import pickle
import os

def extract_and_process_data(zip_path, csv_filename, chunk_size=100000):
    if os.path.exists("processed_data.pkl"):
        print("Loading processed data from pickle file...")
        with open("processed_data.pkl", "rb") as file:
            processed_df = pickle.load(file)
    else:
        # Initialize an empty list to store the chunks
        processed_data = []

        # Open the zip file
        with zipfile.ZipFile(zip_path, "r") as z:
            # Open the CSV file inside the zip file
            with z.open(csv_filename) as f:
                # Read the specified rows in chunks from the CSV file
                for chunk in pd.read_csv(
                    f,
                    sep="|",
                    encoding="windows-1252",
                    encoding_errors="ignore",
                    on_bad_lines="skip",
                    chunksize=chunk_size,
                ):
                    print("Added chunk")
                    processed_data.append(chunk)

            # Reset the pointer to the beginning of the CSV file for the header read
            with z.open(csv_filename) as f:
                headers = pd.read_csv(f, nrows=1, sep="|")

        # Concatenate the chunks into a single DataFrame
        processed_df = pd.concat(processed_data)
        processed_df.columns = headers.columns
        print("Processing data...")

        # Save the processed DataFrame as a pickle file
        with open("processed_data.pkl", "wb") as file:
            pickle.dump(processed_df, file)

    return processed_df


