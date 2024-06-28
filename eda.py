import pandas as pd
import os
import pickle

def perform_eda():
    if os.path.exists("channels.pkl"):
        with open("channels.pkl", "rb") as file:
            channels = pickle.load(file)
    else:
        # Load the processed DataFrame from the pickle file
        with open("processed_data.pkl", "rb") as file:
            processed_df = pickle.load(file)

        # Calculate the value counts for the 'Channel' column
        channel_value_counts = processed_df['Channel'].value_counts()

        # Get the list of channels
        channels = channel_value_counts.index.tolist()

        # Save the channels as a pickle file
        with open("channels.pkl", "wb") as file:
            pickle.dump(channels, file)

    return channels

def get_all_channels():
    if os.path.exists("channels.pkl"):
        with open("channels.pkl", "rb") as file:
            channels = pickle.load(file)
            return channels
    else:
        return perform_eda()
