import pandas as pd
import os
import pickle


def perform_eda():
    print("performing EDA...")
    if os.path.exists("channels.pkl"):
        with open("channels.pkl", "rb") as file:
            channels = pickle.load(file)
    else:
        with open("processed_data.pkl", "rb") as file:
            processed_df = pickle.load(file)

        channel_value_counts = processed_df['Channel'].value_counts()
        channels = channel_value_counts.index.tolist()

        with open("channels.pkl", "wb") as file:
            pickle.dump(channels, file)

    return channels


def get_all_channels():
    print("getting all channels...")
    if os.path.exists("channels.pkl"):
        with open("channels.pkl", "rb") as file:
            channels = pickle.load(file)
            return channels
    else:
        return perform_eda()


def get_clinics_by_channel(channel):
    print(f"getting clinics for {channel}...")
    pickle_filename = f"{channel}_clinics.pkl"

    if os.path.exists(pickle_filename):
        with open(pickle_filename, "rb") as file:
            clinics = pickle.load(file)
    else:
        with open("processed_data.pkl", "rb") as file:
            processed_df = pickle.load(file)

        clinics = processed_df[processed_df["Channel"].str.contains(
            channel, case=False, na=False)]["Clinic Name"].unique().tolist()

        with open(pickle_filename, "wb") as file:
            pickle.dump(clinics, file)

    return clinics
