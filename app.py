
# Streamlit components
import streamlit as st
import pandas as pd
from data_extraction import extract_and_process_data
from eda import get_all_channels
from data_processing import process_data, train_model

# Initialize session state variables
if 'channel_selected' not in st.session_state:
    st.session_state.channel_selected = False
if 'clinics_fetched' not in st.session_state:
    st.session_state.clinics_fetched = False
if 'selected_channel' not in st.session_state:
    st.session_state.selected_channel = None
if 'clinics' not in st.session_state:
    st.session_state.clinics = []
if 'selected_clinics' not in st.session_state:
    st.session_state.selected_clinics = []

# Extract and process data


@st.cache_data
def cached_extract_and_process_data(zip_path, csv_filename):
    return extract_and_process_data(zip_path, csv_filename)


@st.cache_data
def cached_get_all_channels():
    return get_all_channels()


@st.cache_data
def cached_process_data(channel, processed_df_1):
    return process_data(channel, processed_df_1)


zip_path = "consult_det_report_gen.zip"
csv_filename = "consult_det_report_gen.csv"
processed_df_1 = cached_extract_and_process_data(zip_path, csv_filename)

channels = cached_get_all_channels()

st.title('Health Automation App')

# Step 1: Select Channel
if not st.session_state.channel_selected:
    selected_channel = st.selectbox('Select Channel', channels)
    if st.button('Proceed'):
        st.session_state.selected_channel = selected_channel
        st.session_state.channel_selected = True
        combined_df, clinics = process_data(
            selected_channel, processed_df_1)
        st.session_state.combined_df = combined_df
        st.session_state.clinics = clinics
        st.session_state.clinics_fetched = True

# Step 2: Select Clinics and Predict
if st.session_state.channel_selected and st.session_state.clinics_fetched:
    selected_channel = st.session_state.selected_channel
    st.write(f'You selected: {selected_channel}')

    clinics = st.session_state.clinics.tolist()
    clinics.insert(0, "All Clinics")  # Option to select all clinics

    # Multi-select for selecting clinics
    selected_clinics = st.multiselect(
        'Select Clinics', clinics, default="All Clinics")

    if st.button('Process and Predict'):
        if "All Clinics" in selected_clinics:
            selected_clinics = clinics[1:]  # Exclude "All Clinics"
        st.session_state.selected_clinics = selected_clinics
        st.write(f'Processing and predicting for channel: {selected_channel}')
        result_file = train_model(
            st.session_state.combined_df, selected_clinics, selected_channel)
        st.write('Prediction complete.')
        st.write(result_file)
        with open(result_file, 'r') as file:
            st.download_button('Download Prediction Results',file, file_name=result_file)
