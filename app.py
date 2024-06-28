import streamlit as st
import pandas as pd
import os
import pickle
from data_extraction import extract_and_process_data
from eda import get_all_channels
from data_processing import process_and_predict

# Extract and process data
zip_path = "consult_det_report_gen.zip"
csv_filename = "consult_det_report_gen.csv"
extract_and_process_data(zip_path, csv_filename)

# Perform EDA to get channels
channels = get_all_channels()

# Streamlit app
st.title('Health Automation App')

# Dropdown for selecting channel
selected_channel = st.selectbox('Select Channel', channels)

if st.button('Process and Predict'):
    st.write(f'Processing and predicting for channel: {selected_channel}')
    result = process_and_predict(selected_channel)
    st.write('Prediction complete.')
    st.write(result)
