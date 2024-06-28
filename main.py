import data_extraction
import eda

def main():
    # Extract and process data
    zip_path = "consult_det_report_gen.zip"
    csv_filename = "consult_det_report_gen.csv"
    data_extraction.extract_and_process_data(zip_path, csv_filename)
    
    # Perform EDA to get channels
    eda.perform_eda()

if __name__ == "__main__":
    main()
