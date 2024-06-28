import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

import calendar as cal
from medicine_dets import (
    medicine_to_include_lower,
    medicine_to_include,
    mappings,
    min_months_required,
    has_sufficient_data,
)
import re


def parse_date(date_str):
    date_str = date_str.replace(" ", "-")
    date_str = date_str.replace("/", "-")
    date_str = date_str.replace(".", "-")

    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y-%b-%d",
        "%d-%b-%Y",
        "%b-%d-%Y",
        "%Y-%B-%d",
        "%d-%B-%Y",
        "%B-%d-%Y",
    ]

    for fmt in formats:
        try:
            d = pd.to_datetime(date_str, format=fmt)
            # If date is valid but falls on an invalid leap year date, continue to next format
            if d.month == 2 and d.day == 29 and not cal.isleap(d.year):
                continue
            return d.strftime("%Y:%m:%d")
        except ValueError:
            pass
    return pd.NaT


def extract_medication_details(prescription, clinic, date):
    med_details = []
    if isinstance(prescription, str):
        try:
            prescriptions = json.loads(prescription)["pres"]
            for prescription in prescriptions:
                med_name = prescription.get("mName", "")
                duration = prescription.get("dur", "")
                morning = 1 if prescription.get("m") else 0
                afternoon = 1 if prescription.get("a") else 0
                evening = 1 if prescription.get("e") else 0
                night = 1 if prescription.get("n") else 0
                frequency = prescription.get("freq", "")
                med_details.append(
                    (
                        med_name,
                        duration,
                        morning,
                        afternoon,
                        evening,
                        night,
                        frequency,
                        clinic,
                        date,
                    )
                )
        except (json.JSONDecodeError, KeyError):
            pass  # Skip non-JSON entries or entries with missing keys
    return med_details


def abc(data):
    medication_details = []

    for idx, row in data.iterrows():
        med_details = extract_medication_details(
            row["Prescrption json"], row["Clinic Name"], row["Date"]
        )
        medication_details.extend(med_details)

    # Create a DataFrame from the medication details
    medication_df = pd.DataFrame(
        medication_details,
        columns=[
            "Medicine Name",
            "Duration",
            "Morning",
            "Afternoon",
            "Evening",
            "Night",
            "Frequency",
            "Clinic Name",
            "Date",
        ],
    )

    medication_df["Clinic Name"] = medication_df["Clinic Name"].fillna("Master Clinic")
    medication_df.dropna(inplace=True)
    print(medication_df.head())

    return medication_df


def xyz(medication_df):
    filtered_df = medication_df[
        medication_df["Medicine Name"]
        .str.lower()
        .apply(lambda x: any(med_name in x for med_name in medicine_to_include_lower))
    ]
    return filtered_df


def categorize_medicine(row, medicines_to_include):
    row_lower = row.lower().strip()

    # Normalize spacing and format
    row_lower = re.sub(r"\s+", " ", row_lower)

    # Specific condition for ORS
    if row_lower.startswith("ors"):
        return "solution"

    for med in medicines_to_include:
        med_pattern = re.escape(med)
        if re.search(med_pattern, row_lower):
            if any(
                x in row_lower for x in ["syrup", "ml", "sry", "srp", "(ml)", "syp"]
            ):
                return "syrup"
            elif any(
                x in row_lower
                for x in ["mg", "tab", "tabs", "tablets", "caps", "capsule", "cap"]
            ):
                return "tab"
            elif any(x in row_lower for x in ["drops", "drop"]):
                return "drops"
            elif any(x in row_lower for x in ["cream", "creams"]):
                return "cream"
            elif any(x in row_lower for x in ["ointment", "ointments", "ointmnt"]):
                return "ointment"
            elif any(x in row_lower for x in ["gels", "gel"]):
                return "gel"
            elif any(x in row_lower for x in ["soaps", "soap"]):
                return "soap"

    return None


def match_and_replace_medicine_name(medicine_name, medicines_to_include):
    for med in medicines_to_include:
        med_pattern = re.compile(re.escape(med), re.IGNORECASE)
        if med_pattern.search(medicine_name):
            return med
    return medicine_name


def mno(filtered_df):
    numeric_cols = ["Duration", "Morning", "Afternoon", "Evening", "Night"]

    # Convert columns to numeric data type
    for col in numeric_cols:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce")

    # Display the DataFrame with converted columns
    filtered_df.info()
    combined_df = filtered_df[filtered_df["Frequency"] == "d"]
    combined_df["Quantity"] = (
        combined_df["Morning"]
        + combined_df["Afternoon"]
        + combined_df["Evening"]
        + combined_df["Night"]
    )

    combined_df["Total Requirement"] = combined_df["Quantity"] * combined_df["Duration"]
    syrup_columns = [
        "Multivitamin Syrup",
        "Cough Syrup",
        "Diclofenac Gel",
    ]  # Replace with your actual column names

    def convert_values(row):
        if row["Medicine Name"] in syrup_columns:
            return 2 if row["Total Requirement"] > 30 else 1
        return row["Total Requirement"]

    combined_df["Total Requirement"] = combined_df.apply(convert_values, axis=1)
    combined_df.sort_values(by="Date")
    combined_df["Date"] = pd.to_datetime(combined_df["Date"], format="%d-%m-%Y")
    combined_df = combined_df[combined_df["Date"] >= "2023-01-01"]
    return combined_df


def pivot(combined_df):

    # Convert the 'Date' column to datetime format and extract year and month
    combined_df["Year"] = combined_df["Date"].dt.year
    combined_df["Month"] = combined_df["Date"].dt.strftime("%b")

    # Create the pivot table
    pivot_table = combined_df.pivot_table(
        values="Total Requirement",
        index=["Clinic Name", "Medicine Name", "Year", "Month"],
        aggfunc="sum",
        margins=True,
        margins_name="Sum of Total Requirement",
    )
    pivot_table = pivot_table.iloc[:-1]
    pivot_table.reset_index(inplace=True)
    pivot_table["Date"] = pd.to_datetime(
        pivot_table["Year"].astype(str) + "-" + pivot_table["Month"],
        format="%Y-%b",
        errors="coerce",
    )

    # Apply the function and filter the pivot_table
    filtered_data = pivot_table.groupby(["Clinic Name", "Medicine Name"]).filter(
        has_sufficient_data
    )
    unique_clinics = filtered_data["Clinic Name"].unique()
    print(len(unique_clinics))

    # Save the filtered pivot_table to a new CSV (optional)
    return filtered_data



