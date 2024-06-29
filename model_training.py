import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
def encoding(filtered_data):

    # Create label encoder objects
    clinic_encoder = LabelEncoder()
    medicine_encoder = LabelEncoder()

    # Fit and transform the categorical variables
    filtered_data["Clinic Name Encoded"] = clinic_encoder.fit_transform(
        filtered_data["Clinic Name"]
    )
    filtered_data["Medicine Name Encoded"] = medicine_encoder.fit_transform(
        filtered_data["Medicine Name"]
    )

    # Optionally, keep the mappings for reference
    clinic_mapping = dict(
        zip(clinic_encoder.classes_, clinic_encoder.transform(clinic_encoder.classes_))
    )
    medicine_mapping = dict(
        zip(
            medicine_encoder.classes_,
            medicine_encoder.transform(medicine_encoder.classes_),
        )
    )
    # Convert Year and Month to numeric values if necessary
    filtered_data["Year"] = filtered_data["Year"].astype(int)
    # Month might already be in a numeric format (e.g., 'Jan' -> 1), but ensure it's consistent
    month_mapping = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    filtered_data["Month"] = filtered_data["Month"].map(month_mapping)
    return filtered_data

def evaluate_models(X_train, y_train, X_test, y_test, models, param, random_state=42):
    report = {}

    for i in range(len(list(models))):
        model = list(models.values())[i]
        para = param[list(models.keys())[i]]

        gs = GridSearchCV(model, para, cv=3)
        gs.fit(X_train, y_train)

        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        test_model_score = r2_score(y_test, y_test_pred)
        report[list(models.keys())[i]] = test_model_score


    return report
def automate_forecasting(filtered_data, start_index=0, end_index=None, random_state=42):
    print("inside automate forecasting.")
    results = []
    unique_clinics = filtered_data["Clinic Name"].unique()
    print("processing clinics below")
    print(unique_clinics)
    if end_index is None:
        end_index = len(unique_clinics)

    for idx, clinic in enumerate(
        unique_clinics[start_index:end_index], start=start_index
    ):
        print(f"Processing clinic {idx + 1} out of {len(unique_clinics)}")
        print(clinic)
        clinic_data = filtered_data[filtered_data["Clinic Name"] == clinic]

        # Prepare training and testing data
        train_data = clinic_data[
            (clinic_data["Year"] >= 2023)
            & ((clinic_data["Year"] < 2024) | (clinic_data["Month"].isin([1, 2, 3,4])))
        ].copy()
        test_data = clinic_data[
            (clinic_data["Year"] == 2024) & (clinic_data["Month"].isin([5]))
        ].copy()
        if len(train_data) == 0 or len(test_data) == 0:
            print(f"Skipping clinic '{clinic}' due to missing data.")
            continue
        X_train = train_data[
            ["Clinic Name Encoded", "Medicine Name Encoded", "Year", "Month"]
        ]
        y_train = train_data["Total Requirement"]
        X_test = test_data[
            ["Clinic Name Encoded", "Medicine Name Encoded", "Year", "Month"]
        ]
        y_test = test_data["Total Requirement"]

        # Define models and parameters
        models = {
            "Random Forest": RandomForestRegressor(random_state=random_state),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBRegressor": xgb.XGBRegressor(random_state=random_state),
            "AdaBoost Regressor": AdaBoostRegressor(random_state=random_state),
            "CatBoosting Regressor": CatBoostRegressor(
                verbose=False, random_seed=random_state
            ),
        }

        params = {
            "Decision Tree": {
                "criterion": [
                    "squared_error",
                    "friedman_mse",
                    "absolute_error",
                    "poisson",
                ],
                "max_depth": [None, 10, 20, 30, 40, 50],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "Random Forest": {"n_estimators": [8, 16, 32, 64, 128, 256]},
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Linear Regression": {},
            "XGBRegressor": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "AdaBoost Regressor": {
                "learning_rate": [0.1, 0.01, 0.5, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "CatBoosting Regressor": {
                "depth": [6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "iterations": [30, 50, 100],
            },
        }

        # Evaluate models and select the best one
        model_report = evaluate_models(
            X_train, y_train, X_test, y_test, models, params, random_state=random_state
        )
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        y_pred_test = best_model.predict(X_test)
        y_pred_test = np.round(y_pred_test)
        y_pred_test_list = y_pred_test.tolist()

        # Retrain the best model on all available data before forecasting June 2024
        X_all = clinic_data[
            ["Clinic Name Encoded", "Medicine Name Encoded", "Year", "Month"]
        ]
        y_all = clinic_data["Total Requirement"]
        best_model.fit(X_all, y_all)

        # Predicting with the best model for test data (April, May 2024)
        if best_model_score > 0.3:
            print("Taken")
            # Generate forecast for June 2024
            june_2024_data = test_data.copy()
            X_june_2024 = june_2024_data[
                ["Clinic Name Encoded", "Medicine Name Encoded", "Year"]
            ]
            X_june_2024["Month"] = 6
            y_pred_june = best_model.predict(X_june_2024)
            y_pred_june = np.round(y_pred_june)
            y_pred_june_list = y_pred_june.tolist()

            # Store the results
            for idx, (actual, prediction_test, prediction_june) in enumerate(
                zip(y_test, y_pred_test_list, y_pred_june_list)
            ):
                test_month = test_data.iloc[idx]["Month"]
                test_year = test_data.iloc[idx]["Year"]
                medicine_name = test_data.iloc[idx]["Medicine Name"]
                results.append(
                    {
                        "Clinic Name": clinic,
                        "Medicine Name": medicine_name,
                        f"{test_month}/{test_year} Actual Data": actual,
                        f"{test_month}/{test_year} Predicted Data": prediction_test,
                        "June 2024 Forecasted Data": prediction_june,
                    }
                )

    # Convert results to a DataFrame for better visualization
    results_df = pd.DataFrame(results)
    return results_df

    # Save results to a CSV file

