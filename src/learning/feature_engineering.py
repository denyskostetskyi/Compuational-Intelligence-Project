import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import re

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import constants

def process_dataset_for_training(df: pd.DataFrame) -> pd.DataFrame:
    return __process_dataset(df, True)

def process_dataset_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    return __process_dataset(df, False)

def __process_dataset(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = __rename_columns(df)
    df = __select_columns(df, is_train)
    __impute_missing_data(df, is_train)
    __encode_categorical_columns(df, is_train)
    __transform_column_types(df, is_train)
    __replace_outliers(df)
    __scale_numerical_values(df, is_train)
    __save_processed_dataset(df, is_train)
    return df

def __rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(
        columns={
            "Daytime/evening attendance\t": "Daytime attendance",
            "Nacionality": "Nationality",
            "Gender": "Male",
            "Age at enrollment": "Age",
        },
        inplace=True,
    )
    df = df.iloc[:, 1:]  # remove index column
    df.columns = [__to_snake_case(col) for col in df.columns]
    return df

def __to_snake_case(s: str) -> str:
    s = re.sub(r"[’'\"()]", "", s)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    return s.strip("_").lower()

def __select_columns(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    selected_columns = constants.COLUMNS_SELECTED.copy()
    # remove target column for prediction
    if not is_train:
        selected_columns.remove(constants.COLUMN_TARGET)
    missing_cols = [col for col in selected_columns if col not in df.columns]
    if is_train and missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    return df[selected_columns].copy()


def __impute_missing_data(df: pd.DataFrame, is_train: bool) -> None:
    missing_numerical = []
    missing_categorical = []

    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(f"Missing values in {col}: {missing_count}")
            if col in constants.COLUMNS_NUMERICAL:
                missing_numerical.append(col)
            elif col in constants.COLUMNS_CATEGORICAL:
                missing_categorical.append(col)
            else:
                print(f"Warning: column '{col}' not found in defined constants.")

    if not (missing_numerical or missing_categorical):
        print("No missing values in the dataset.")
        return

    if missing_numerical:
        if is_train:
            imputer_numerical = SimpleImputer(strategy='median')
            df[missing_numerical] = imputer_numerical.fit_transform(df[missing_numerical])
            joblib.dump(imputer_numerical, constants.PATH_IMPUTER_NUMERICAL)
        else:
            imputer_numerical: SimpleImputer = joblib.load(constants.PATH_IMPUTER_NUMERICAL)
            df[missing_numerical] = imputer_numerical.transform(df[missing_numerical])

    if missing_categorical:
        if is_train:
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            df[missing_categorical] = imputer_categorical.fit_transform(df[missing_categorical])
            joblib.dump(imputer_categorical, constants.PATH_IMPUTER_CATEGORICAL)
        else:
            imputer_categorical: SimpleImputer = joblib.load(constants.PATH_IMPUTER_CATEGORICAL)
            df[missing_categorical] = imputer_categorical.transform(df[missing_categorical])

    print("All missing values have been imputed.")

def __encode_categorical_columns(df: pd.DataFrame, is_train: bool) -> None:
    if is_train:
        df[constants.COLUMN_TARGET] = df[constants.COLUMN_TARGET].map(constants.TARGET_MAPPING)

def __transform_column_types(df: pd.DataFrame, is_train: bool) -> None:
    integer_columns = constants.COLUMNS_INTEGER.copy()
    if not is_train:
        integer_columns.remove(constants.COLUMN_TARGET)
    for col in integer_columns:
        df[col] = df[col].astype(int)

def __replace_outliers(df: pd.DataFrame) -> None:
    for column_name, valid_values in constants.COLUMNS_CATEGORICAL_WITH_VALUES:
        __validate_categorical_column(df, column_name, valid_values)
    for column_name, min_value, max_value in constants.COLUMNS_NUMERICAL_WITH_VALUES:
        __validate_numerical_column(df, column_name, min_value, max_value)

def __validate_categorical_column(df: pd.DataFrame, column_name: str, valid_values: list) -> None:
    invalid_values = df[column_name][~df[column_name].isin(valid_values)]
    if not invalid_values.empty:
        mode_value = df[column_name].mode()[0]
        df.loc[~df[column_name].isin(valid_values), column_name] = mode_value
        print(f"Replaced {len(invalid_values)} invalid values using trained categorical imputer in '{column_name}'.")

def __validate_numerical_column(df: pd.DataFrame, column_name: str, min_value: float, max_value: float) -> None:
    out_of_bounds = (df[column_name] < min_value) | (df[column_name] > max_value)
    if out_of_bounds.any():
        median_value = df[column_name].median()
        df.loc[out_of_bounds, column_name] = median_value
        print(f"Replaced {out_of_bounds.sum()} out-of-bounds values using trained numerical imputer in '{column_name}'.")

def __scale_numerical_values(df: pd.DataFrame, is_train: bool) -> None:
    if is_train:
        scaler_numerical = StandardScaler()
        scaled_values = scaler_numerical.fit_transform(df[constants.COLUMNS_NUMERICAL])
        scaled_df = pd.DataFrame(scaled_values, columns=constants.COLUMNS_NUMERICAL)
        df[constants.COLUMNS_NUMERICAL] = scaled_df
        joblib.dump(scaler_numerical, constants.PATH_SCALER_NUMERICAL)
    else:
        scaler_numerical: StandardScaler = joblib.load(constants.PATH_SCALER_NUMERICAL)
        scaled_values = scaler_numerical.transform(df[constants.COLUMNS_NUMERICAL])
        scaled_df = pd.DataFrame(scaled_values, columns=constants.COLUMNS_NUMERICAL)
        df[constants.COLUMNS_NUMERICAL] = scaled_df

def __save_processed_dataset(df: pd.DataFrame, is_train: bool) -> None:
    if is_train:
        path = constants.PATH_PROCESSED_DATASET
        df.to_csv(path, index=False)
        print(f"Processed dataset saved to {path}.")
