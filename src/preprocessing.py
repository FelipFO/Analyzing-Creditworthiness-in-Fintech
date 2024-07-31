from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

def preprocess_data(
    train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses data for modeling. Receives train, val, and test dataframes
    and returns cleaned up dataframes with feature engineering already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : pd.DataFrame
        val : pd.DataFrame
        test : pd.DataFrame
    """
    # Make a copy of the dataframes
    working_train_data = train_data.copy()
    working_val_data = val_data.copy()
    working_test_data = test_data.copy()

    # 1. Correct outliers/anomalous values in numerical columns (`DAYS_EMPLOYED` column).
    working_train_data["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_data["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_data["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. Encode string categorical features
    categorical_cols = working_train_data.select_dtypes(include=['object']).columns.tolist()
    binary_cols = []
    multi_cols = []

    for col in categorical_cols:
        if working_train_data[col].nunique() == 2:
          binary_cols.append(col)
        else:
          multi_cols.append(col)

    ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=np.nan)
    ordinal_encoder.fit(working_train_data[binary_cols])
    working_train_data[binary_cols] = ordinal_encoder.transform(working_train_data[binary_cols])
    working_val_data[binary_cols] = ordinal_encoder.transform(working_val_data[binary_cols])
    working_test_data[binary_cols] = ordinal_encoder.transform(working_test_data[binary_cols])
    print("Shape after ordinal set:")
    print("Train:", working_train_data.shape)
    print("Validation:", working_val_data.shape)
    print("Test:", working_test_data.shape)

    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    onehot_encoder.fit(working_train_data[multi_cols])
    onehot_encoded_cols_train = onehot_encoder.transform(working_train_data[multi_cols])
    onehot_encoded_data_train = pd.DataFrame(onehot_encoded_cols_train, columns=onehot_encoder.get_feature_names_out())
    working_train_data.drop(columns=multi_cols, inplace=True)
    working_train_data = working_train_data.join(onehot_encoded_data_train)

    onehot_encoded_cols_val = onehot_encoder.transform(working_val_data[multi_cols])
    onehot_encoded_data_val = pd.DataFrame(onehot_encoded_cols_val, columns=onehot_encoder.get_feature_names_out())
    working_val_data.drop(columns=multi_cols, inplace=True)
    working_val_data = working_val_data.join(onehot_encoded_data_val)

    onehot_encoded_cols_test = onehot_encoder.transform(working_test_data[multi_cols])
    onehot_encoded_data_test = pd.DataFrame(onehot_encoded_cols_test, columns=onehot_encoder.get_feature_names_out())
    working_test_data.drop(columns=multi_cols, inplace=True)
    working_test_data = working_test_data.join(onehot_encoded_data_test)

    print("Shape after encoding and feature engineering:")
    print("Train:", working_train_data.shape)
    print("Validation:", working_val_data.shape)
    print("Test:", working_test_data.shape)

    # 3. Impute values for all columns with missing data using median
    imputer = SimpleImputer(strategy='median')
    imputer.fit(working_train_data)
    working_train_data = imputer.transform(working_train_data)
    working_val_data = imputer.transform(working_val_data)
    working_test_data = imputer.transform(working_test_data)

    # 4. Feature scaling with Min-Max scaler
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(working_train_data)
    working_train_data = minmax_scaler.transform(working_train_data) 
    working_val_data = minmax_scaler.transform(working_val_data)
    working_test_data = minmax_scaler.transform(working_test_data)

    print("Shape after imputation and scaling:")
    print("Train:", working_train_data.shape)
    print("Validation:", working_val_data.shape)
    print("Test:", working_test_data.shape)

    return working_train_data, working_val_data, working_test_data
