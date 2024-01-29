import pandas as pd
import numpy as np
import os


def shuffle_data(df1, df2):
    """
    Shuffles the data by concatenating two DataFrames and then randomly permuting the rows.

    Parameters:
    - df1 (pd.DataFrame): First DataFrame to shuffle.
    - df2 (pd.DataFrame): Second DataFrame to shuffle.

    Returns:
    pd.DataFrame: Shuffled DataFrame.
    """
    df_combined = pd.concat([df1, df2], ignore_index = True)

    # Frac = 1 corresponds to 100% of the data. drop = True corresponds to not preserving old indexes.
    df_combined_shuffled = df_combined.sample(frac = 1, random_state = 42).reset_index(drop = True)

    return df_combined_shuffled


def separate_data_frame(df, percentage_training = 75, percentage_testing = 15):
    """
    Separates a DataFrame into training, testing, and validation sets based on the provided percentages.

    Parameters:
    - df (pd.DataFrame): DataFrame to be split.
    - percentage_training (float): Percentage of data for the training set.
    - percentage_testing (float): Percentage of data for the testing set.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple containing the training, testing, and validation DataFrames.
    """
    total_samples = len(df)
    train_size = int((percentage_training/100)*total_samples)
    test_size = int((percentage_testing/100)*total_samples)

    df_train = df[:train_size]
    df_test = df[train_size:(train_size + test_size)]
    df_validation = df[(train_size + test_size):]

    return df_train, df_test, df_validation


def separate_and_clean_data(name, data_frame, already_cleaned = False):
    """
    Separates the input DataFrame into X (inputs) and y (outputs), applying cleaning operations if not already cleaned.

    Parameters:
    - name (str): Name identifier for the DataFrame.
    - data_frame (pd.DataFrame): Input DataFrame.
    - already_cleaned (bool): Flag indicating whether the DataFrame is already cleaned.

    Returns:
    Tuple[pd.DataFrame, pd.Series]: Tuple containing X (inputs) and y (outputs).
    """
    if already_cleaned == False:
        # Convert all data to numeric type (or null if not possible) and get rid of all without/null-output and repeated rows.
        data_frame = data_frame.apply(pd.to_numeric, errors = 'coerce')
        data_frame = data_frame.dropna(subset = ['output'])
        data_frame.drop_duplicates(inplace = True)

    # Separate the data frame into X (inputs) and y (outputs) and see their sizes.
    X = data_frame.iloc[:, 1:]
    y = data_frame.iloc[:, 0]
    print(f'Sizes of the {name}: X = {X.shape}, y = {y.shape}.\n')
    
    return X, y


def fill_null_values(X, y, name, means = None):
    """
    Fills null values in X with means and provides statistics about the data.

    Parameters:
    - X (pd.DataFrame): Input features.
    - y (pd.Series): Output labels.
    - name (str): Name identifier for the data.
    - means (pd.Series): Means for imputing null values.

    Returns:
    Tuple[pd.DataFrame, pd.Series]: Tuple containing filled X and means used for imputation.
    """
    df = pd.concat([X, y], axis = 1)

    if means is None:
        means = X.mean()

    # Replace null values with the average (mean) of the column
    X = X.fillna(means)

    # Statistics of the data set.
    stats = df.describe()

    # Confirm that all columns have the same amount of data and that there are no null values.
    first_column_count = stats.loc['count'].iloc[0]
    null_values = df.isnull().any().any()
    print(f'\nThe total amount of data for the {name} is equal to {first_column_count} in all columns.')
    print(f'It is {null_values} that there are null values, and the means of the inputs in the {name} is')
    X_mean_table = X.mean().to_numpy().reshape(7, 3)
    print(X_mean_table)
    
    return X, means


def remove_outliers(X, y, name, threshold = 3.75):
    """
    Removes outliers from the DataFrame.

    Parameters:
    - X (pd.DataFrame): Input features.
    - y (pd.Series): Output labels.
    - name (str): Name identifier for the data.
    - threshold (float): Threshold for identifying outliers.

    Returns:
    Tuple[pd.DataFrame, pd.Series]: Tuple containing features (X) and labels (y) after removing outliers.
    """
    df = pd.concat([y, X], axis = 1)

    input_columns = X.columns
    rows_before = len(df)

    for i in range(5):
        for column in input_columns:

            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            df = df[~outliers]

    rows_after = len(df)
    rows_removed = rows_before - rows_after
    percentage_removed = (rows_removed/rows_before)*100
    print(f'{rows_removed} rows have been removed from {name}, which is the {percentage_removed}% percent of the original data.')

    return df.iloc[:, 1:], df.iloc[:, 0]


def save_data(data_path, name_file, X, y):
    """
    Saves the provided features (X) and labels (y) to a CSV file.

    Parameters:
    - data_path (str): Path to save the file.
    - name_file (str): Name of the file.
    - X (pd.DataFrame): Input features.
    - y (pd.Series): Output labels.
    """
    try:
        # Convert X and y to pandas.Series or pandas.DataFrame if they are not already
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            X = pd.DataFrame(X)
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.DataFrame(y)

        combined_df = pd.concat([y, X], axis = 1)
        combined_df = combined_df.sort_index()
        column_names = ['output'] + [f'input{i}' for i in range(1, 22)]
        combined_df.columns = column_names
        combined_df.to_csv(os.path.join(data_path, name_file), index = False)
        print(f"The {name_file} has been saved in {data_path}.\n")

    except Exception as e:
        print(f"Could not save {name_file} in project/data/processed. Error: {str(e)}\n")