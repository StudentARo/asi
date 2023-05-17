"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.8
"""
import pandas as pd

def remove_unimportant_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop("Flight Distance", axis=1)
    data = data.drop("Class", axis=1)
    data = data.drop("Type of Travel", axis=1)
    data = data.drop("Age", axis=1)
    data = data.drop("Customer Type", axis=1)
    data = data.drop("Gender", axis=1)
    data = data.drop("id", axis=1)
    return data

def drop_outliers(data: pd.DataFrame) -> pd.DataFrame:
    arrival_percentile = data['Arrival Delay in Minutes'].quantile(0.75)
    data = data[data['Arrival Delay in Minutes'] < arrival_percentile]
    departure_percentile = data['Departure Delay in Minutes'].quantile(0.75)
    data = data[data['Departure Delay in Minutes'] < departure_percentile]
    return data

def preprocess_flights(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data for flights.
    Drops: id, Gender, Customer Type, Age, Type of Travel, Class, Flight Distance
    Removes outliers from Arrival Delay in Minutes and Departure Delay in Minute
    Changes satisfaction to dummy column where:
        neutral or dissatisfied : 0
        satisfied               : 1
    
    Args:
        data: Raw data.
    Returns:
        Preprocessed data.
    """
    data.fillna(0, inplace=True)
    data = remove_unimportant_columns(data)
    data = drop_outliers(data)
    data['satisfaction'] = data['satisfaction'].astype('category')
    data = pd.get_dummies(data)
    data.drop("satisfaction_neutral or dissatisfied", axis=1, inplace = True)
    data.rename(columns={'satisfaction_satisfied': 'satisfaction'}, inplace=True)
    return data