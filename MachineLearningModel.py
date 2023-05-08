import argparse
from argparse import Namespace
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix


def main(args: Namespace) -> None:
    """
    This is the main function of this module, it runs machine learning algorithm

    :param args: Namespace storing all arguments from command line
    :returns: None
    """
    training_data = pd.read_csv("Dataset/train.csv")
    testing_data = pd.read_csv("Dataset/test.csv")

    training_data = prepare_data(training_data)
    testing_data = prepare_data(testing_data)

    y_train = training_data["satisfaction_satisfied"]
    X_train = training_data.drop("satisfaction_satisfied", axis=1)
    y_test = testing_data["satisfaction_satisfied"]
    X_test = testing_data.drop("satisfaction_satisfied", axis=1)

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_train)
    print(confusion_matrix(y_train, y_pred))


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    This is a function that prepares data for machine learning algorithm.

    :param: pd.DataFrame
    :returns: pd.DataFrame prepared for machine learning algorithm.
    """
    # Fill NaNs with 0
    data.fillna(0, inplace=True)

    # Change Object data type to Category
    data['Gender'] = data['Gender'].astype('category')
    data['Customer Type'] = data['Customer Type'].astype('category')
    data['Type of Travel'] = data['Type of Travel'].astype('category')
    data['Class'] = data['Class'].astype('category')
    data['satisfaction'] = data['satisfaction'].astype('category')

    # Make bools from categories
    data = pd.get_dummies(data)
    return data


def parse_arguments() -> Namespace:
    """
    This is a function that parses arguments from command line.

    :param: None
    :returns: Namespace storing all arguments from command line
    """
    parser = argparse.ArgumentParser(
        description='This is machine learning algorithm for ASI')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
