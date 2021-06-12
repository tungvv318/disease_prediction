import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, RFE, chi2, f_regression

from sklearn import preprocessing as pp
from models import Input, Output


weights = []

def load_data_breast_cancer(path):
    """

    :param path:
    :return:
    :rtype: (list of models.Input, list of models.Output)
    """
    df = pd.read_csv(path)
    id = df['id']
    df.drop('id', axis=1, inplace=True)
    df.drop('Unnamed: 32', axis=1, inplace=True)

    diagnosis = df['diagnosis'].map({'M': 1, 'B': 0})
    df.drop('diagnosis', axis=1, inplace=True)

    # normalize data
    inputs = [Input(e) for e in zip(id, df.values.tolist())]
    outputs = [Output(value) for _, value in diagnosis.items()]
    return inputs, outputs


def preprocess(inputs):
    """

    :param list of models.Input inputs:
    :return:
    :rtype: list of models.Input
    """
    return inputs


def load_test(path):
    """

    :param path:
    :return:
    :rtype: (list of models.Input, list of models.Output)
    """
    inputs = []
    df = pd.read_csv(path)
    ids = df['id'].to_frame('id')
    df.drop('id', axis=1, inplace=True)
    df.drop('Unnamed: 32', axis=1, inplace=True)

    # normalize data
    inputs = [Input(e) for e in df.values.tolist()]
    return inputs, ids


# def get_min_max(df):
#     inputs = []
#     # duyet qua cac cot
#     for (_, column_data) in df.iteritems():
#         list_item_in_column = column_data.values
#         max_value = list_item_in_column.max()
#         min_value = list_item_in_column.min()
#         inputs.append([(item - min_value) / (max_value-min_value) for item in list_item_in_column])
#     return np.array(inputs).T
