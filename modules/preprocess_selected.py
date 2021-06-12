import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, RFE, chi2, f_regression
from sklearn.tree import DecisionTreeClassifier

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
    diagnosis = df['diagnosis'].map({'M': 1, 'B': 0})
    df = df.iloc[:, [2, 4, 5, 6, 7, 8, 10, 11, 12, 14, 16, 18, 19, 21, 22, 23, 24, 25, 28, 29]]

    # feature selection with SelectKBest + chi2
    # chi2_scores = chi2(df, diagnosis)
    # weights.append(SelectKBest(score_func=chi2, k='all').fit(df, diagnosis).scores_)
    # df = df.mul(weights[0], axis='columns')

    # normalize data
    scaler = pp.MinMaxScaler()
    min_max = scaler.fit_transform(df)
    inputs = [Input(e) for e in min_max]
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
    df = df.iloc[:, [2, 4, 5, 6, 7, 8, 10, 11, 12, 14, 16, 18, 19, 21, 22, 23, 24, 25, 28, 29]]

    # normalize data
    scaler = pp.MinMaxScaler()
    minmax = scaler.fit_transform(df)
    inputs = [Input(e) for e in minmax]
    return inputs, ids

