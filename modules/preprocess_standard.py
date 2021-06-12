import pandas as pd
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
    scaler = pp.StandardScaler()
    standard = scaler.fit_transform(df)
    standard = zip(id, standard)
    inputs = [Input(e) for e in standard]
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
    scaler = pp.StandardScaler()
    standard = scaler.fit_transform(df)
    inputs = [Input(e) for e in standard]
    return inputs, ids