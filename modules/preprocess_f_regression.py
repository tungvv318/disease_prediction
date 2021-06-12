import pandas as pd
from sklearn.feature_selection import f_regression

from sklearn import preprocessing as pp
from models import Input, Output


weights = []
indexs = []
def load_data_breast_cancer(path):
    """

    :param path:
    :return:
    :rtype: (list of models.Input, list of models.Output)
    """
    df = pd.read_csv(path)
    df.drop('id', axis=1, inplace=True)
    df.drop('Unnamed: 32', axis=1, inplace=True)

    diagnosis = df['diagnosis'].map({'M': 1, 'B': 0})
    df.drop('diagnosis', axis=1, inplace=True)

    # feature selection with SelectKBest + chi2
    weight, index = get_top_f_regression(10, df, diagnosis)
    weights.append(weight)
    indexs.append(index)
    df = df.iloc[:, indexs[0]]
    df = df.mul(weights[0], axis='columns')

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
    df = pd.read_csv(path)
    ids = df['id'].to_frame('id')
    df.drop('id', axis=1, inplace=True)
    df.drop('Unnamed: 32', axis=1, inplace=True)

    df = df.iloc[:, indexs[0]]
    df = df.mul(weights[0], axis='columns')

    # normalize data
    scaler = pp.MinMaxScaler()
    minmax = scaler.fit_transform(df)
    inputs = [Input(e) for e in minmax]
    return inputs, ids


def get_top_f_regression(n_feature, X, y):
    chi2_scores = list(zip(f_regression(X, y)[0], range(len(X.columns))))
    chi2_scores.sort(key=lambda x: -x[0])
    weights = []
    index = []
    i = 1
    for tuple in chi2_scores:
        if (i > n_feature):
            break
        weights.append(tuple[0])
        index.append(tuple[1])
        i += 1

    # print weight of column name
    weight_file = pd.DataFrame()
    weight_file['col_index'] = index
    weight_file['col_name'] = X.columns[index].tolist()
    weight_file['weight'] = weights
    weight_file.to_csv("C:/Users/Tung/Desktop/Shopee/disease_prediction/data/raw_data/weight_f_regression.csv", index=False)
    return weights, index
