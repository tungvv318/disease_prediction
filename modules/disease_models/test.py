import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_min_max(df):
    inputs = []
    # duyet qua cac cot
    for (_, column_data) in df.iteritems():
        list_item_in_column = column_data.values
        max_value = list_item_in_column.max()
        min_value = list_item_in_column.min()
        inputs.append([(item - min_value) / (max_value-min_value) for item in list_item_in_column])
    return np.array(inputs).T


def get_weights(path="C:/Users/Tung/Desktop/Shopee/disease_prediction/data/raw_data/weight.txt"):
    weights = []
    lines = [line.strip() for line in open(path, 'r')]
    for line in lines:
        items = line.split(' ')
        for item in items:
            weights.append(float(item))
    return weights


list = [1, 2, 3, 4]
data = []
# remove data correlative > 0.8
correlated_features = set()
correlation_matrix = data.drop('Survived', axis=1).corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)