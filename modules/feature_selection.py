import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import chi2


def feature_evaluate(path):
    df = pd.read_csv(path)
    df.drop('id', axis=1, inplace=True)
    df.drop('Unnamed: 32', axis=1, inplace=True)

    diagnosis = df['diagnosis'].map({'M': 1, 'B': 0})
    df.drop('diagnosis', axis=1, inplace=True)

    # feature selection with SelectKBest + chi2
    chi2_scores = chi2(df, diagnosis)
    chi2_stat = pd.Series(chi2_scores[0], index=df.columns)
    chi2_stat.sort_values(ascending=False, inplace=True)
    chi2_stat.plot.bar()
    plt.show()


feature_evaluate('C:/Users/Tung/Desktop/Shopee/disease_prediction/data/raw_data/data.csv')
