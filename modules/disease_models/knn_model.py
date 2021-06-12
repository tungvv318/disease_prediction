from sklearn.neighbors import KNeighborsClassifier

import numpy as np

from modules.models import Model


class BreastCancerKNNModel(Model):
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors = 2)

    def _represent(self, inputs):
        """

        :param list of models.Input inputs:
        :return:
        """
        return np.array([data.features for data in inputs])

    def train(self, inputs, outputs):
        """

        :param list of models.Input inputs:
        :param list of models.AspectOutput outputs:
        :return:
        """
        self.model.fit(inputs, outputs)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def predict(self, inputs):
        """

        :param inputs:
        :return:
        :rtype: list of models.Output
        """

        predicts = self.model.predict(inputs)
        return predicts

