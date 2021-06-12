from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np

from modules.models import Model

class BreastCancerSVMModel(Model):
    def __init__(self):
        # self.model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        self.model = SVC(random_state=1)
        # self.model = SVC(C=1.0, gamma=0.002, kernel='linear')

    def _represent(self, inputs):
        """

        :param list of models.Input inputs:
        :return:
        """
        return np.array([data.features[1] for data in inputs])

    def train(self, inputs, outputs):
        """

        :param list of models.Input inputs:
        :param list of models.AspectOutput outputs:
        :return:
        """
        X = self._represent(inputs)
        y = np.array([output.score for output in outputs])

        self.model.fit(X, y)

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
        X = self._represent(inputs)

        predicts = self.model.predict(X)
        return predicts
