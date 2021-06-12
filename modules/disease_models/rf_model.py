from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np

from modules.models import Model


class BreastCancerRFModel(Model):
    def __init__(self):
        # self.model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=2)
        self.model = RandomForestClassifier()

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

        outputs = []
        predicts = self.model.predict(X)
        # for ps in zip(*predicts):
        #     score = ps
        #     outputs.append(Output(score))
        return predicts
