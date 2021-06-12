from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import sklearn.model_selection as model_selection
import numpy as np

from modules.models import Model


class BreastCancerSGDModel(Model):
    def __init__(self):
        self.model = SGDClassifier()

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
