from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

import numpy as np

from modules.models import Model


class BreastCancerVotingModel(Model):
    def __init__(self):
        # clf1 = GradientBoostingClassifier()
        clf2 = RandomForestClassifier()
        # clf3 = HistGradientBoostingClassifier()
        clf1 = LogisticRegression()
        # clf2 = SGDClassifier()
        clf3 = SVC(random_state=1)
        self.model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svm', clf3)], voting='hard')

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

