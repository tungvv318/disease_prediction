import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics
import warnings
import sklearn.model_selection as model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, BaggingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier


# from modules.preprocess import load_data_breast_cancer, preprocess, load_test
# from modules.preprocess_min_max import load_data_breast_cancer, preprocess, load_test
# from modules.preprocess_standard import load_data_breast_cancer, preprocess, load_test
# from modules.preprocess_chi2 import load_data_breast_cancer, preprocess, load_test
# from modules.preprocess_f_regression import load_data_breast_cancer, preprocess, load_test
from modules.preprocess_f_classif import load_data_breast_cancer, preprocess, load_test

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    inputs, outputs = load_data_breast_cancer('data/raw_data/data.csv')
    test_inputs, ids = load_test('data/raw_data/test.csv')
    inputs = preprocess(inputs)

    X = np.array([data.features[1] for data in inputs])
    y = np.array([output.score for output in outputs])
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}

    kfold = model_selection.KFold(n_splits=5, random_state=14, shuffle=True)
    # model = LogisticRegression()
    # model = DecisionTreeClassifier()
    # model = RandomForestClassifier()
    # model = SVC(random_state=1)
    model = SGDClassifier()
    # model = GradientBoostingClassifier()
    # model = HistGradientBoostingClassifier()
    # model = BaggingClassifier()
    #
    # # voting model
    # # clf1 = GradientBoostingClassifier()
    # clf2 = RandomForestClassifier()
    # # # clf3 = HistGradientBoostingClassifier()
    # clf1 = LogisticRegression()
    # # # clf2 = SGDClassifier()
    # clf3 = SVC(random_state=1)
    # model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svm', clf3)], voting='hard')

    results = model_selection.cross_validate(estimator=model,
                                              X=X,
                                              y=y,
                                              cv=kfold,
                                              scoring=scoring)
    df = pd.DataFrame({'acc': [round(np.mean(results['test_accuracy']), 4)],
                   'p': [round(np.mean(results['test_precision']), 4)],
                   'r': [round(np.mean(results['test_recall']), 4)],
                                   'f1': round(np.mean(results['test_f1_score']), 4)})
    df.to_csv('data/raw_data/output.csv', mode='a', encoding='utf-8', index=False, header=False)

    print('accuracy:', round(np.mean(results['test_accuracy']), 4))
    print('precision:', round(np.mean(results['test_precision']), 4))
    print('recall:', round(np.mean(results['test_recall']), 4))
    print('f1_score:', round(np.mean(results['test_f1_score']), 4))