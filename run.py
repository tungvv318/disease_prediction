import numpy as np
import pandas as pd
from sklearn import metrics
import warnings
from sklearn.model_selection import train_test_split
from modules.disease_models.lr_model import BreastCancerLRModel
from modules.disease_models.dt_model import BreastCancerDTModel
from modules.disease_models.rf_model import BreastCancerRFModel
from modules.disease_models.svm_model import BreastCancerSVMModel
from modules.disease_models.sgd_model import BreastCancerSGDModel
from modules.disease_models.gbc_model import BreastCancerGBCModel
from modules.disease_models.hgbc_model import BreastCancerHGBCModel
from modules.disease_models.bc_model import BreastCancerBCModel
from modules.disease_models.voting_model import BreastCancerVotingModel

# from modules.preprocess import load_data_breast_cancer, preprocess, load_test
# from modules.preprocess_min_max import load_data_breast_cancer, preprocess, load_test
# from modules.preprocess_chi2 import load_data_breast_cancer, preprocess, load_test
from modules.preprocess_f_classif import load_data_breast_cancer, preprocess, load_test

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    inputs, outputs = load_data_breast_cancer('data/raw_data/data.csv')

    inputs = preprocess(inputs)

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=14)
    # change model
    # model = BreastCancerLRModel()
    # model = BreastCancerDTModel()
    # model = BreastCancerRFModel()
    # model = BreastCancerSVMModel()
    # model = BreastCancerSGDModel()
    # model = BreastCancerGBCModel()
    model = BreastCancerHGBCModel()
    # model = BreastCancerBCModel()
    # model = BreastCancerVotingModel()
    model.train(X_train, y_train)

    predicts = model.predict(X_test)
    for id, y_tes, predict in list(zip([e.features[0] for e in X_test], [output.score for output in y_test], predicts)):
        if y_tes != predict:
            df = pd.DataFrame({'id': [id],
                               'actual': [y_tes],
                               'predict': [predict]})
            df.to_csv('data/raw_data/output.csv', mode='a', encoding='utf-8', index=False, header=False)
            print("id: ", id, "actual: ", y_tes, "predict:", predict)


    # test_inputs, ids = load_test('data/raw_data/test.csv')
    # predicts = model.predict(test_inputs)
    #
    # # Save to file
    # ids['diagnosis'] = ['M' if predict == 1.0 else 'B' for predict in predicts]
    # # ids['diagnosis'] = [predict for predict in predicts]
    # ids.to_csv('data/raw_data/submission.csv', encoding='utf-8', index=False)