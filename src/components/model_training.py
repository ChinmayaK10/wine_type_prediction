import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomError
import sys
import os

from sklearn.metrics import accuracy_score,f1_score
# models

# Extra Trees (faster than RandomForest sometimes)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
# Linear models
from sklearn.linear_model import LogisticRegression
# Tree-based
from sklearn.tree import DecisionTreeClassifier
# Ensemble models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
# Support Vector Machine
from sklearn.svm import SVC
# Neighbors
from sklearn.neighbors import KNeighborsClassifier
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Split training and test into input and output data")
            x_train=train_array[:,:-1]
            x_test=test_array[:,:-1]
            Y_train=train_array[:,-1]
            Y_test=test_array[:,-1]

            svc=SVC()
            svc.fit(x_train,Y_train)
            y_pred=svc.predict(x_test)
            accuracy=accuracy_score(y_pred=y_pred,y_true=Y_test)
            f1_val=f1_score(y_pred=y_pred,y_true=Y_test)

            save_object(self.model_trainer_config.trained_model_file_path,
                        obj=svc)
            return {'accuracy':accuracy,'f1_val':f1_val}
            
        



        except Exception as e:
            raise CustomError(e,sys)
        