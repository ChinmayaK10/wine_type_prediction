import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.logger import logging
from  src.exception import CustomError
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    proccessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transfromation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_feature=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides','free_sulfur_dioxide','total_sulfur_dioxide', 'density',
            'pH', 'sulphates', 'alcohol', 'quality']
            
            num_pipeline=Pipeline(
                steps=[
                    ("scaler",StandardScaler()) 
                ]
            )
            logging.info("standard scaler done")
        
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_feature)
            ])
            return preprocessor
        except Exception as e:
            raise CustomError(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("reading the train and test data")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("obtaining preprocessor")
            preprocessor_obj=self.get_data_transformer_object()

            target_column="type"
            logging.info("separting the X and Y")
            input_feature_train_df=train_df.drop(columns=[target_column])
            target_feature_train_df=train_df[target_column].map({'red':1,"white":0})
            
            input_feature_test_df=test_df.drop(columns=[target_column])
            target_feature_test_df=test_df[target_column].map({'red':1,"white":0})
            
            logging.info("applying preprocessor in the new dataframes")

            input_feature_processed_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_processed_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_processed_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_processed_test_arr,np.array(target_feature_test_df)
            ]

            logging.info('saved preprocessing object')

            save_object(
                file_path=self.data_transfromation_config.proccessor_obj_file_path,
                obj=preprocessor_obj
            )
            return(train_arr,test_arr,self.data_transfromation_config.proccessor_obj_file_path)

        except Exception as e:
            raise CustomError (e,sys)

