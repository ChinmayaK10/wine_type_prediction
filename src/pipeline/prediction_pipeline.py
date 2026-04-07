import sys
import pandas as pd
from src.exception import CustomError

from src.utils import load_obj
import os
from src.logger import logging

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts",'model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            logging.info("Loading the model")
            model=load_obj(file_path=model_path)
            preprocessor=load_obj(preprocessor_path)

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
             raise CustomError(e,sys)
            
class CustomData:

    def __init__(self,fixed_acidity:float,volatile_acidity:float,citric_acid:float,
                 residual_sugar:float,chlorides:float,freesulfur_dioxide:float,
                 totalsulfur_dioxide:float,density:float,pH:float,sulphates:float,
                alcohol:float,quality:int):
        
            self.fixed_acidity=fixed_acidity
            self.volatile_acidity=volatile_acidity
            self.citric_acid=citric_acid
            self. residual_sugar= residual_sugar
            self.chlorides=chlorides
            self.freesulfur_dioxide=freesulfur_dioxide
            self.totalsulfur_dioxide=totalsulfur_dioxide
            self.density=density
            self.pH=pH
            self.sulphates=sulphates
            self.alcohol=alcohol
            self.quality=quality

    def get_data_as_data_frame(self):
        try:
                custom_data_input_dict={
                    'fixed_acidity':[self.fixed_acidity], 
                    'volatile_acidity':[self.volatile_acidity],
                    'citric_acid':[self.citric_acid],
                    'residual_sugar':[self.residual_sugar],
                    'chlorides':[self.chlorides],
                    'free_sulfur_dioxide':[self.freesulfur_dioxide], 
                    'total_sulfur_dioxide':[self.totalsulfur_dioxide], 
                    'density':[self.density],
                    'pH':[self.pH],
                    'sulphates':[self.sulphates],
                    'alcohol':[self.alcohol],
                    'quality':[self.quality]
                    }
                logging.info('sending data back as dataframe')
                return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
             raise CustomError(e,sys)
