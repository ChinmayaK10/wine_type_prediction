import os 
import sys 
from src.exception import CustomError
import pandas as pd
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')
        os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
     
        try:

            logging.info('reading the dataset as df')
            df=pd.read_csv(r'notebook\wine-quality-white-and-red.csv')
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info(" doing train test")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,header=True,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,header=True,index=False)
            
            logging.info('data saved')
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomError(e,sys)
if __name__=="__main__":
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()
    

    preporcessor=DataTransformation()
    train_arr,test_arr,preporcessor_path=preporcessor.initiate_data_transformation(train_path=train_path,test_path=test_path)

    model=ModelTrainer()
    model.initiate_model_trainer(train_array=train_arr,test_array=test_arr,preprocessor_path=preporcessor_path)