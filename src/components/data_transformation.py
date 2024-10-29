import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
        
    def get_data_transformer_object(self):
        #creates pickle files for column tansformer,onehotencoder
        try:
            num_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
             
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler',StandardScaler())
            ])
            
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('Encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])
            
            logging.info("Numerical Columns Encoded")
            logging.info("Categorical Columns Encoded")
            
            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,num_features),
                ("cat_pipeline",cat_pipeline,categorical_features)
            ])
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df =pd.read_csv(test_path)
            
            logging.info("Train and test df are read")
            
            logging.info("Obtaining Preprocessor")
            
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math_score"
            num_features = ["reading_score","writing_score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying Preprocessor ObjecT Now")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info("Preprocessing Completed & Saved")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj) 
            
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
            
        except Exception as e:
            raise CustomException(e,sys)