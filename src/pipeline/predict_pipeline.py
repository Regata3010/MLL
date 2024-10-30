import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_objects
import os


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preproccessor_path = os.path.join('artifacts','preprocessor.pkl')
            model = load_objects(file_path = model_path)
            preproccessor = load_objects(file_path = preproccessor_path)
            data_scaled = preproccessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys) 
    
    
class CustomData:
    def __init__(self,gender:str,race_ethnicity:str,lunch: str,parental_level_of_education,test_preparation_course:int,writing_score: int,reading_score:int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.lunch = lunch
        self.parental_level_of_education = parental_level_of_education
        self.test_preparation_course = test_preparation_course
        self.writing_score = writing_score
        self.reading_score = reading_score
        
    def get_data_as_frame(self):
        try:
            custom_input_data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "lunch": [self.lunch],
                "parental_level_of_education": [self.parental_level_of_education],
                "test_preparation_course": [self.test_preparation_course],
                "writing_score": [self.writing_score],
                "reading_score": [self.reading_score]
                }
            return pd.DataFrame(custom_input_data_dict)
        except Exception as e:
            pass