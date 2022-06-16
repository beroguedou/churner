import sys
sys.path.append("/Users/bguedou/churner-ml/")

import os
import pickle
import pandas as pd
from churner.ml.utils import preprocessor


class Predictor():

    def __init__(self, config):
        self.config = config  
        
    def preprocess(self, input_dict):
        # Create a dataframe with parameters data
        input_dict = {k: [v] for (k,v) in input_dict.items()}
        df = pd.DataFrame(input_dict)
        # Make the preprocessings 
        # Separate variables
        X = preprocessor(df,
                         self.config, 
                         option_train='inference', 
                         option_output='inference')

        return X
    
    def predict(self, X):
        # Load model with pickle
        model_path = os.path.join(self.config['model']['savepath'], 
                                  self.config['model']['name'])
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        # Predict 
        predicted_proba = model.predict_proba(X)[0, 1]
        return predicted_proba
    
    def postprocess(self, proba):
        threshold = self.config['inference']['alert_threshold']
        if proba >= threshold:
            message = "This client will leave us. Call him !" 
        else:
            message = " This client is safe !" 
          
        response = {
            'message': message, 
            'probability': round(proba, 4)
        }
        return response

    def inference_pipeline(self, input_dict):
        X = self.preprocess(input_dict)
        proba = self.predict(X)
        response = self.postprocess(proba)
        return response
    
    