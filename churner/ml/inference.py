import os
import pickle
import pandas as pd



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
        model_path = os.path.join(self.config['savepath'], 
                                  'churner_calibrated_model.pkl')
        with open('model_path', 'rb') as file:
            model = pickle.load(file)
        # Predict 
        predicted_proba = model.predict_proba(X)[0, 1]
        return predicted_proba
    
    def postprocess(self, proba):
        threshold = self.config['inference']['alert_threshold']
        if proba > threshold:
            message = "This client will leave us. Call him !" 
        else:
            message = " This client is safe !" 
          
        response = {
            'message': message, 
            'probability': proba
        }
        return response

    def inference(self, input_dict):
        X = preprocess(input_dict)
        proba = predict(X)
        response = postprocess(proba)
        return response
    
    