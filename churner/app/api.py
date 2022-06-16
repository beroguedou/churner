import yaml
from flask import Flask, request
from inference import Predictor

# Paramètres
config_path = "/Users/bguedou/churner-ml/config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
predictor = Predictor(config)

app = Flask(__name__)


@app.route('/serving/predict', methods=['POST'])
def prediction_pipeline():
    input_dict = request.json
    response = predictor.inference_pipeline(input_dict)
    return response

@app.route('/serving/healthcheck', methods=['GET'])
def healthcheck():
    return 'Hello I am well !'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)