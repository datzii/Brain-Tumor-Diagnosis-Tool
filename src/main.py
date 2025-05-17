
import os
import time

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from tensorflow.keras.models import load_model
from services.brain_tumor_classification_service import start_brain_tumor_classifier


app = Flask(__name__)
CORS(app,
     allow_origins=["*"],
     allow_headers=["Content-Type, Authorization, X-Auth-Token"],
     allow_methods=["GET, POST, PUT, DELETE, OPTIONS"],
     supports_credentials=True)


@app.route("/", methods=['GET', 'OPTIONS'])
@app.route("/v1/status/", methods=['GET', 'OPTIONS'])
@cross_origin()
def status():
    start = time.time()

    took_ms = round((time.time() - start) * 1000) + 1

    return jsonify(
        kudos="up",
        took_ms=took_ms
    ), 200


@app.route("/diagnosis_tool/get_prediction", methods = ['GET', 'POST', 'PUT','DELETE', 'OPTIONS'])
@cross_origin()
def api():
    data = request.json
    print(data)   
    answer_evaluator = start_brain_tumor_classifier(data, diagnosis_models)
    return answer_evaluator


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', "Content-Type, Authorization, X-Auth-Token")
    response.headers.add('Access-Control-Allow-Methods', "GET, POST, PUT, DELETE, OPTIONS")
    return response

global models
models = []

def initialize_model():
    global models
    paths = ['../models/resNet/resNet50-09-0.98-0.06.h5', '../models/resNet100/model-06-0.97-0.11.h5', '../models/combined_100/model-11-0.98-0.08.h5']
    for path in paths:
        model = load_model(path)
        models.append(model)
        print('model loaded')    
    return models

if __name__ == "__main__":

    port = 5600
    
    diagnosis_models = initialize_model()

    if os.getenv('PORT') is not None and os.getenv('PORT') != '':
        port = int(os.getenv('PORT'))

    app.run(host="0.0.0.0", port=port, debug=False)
