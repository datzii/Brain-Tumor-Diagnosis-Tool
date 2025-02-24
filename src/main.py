
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


@app.route("/v1/evaluator/", methods = ['GET', 'POST', 'PUT','DELETE', 'OPTIONS'])
@cross_origin()
def api():
    data = request.json    
    answer_evaluator = start_brain_tumor_classifier(data, diagnosis_model)
    return answer_evaluator


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', "Content-Type, Authorization, X-Auth-Token")
    response.headers.add('Access-Control-Allow-Methods', "GET, POST, PUT, DELETE, OPTIONS")
    return response

def initialize_model():
    model_path = '../models/resNet/model-14-0.99-0.05.h5'
    model = load_model(model_path)
    print('model loaded')
    return model

global diagnosis_model
diagnosis_model = initialize_model()

if __name__ == "__main__":

    port = 5600
    
    if os.getenv('PORT') is not None and os.getenv('PORT') != '':
        port = int(os.getenv('PORT'))

    app.run(host="0.0.0.0", port=port, debug=False)
