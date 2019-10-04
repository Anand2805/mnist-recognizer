from flask import Flask, request, json, render_template, send_from_directory
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify
import base64
import io
from PIL import Image
import numpy as np
import createMnistFormat as recognizer
import randomForest as rfc

app = Flask(__name__,template_folder='template')
api = Api(app)
CORS(app)

@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/recognizeImage", methods=['POST'])
def recognizeImage():
    print("Image recieved")
    data_url = request.values['imageBase64']
    data_url = data_url.split(",")[1]
    img_bytes = base64.b64decode(str(data_url))
    img = Image.open(io.BytesIO(img_bytes))
    predictedData = recognizer.readAndSendRecognizedData(img, nRFC)
    print('predicted is : {}'.format(predictedData))
    return jsonify({'n': nRFC, 'accuracy': accuracy, 'predicted': predictedData.item(0)})


if __name__ == '__main__':
    global nRFC, accuracy
    nRFC, accuracy = rfc.findNEstimators()
    print(nRFC, accuracy)
    app.run(port=5002)
