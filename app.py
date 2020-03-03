from flask import Flask, request
import pickle
import numpy
import os
from pandas.io.json import json_normalize
import numpy as np

app = Flask(__name__)

max_length = 7424


loaded_model_1 = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/models/model_1.pkl', 'rb'))
loaded_model_2 = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/models/model_2.pkl', 'rb'))
loaded_model_3 = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/models/model_3.pkl', 'rb'))
loaded_model_4 = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/models/model_4.pkl', 'rb'))


@app.route('/', methods=['POST'])
def predict():
    training_entry = request.get_json()

    result = json_normalize(training_entry, 'keypoints', ['score'], record_prefix='keypoints.')
    #result = result.reindex(columns=['score', 'keypoints.part', 'keypoints.score', 'keypoints.position.x', 'keypoints.position.y']) # Take away part pt 2
    result = result.reindex(columns=['score', 'keypoints.score', 'keypoints.position.x', 'keypoints.position.y'])
    result = np.array(result)
    #result = np.reshape(result, (-1, 85)) # Take away part pt 2
    result = np.reshape(result, (-1, 68))
    result = result[:, 12:44] # Narrow features to ignore hips and some facial features
    result = list(result.ravel())
    
    # 0 Padding
    result.extend([0] * (max_length - len(result)))
    
    result_1 = loaded_model_1.predict([result])
    result_2 = loaded_model_2.predict([result])
    result_3 = loaded_model_3.predict([result])
    result_4 = loaded_model_4.predict([result])

    print(result_1)
    print(result_2)
    print(result_3)
    print(result_4)
    return {'1': result_1[0], '2': result_2[0], '3': result_3[0], '4': result_4[0]}
