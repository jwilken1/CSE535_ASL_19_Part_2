from flask import Flask, request
import pickle
import numpy
import os
from pandas.io.json import json_normalize
import numpy as np

app = Flask(__name__)

max_length = 7424
max_length_2 = 232

loaded_model_1 = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/models/model_1.pkl', 'rb'))
loaded_model_2 = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/models/model_2.pkl', 'rb'))
loaded_model_3 = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/models/model_3.pkl', 'rb'))
loaded_model_4 = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/models/model_4.pkl', 'rb'))

# Takes 1 video data set and 1 data identifier and gives you a feature vector!
def getFeatureVector(training_feature_data):
    zeroCrossingArray = numpy.array([])
    maxDiffArray = numpy.array([])
    rY = training_feature_data
    normRawData = (rY - numpy.mean(rY))/(numpy.max(rY-numpy.mean(rY))-numpy.min(rY-numpy.mean(rY)))
    diffNormRawData = numpy.diff(normRawData)

    if diffNormRawData[0] > 0:
        initSign = 1
    else:
        initSign = 0

    windowSize = 5;
    
    for x in range(1, len(diffNormRawData)):
        if diffNormRawData[x] > 0:
            newSign = 1
        else:
            newSign = 0
            
        if initSign != newSign:
            zeroCrossingArray = numpy.append(zeroCrossingArray, x)
            initSign = newSign
            maxIndex = numpy.minimum(len(diffNormRawData),x+windowSize)
            minIndex = numpy.maximum(0,x - windowSize)
            maxVal = numpy.amax(diffNormRawData[minIndex:maxIndex])
            minVal = numpy.amin(diffNormRawData[minIndex:maxIndex])
            maxDiffArray = numpy.append(maxDiffArray, (maxVal - minVal))
			

    index = numpy.argsort(-maxDiffArray)

    feature_vector = numpy.array([])
    
    if (len(diffNormRawData) < (max_length_2 - 1)):
        zeros = numpy.zeros(max_length_2 - 1 - len(diffNormRawData))
        diffNormRawData = numpy.append(diffNormRawData, zeros)
    feature_vector = numpy.append(feature_vector, diffNormRawData[0:231])
    
    temp_array = zeroCrossingArray[index[0:5]]
    if (len(temp_array) < 5):
        zeros = numpy.zeros(5 - len(temp_array))
        temp_array = numpy.append(temp_array, zeros)
    feature_vector = numpy.append(feature_vector, temp_array)
    
    temp_array = maxDiffArray[index[0:5]]
    if (len(temp_array) < 5):
        zeros = numpy.zeros(5 - len(temp_array))
        temp_array = numpy.append(temp_array, zeros)
    feature_vector = numpy.append(feature_vector, temp_array)
    
    return (feature_vector)
    

# Takes 1 video data and gives you total features Matrix!
def getFeatureMatrix(request_json):
    rightWrist_y = []
    rightWrist_x = []
    leftWrist_y = []
    leftWrist_x = []
    leftElbow_x = []
    leftElbow_y = []
    rightElbow_x = []
    rightElbow_y = []

    for frame in request_json:
        for keypoint in frame['keypoints']:
            if (keypoint['part'] == 'rightWrist'):
                rightWrist_y.append(keypoint['position']['y'])
            if (keypoint['part'] == 'rightWrist'):
                rightWrist_x.append(keypoint['position']['x'])
            if (keypoint['part'] == 'leftWrist'):
                leftWrist_y.append(keypoint['position']['y'])
            if (keypoint['part'] == 'leftWrist'):
                leftWrist_x.append(keypoint['position']['x'])
            #if (keypoint['part'] == 'leftElbow'):
            #    leftElbow_x.append(keypoint['position']['x'])
            #if (keypoint['part'] == 'leftElbow'):
            #    leftElbow_y.append(keypoint['position']['y'])
            #if (keypoint['part'] == 'rightElbow'):
            #    rightElbow_x.append(keypoint['position']['x'])
            #if (keypoint['part'] == 'rightElbow'):
            #    rightElbow_y.append(keypoint['position']['y'])
        
    feature_matrix = numpy.array([])
    feature_vector = getFeatureVector(rightWrist_y)
    feature_matrix = numpy.append(feature_matrix, feature_vector)
    feature_vector = getFeatureVector(rightWrist_x)
    feature_matrix = numpy.append(feature_matrix, feature_vector)
    feature_vector = getFeatureVector(leftWrist_y)
    feature_matrix = numpy.append(feature_matrix, feature_vector)
    feature_vector = getFeatureVector(leftWrist_x)
    feature_matrix = numpy.append(feature_matrix, feature_vector)
    #feature_vector = getFeatureVector(leftElbow_x)
    #feature_matrix = numpy.append(feature_matrix, feature_vector)
    #feature_vector = getFeatureVector(leftElbow_y)
    #feature_matrix = numpy.append(feature_matrix, feature_vector)
    #feature_vector = getFeatureVector(rightElbow_x)
    #feature_matrix = numpy.append(feature_matrix, feature_vector)
    #feature_vector = getFeatureVector(rightElbow_y)
    #feature_matrix = numpy.append(feature_matrix, feature_vector)
        
    return (feature_matrix)


@app.route('/', methods=['POST'])
def predict():
    training_entry = request.get_json()
    feature_matrix = getFeatureMatrix(training_entry)
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
    
    result_1 = loaded_model_1.predict([feature_matrix])
    result_2 = loaded_model_2.predict([result])
    result_3 = loaded_model_3.predict([feature_matrix])
    result_4 = loaded_model_4.predict([feature_matrix])

    print(result_1)
    print(result_2)
    print(result_3)
    print(result_4)
    return {'1': result_1[0], '2': result_2[0], '3': result_3[0], '4': result_4[0]}
