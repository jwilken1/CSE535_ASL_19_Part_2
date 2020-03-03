from flask import Flask, request
import pickle
import numpy

app = Flask(__name__)

max_length = 232
training_features = ['rightWrist_y', 'rightWrist_x', 'leftWrist_y', 'leftWrist_x', 'leftElbow_x', 'leftElbow_y', 'rightElbow_x', 'rightElbow_y']

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
    
    if (len(diffNormRawData) < (max_length - 1)):
        zeros = numpy.zeros(max_length - 1 - len(diffNormRawData))
        diffNormRawData = numpy.append(diffNormRawData, zeros)
    feature_vector = numpy.append(feature_vector, diffNormRawData)
    
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
    
    for frame in training_entry:
    for keypoint in frame['keypoints']:
        if (keypoint['part'] == 'rightWrist'):
            #print(keypoint['position']['y'])
            print (keypoint)
            #testing_data.append(keypoint['position']['y'])
    for keypoint in frame['keypoints']:
        if (keypoint['part'] == 'leftWrist'):
            print (keypoint)
    
    
    counter = 1
    feature_matrix = numpy.array([])
    for training_feature in training_features:
        feature_vector = getFeatureVector(raw_data, training_feature)
        feature_matrix = numpy.append(feature_matrix, feature_vector)
        
    return (feature_matrix)


@app.route('/', methods=['POST'])
def predict():
    request_json = request.get_json()

    # Model 1 (Jane Ivanova)
    loaded_model_1 = pickle.load(open('models/model_1.pkl', 'rb'))
    loaded_model_2 = pickle.load(open('models/model_2.pkl', 'rb'))
    loaded_model_3 = pickle.load(open('models/model_3.pkl', 'rb'))
    loaded_model_4 = pickle.load(open('models/model_4.pkl', 'rb'))
    
    feature_matrix = getFeatureMatrix(request_json)

    result_1 = loaded_model_1.predict([feature_matrix])
    result_2 = loaded_model_2.predict([feature_matrix])
    result_3 = loaded_model_3.predict([feature_matrix])
    result_4 = loaded_model_4.predict([feature_matrix])

    print(result_1[0])
    print(result_2[0])
    print(result_3[0])
    print(result_4[0])
    return {'1': result_1[0], '2': result_2[0], '3': result_3[0], '4': result_4[0]}
