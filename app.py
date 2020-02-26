from flask import Flask, request
import pickle
import numpy

app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    request_json = request.get_json()

    # Model 1 (Jane Ivanova)
    loaded_model_1 = pickle.load(open('models/model_1.pkl', 'rb'))
    max_records = 34
    index = 0
    testing_data = []
    for frame in request_json:
        index = index + 1
        for keypoint in frame['keypoints']:
            if (keypoint['part'] == 'rightWrist'):
                testing_data.append(keypoint['position']['y'])
        if index == max_records: break

    norm_raw_data = (testing_data - numpy.mean(testing_data)) / (
            numpy.max(testing_data - numpy.mean(testing_data)) - numpy.min(testing_data - numpy.mean(testing_data)))
    diff_norm_raw_data = numpy.diff(norm_raw_data)
    feature_vector = numpy.append(numpy.array([]), diff_norm_raw_data)
    result_1 = loaded_model_1.predict([feature_vector])

    print(result_1[0])
    return {'1': result_1[0]}
