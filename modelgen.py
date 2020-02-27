import json
import os
from pandas.io.json import json_normalize
import numpy as np

def load():
    training_folders = ['buy','fun','hope','really','communicate','mother']
    training_folder_location = './training_data/'
    all_data = []
    all_data_labels = []
    traning_data = []
    test_data= []
    training_files =[]
    data_lengths = []

    for training_label in training_folders:
        file_list = os.listdir(training_folder_location + training_label + "/")
        for file in file_list:
            with open(training_folder_location + training_label + "/" + file) as video_entry:
                training_entry = json.load(video_entry)

            result = json_normalize(training_entry, 'keypoints', ['score'], record_prefix='keypoints.')
            #result = result.reindex(columns=['score', 'keypoints.part', 'keypoints.score', 'keypoints.position.x', 'keypoints.position.y']) # Take away part pt 2
            result = result.reindex(columns=['score', 'keypoints.score', 'keypoints.position.x', 'keypoints.position.y'])
            result = np.array(result)
            #result = np.reshape(result, (-1, 85)) # Take away part pt 2
            result = np.reshape(result, (-1, 68))
            result = result[:, 12:44] # Narrow features to ignore hips and some facial features
            result = list(result.ravel())
            all_data.append(result)
            data_lengths.append(len(result))
            all_data_labels.append(training_label)

    # 0 Padding
    max_data_size = max(data_lengths)
    for item in all_data:
        item.extend([0] * (max_data_size - len(item)))

    print(all_data[0])
    print(all_data_labels)

if __name__ == "__main__":
	load()