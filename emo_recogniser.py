from __future__ import division, absolute_import
import re
import os
import random
import sys
import cv2
import numpy as np
import tflearn
from decouple import config
from sklearn.model_selection import train_test_split
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

save_dir = config('SAVE_DIRECTORY')
dataset_images = config('SAVE_DATASET_IMAGES_FILENAME')
dataset_labels = config('SAVE_DATASET_LABELS_FILENAME')
dataset_csv_filename = config('DATASET_CSV_FILENAME')
size_face = int(config('SIZE_FACE'))
cascade_path = config('CASC_PATH')
emotions = ['angry', 'disgusted', 'fearful','happy', 'sad', 'surprised', 'neutral']
save_model_file = config('MODEL_FILE')


class DatasetLoader:
    def load_from_save(self):
        images = np.load(os.path.join(save_dir, dataset_images))
        images = images.reshape([-1, size_face, size_face, 1])
        labels = np.load(os.path.join(save_dir, dataset_labels))
        labels = labels.reshape([-1, len(emotions)])
        self._images, self._images_test, self._labels, self._labels_test = train_test_split(images, labels, test_size=0.20, random_state=42)

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def images_test(self):
        return self._images_test

    @property
    def labels_test(self):
        return self._labels_test


class EmotionRecognition:
    def __init__(self):
        self.dataset = DatasetLoader()

    def build_network(self):
        
        print('Building Neural Network')
        self.network = input_data(shape=[None, size_face, size_face, 1])
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 128, 4, activation='relu')
        self.network = dropout(self.network, 0.3)
        self.network = fully_connected(self.network, 3072, activation='relu')
        self.network = fully_connected(
            self.network, len(emotions), activation='softmax')
        self.network = regression(
            self.network,
            optimizer='momentum',
            loss='categorical_crossentropy'
        )
        self.model = tflearn.DNN(
            self.network,
            checkpoint_path = save_dir + '/emotion_recognition',
            max_checkpoints = 1,
            tensorboard_verbose=2
        )
        self.load_model()

    def load_saved_dataset(self):
        self.dataset.load_from_save()
        print('Dataset loaded')

    def start_training(self):
        self.load_saved_dataset()
        self.build_network()
        if self.dataset is None:
            self.load_saved_dataset()
        # Training
        print('Training network')
        self.model.fit(self.dataset.images, self.dataset.labels,
            validation_set=(self.dataset.images_test,self.dataset.labels_test),n_epoch=100,
            batch_size=50,
            shuffle=True,
            show_metric=True,
            snapshot_step=200,
            snapshot_epoch=True,
            run_id='emotion_recognition'
        )

    def predict(self, image):
        if image is None:
            return None
        image = image.reshape([-1, size_face, size_face, 1])
        return self.model.predict(image)

    def save_model(self):
        self.model.save(os.path.join(save_dir, save_model_file))
        print('Model trained and saved at ' + save_model_file)

    def load_model(self):
        if os.path.isfile(os.path.join(save_dir, save_model_file)):
            self.model.load(os.path.join(save_dir, save_model_file))
            print('Model loaded from ' + save_model_file)


def show_usage():
    # I din't want to have more dependecies
    print('[!] Usage: python emotion_recognition.py')
    print('\t emotion_recognition.py train \t Trains and saves model with saved dataset')
    


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        show_usage()
        exit()
    network = EmotionRecognition()
    if sys.argv[1] == 'train':
        network.start_training()
        network.save_model()
    else:
        show_usage()

