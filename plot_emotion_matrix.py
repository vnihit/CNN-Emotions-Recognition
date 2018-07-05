# -*- coding: utf-8 -*-
import cv2
import sys
from decouple import config
from emo_recogniser import EmotionRecognition
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

save_dir = config('SAVE_DIRECTORY')
dataset_images = config('SAVE_DATASET_IMAGES_FILENAME')
dataset_labels = config('SAVE_DATASET_LABELS_FILENAME')
size_face = int(config('SIZE_FACE'))
emotions = ['angry', 'disgusted', 'fearful','happy', 'sad', 'surprised', 'neutral']


# Load Model
network = EmotionRecognition()
network.build_network()

images = np.load(join(save_dir, dataset_images))
labels = np.load(join(save_dir, dataset_labels))
images = images.reshape([-1, size_face, size_face, 1])
labels = labels.reshape([-1, len(emotions)])

print('[+] Loading Data')
data = np.zeros((len(emotions), len(emotions)))
for i in range(images.shape[0]):
    result = network.predict(images[i])
    data[np.argmax(labels[i]), result[0].tolist().index(max(result[0]))] += 1
    # print x[i], ' vs ', y[i]

# Take % by column
for i in range(len(data)):
    total = np.sum(data[i])
    for x in range(len(data[0])):
        data[i][x] = data[i][x] / total
print(data)

print('[+] Generating graph')
c = plt.pcolor(data, edgecolors='k', linewidths=4,
               cmap='Blues', vmin=0.0, vmax=1.0)


def show_values(pc, fmt="%.2f", **kw):
    pc.update_scalarmappable()
    ax = pc.axes
    ax.set_yticks(np.arange(len(emotions)) + 0.5, minor=False)
    ax.set_xticks(np.arange(len(emotions)) + 0.5, minor=False)
    ax.set_xticklabels(emotions, minor=False)
    ax.set_yticklabels(emotions, minor=False)
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


show_values(c)
plt.xlabel('Predicted Emotion')
plt.ylabel('Real Emotion')
plt.show()
