from decouple import config
import pandas as pd 
import numpy as np 
import cv2 
from PIL import Image
import os

save_dir = config('SAVE_DIRECTORY')
dataset_images = config('SAVE_DATASET_IMAGES_FILENAME')
dataset_labels = config('SAVE_DATASET_LABELS_FILENAME')
dataset_csv_filename = config('DATASET_CSV_FILENAME')
size_face = int(config('SIZE_FACE'))
cascade_path = config('CASC_PATH')
emotions = ['angry', 'disgusted', 'fearful','happy', 'sad', 'surprised', 'neutral']


data = pd.read_csv(os.path.join(save_dir, dataset_csv_filename))
labels = []
images = []
index = 1
total = data.shape[0]

cascade_classifier = cv2.CascadeClassifier(cascade_path)

def format_face_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    gray_border = np.zeros((150, 150), np.uint8)
    gray_border[:, :] = 200
    gray_border[
        int((150/2) - (size_face/2)): int((150/2) + (size_face/2)),
        int((150/2) - (size_face/2)): int((150/2) + (size_face/2))
    ] = image
    image = gray_border
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    #  None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size

    try:
        image = cv2.resize(image, (size_face, size_face),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("Problem during resize")
        return None
    return image


for index, row in data.iterrows():

    #convert emotions column to a vector
    
    d = np.zeros(len(emotions))
    d[row['emotion']] = 1.0
    emotion = d
    #convert pixel data into image
    data_image = np.fromstring(str(row['pixels']), dtype=np.uint8, sep=' ')
    data_image = data_image.reshape((size_face, size_face))
    data_image = Image.fromarray(data_image).convert('RGB')
    #print(data_image)
    data_image = np.array(data_image)[:, :, ::-1].copy()
    data_image = format_face_image(data_image)
    #print(data_image)
    if data_image is not None:
        labels.append(emotion)
        images.append(data_image)
    # if index==10:
    #     break
    index+=1
    #print("Progress: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))
    print("Progress:", index*100/total)
print("Total: " + str(len(images)))
np.save(os.path.join(save_dir, dataset_images), images)
np.save(os.path.join(save_dir, dataset_labels), labels)
