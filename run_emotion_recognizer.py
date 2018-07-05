import cv2
import sys
from decouple import config
from emo_recogniser import EmotionRecognition
import numpy as np



cascade_path = config('CASC_PATH')
size_face = int(config('SIZE_FACE'))
emotions = ['angry', 'disgusted', 'fearful','happy', 'sad', 'surprised', 'neutral']

cascade_classifier = cv2.CascadeClassifier(cascade_path)


def format_image(image, faces):

    # None is we don't found an image
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
        print("[+] Problem during resize")
        return None
    # cv2.imshow("Lol", image)
    # cv2.waitKey(0)
    return image


# Load Model
network = EmotionRecognition()
network.build_network()

video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

feelings_faces = []
for index, emotion in enumerate(emotions):
    feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # image = format_image(frame)
    #
    # # Predict result with network
    # result = network.predict(image)

    if len(frame.shape) > 2 and frame.shape[2] == 3:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(frame, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=3)

    image = format_image(image, faces)

    # Predict result with network
    result = network.predict(image)

    # Draw rectangle around face
    for (x,y,w,h) in faces:
      cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    
    if result is not None:
        for index, emotion in enumerate(emotions):
            cv2.putText(frame, emotion, (10, index * 20 + 20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        
            cv2.rectangle(frame, (130, index * 20 + 10), (130 +
                                                          int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)

        face_image = feelings_faces[np.argmax(result[0])]


        for c in range(0, 3):
            frame[200:320, 10:130, c] = face_image[:, :, c] * \
                (face_image[:, :, 3] / 255.0) + frame[200:320,10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
