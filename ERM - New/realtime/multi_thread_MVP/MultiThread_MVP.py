import os

import cv2
import numpy as np
import time
import tensorflow as tf
from retrieve_frozen_graphs import get_frozen_model
from multiprocessing.pool import ThreadPool

# Get the frozen model
frozen_func = get_frozen_model('models/frozen_models/frozen_graph.pb')


# Define predict function
def predict(data):
    result1 = frozen_func(x=tf.constant(data, dtype=tf.float32))[0]
    return result1

# Setting video capture device
video=cv2.VideoCapture(0)

# Loading face detection model
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Setting labels dictionary
labels_dict={0:'Angry',1:'Contempt',2:'Disgust',3:'Fear',4:'Happy',5:'Neutral',6:'Sad',7:'Surprise'}
#labels_dict={0:'Angry',1:'Fear',2:'Happy',3:'Neutral',4:'Sad',5:'Surprise'}


# will create as many threads as there are CPUs by default
pool = ThreadPool()

while True:
    start = time.time()

    # Reading video feed in by frame
    ret,frame=video.read()
    frame = cv2.flip(frame, 1)

    # Passing frame into face detection model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()

    # Getting height and width of frame
    h, w = frame.shape[:2]

    # Input list to contain (numpy array of face, 4-element tuple of coordinates )
    input_list = []

    # For loop number of faces
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            # Getting sub image of face
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            sub_face_img = frame[y:y1, x:x1]

            # If face sub image is not empty, resize and reshape into numpy array
            # to pass into input list
            if sub_face_img.shape[0] != 0 and sub_face_img.shape[1] != 0:
                resized = cv2.resize(sub_face_img, (224, 224))
                reshaped = np.reshape(resized, (1, 224, 224, 3))
                input_list.append((reshaped, (x, y, x1, y1)))

    # Check if input list is not empty
    if input_list:
        # Creating numpy array of each face sub image from input list
        faces_list = np.asarray([i[0] for i in input_list])

        # Check if faces_list is not empty
        if np.any(faces_list):
            # Pass in tensor of faces into model for predictions

            # pool = ThreadPool(faces_list.shape[0])
            result = pool.map(predict, faces_list)


            # For loop number of predictions
            for i in range(len(result)):
                # Get box coordinates of face from input list
                x, y, x1, y1 = input_list[i][1]

                # Get highest prediction
                label = np.argmax(result[i])

                # Drawing box around face
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

                # Drawing text for current emotion
                scaling_font_size = ((x1-x) * 0.0047)
                font_size = scaling_font_size if scaling_font_size >= 0.5 else 0.5
                text_width = cv2.getTextSize(labels_dict[label], cv2.FONT_HERSHEY_SIMPLEX, font_size, 4)[0]
                rectangle_mid = int((x1 - x) / 2)
                cv2.putText(frame, labels_dict[label], (((rectangle_mid + x) - (text_width[0] // 2)), y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 4)
                cv2.putText(frame, labels_dict[label], (((rectangle_mid + x) - (text_width[0] // 2)), y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
    # frame = cv2.resize(frame, (1366, 1080), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    end = time.time()
    print(f"inference time: {round(end - start, 2)} seconds")
    # print(f"FPS: {round(1//(end - start), 2)}")
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()