import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# model=load_model('../models/ENetB0_E30_B64_ImageNet.h5')
model=load_model('../models/ENetB0_6Class_ValAcc5780.h5')

modelFile = "../models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "../models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# labels_dict={0:'Angry',1:'Contempt',2:'Disgust',3:'Fear',4:'Happy',5:'Neutral',6:'Sad',7:'Surprise'}
labels_dict={0:'Angry',1:'Fear',2:'Happy',3:'Neutral',4:'Sad',5:'Surprise'}

# len(number_of_image), image_height, image_width, channel

frame=cv2.imread("../images/professors-2.png")
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                             1.0, (300, 300), (104.0, 117.0, 123.0))
net.setInput(blob)
faces = net.forward()
h, w = frame.shape[:2]
for i in range(faces.shape[2]):
    confidence = faces[0, 0, i, 2]
    if confidence > 0.5:

        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, x1, y1) = box.astype("int")
        sub_face_img = frame[y:y1, x:x1]
        resized = cv2.resize(sub_face_img, (224, 224))
        reshaped = np.reshape(resized, (1, 224, 224, 3)) / 255
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # Drawing box around face
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

        # Drawing text for current emotion
        scaling_font_size = ((x1 - x) * 0.0047)
        font_size = scaling_font_size if scaling_font_size >= 0.5 else 0.5
        text_width = cv2.getTextSize(labels_dict[label], cv2.FONT_HERSHEY_SIMPLEX, font_size, 4)[0]
        rectangle_mid = int((x1 - x) / 2)
        cv2.putText(frame, labels_dict[label], (((rectangle_mid + x) - (text_width[0] // 2)), y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 4)
        cv2.putText(frame, labels_dict[label], (((rectangle_mid + x) - (text_width[0] // 2)), y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)



cv2.imshow("Frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()