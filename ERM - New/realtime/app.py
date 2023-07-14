import cv2
from flask import Flask, render_template, Response, json, request, jsonify
from flask_sock import Sock
import numpy as np
import time
from tensorflow.python.keras.utils import multi_gpu_utils
from tensorflow.keras.models import load_model

app = Flask(__name__)
sock = Sock(app)

LABELS = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
mutex = False
camera_port = 0
camera = cv2.VideoCapture(camera_port)

# Loading the emotion recognition model
model = load_model('../models/ENetB0_6Class_ValAcc5780.h5')

# Loading face detection model (OpenCV DNN)
model_file = "../models/res10_300x300_ssd_iter_140000.caffemodel"
config_file = "../models/deploy.prototxt.txt"

net = cv2.dnn.readNetFromCaffe(config_file, model_file)
predictions_buffer = []


def predict_emotions(faces, frame):
    # List for keeping track of each faces' size, emotion labels sorted, probabilities, rectangle coordinates,
    # and best label
    predictions = []

    h, w = frame.shape[:2]
    for i in range(faces.shape[2]):
        face_prediction = dict()
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            # Getting sub image of face
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            sub_face_img = frame[y:y1, x:x1]

            try:
                # Reshaping sub image to pass into OpenCV DNN model
                face_size = abs((x1 - x) * (y1 - y))
                resized = cv2.resize(sub_face_img, (224, 224))
                reshaped = np.reshape(resized, (1, 224, 224, 3))

                # Passing into model to predict emotion
                start = time.time()
                result = model.predict(reshaped / 255)

                end = time.time()
                # print(f"inference time: {round(end - start, 2)} seconds")
                label = np.argmax(result, axis=1)[0]
                probabilities = np.flip(np.sort(result[0])).tolist()

                sorted_result = np.argsort(-result[0])
                # top_3_ind = np.argsort(-result[0])[:3]
                emotions_sorted = [LABELS[x] for x in sorted_result]

                frame.flags.writeable = True

                face_prediction["object_detected"] = "face"
                face_prediction["face_size"] = face_size
                face_prediction["emotions_sorted"] = emotions_sorted
                face_prediction["probabilities"] = probabilities
                face_prediction["box_coords"] = (x, y, x1, y1)
                face_prediction["label"] = label
                predictions.append(face_prediction)
                # predictions.append(
                #     tuple((face_size, emotions_sorted, probabilities, (x, y, x1, y1), label)))
            except Exception as e:
                print(str(e))
    return predictions


def draw_frame_graphics(frame, predictions):
    process_frame = frame
    if len(predictions) > 0:
        average_sizes = [x["face_size"] for x in predictions]
        max_index = average_sizes.index(max(average_sizes))

        for i in range(len(predictions)):
            face = predictions[i]
            x, y, x1, y1 = face["box_coords"]
            label = face["label"]
            cv2.rectangle(process_frame, (x, y), (x1, y1), (0, 0, 255), 2)

            if i != max_index:
                scaling_font_size = ((x1 - x) * 0.0047)
                font_size = scaling_font_size if scaling_font_size >= 0.5 else 0.5
                text_width = cv2.getTextSize(LABELS[label], cv2.FONT_HERSHEY_SIMPLEX, font_size, 4)[0]
                rectangle_mid = int((x1 - x) / 2)
                cv2.putText(process_frame, LABELS[label], (((rectangle_mid + x) - (text_width[0] // 2)), y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 4)
                cv2.putText(process_frame, LABELS[label], (((rectangle_mid + x) - (text_width[0] // 2)), y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
            else:
                offset = 10
                scaling_font_size = ((x1 - x) * 0.0047)
                font_size = scaling_font_size if scaling_font_size >= 0.35 else 0.35
                rectangle_mid = int((x1 - x) / 2)

                for j in range(len(face["emotions_sorted"])):
                    emotion_prob_string = ""
                    emotion_prob_string += str(face["emotions_sorted"][j]) + ": "
                    emotion_prob_string += str(round(face["probabilities"][j], 2))
                    text_width = cv2.getTextSize(emotion_prob_string, cv2.FONT_HERSHEY_SIMPLEX, font_size, 4)[0]
                    cv2.putText(process_frame, emotion_prob_string,
                                (((rectangle_mid + x) - (text_width[0] // 2)), y - offset),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 4)
                    cv2.putText(process_frame, emotion_prob_string,
                                (((rectangle_mid + x) - (text_width[0] // 2)), y - offset),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
                    offset += 30
                    font_size -= 0.06
        return process_frame


def detect_faces():
    # Setting camera capture device
    # camera = cv2.VideoCapture(0)
    global predictions_buffer

    while True:
        # Reading camera feed in by frame
        ret, frame = read_camera()
        frame = cv2.flip(frame, 1)

        # Passing frame into face detection model
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     1.0, (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces = net.forward()

        frame.flags.writeable = False

        predictions = predict_emotions(faces, frame)
        predictions_buffer = predictions

        process_frame = draw_frame_graphics(frame, predictions)

        if process_frame is not None:
            frame = process_frame

        frame = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

        imgencode = cv2.imencode('.jpg', frame)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')

        k = cv2.waitKey(1)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
    del camera


def read_camera():
    global mutex
    while mutex:
        pass
    mutex = True
    items = camera.read()
    mutex = False
    return items


def get_frame():
    while True:
        retval, im = read_camera()
        imgencode = cv2.imencode('.jpg', im)[1]
        stringData = imgencode.tostring()
        # return_JSON()
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')
    del (camera)


@app.route('/')
def index():
    return render_template('test.html')


@app.route('/vid')
def vid():
    return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotion')
def emotion():
    return Response(detect_faces(),mimetype='multipart/x-mixed-replace; boundary=frame')


@sock.route('/echo')
def echo(sock):
    global predictions_buffer
    while True:
        data = sock.receive()
        sock.send(predictions_buffer)


