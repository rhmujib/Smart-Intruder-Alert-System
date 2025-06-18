import cv2
import time
import datetime
import os
import pygame
import threading
import json
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from flask_socketio import SocketIO
from skimage.metrics import structural_similarity as ssim

from src.detector import detect_intruder
from src.gui_overlay import draw_overlays

# Paths
PROTO_PATH = "models/mobilenetssd/deploy.prototxt"
MODEL_PATH = "models/mobilenetssd/mobilenet_iter_73000.caffemodel"
ALARM_PATH = "alarm/alarm.mp3"
AUTHORIZED_FACES_DIR = "authorized_faces"
HISTORY_LOG_PATH = "history_log.json"

# Paths for Dlib models
DLIB_SHAPE_PREDICTOR_PATH = "models/dlib/shape_predictor_68_face_landmarks.dat"
DLIB_FACE_RECOGNITION_MODEL_PATH = "models/dlib/dlib_face_recognition_resnet_model_v1.dat"

# Initialize Dlib models
import dlib
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(DLIB_SHAPE_PREDICTOR_PATH)
face_recognition_model = dlib.face_recognition_model_v1(DLIB_FACE_RECOGNITION_MODEL_PATH)

# Create necessary folders and files
if not os.path.exists("recordings"):
    os.makedirs("recordings")
if not os.path.exists(HISTORY_LOG_PATH):
    with open(HISTORY_LOG_PATH, "w") as f:
        json.dump([], f)

# Load authorized faces
authorized_faces = []
authorized_names = []

for file in os.listdir(AUTHORIZED_FACES_DIR):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(AUTHORIZED_FACES_DIR, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (200, 200))
        authorized_faces.append(img)
        authorized_names.append(os.path.splitext(file)[0])

# Load and encode authorized faces
authorized_encodings = []
for file in os.listdir(AUTHORIZED_FACES_DIR):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(AUTHORIZED_FACES_DIR, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_img)
        if len(faces) == 1:  # Ensure only one face per image
            shape = shape_predictor(gray_img, faces[0])
            encoding = np.array(face_recognition_model.compute_face_descriptor(img, shape))
            authorized_encodings.append((encoding, os.path.splitext(file)[0]))

# Initialize pygame mixer for alarm
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(ALARM_PATH)
alarm_playing = False

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Globals
detection_timer_start = None
recording = False
out = None
fps_time = time.time()
frame_count = 0
fps = 0
detection_active = False
current_intruder_count = 0  # Current number of intruders in the frame
authorized_count = 0  # Count of authorized persons detected
total_detections = 0  # Total detections (authorized + intruders)
last_detection_time = None  # To track the last detection time

# Flask app and SocketIO setup
app = Flask(__name__)
socketio = SocketIO(app)

# Load the DNN model with CPU backend
net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Use OpenCV backend
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)       # Use CPU for inference

# Function to recognize faces
def recognize_faces(frame):
    global authorized_count
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_frame)
    recognized_names = []
    for face in faces:
        shape = shape_predictor(gray_frame, face)
        encoding = np.array(face_recognition_model.compute_face_descriptor(frame, shape))
        matches = []
        for auth_encoding, name in authorized_encodings:
            distance = np.linalg.norm(auth_encoding - encoding)
            if distance < 0.6:  # Threshold for face recognition
                matches.append(name)
        if matches:
            recognized_names.append(matches[0])
    authorized_count = len(recognized_names)
    return recognized_names

def gen_frames():
    global detection_timer_start, recording, out, alarm_playing, fps_time, frame_count, fps, detection_active, current_intruder_count, authorized_count, total_detections, last_detection_time

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        current_time = time.time()

        if current_time - fps_time >= 1:
            fps = frame_count / (current_time - fps_time)
            frame_count = 0
            fps_time = current_time

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if detection_active:
            # Preprocess the frame for the DNN model
            resized_frame = cv2.resize(frame, (300, 300))  # Resize to match model input size
            blob = cv2.dnn.blobFromImage(resized_frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True)
            net.setInput(blob)
            detections = net.forward()  # Perform inference using CPU

            patience_remaining = 0
            if detection_timer_start is not None:
                patience_elapsed = current_time - detection_timer_start
                patience_remaining = max(0, 3 - patience_elapsed)

            frame = draw_overlays(frame, detection_timer_start, datetime.datetime.now(), fps, patience_remaining)

            # Update the current intruder count based on the number of bounding boxes
            current_intruder_count = 0
            boxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:  # Lower confidence threshold for better sensitivity
                    idx = int(detections[0, 0, i, 1])
                    if idx == 15:  # Person class
                        current_intruder_count += 1
                        box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                        boxes.append(box.astype("int"))

            # Recognize faces in the frame
            recognized_names = recognize_faces(frame)

            if current_intruder_count > 0:
                total_detections += current_intruder_count  # Update total detections dynamically

                # Calculate detection accuracy
                detection_accuracy = int((authorized_count / total_detections) * 100 if total_detections > 0 else 0)

                # Emit real-time detection event
                socketio.emit('person_detected', {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'intruders': current_intruder_count,
                    'authorized': authorized_count,
                    'accuracy': detection_accuracy,
                    'recognized_names': recognized_names
                })

                # Save to history log only if a new detection occurs
                current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if last_detection_time != current_time_str:  # Avoid duplicate entries
                    with open(HISTORY_LOG_PATH, "r+") as f:
                        history = json.load(f)
                        history.append({
                            "date": current_time_str,
                            "intruders": current_intruder_count,
                            "authorized": authorized_count,
                            "accuracy": detection_accuracy,
                            "recognized_names": recognized_names
                        })
                        f.seek(0)
                        json.dump(history, f, indent=4)
                    last_detection_time = current_time_str  # Update the last detection time

                # Draw bounding boxes for all detected intruders
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Person: Intruder", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if detection_timer_start is None:
                    detection_timer_start = time.time()
                elif current_time - detection_timer_start > 3:
                    if not alarm_playing:
                        alarm_sound.play(loops=-1)
                        alarm_playing = True
                    if not recording:
                        filename = datetime.datetime.now().strftime("recordings/%Y-%m-%d_%H-%M-%S.avi")
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))
                        recording = True
            else:
                current_intruder_count = 0  # Reset intruder count when no intruders are detected
                detection_timer_start = None
                if alarm_playing:
                    alarm_sound.stop()
                    alarm_playing = False
                if recording:
                    out.release()
                    recording = False
        else:
            frame = draw_overlays(frame, None, datetime.datetime.now(), fps, None)
            if alarm_playing:
                alarm_sound.stop()
                alarm_playing = False
            if recording:
                out.release()
                recording = False

        if recording and out is not None:
            out.write(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def get_history():
    """API endpoint to fetch historical data for the graph."""
    if os.path.exists(HISTORY_LOG_PATH):
        with open(HISTORY_LOG_PATH, "r") as f:
            history = json.load(f)
        return jsonify(history)
    return jsonify([])

@app.route('/reset_history', methods=['POST'])
def reset_history():
    """API endpoint to reset the historical graph data."""
    if os.path.exists(HISTORY_LOG_PATH):
        with open(HISTORY_LOG_PATH, "w") as f:
            json.dump([], f)  # Clear the history log
    return jsonify({"status": "success", "message": "History reset successfully."})

@app.route('/start')
def start_detection():
    global detection_active, current_intruder_count, authorized_count, total_detections, last_detection_time
    detection_active = True
    current_intruder_count = 0  # Reset intruder count for the new session
    authorized_count = 0  # Reset authorized count for the new session
    total_detections = 0  # Reset total detections for the new session
    last_detection_time = None  # Reset last detection time
    return redirect(url_for('index'))

@app.route('/stop')
def stop_detection():
    global detection_active, detection_timer_start, alarm_playing, recording, out
    detection_active = False
    detection_timer_start = None
    if alarm_playing:
        alarm_sound.stop()
        alarm_playing = False
    if recording and out:
        out.release()
        recording = False
    return redirect(url_for('index'))

@app.route('/terminate')
def terminate():
    os._exit(0)

if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False)