import cv2, json
import numpy as np
from tensorflow.keras.models import load_model
from pygame import mixer
import time
import base64
from PIL import Image
import io
from flask import Flask
from flask_socketio import SocketIO
from io import BytesIO
from PIL import Image

# Initializations
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load model and face cascade classifiers
model = load_model('models/cnncat2.h5')
face = cv2.CascadeClassifier(r'haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(r'haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(r'haar cascade files\haarcascade_righteye_2splits.xml')

# global count, score, alarm_played, alarm_start_time, rpred, lpred, thicc
lbl = ['Close', 'Open']
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2

alarm_played = False  # Flag to check if the alarm was played
alarm_start_time = 0  # Time when the alarm started playing

# Flask and SocketIO setup
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return "Deep Learning Server is running!"

global scoreTotal
scoreTotal = 0
@socketio.on('send_image')
def handle_frame(data):
    global scoreTotal
    print("Received image data from client")
    image = None
    if 'image' in data:
        
        # ถอดรหัส Base64 เป็น bytes
        image_data = base64.b64decode(data['image'])
        # ใช้ PIL เพื่อแปลงเป็น Image
        image = Image.open(BytesIO(image_data))
        # บันทึกภาพเป็นไฟล์
        image.save("output.jpg")
    if image == None:
        result = {"message": f"Error: image not found."}
    
    score = loadImage(image)
    if score == 0:
        scoreTotal = 0
    else:
        scoreTotal += score
    
    print(f"scoreTotal === {scoreTotal}")
    result = {"message": f"Frame processed successfully {scoreTotal}", "score": scoreTotal}
    socketio.emit('result', result)

@socketio.on('connect')
def on_connect():
    print("Client connected")

@socketio.on('disconnect')
def on_disconnect():
    print("Client disconnected")

def reSizeImage(image):
    width, height = image.size
    new_width = 500
    new_height = int((new_width / width) * height)

    # ปรับขนาด
    resized_image = image.resize((new_width, new_height))
    return resized_image

def loadImage(image):
    reSizeImage(image)
    global count, score, rpred, lpred, alarm_played, alarm_start_time
    rpred = [99]
    lpred = [99]
    # แปลงภาพเป็น NumPy array
    frame = np.array(image)
    # ret, frame = cap.read()
    height, width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict_classes(r_eye)
        if rpred[0] == 1:
            lbl = 'Open'
        if rpred[0] == 0:
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict_classes(l_eye)
        if lpred[0] == 1:
            lbl = 'Open'   
        if lpred[0] == 0:
            lbl = 'Closed'
        break
    print("rpred ==== ", rpred[0])
    print("lpred ==== ", lpred[0])

    # ตั้งคะแนนเป็น 0 เมื่อดวงตาเปิด
    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    elif rpred[0] == 1 or lpred[0] == 1:  # หากดวงตาเปิด
        score = 0  # ตั้งคะแนนเป็น 0
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # กำหนดเงื่อนไขให้คะแนนไม่ติดลบ
    if score < 0:
        score = 0

    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # เมื่อคะแนนเกิน 15
    if score > 15:
        if not alarm_played:  # ถ้ายังไม่เคยเล่นเสียง
            # หากดวงตายังปิด
            if rpred[0] == 0 and lpred[0] == 0:
                # เล่นเสียงต่อเนื่อง
                try:
                    sound.play()
                    alarm_played = True
                    alarm_start_time = time.time()  # บันทึกเวลาเริ่มเล่นเสียง
                except:
                    pass
            else:  # หากดวงตาเปิดแล้ว
                # เล่นเสียงแค่ 1 วินาที
                try:
                    sound.play()
                    alarm_played = True
                    alarm_start_time = time.time()  # บันทึกเวลาเริ่มเล่นเสียง
                except:
                    pass

        # ตรวจสอบเวลาที่ผ่านไปหลังจากเสียงเริ่มเล่น
        if alarm_played and time.time() - alarm_start_time >= 1:  # หากผ่านไป 1 วินาที
            sound.stop()  # หยุดเสียง
            alarm_played = False  # รีเซ็ตสถานะการเล่นเสียง

        if thicc < 16:
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    return score

if __name__ == "__main__":
    # image = Image.open("output2.jpg")
    # if image == None:
    #     result = {"message": f"Error: image not found."}
    # for i in range(0, 9):
    #     loadImage(image)
    # ใช้ Eventlet เพื่อรองรับ WebSocket
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
