import cv2
import mediapipe as mp

from flask import Flask, render_template, jsonify
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)

camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

points = [0 for i in range(21)]

def get_frame():
    good, img = camera.read()
    if not good:
        return None

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            for id, point in enumerate(handLms.landmark):
                w, h, color = img.shape
                w, h = int(point.x * h), int(point.y * w)
                points[id] = h
                if id == 8:
                    cv2.circle(img, (w, h), 20, (10, 255, 10), cv2.FILLED)
                    print(f'Палец поднят и его координаты => {w}')
                    cv2.putText(img, f'koordinate => {w}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 222), 2)
            distance_0_5 = abs(points[0] - points[5])
            distance_0_8 = abs(points[0] - points[8])
            distanceGood = distance_0_5 + (distance_0_5/2)
            if distance_0_8 < distanceGood:
                cv2.putText(img, 'finger opushen', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 222), 2)
                print("палец опущен")

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image')
def image():
    img_data = get_frame()
    if img_data:
        return jsonify({'img_data': img_data})
    else:
        return jsonify({'error': 'Could not capture image'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7654, debug=True)
