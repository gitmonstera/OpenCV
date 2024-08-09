from flask import Flask, render_template, jsonify
import cv2
import numpy as np
import mss
import base64
from io import BytesIO
from PIL import Image

import keyboard
from time import sleep

app = Flask(__name__)

def capture_and_process_screen():
    # Создаем объект для захвата экрана
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)

        # Конвертируем изображение в формат, поддерживаемый OpenCV
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Конвертация изображения из BGR в HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Определение расширенного диапазона для желтого цвета
        low_yellow = np.array([0, 0, 200])
        high_yellow = np.array([35, 255, 255])

        # Применение маски для выделения заданного цветового диапазона
        img_tmp = cv2.inRange(hsv_img, low_yellow, high_yellow)

        # Поиск контуров желтых объектов
        contours, _ = cv2.findContours(img_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Рисование прямоугольников вокруг найденных объектов
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 800:  # Фильтр по площади
                # keyboard.press("w")
                # sleep(1)
                # keyboard.release("w")
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Зеленый прямоугольник

        # Конвертируем изображение в формат, подходящий для передачи на веб-страницу
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")


        return img_str

@app.route('/')
def index():
    img_data = capture_and_process_screen()
    return render_template('index.html', img_data=img_data)


@app.route('/image')
def image():
    img_data = capture_and_process_screen()
    return jsonify({'img_data': img_data})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7654, debug=True)
