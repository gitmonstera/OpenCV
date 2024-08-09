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


def angle_between_rects(rect1, rect2):
    """ Функция для вычисления угла между двумя прямоугольниками """
    angle1 = rect1[2]
    angle2 = rect2[2]

    # Нормализуем углы в диапазоне [0, 180]
    angle1 = angle1 % 180
    angle2 = angle2 % 180

    # Вычисляем разницу между углами
    angle_diff = abs(angle1 - angle2)
    if angle_diff > 90:
        angle_diff = 180 - angle_diff

    return angle_diff

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

        # Определение расширенного диапазона для белого цвета
        low_yellow = np.array([0, 0, 200])
        high_yellow = np.array([35, 255, 255])

        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])

        # Применение маски для выделения заданного цветового диапазона
        img_tmp = cv2.inRange(hsv_img, low_yellow, high_yellow)

        img_tmp_car = cv2.inRange(hsv_img, lower_blue, upper_blue)

        # Поиск контуров белых объектов
        contours, _ = cv2.findContours(img_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours1, _ = cv2.findContours(img_tmp_car, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []
        car_rectangles = []

        # Рисование прямоугольников вокруг найденных объектов
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 800:  # Фильтр по площади
                rect = cv2.minAreaRect(contour)
                rectangles.append(rect)

        for contour in contours1:
            area = cv2.contourArea(contour)
            if area > 800:  # Фильтр по площади
                rect = cv2.minAreaRect(contour)
                car_rectangles.append(rect)
                box = cv2.boxPoints(rect).astype(int)
                cv2.drawContours(img, [box], 0, (0, 0, 255), 2)


        for i, rect1 in enumerate(rectangles):
            for j, rect2 in enumerate(rectangles):
                if i != j:
                    angle_diff = angle_between_rects(rect1, rect2)
                    if angle_diff < 5:
                        box1 = cv2.boxPoints(rect1).astype(int)
                        box2 = cv2.boxPoints(rect2).astype(int)

                        # Проверка пересичения синего прямоугольника с остальными
                        for car_rect in car_rectangles:
                            car_box = cv2.boxPoints(car_rect).astype(int)
                            intersection_type, intersection_point = cv2.rotatedRectangleIntersection(rect1, car_rect)
                            if intersection_type != cv2.INTERSECT_NONE:
                                print("Синий прямоугольник пересекается с одним из параллельных зеленых")
                                cv2.putText(img, "YES", (car_box[0][0], car_box[0][1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        # print("forward")
                        x1, y1, width1, height1 = cv2.boundingRect(box1)
                        x2, y2, width2, height2 = cv2.boundingRect(box2)
                        if (width1 and width2) >= 5 and (height1 and height2) >= 50:

                            print(f"Ширина: {width1}, Высота: {height1}")
                            cv2.drawContours(img, [box1], 0, (0, 255, 0), 2)  # Зеленый прямоугольник
                            cv2.putText(img, f'W:{width1} H:{height1}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            print(f"Ширина: {width2}, Высота: {height2}")
                            cv2.drawContours(img, [box2], 0, (0, 0, 222), 2)  # Красный прямоугольник
                            cv2.putText(img, f'W:{width2} H:{height2}', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 222), 2)
                    else:
                        print("stop")

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
