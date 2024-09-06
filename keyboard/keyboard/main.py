import cv2
from time import sleep
import numpy as np
import cvzone
from pynput.keyboard import Controller
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, jsonify
from io import BytesIO
from PIL import Image
import base64

from HandTrackingModule import handDetector
from ButtonClass import ButtonKeyboard, keys_en, keys_ru, currentLanguage

app = Flask(__name__)

camera = cv2.VideoCapture(0)
camera.set(3, 1280)
camera.set(4, 720)

executor = ThreadPoolExecutor()
detector = handDetector(detectionCon=0.8)

finalText = ""

keyboardVisible = True
keyboard = Controller()

buttonList = []
for i in range(len(keys_en)):
    for j, key in enumerate(keys_en[i]):
        buttonList.append(ButtonKeyboard([100 * j + 50, 100 * i + 50], key))

collapse_img = cv2.imread('img/Collapse.png')
buttonList.append(ButtonKeyboard([1100, 50], "Collapse/Expand", size=[90, 90], img=collapse_img))


def update_keys():
    global buttonList
    buttonList = []

    # Выбираем нужную раскладку в зависимости от текущего языка
    keys = keys_en if currentLanguage == "EN" else keys_ru

    # Перезаполняем список кнопок для клавиатуры
    for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            buttonList.append(ButtonKeyboard([100 * j + 50, 100 * i + 50], key))

    # Кнопка сворачивания/разворачивания клавиатуры
    buttonList.append(ButtonKeyboard([1100, 50], "Collapse/Expand", size=[90, 90], img=collapse_img))

    # Обновляем кнопку переключения языка
    language_button_text = "RU" if currentLanguage == "EN" else "EN"
    buttonList.append(ButtonKeyboard([1100, 150], language_button_text))


def drawALL(img, buttonList, keyboardVisible):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        w, h = button.size

        # Если есть изображение, рисуем его, а не текст
        if button.text == "Collapse/Expand" and button.img is not None:
            # Рисуем рамку
            cvzone.cornerRect(imgNew, (x, y, w, h), 20, rt=0)
            # Рисуем изображение на кнопке
            imgNew[y:y + h, x:x + w] = cv2.resize(button.img, (w, h))
        elif keyboardVisible or button.text == "Collapse/Expand":  # Рисуем обычные кнопки
            cvzone.cornerRect(imgNew, (x, y, w, h), 20, rt=0)
            cv2.rectangle(imgNew, button.pos, (x + w, y + h), (0, 0, 255), cv2.FILLED)
            cv2.putText(imgNew, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out


def press_btn_threading(btn):
    global finalText
    if btn.text == "<":
        finalText = finalText[:-1]
        keyboard.press('\010')
        # keyboard.release('\010')
    elif btn.text == 'AC':
        finalText = ""
    else:
        finalText += btn.text
        keyboard.press(btn.text)
        # keyboard.release(btn.text)
    sleep(0.15)


def get_frame():
    global keyboardVisible, currentLanguage
    success, img = camera.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bboxInfo = detector.findPosition(img)

    # Рисуем только видимые элементы
    img = drawALL(img, buttonList, keyboardVisible)

    # Если есть обнаруженные координаты руки
    if lmList:
        for btn in buttonList:
            x, y = btn.pos
            w, h = btn.size

            # Проверяем только кнопку "Collapse/Expand" и другие элементы
            if btn.text == "Collapse/Expand" or keyboardVisible:
                # Проверяем, находится ли палец над кнопкой
                if x < lmList[8][1] < x + w and y < lmList[8][2] < y + h:
                    cv2.rectangle(img, btn.pos, (x + w, y + h), (175, 0, 175), cv2.FILLED)
                    cv2.putText(img, btn.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    l, _, _ = detector.findDistance(8, 12, img, draw=False)

                    # Если расстояние между пальцами маленькое, происходит нажатие
                    if l < 40:
                        cv2.rectangle(img, btn.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, btn.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                        # Переключение раскладки клавиатуры
                        if btn.text == "EN" or btn.text == "RU":
                            currentLanguage = "RU" if currentLanguage == "EN" else "EN"
                            update_keys()  # Перерисовка кнопок
                            print(f"Language switched to {currentLanguage}")

                        elif btn.text == "Collapse/Expand":
                            keyboardVisible = not keyboardVisible  # Переключаем видимость клавиатуры
                        else:
                            executor.submit(press_btn_threading, btn)

    # Отображаем текст
    cv2.rectangle(img, (50, 710), (900, 610), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, finalText, (60, 690), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    # Преобразуем изображение для отправки на страницу
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

