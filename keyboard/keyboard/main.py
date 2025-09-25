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
import threading
import time
import datetime
import sys
import os

from HandTrackingModule import handDetector
from ButtonClass import ButtonKeyboard, keys_en, keys_ru, currentLanguage

# =============================================
# КОНФИГУРАЦИЯ ЛОГГИРОВАНИЯ
# =============================================

class ColorLogger:
    """Класс для цветного логирования с временными метками"""
    
    # Цвета для разных уровней логов
    COLORS = {
        'INFO': '\033[94m',     # Синий
        'SUCCESS': '\033[92m',  # Зеленый
        'WARNING': '\033[93m',  # Желтый
        'ERROR': '\033[91m',    # Красный
        'LOG': '\033[95m',      # Фиолетовый
        'RESET': '\033[0m'      # Сброс цвета
    }
    
    @staticmethod
    def _get_timestamp():
        """Возвращает текущее время в формате HH:MM:SS"""
        return datetime.datetime.now().strftime("%H:%M:%S")
    
    @staticmethod
    def info(message):
        print(f"{ColorLogger.COLORS['INFO']}[INFO] {ColorLogger._get_timestamp()} {message}{ColorLogger.COLORS['RESET']}")
    
    @staticmethod
    def success(message):
        print(f"{ColorLogger.COLORS['SUCCESS']}[SUCCESS] {ColorLogger._get_timestamp()} {message}{ColorLogger.COLORS['RESET']}")
    
    @staticmethod
    def warning(message):
        print(f"{ColorLogger.COLORS['WARNING']}[WARNING] {ColorLogger._get_timestamp()} {message}{ColorLogger.COLORS['RESET']}")
    
    @staticmethod
    def error(message):
        print(f"{ColorLogger.COLORS['ERROR']}[ERROR] {ColorLogger._get_timestamp()} {message}{ColorLogger.COLORS['RESET']}")
    
    @staticmethod
    def log(message):
        print(f"{ColorLogger.COLORS['LOG']}[LOG] {ColorLogger._get_timestamp()} {message}{ColorLogger.COLORS['RESET']}")
    
    @staticmethod
    def print_banner():
        """Печать красивого баннера при запуске"""
        banner = f"""
{ColorLogger.COLORS['SUCCESS']}
╔══════════════════════════════════════════════════════════════╗
║                   VIRTUAL KEYBOARD v1.0                      ║
║                 Hand Gesture Recognition                     ║
╚══════════════════════════════════════════════════════════════╗
{ColorLogger.COLORS['RESET']}"""
        print(banner)

# =============================================
# ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ
# =============================================

app = Flask(__name__)

# Глобальные переменные
camera_lock = threading.Lock()
camera = None
finalText = ""
keyboardVisible = True
keyboard = Controller()

# =============================================
# ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ
# =============================================

def initialize_components():
    """Инициализация всех компонентов системы"""
    ColorLogger.info("Инициализация компонентов...")
    
    # Инициализация детектора рук
    global detector, executor, buttonList
    detector = handDetector(detectionCon=0.8)
    executor = ThreadPoolExecutor(max_workers=4)
    
    # Инициализация кнопок клавиатуры
    buttonList = []
    for i in range(len(keys_en)):
        for j, key in enumerate(keys_en[i]):
            buttonList.append(ButtonKeyboard([100 * j + 50, 100 * i + 50], key))

    # Загрузка изображения для кнопки свертывания
    try:
        collapse_img = cv2.imread('img/Collapse.png')
        if collapse_img is None:
            ColorLogger.warning("Не удалось загрузить Collapse.png, создаем черный квадрат")
            collapse_img = np.zeros((90, 90, 3), dtype=np.uint8)
        else:
            ColorLogger.success("Изображение Collapse.png успешно загружено")
    except Exception as e:
        ColorLogger.error(f"Ошибка загрузки изображения: {e}")
        collapse_img = np.zeros((90, 90, 3), dtype=np.uint8)

    buttonList.append(ButtonKeyboard([1100, 50], "Collapse/Expand", size=[90, 90], img=collapse_img))
    ColorLogger.success("Клавиатура инициализирована")

# =============================================
# УПРАВЛЕНИЕ КАМЕРОЙ
# =============================================

def get_camera():
    """Получение объекта камеры с обработкой ошибок"""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            try:
                ColorLogger.info("Попытка открыть камеру...")
                camera = cv2.VideoCapture(0)
                
                if not camera.isOpened():
                    ColorLogger.warning("Попытка открыть камеру через V4L2...")
                    camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
                
                if not camera.isOpened():
                    ColorLogger.warning("Попытка открыть камеру через ANY...")
                    camera = cv2.VideoCapture(0, cv2.CAP_ANY)
                
                if camera.isOpened():
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    ColorLogger.success("Камера успешно инициализирована")
                else:
                    ColorLogger.error("Не удалось открыть камеру")
                    return None
                    
            except Exception as e:
                ColorLogger.error(f"Ошибка при открытии камеры: {e}")
                return None
        return camera

def release_camera():
    """Освобождение ресурсов камеры"""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            ColorLogger.info("Камера освобождена")

# =============================================
# ФУНКЦИОНАЛ КЛАВИАТУРЫ
# =============================================

def update_keys():
    """Обновление раскладки клавиатуры"""
    global buttonList
    buttonList = []
    keys = keys_en if currentLanguage == "EN" else keys_ru

    for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            buttonList.append(ButtonKeyboard([100 * j + 50, 100 * i + 50], key))

    buttonList.append(ButtonKeyboard([1100, 50], "Collapse/Expand", size=[90, 90], img=collapse_img))
    
    language_button_text = "RU" if currentLanguage == "EN" else "EN"
    buttonList.append(ButtonKeyboard([1100, 150], language_button_text))
    
    ColorLogger.log(f"Раскладка клавиатуры обновлена: {currentLanguage}")

def drawALL(img, buttonList, keyboardVisible):
    """Отрисовка всей клавиатуры на изображении"""
    if img is None:
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        
    imgNew = np.zeros_like(img, np.uint8)
    
    for button in buttonList:
        x, y = button.pos
        w, h = button.size

        if button.text == "Collapse/Expand" and button.img is not None:
            cvzone.cornerRect(imgNew, (x, y, w, h), 20, rt=0)
            resized_img = cv2.resize(button.img, (w, h))
            imgNew[y:y + h, x:x + w] = resized_img
        elif keyboardVisible or button.text == "Collapse/Expand":
            cvzone.cornerRect(imgNew, (x, y, w, h), 20, rt=0)
            cv2.rectangle(imgNew, (x, y), (x + w, y + h), (0, 0, 255), cv2.FILLED)
            text_size = cv2.getTextSize(button.text, cv2.FONT_HERSHEY_PLAIN, 4, 4)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            cv2.putText(imgNew, button.text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

def press_btn_threading(btn):
    """Обработка нажатия кнопки в отдельном потоке"""
    global finalText
    try:
        if btn.text == "<":
            finalText = finalText[:-1]
            keyboard.press('\010')
            ColorLogger.log(f"Удален символ. Текст: {finalText}")
        elif btn.text == 'AC':
            finalText = ""
            ColorLogger.log("Текст очищен")
        else:
            finalText += btn.text
            keyboard.press(btn.text)
            ColorLogger.log(f"Добавлен символ: '{btn.text}'. Текст: {finalText}")
        sleep(0.15)
    except Exception as e:
        ColorLogger.error(f"Ошибка при нажатии кнопки: {e}")

# =============================================
# ОБРАБОТКА ВИДЕОПОТОКА
# =============================================

def get_frame():
    """Получение и обработка кадра с камеры"""
    global keyboardVisible, currentLanguage
    
    cam = get_camera()
    if cam is None:
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(img, "КАМЕРА НЕДОСТУПНА", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        with camera_lock:
            success, img = cam.read()
        
        if not success:
            ColorLogger.warning("Ошибка захвата кадра, попытка переподключения...")
            release_camera()
            cam = get_camera()
            if cam:
                with camera_lock:
                    success, img = cam.read()
            
            if not success:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(img, "ОШИБКА ЗАХВАТА КАДРА", (350, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if success and img is not None:
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bboxInfo = detector.findPosition(img, draw=False)  

        img = drawALL(img, buttonList, keyboardVisible)

        if lmList and len(lmList) >= 21:  
            for btn in buttonList:
                x, y = btn.pos
                w, h = btn.size

                if btn.text == "Collapse/Expand" or keyboardVisible:
                    if x < lmList[8][1] < x + w and y < lmList[8][2] < y + h:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (175, 0, 175), cv2.FILLED)
                        text_size = cv2.getTextSize(btn.text, cv2.FONT_HERSHEY_PLAIN, 4, 4)[0]
                        text_x = x + (w - text_size[0]) // 2
                        text_y = y + (h + text_size[1]) // 2
                        cv2.putText(img, btn.text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        
                        l, _, _ = detector.findDistance(8, 12, img, draw=False)

                        if l < 40:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, btn.text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                            if btn.text in ["EN", "RU"]:
                                currentLanguage = "RU" if currentLanguage == "EN" else "EN"
                                update_keys()
                                ColorLogger.log(f"Язык переключен на {currentLanguage}")
                                sleep(0.5)  

                            elif btn.text == "Collapse/Expand":
                                keyboardVisible = not keyboardVisible
                                ColorLogger.log(f"Клавиатура {'скрыта' if not keyboardVisible else 'показана'}")
                                sleep(0.5)
                            else:
                                executor.submit(press_btn_threading, btn)
                                sleep(0.3)

        cv2.rectangle(img, (50, 610), (900, 710), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, finalText, (60, 680), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)
    
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG", quality=80)
        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        return img_str
    except Exception as e:
        ColorLogger.error(f"Ошибка конвертации изображения: {e}")
        empty_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        pil_img = Image.fromarray(empty_img)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")

# =============================================
# ВЕБ-ИНТЕРФЕЙС (FLASK ROUTES)
# =============================================

@app.route('/')
def index():
    """Главная страница"""
    ColorLogger.log("GET / - Главная страница")
    return render_template('index.html')

@app.route('/image')
def image():
    """API для получения изображения"""
    try:
        start_time = time.time()
        img_data = get_frame()
        processing_time = (time.time() - start_time) * 1000
        ColorLogger.log(f"GET /image - Обработано за {processing_time:.1f}ms")
        return jsonify({'img_data': img_data})
    except Exception as e:
        ColorLogger.error(f"Ошибка в маршруте /image: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health():
    """Проверка статуса системы"""
    ColorLogger.log("GET /health - Проверка здоровья системы")
    cam = get_camera()
    status = "healthy" if cam and cam.isOpened() else "unhealthy"
    return jsonify({
        'status': status, 
        'camera_available': cam is not None and cam.isOpened(),
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/status')
def status():
    """Полный статус системы"""
    ColorLogger.log("GET /status - Полный статус системы")
    cam = get_camera()
    return jsonify({
        'camera_available': cam is not None and cam.isOpened(),
        'keyboard_visible': keyboardVisible,
        'current_language': currentLanguage,
        'text_length': len(finalText),
        'timestamp': datetime.datetime.now().isoformat()
    })

# =============================================
# ЗАВЕРШЕНИЕ РАБОТЫ
# =============================================

def cleanup():
    """Очистка ресурсов при завершении работы"""
    ColorLogger.info("Завершение работы приложения...")
    release_camera()
    executor.shutdown(wait=False)
    ColorLogger.success("Ресурсы освобождены")

import atexit
atexit.register(cleanup)

# =============================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# =============================================

if __name__ == '__main__':
    # Печать баннера
    ColorLogger.print_banner()
    
    ColorLogger.info("Запуск приложения Virtual Keyboard...")
    
    # Инициализация компонентов
    initialize_components()
    
    # Проверка камеры
    ColorLogger.info("Проверка камеры...")
    cam = get_camera()
    if cam and cam.isOpened():
        ColorLogger.success("✅ Камера готова к работе")
    else:
        ColorLogger.warning("⚠️ Камера недоступна, приложение запустится в демо-режиме")
    
    # Запуск веб-сервера
    ColorLogger.info("Запуск веб-сервера Flask...")
    ColorLogger.success("🌐 Сервер запущен на http://0.0.0.0:7654")
    ColorLogger.info("📊 Доступные эндпоинты:")
    ColorLogger.info("   • http://localhost:7654/ - Главная страница")
    ColorLogger.info("   • http://localhost:7654/health - Проверка здоровья")
    ColorLogger.info("   • http://localhost:7654/status - Статус системы")
    ColorLogger.info("   • http://localhost:7654/image - Поток видео")
    
    ColorLogger.info("Для остановки сервера нажмите Ctrl+C")
    print("")  # Пустая строка для разделения
    
    try:
        app.run(host='0.0.0.0', port=7654, debug=False, threaded=True)
    except KeyboardInterrupt:
        ColorLogger.info("Получен сигнал прерывания...")
    except Exception as e:
        ColorLogger.error(f"Ошибка при запуске сервера: {e}")
    finally:
        cleanup()