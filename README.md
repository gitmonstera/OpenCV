# OpenCV

## 🛠 Технологии проекта

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0F9D58?style=for-the-badge&logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![CVZone](https://img.shields.io/badge/CVZone-orange?style=for-the-badge)

## Для старта 
Проект раюотает на системах:

![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-000000?style=for-the-badge&logo=apple&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)

### Инструкция для запуска в VSCode

1. Необходимо создать окружение 
```bash
    python3 -m venv venv
    source venv/bin/activate
```

2. Необходимо установить все зависимости 
```bash
    pip install opencv-python-headless mediapipe numpy cvzone pynput flask pillow
```

3. Запустить проект 

    3.1 Подготовка запуска клавиатуры
    
    Надо проверить и узнать работает ли камера а так же узнать ее индекс, для этого есть `test_camera.py`
    ```bash
        python test_camera.py
    ```

    после этого если камера не по дефолтному индексу (то есть не по индексу 0) необходимо подставить выш индекс камеры во все аналогичные строки кода `cv2.VideoCapture(index)`

    3.2 Запуск клавиатуры

    ```bash
        python main.py
    ```
    вся информация для использования будет в терминале 
    
    для перехода к клавиатуре перейдите по сыылке 
    * Running on http://127.0.0.1:7654
    (ссылка есть в логах программы)

--- 

## Разработчик: Карташов Антон Алексеевич

### Контакты

| Платформа | Ссылка | Описание |
|-----------|--------|----------|
| <img src="https://img.icons8.com/ios-filled/24/ffffff/github.png" width="20"> **GitHub** | [gitmonstera](https://github.com/gitmonstera) | Мои проекты и код |
| <img src="https://img.icons8.com/ios-filled/24/ffffff/telegram.png" width="20"> **Telegram** | [telegram](https://t.me/ant0ndevel0per) | Для связи |

