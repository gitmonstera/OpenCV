import cv2
import sys

print("=== ТЕСТ ДОСТУПНОСТИ КАМЕРЫ ===")

# Проверим доступные камеры
for i in range(10):
    try:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Камера {i} ДОСТУПНА - разрешение: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print(f"⚠️ Камера {i} подключена, но не может захватить изображение")
            cap.release()
        else:
            print(f"❌ Камера {i} недоступна")
    except Exception as e:
        print(f"❌ Ошибка при проверке камеры {i}: {e}")

print("\n=== ПРОВЕРКА OpenCV ===")
print(f"Версия OpenCV: {cv2.__version__}")

# Проверим бэкенды
backends = [cv2.CAP_ANY, cv2.CAP_V4L2, cv2.CAP_FFMPEG, cv2.CAP_DSHOW]
backend_names = ['ANY', 'V4L2', 'FFMPEG', 'DSHOW']

for backend, name in zip(backends, backend_names):
    try:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            print(f"✅ Бэкенд {name} работает")
            cap.release()
        else:
            print(f"❌ Бэкенд {name} не работает")
    except:
        print(f"❌ Ошибка бэкенда {name}")