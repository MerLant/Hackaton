from ultralytics import YOLO
import cv2
import torch
import numpy as np

model_path = 'runs/segment/train13/weights/best.pt'
model = YOLO(model_path)

image_path = 'dataset/test/images/fdbfb637-photo_2024-04-06_23.41.12.jpeg'
image = cv2.imread(image_path)

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for result in results:
    if result.masks is not None:  # Проверка наличия объектов
        for j, mask in enumerate(result.masks.data):
            mask = mask.cpu().numpy() * 255  # Перемещение тензора на CPU и конвертация в массив NumPy
            mask = cv2.resize(mask, (W, H)).astype(np.uint8)  # Преобразование в тип данных uint8

            # Создание маски в формате BGR
            mask_bgr = cv2.merge([mask, mask, mask])

            # Наложение маски на оригинальное изображение
            overlay = cv2.addWeighted(image.astype(np.uint8), 0.5, mask_bgr.astype(np.uint8), 0.5, 0)

            cv2.imwrite('./output.png', overlay)
    else:
        print("No objects detected")
