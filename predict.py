from ultralytics import YOLO
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize

def create_bezier_curve(mask, n_points=100):
    # Преобразуем изображение в градации серого
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Бинаризуем изображение
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Находим скелет маски
    skeleton = skeletonize(binary // 255)

    # Находим координаты пикселей скелета
    y, x = np.where(skeleton)

    # Преобразуем координаты в формат, подходящий для splprep
    cnt = np.column_stack((x, y)).squeeze().transpose()

    # Создаем кривую Безье
    tck, u = splprep(cnt, u=None, s=1.0, per=0)
    u_new = np.linspace(u.min(), u.max(), n_points)
    x_new, y_new = splev(u_new, tck, der=0)

    # Возвращаем новые точки
    return np.column_stack((x_new, y_new)).astype(int)



def displayLineBiz(road_mask):
    # Создаем кривую Безье
    curve_points = create_bezier_curve(road_mask)

    print(curve_points)

    # Создаем изображение для кривой
    curve_image = np.zeros_like(road_mask)

    # Рисуем кривую на изображении
    cv2.polylines(curve_image, [curve_points], False, (255, 255, 255), thickness=2)

    # Отображение исходного изображения и кривой
    cv2.imshow('Original Image', road_mask)
    cv2.imshow('Bezier Curve', curve_image)

    # Ожидание нажатия клавиши для закрытия окон
    cv2.waitKey(0)
    cv2.destroyAllWindows()


model_path = 'runs/segment/train13/weights/best.pt'
model = YOLO(model_path)

image_path = 'dataset/test/images/ebe9a0ed-photo_2024-04-07_01.04.03.jpeg'
image = cv2.imread(image_path)

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for result in results:
    if result.masks is not None:
        for j, mask in enumerate(result.masks.data):
            mask = mask.cpu().numpy() * 255
            mask = cv2.resize(mask, (W, H)).astype(np.uint8)

            mask_bgr = cv2.merge([mask, mask, mask])

            displayLineBiz(mask_bgr)

            overlay = cv2.addWeighted(image.astype(np.uint8), 0.5, mask_bgr.astype(np.uint8), 0.5, 0)

            cv2.imwrite('./output/output.png', overlay)
    else:
        print("No objects detected")
