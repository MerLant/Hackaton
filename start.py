import cv2
from ultralytics import YOLO

# Функция для разбиения изображения на полосы
def split_image_into_strips(image_path, strip_height=100):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    strips = []
    for y in range(0, height, strip_height):
        end_y = y + strip_height if (y + strip_height) < height else height
        strip = image[y:end_y, :]
        strips.append(strip)
    return strips

# Функция для обработки полос и объединения их в одно изображение
def process_strips_and_combine(strips, model):
    # Создаем пустой список для хранения обработанных полос
    processed_strips = []

    for strip in strips:
        # Применяем модель YOLO к каждой полосе
        results = model(strip)
        result = results[0]
        for box in result.boxes:
            box_coords = box.xyxy.tolist()
            x1, y1, x2, y2 = box_coords[0]
            class_id = box.cls.tolist()[0]
            class_name = result.names[int(class_id)]
            if class_name == "Road":
                cv2.rectangle(strip, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(strip, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        processed_strips.append(strip)

    # Объединяем все обработанные полосы в одно изображение
    combined_image = cv2.vconcat(processed_strips)
    return combined_image

# Основной код
model_path = 'runs/segment/train13/weights/best.pt'
model = YOLO(model_path)
image_path = 'dataset/test/images/e7b8691a-Property_1Forest.jpg'
output_path = "output/combined_image.jpg"

strips = split_image_into_strips(image_path, strip_height=1000)
combined_image = process_strips_and_combine(strips, model)
cv2.imwrite(output_path, combined_image)
