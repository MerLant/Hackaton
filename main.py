from ultralytics import YOLO
import os


def train_and_export_model():
    dataset_yaml_path = './dataset.yaml'

    model = YOLO('yolov8n-seg.pt')

    results = model.train(data=dataset_yaml_path, epochs=50)

    val_results = model.val()

    export_success = model.export(format='onnx')

    model('dataset/test/images/f9177ace-photo_2024-04-06_23.41.14_1.jpeg')

    # Если API не поддерживает прямое указание пути, переместите файл после экспорта


if __name__ == '__main__':
    train_and_export_model()
