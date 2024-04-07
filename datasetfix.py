import os

def clamp(value, min_value=0.0, max_value=1.0):
    """Ограничивает значение заданным диапазоном."""
    return max(min_value, min(max_value, value))

def fix_annotations(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Рассматриваем только файлы аннотаций
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r+') as file:
                lines = file.readlines()
                file.seek(0)  # Перемещаем указатель в начало файла
                file.truncate()  # Очищаем файл для записи исправленных данных

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:  # Убедимся, что строка соответствует ожидаемому формату
                        # Исправляем координаты, если они выходят за пределы [0, 1]
                        parts[1] = str(clamp(float(parts[1])))
                        parts[2] = str(clamp(float(parts[2])))
                        parts[3] = str(clamp(float(parts[3])))
                        parts[4] = str(clamp(float(parts[4])))

                    # Записываем исправленные данные обратно в файл
                    file.write(' '.join(parts) + '\n')

# Укажите путь к директории с файлами аннотаций
annotations_directory = 'F:/Python/Hack/dataset/labels'
fix_annotations(annotations_directory)

print("Исправление аннотаций завершено.")
