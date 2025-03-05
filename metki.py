import os

# Список директорий с аудиофайлами
directories = [
    'path/to/class1',
    'path/to/class2',
    'path/to/class3',
]

# Список меток
labels = []

# Присвоение меток вручную
for i, directory in enumerate(directories):
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            # Прослушайте файл и присвойте метку
            print(f"Прослушайте файл {filename} и введите метку (например, {i}):")
            label = int(input("Введите номер метки: "))
            labels.append(label)

# Сохраните метки в файл или используйте их для обучения модели
