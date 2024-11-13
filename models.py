import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers, models


def normalize_spectrogram(spectrogram):
    """Нормализует спектрограмму с использованием Min-Max нормализации."""
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    normalized = (spectrogram - min_val) / (max_val - min_val)
    return normalized


def load_wav_files_and_convert_to_spectrograms(directory):
    """
    Загружает аудиофайлы с расширением .wav из указанной директории
    и преобразует их в мел-спектрограммы.

    :param directory: Путь к директории с аудиофайлами
    :return: Список спектрограмм и список названий файлов
    """
    spectrograms = []
    file_names = []

    # Проходим по всем файлам в указанной директории
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):  # Проверяем, имеет ли файл расширение .wav
            file_path = os.path.join(directory, filename)  # Получаем полный путь к файлу

            # Загружаем аудиофайл и создаем мел-спектрограмму
            audio, sr = librosa.load(file_path, sr=None)
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

            # Нормализация спектрограммы
            normalized_spectrogram = normalize_spectrogram(spectrogram_db)

            # Добавляем ось канала для совместимости с CNN
            spectrogram_with_channel = np.expand_dims(normalized_spectrogram, axis=-1)

            # Сохраняем спектрограмму и имя файла
            spectrograms.append(spectrogram_with_channel)
            file_names.append(filename)

            # Визуализация спектрограммы (по желанию)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(normalized_spectrogram, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Нормализованная мел-спектрограмма: {filename}')
            plt.tight_layout()
            plt.show()

    return spectrograms, file_names


def split_data(spectrograms, labels, test_size=0.2):
    """
    Разделяет данные на обучающую и тестовую выборки.

    :param spectrograms: Список спектрограмм
    :param labels: Список меток классов
    :param test_size: Доля тестовой выборки (по умолчанию 20%)
    :return: Обучающие и тестовые выборки для спектрограмм и меток
    """
    return train_test_split(spectrograms, labels, test_size=test_size, random_state=42)


def pad_sequences_to_same_length(spectrograms):
    """
    Дополняет спектрограммы до одинаковой длины.

    :param spectrograms: Список спектрограмм
    :return: Дополненные спектрограммы в виде массива NumPy
    """
    max_length = max(s.shape[0] for s in spectrograms)  # Получаем максимальную длину

    # Дополняем спектрограммы до одинаковой длины с помощью pad_sequences
    padded_spectrograms = pad_sequences([s.reshape(-1) for s in spectrograms], maxlen=max_length * 128, padding='post',
                                        dtype='float32')

    return padded_spectrograms.reshape(-1, max_length, 128, 1)


def create_cnn_model(input_shape, num_classes):
    """
    Создает модель CNN для классификации звуков.

    :param input_shape: Форма входных данных (включая ось канала)
    :param num_classes: Количество классов для классификации
    :return: Скомпилированная модель CNN
    """
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(num_classes, activation='softmax'))  # Выходной слой для многоклассовой классификации

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# Пример использования функций
directory_path = 'C:/Users/secinstaller/Documents/Python Scripts/AI_models_micro/archive/'  # Укажите путь к вашей директории с аудиофайлами
spectrograms, names = load_wav_files_and_convert_to_spectrograms(directory_path)

# Создаем метки для классов (например, извлекая их из имен файлов или используя другую логику)
labels = [name.split('_')[0] for name in names]  # Предположим, что метка класса - это часть имени файла

# Преобразуем метки в числовой формат с помощью LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)  # Преобразование строковых меток в числовые

# Разделяем данные на обучающую и тестовую выборки
X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_data(spectrograms, y_encoded)

# Дополняем спектрограммы до одинаковой длины
X_train = pad_sequences_to_same_length(X_train_raw)
X_test = pad_sequences_to_same_length(X_test_raw)

# Определяем параметры модели
input_shape = X_train.shape[1:]  # Форма входных данных (включая ось канала)
num_classes = len(np.unique(y_encoded))  # Количество уникальных классов

# Создаем модель CNN
model = create_cnn_model(input_shape=input_shape, num_classes=num_classes)

# Обучаем модель на обучающих данных
model.fit(X_train.astype('float32'), y_train_raw.astype('int'), epochs=10,
          validation_data=(X_test.astype('float32'), y_test_raw.astype('int')))

print("Модель обучена.")