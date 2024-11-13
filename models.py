import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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


# Пример использования функций
directory_path = 'C:/Users/secinstaller/Documents/Python Scripts/AI_models_micro/1/'#кажите путь к вашей директории с аудиофайлами
spectrograms, names = load_wav_files_and_convert_to_spectrograms(directory_path)

# Создаем метки для классов (например, извлекая их из имен файлов или используя другую логику)
labels = [name.split('_')[0] for name in names]  # Предположим, что метка класса - это часть имени файла

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = split_data(spectrograms, labels)

print(f'Форма обучающей выборки: {len(X_train)}')
print(f'Форма тестовой выборки: {len(X_test)}')