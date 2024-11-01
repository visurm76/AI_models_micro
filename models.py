import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def find_wav_files(directory):
    """Проверяет расширение файла '.wav' в директории и добавляет полный путь к файлам в список."""
    wav_files = []

    # Проходим по всем файлам в указанной директории
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):  # Проверяем, имеет ли файл расширение .wav
            full_path = os.path.join(directory, filename)  # Получаем полный путь к файлу
            wav_files.append(full_path)  # Добавляем полный путь в список

    return wav_files


def create_spectrogram(file_path):
    """Создание мел-спектрограммы из аудиофайла и визуализация."""
    audio, sr = librosa.load(file_path, sr=None)

    # Создание спектрограммы
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Добавление оси канала
    spectrogram_with_channel = np.expand_dims(spectrogram_db, axis=-1)

    # Визуализация спектрограммы
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Мел-спектрограмма: {os.path.basename(file_path)}')
    plt.tight_layout()
    plt.show()

    return spectrogram_with_channel


def prepare_data():
    """Подготовка данных для обучения модели CNN."""
    download_files = find_wav_files(directory_path)  # Загружаем аудиофайлы
    labels = []
    audio_files = []
    X = []
    y = []

    for file in download_files:
        spectrogram = create_spectrogram(file)  # Создаем спектрограмму
        X.append(spectrogram)
        y.append(labels[audio_files.index(file)])  # Добавляем метку

    X = np.array(X)
    y = np.array(y)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Пример использования функций
directory_path = 'C:/Users/secinstaller/Documents/Python Scripts/CNN_models/archive/'  # Укажите путь к вашей директории с аудиофайлами
wav_files_list = find_wav_files(directory_path)

print("Найденные WAV файлы:")
for wav_file in wav_files_list:
    print(wav_file)
    create_spectrogram(wav_file)  # Создаем и отображаем спектрограмму для каждого найденного файла

