import numpy as np
import pandas as pd
import librosa
import os
from sklearn.model_selection import train_test_split


def load_audio_files(data_dir):
    """Загрузка аудиофайлов и создание меток."""
    audio_files = []
    labels = []

    # Проходим по всем файлам в указанной директории
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)

        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(label_dir, file))
                    labels.append(label)  # Добавляем метку из имени папки

    return audio_files, labels


def create_spectrogram(file_path):
    """Создание мел-спектрограммы из аудиофайла."""
    audio, sr = librosa.load(file_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Добавление оси канала
    spectrogram_with_channel = np.expand_dims(spectrogram_db, axis=-1)

    return spectrogram_with_channel
# Визуализация спектрограммы
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Мел-спектрограмма')
    plt.tight_layout()
    plt.show()

    return spectrogram_with_channel


def prepare_data(data_dir):
    """Подготовка данных для обучения модели."""
    audio_files, labels = load_audio_files(data_dir)

    X = []
    y = []

    for file in audio_files:
        spectrogram = create_spectrogram(file)
        X.append(spectrogram)

    X = np.array(X)
    y = np.array(labels)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Пример использования
data_directory = 'C:/Users/secinstaller/Documents/Python Scripts/AI_models_micro/archive'  # Укажите путь к вашей директории с аудиофайлами
X_train, X_test, y_train, y_test = prepare_data(data_directory)

print(f'Форма обучающей выборки: {X_train.shape}')
print(f'Форма тестовой выборки: {X_test.shape}')