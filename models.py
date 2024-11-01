import numpy as np
import librosa
import matplotlib.pyplot as plt


# Функция для загрузки аудио и создания спектрограммы
def load_audio_and_create_spectrogram(file_path):
    # Загрузка аудиофайла
    audio, sr = librosa.load(file_path, sr=None)

    # Создание спектрограммы
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    return spectrogram_db


# Пример загрузки аудио и создания спектрограммы
file_path = 'C:/Users/secinstaller/Documents/Python Scripts/CNN_models/archive/0 (1).wav'  # Путь к вашему аудиофайлу
spectrogram = load_audio_and_create_spectrogram(file_path)

# Изменение формы данных для CNN (добавление оси канала)
spectrogram_with_channel = np.expand_dims(spectrogram, axis=-1)  # Добавляет ось канала в конец





# Визуализация спектрограммы
audio, sr = librosa.load(file_path, sr=None)
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Мел-спектрограмма')
plt.tight_layout()
plt.show()

# Проверка формы данных
print(f'Форма спектрограммы с добавленной осью канала: {spectrogram_with_channel.shape}')
