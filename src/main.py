import os

import settings
from src.model import CNN
import librosa


if __name__ == '__main__':
    audio_path = settings.DATA_DIR_GUITAR
    audio_chord_file = os.path.join(audio_path, 'a124113285.wav')
    y, sr = librosa.load(audio_chord_file, sr=44100, duration=2)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    print(spectrogram.shape)
    # cnn = CNN((128, 173))
    # cnn.load_model()
    # chord = cnn.predict(audio_chord_file)
    # print(chord)

