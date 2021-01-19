import numpy
import librosa, librosa.display
import settings


def extract_chroma_feature_vector(filepath):
    y, sr = librosa.load(filepath)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    meaned_chroma_vector = []
    for semitones in chromagram:
        meaned_chroma_vector.append(numpy.mean(semitones))
    return meaned_chroma_vector
