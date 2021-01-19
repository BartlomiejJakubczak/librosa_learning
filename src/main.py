import os

import settings
import audio_converter
import csv


if __name__ == '__main__':
    audio_path = settings.DATA_DIR_GUITAR
    other_path = settings.DATA_DIR_OTHER
    chroma_vectors = []


    def read_chroma_vectors(audio_path):
        directories = os.listdir(audio_path)
        for pitch_directory in directories:
            current_directory_path = os.path.join(audio_path, pitch_directory)
            if os.path.isdir(current_directory_path):
                for filename in os.listdir(current_directory_path):
                    current_directory_path_filename = os.path.join(current_directory_path, filename)
                    chroma_vector = audio_converter.extract_chroma_feature_vector(current_directory_path_filename)
                    chroma_vector.append(settings.CLASSES_MAP[pitch_directory])
                    chroma_vectors.append(chroma_vector)

    def write_csv_file(path, chroma_vectors):
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            # writer.writerow(["a", "ais", "h", "c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "class"])
            for chroma in chroma_vectors:
                writer.writerow(
                    [chroma[0], chroma[1], chroma[2], chroma[3], chroma[4], chroma[5], chroma[6], chroma[7], chroma[8],
                     chroma[9], chroma[10], chroma[11], chroma[12]])

    read_chroma_vectors(audio_path)

    write_csv_file(settings.RAW_METADATA_DIR, chroma_vectors)
    chroma_vectors = []
    read_chroma_vectors(other_path)
    write_csv_file(settings.RAW_METADATA_DATA_DIR_OTHER_CSV_FILE, chroma_vectors)
