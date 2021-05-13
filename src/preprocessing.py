import librosa
import pandas as pd
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
import numpy as np
from settings import DATA_DIR_GUITAR, CLASSES_MAP, RAW_METADATA_DIR, PROCESSED_METADATA_DIR, DATA_DIR_MY_SAMPLES, DATA_DIR_ALL_SAMPLES
import glob
import os
from imblearn.over_sampling import RandomOverSampler

def df_info(f):
    def inner(df, *args, **kwargs):
        result = f(df, *args, **kwargs)
        print(f"After applying {f.__name__}, shape of df = {result.shape }")
        print(f"Columns of df are {df.columns}\n")
        return result
    return inner


def construct_dataframe(df):
    file_path = glob.glob(DATA_DIR_ALL_SAMPLES + "**\\*.wav")
    df['file_path'] = file_path
    df['file_path'] = df['file_path'].map(lambda x: x[x.rindex('Only\\') + len('Only\\'):])
    df['file_name'] = df['file_path'].map(lambda x: x[x.rindex('\\') + 1:])
    df['class_name'] = df['file_path'].map(lambda x: x[:x.index('\\')])
    df['class_ID'] = df['class_name'].map(lambda x: CLASSES_MAP[x])
    print(df)
    return df.copy()


def get_spectrogram(df):
    """Extract spectrogram from audio"""
    df['audio_series'] = df['file_path'].map(lambda x: librosa.load(DATA_DIR_ALL_SAMPLES \
                                                                    + x, duration=2, sr=44100))
    df['y'] = df['audio_series'].map(lambda x: x[0])
    df['sr'] = df['audio_series'].map(lambda x: x[1])
    df['spectrogram'] = df.apply(lambda row: librosa.feature.melspectrogram(y=row['y'],\
         sr=row['sr']), axis=1)
    df.drop(columns='audio_series', inplace=True)
    print(df)
    return df


def add_shape(df):
    df['shape'] = df['spectrogram'].map(lambda x: x.shape)
    print(df)
    return df


def process(df):
    df = (df.pipe(clean_shape)
                .pipe(over_sample)
    )
    df = df[['spectrogram','class_ID', 'class_name']]
    print(df)
    return df


def get_most_shape(df):
    most_shape = df['spectrogram'].map(lambda x: x.shape).value_counts().index[0]
    print(f"The most frequent shape is {most_shape}")
    return most_shape


def clean_shape(df):
    most_shape = get_most_shape(df)
    df = df[df['shape'] == most_shape]
    df.drop(columns='shape', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("Data frame po oczysczeniu, zaraz nastapi overampling")
    print(df)
    print(df['class_ID'].value_counts())
    return df


def get_count(df):
    return df['class_name'].value_counts()


def get_class(class_ID):
    return list(CLASSES_MAP.keys())[list(CLASSES_MAP.values()).index(class_ID)]


def over_sample(df):
    oversample = RandomOverSampler(sampling_strategy='auto')
    X, y = df['spectrogram'].values, df['class_ID'].values
    X = X.reshape(-1, 1)
    X, y = oversample.fit_resample(X, y)
    df = pd.DataFrame()
    df['spectrogram'] = pd.Series([np.array(x[0]) for x in X])
    df['class_ID'] = pd.Series(y)
    df['class_name'] = df['class_ID'].map(lambda x: get_class(x))
    print("Po oversamplingu: ")
    print(df)
    return df


def run():
    data_df_raw = (pd.DataFrame().pipe(construct_dataframe)
                   .pipe(get_spectrogram)
                   .pipe(add_shape))
    print(data_df_raw)
    data_df_raw.to_csv(os.path.join(RAW_METADATA_DIR, 'data.csv'), index=False)
    data_df_raw.to_pickle(os.path.join(RAW_METADATA_DIR, 'data.pkl'))
    print(get_count(data_df_raw))
    print("Raw data has been saved.")
    data_df_processed = process(data_df_raw)
    data_df_processed.to_csv(os.path.join(PROCESSED_METADATA_DIR, 'data.csv'), index=False)
    data_df_processed.to_pickle(os.path.join(PROCESSED_METADATA_DIR, 'data.pkl'))
    print(get_count(data_df_processed))
    print("Processed data has been saved.")


if __name__ == '__main__':
    run()