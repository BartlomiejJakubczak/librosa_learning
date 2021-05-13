import os

CLASSES = ['a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g']
CLASSES_MAP = {'a':0, 'am':1, 'bm':2, 'c':3, 'd':4, 'dm':5, 'e':6, 'em':7, 'f':8, 'g':9}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_DIR_AUDIO = os.path.join(DATA_DIR, 'audio')
DATA_DIR_GUITAR = os.path.join(DATA_DIR_AUDIO, 'Guitar_Only/')
DATA_DIR_MY_SAMPLES = os.path.join(DATA_DIR_AUDIO, 'My_Samples_Only/')
DATA_DIR_ALL_SAMPLES = os.path.join(DATA_DIR_AUDIO, 'All_Guitar_Samples_Only/')
DATA_DIR_OTHER = os.path.join(DATA_DIR_AUDIO, 'Other_Instruments/')
DATA_DIR_AUGMENTED = os.path.join(DATA_DIR_AUDIO, 'augmented')

METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
RAW_METADATA_DIR = os.path.join(DATA_DIR, 'raw')
AUGMENTED_METADATA_DIR = os.path.join(METADATA_DIR, 'augmented')
PROCESSED_METADATA_DIR = os.path.join(DATA_DIR, 'processed')

AUGMENTED_RAW_METADATA_DIR = os.path.join(AUGMENTED_METADATA_DIR, 'raw')
AUGMENTED_PROCESSED_METADATA_DIR = os.path.join(AUGMENTED_METADATA_DIR, 'processed')

RAW_METADATA_DATA_DIR_OTHER_CSV_FILE = os.path.join(DATA_DIR_OTHER, 'other_instruments.csv')

LOG_DIR = os.path.join(ROOT_DIR, 'logs')
LOG_DIR_TRAINING = os.path.join(LOG_DIR, 'training')

MODEL_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_JSON = os.path.join(MODEL_DIR, 'model.json')
MODEL_H5 = os.path.join(MODEL_DIR, 'model.h5')
MODEL_SAVE_MODEL = MODEL_DIR
