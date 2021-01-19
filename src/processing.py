import numpy as np
import keras


def train_test_split(dataset, augmented=True, split_ratio=0.65):
    print(f"Start train test split with split ratio: {split_ratio}")
    np.random.seed(42)
    sample = np.random.choice(dataset.index, size=int(len(dataset) * split_ratio), replace=False)
    if augmented:
        train_data = dataset.iloc[sample]
        test_data = dataset.drop(sample)
        test_data = test_data[test_data['augmentation'] == 'None']
    else:
        train_data, test_data = dataset.iloc[sample], dataset.drop(sample)
    print(f"Number of training samples is {len(train_data)}")
    print(f"Number of testing samples is {len(test_data)}")
    print(f"Train test split completed")
    return train_data, test_data


def features_target_split(data):
    print(f"Start feature target split")
    feature = data['spectrogram']
    target = data['class_ID']
    print(f"Feature target split completed")
    return feature, target


def reshape_feature_CNN(features):
    print(f"Features reshaped for CNN Input")
    return np.array([feature.reshape( (128, 87, 1) ) for feature in features])


def one_hot_encode(target):
    print(f"Target one hot encoded")
    return np.array(keras.utils.to_categorical(target, 10))
