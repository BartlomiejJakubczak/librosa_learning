import pandas as pd
import os
from settings import PROCESSED_METADATA_DIR, CLASSES
from src.preprocessing import get_most_shape
from src.processing import *
from src.model import CNN
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    dataset = pd.read_pickle(os.path.join(PROCESSED_METADATA_DIR, 'data.pkl'))
    print(f"Number of samples: {len(dataset)}")
    most_shape = get_most_shape(dataset)
    train_data, test_data = train_test_split(dataset, augmented=False, split_ratio=0.65)
    X_train, y_train = features_target_split(train_data)
    X_test, y_test = features_target_split(test_data)
    # Reshape for CNN input
    X_train, X_test = reshape_feature_CNN(X_train), reshape_feature_CNN(X_test)
    # Preserve y_test values
    y_test_values = y_test.copy()
    # One-Hot encoding for classes
    y_train, y_test = one_hot_encode(y_train), one_hot_encode(y_test)
    # Instance of CNN model
    cnn = CNN(most_shape)
    print(str(cnn))
    cnn.train(X_train, y_train, X_test, y_test)
    cnn.evaluate(X_train, y_train, X_test, y_test)
    predictions = cnn.model.predict_classes(X_test)
    conf_matrix = confusion_matrix(y_test_values, predictions, labels=range(10))
    print('Confusion Matrix for classes {}:\n{}'.format(CLASSES, conf_matrix))
    cnn.save_model()
