import os
import datetime
from keras.layers import Activation, Dense, Dropout, Conv2D, \
    Flatten, MaxPooling2D
from keras.models import Sequential
from src.metrics import *
from settings import LOG_DIR_TRAINING, MODEL_JSON, MODEL_DIR, MODEL_H5


class CNN(object):
    def __init__(self, most_shape):
        print("Initializing CNN")
        self.model = Sequential()
        self.input_shape = most_shape + (1,)
        print(f"Input shape = {self.input_shape}")
        self.model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=self.input_shape))
        self.model.add(MaxPooling2D((4, 2), strides=(4, 2)))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(48, (5, 5), padding="valid"))
        self.model.add(MaxPooling2D((4, 2), strides=(4, 2)))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(48, (5, 5), padding="valid"))
        self.model.add(Activation('relu'))

        self.model.add(Flatten())
        self.model.add(Dropout(rate=0.5))

        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(rate=0.5))

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))
        print("CNN Initialized")

    def __str__(self):
        return str(self.model.summary())

    def train(self, X_train, y_train, X_test, y_test):
        print("Start training model")
        self.model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=['accuracy', precision, recall, fmeasure])

        # TensorBoard Logging
        log_dir = os.path.join(LOG_DIR_TRAINING, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        print("Tensorboard Logging Started")
        print(
            "Use the following command in the terminal to view the logs during training: tensorboard --logdir logs/training")

        self.model.fit(
            x=X_train,
            y=y_train,
            epochs=67,
            batch_size=20,
            validation_data=(X_test, y_test),
            callbacks=[tensorboard_callback])

        print("Training completed")

    def evaluate(self, X_train, y_train, X_test, y_test):
        print("Evaluating model")
        self.score_test = self.model.evaluate(
            x=X_test,
            y=y_test)

        self.score_train = self.model.evaluate(
            x=X_train,
            y=y_train)

        print(f'Train loss: {self.score_train[0]}')
        print(f'Train accuracy: {self.score_train[1]}')
        print(f'Train precision: {self.score_train[2]}')
        print(f'Train recall: {self.score_train[3]}')
        print(f'Train f1-score: {self.score_train[4]}')

        print(f'Test loss: {self.score_test[0]}')
        print(f'Test accuracy: {self.score_test[1]}')
        print(f'Test precision: {self.score_test[2]}')
        print(f'Test recall: {self.score_test[3]}')
        print(f'Test f1-score: {self.score_test[4]}')

    def save_model(self):
        print('Saving model')
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(MODEL_JSON, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(MODEL_H5)
        print("Saved model to "+MODEL_DIR)
