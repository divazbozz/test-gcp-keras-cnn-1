from tensorflow.python.lib.io import file_io
import argparse
import numpy as np
import keras as keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Flatten, Conv2D
from keras.layers import Dropout, MaxPooling2D
import keras.backend as K
from keras import callbacks
import tensorflow as tf
K.set_image_data_format('channels_last')


def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        name='conv1',
        input_shape=input_shape
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='maxpool1'))
    model.add(
        Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            name='conv2'
        )
    )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='maxpool2'))
    model.add(Flatten())
    model.add(
        Dense(1024, activation='relu', name='dense_1')
    )
    model.add(Dropout(0.4, name='dropout'))
    model.add(
        Dense(10, activation='softmax', name='dense_2')
    )
    # First input
    model_input = Input(input_shape)
    model_output = model(model_input)
    model = Model(inputs=model_input, outputs=model_output)

    return model


def main(job_dir, **args):
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
    # Setting up the path for saving logs
    logs_path = job_dir + 'logs/tensorboard'

    num_classes = 10
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(
        path="mnist.npz")
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # Initializing the model
    print('X_train/_test dimension')
    print(X_train.shape, X_test.shape)
    print('y_train dimension')
    print(y_train.shape)
    model = build_model(X_train.shape[1:])

    # Compling the model
    model.compile(optimizer="Adam",
                  loss="binary_crossentropy", metrics=["accuracy"])

    # Printing the modle summary
    model.summary()

    # Adding the callback for TensorBoard and History
    tensorboard = callbacks.TensorBoard(
        log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

    # fitting the model
    model.fit(x=X_train, y=y_train, epochs=4, verbose=1, batch_size=100, callbacks=[
        tensorboard], validation_data=(X_test, y_test))

    # Save model.h5 on to google storage
    model.save('model.h5')
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + 'model/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


# Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
