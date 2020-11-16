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
        name='conv1'
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


def main():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(
        path="mnist.npz")
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    # Initializing the model
    print('X_train dimension')
    print(X_train.shape[1:])
    model = build_model(X_train.shape[1:])

    # Compling the model
    model.compile(optimizer="Adam",
                  loss="binary_crossentropy", metrics=["accuracy"])

    # Printing the modle summary
    model.summary()

    # Adding the callback for TensorBoard
    tensorboard = callbacks.TensorBoard(
        log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    # fitting the model
    model.fit(x=X_train, y=y_train, epochs=4, verbose=1, batch_size=100, callbacks=[
        tensorboard], validation_data=(X_test, y_test))


# Running the app
if __name__ == "__main__":
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
    main()
