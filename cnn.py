from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, Activation, Dense, GlobalMaxPooling1D, MaxPooling1D


def CNNModel(input_shape):
    model = Sequential([
        Conv1D(4, (3), input_shape=input_shape),
        Activation('relu'),
        GlobalMaxPooling1D(),
        Dense(16),
        Activation('relu'),
        Dense(1),
        Activation('sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def CNNModel2(input_shape):
    model = Sequential([
        Conv1D(4, (3), input_shape=input_shape),
        Activation('relu'),
        MaxPooling1D(),
        Conv1D(8, (3)),
        Activation('relu'),
        GlobalMaxPooling1D(),
        Dense(16),
        Activation('relu'),
        Dense(1),
        Activation('sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
