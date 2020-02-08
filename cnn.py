from keras.layers import (Activation, Conv1D, Conv2D, Dense,
                          GlobalMaxPooling1D, MaxPooling1D)
from keras.models import Sequential


def CNNModel(input_shape):
    model = Sequential([
        # Conv 1
        Conv1D(4, (3), input_shape=input_shape),
        Activation('relu'),
        GlobalMaxPooling1D(),
        # Dense 1
        Dense(16),
        Activation('relu'),
        # Output
        Dense(1),
        Activation('sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


def CNNModel2(input_shape):
    model = Sequential([
        # Conv 1
        Conv1D(4, (3), input_shape=input_shape),
        Activation('relu'),
        MaxPooling1D(),
        # Conv 2
        Conv1D(8, (3)),
        Activation('relu'),
        GlobalMaxPooling1D(),
        # Dense 1
        Dense(16),
        Activation('relu'),
        # Output
        Dense(1),
        Activation('sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


def CNNModel3(input_shape):
    model = Sequential([
        # Conv 1
        Conv1D(32, 3, input_shape=input_shape),
        Activation('relu'),
        MaxPooling1D(),
        # Conv 2
        Conv1D(64, 3),
        Activation('relu'),
        GlobalMaxPooling1D(),
        # Dense 1
        Dense(128),
        Activation('relu'),
        # Dense 2
        Dense(64),
        Activation('relu'),
        # Dense 3
        Dense(32),
        Activation('relu'),
        # Output
        Dense(1),
        Activation('sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model
