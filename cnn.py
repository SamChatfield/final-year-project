from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, Activation, Dense, GlobalMaxPooling1D, MaxPooling1D


def CNNModel():
    model = Sequential([
        Conv1D(4, (3), input_shape=(10, 3)),
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


def CNNModel2():
    model = Sequential([
        Conv1D(4, (3), input_shape=(10, 3)),
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
