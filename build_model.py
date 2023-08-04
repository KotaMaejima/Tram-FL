from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ReLU, Embedding, LSTM
from keras.initializers import he_normal
import tensorflow_addons as tfa


def build_mnist_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

def build_cifar10_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding="same", input_shape=(32, 32, 3), kernel_initializer=he_normal))
    model.add(tfa.layers.GroupNormalization(groups=1))
    model.add(ReLU())
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=he_normal))
    model.add(tfa.layers.GroupNormalization(groups=2))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=he_normal))
    model.add(tfa.layers.GroupNormalization(groups=2))
    model.add(ReLU())
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=he_normal))
    model.add(tfa.layers.GroupNormalization(groups=2))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer=he_normal))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax', kernel_initializer=he_normal))
    return model

def build_imdb_model():
    model = Sequential()
    model.add(Embedding(10000, 32))
    model.add(LSTM(32, kernel_initializer=he_normal))
    model.add(Dense(2, activation="softmax", kernel_initializer=he_normal))
    return model