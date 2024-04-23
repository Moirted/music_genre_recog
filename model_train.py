import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import keras.backend as K
from keras import layers


def GenreModel(classes=18):
    X_input = keras.Input(shape=(320, 240, 4))

    X = layers.Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(X_input)
    X = layers.BatchNormalization(axis=3)(X)
    X = keras.activations.relu(X)
    X = layers.MaxPooling2D((2, 2))(X)

    X = layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = keras.activations.relu(X)
    X = layers.MaxPooling2D((2, 2))(X)

    X = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = keras.activations.relu(X)
    X = layers.MaxPooling2D((2, 2))(X)

    X = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1))(X)
    X = layers.BatchNormalization(axis=-1)(X)
    X = keras.activations.relu(X)
    X = layers.MaxPooling2D((2, 2))(X)

    X = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1))(X)
    X = layers.BatchNormalization(axis=-1)(X)
    X = keras.activations.relu(X)
    X = layers.MaxPooling2D((2, 2))(X)

    X = layers.Flatten()(X)
    X = layers.Dropout(rate=0.3)(X)

    X = layers.Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    model = keras.Model(inputs=X_input, outputs=X, name='GenreModel')

    return model


# Жанры музыки
genres = 'ambient blues chanson classic country disco drum_n_bass hardcore hiphop hyperpop indie jazz pop rap reggae ' \
         'rock ' \
         'techno trance '
genres = genres.split()

# Генератор для тренировочных данных
train_dir = "spectres//spectograms3sec//train/"
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(320, 240), color_mode="rgba",
                                                    class_mode='categorical', batch_size=128)
# Генератор для тестовых данных
validation_dir = "spectres//spectograms3sec//test/"
vali_datagen = ImageDataGenerator(rescale=1. / 255)
vali_generator = vali_datagen.flow_from_directory(validation_dir, target_size=(320, 240), color_mode='rgba',
                                                  class_mode='categorical', batch_size=128)


def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


model = GenreModel(classes=18)
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', get_f1])
model.summary()
model.fit_generator(train_generator, epochs=70, validation_data=vali_generator)

model.save("lol.h5")
