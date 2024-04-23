import os
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import librosa
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.utils import load_img, img_to_array
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment
from tensorflow import keras

class_labels = ['blues', 'classical', 'country', 'disco', 'pop', 'hiphop', 'metal', 'reggae', 'rock']


def GenreModel(classes=9):
    X_input = keras.Input(shape=(288, 432, 4))

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


model = GenreModel(classes=9)
model.load_weights("old_model.h5")


def extract_relevant(wav_file, t1, t2):
    wav = AudioSegment.from_wav(wav_file)
    wav = wav[1000 * t1:1000 * t2]
    wav.export("extracted.wav", format='wav')


def create_melspectrogram(wav_file):
    y, sr = librosa.load(wav_file, duration=3)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
    plt.savefig('melspectrogram.png')


def predict(image_data, model):
    image = img_to_array(image_data)
    image = np.reshape(image, (1, 288, 432, 4))
    prediction = model.predict(image / 255)
    prediction = prediction.reshape((9,))
    class_label = np.argmax(prediction)
    return class_label, prediction


def show_output(songfile_path):

    cl = {'blues': 0.0, 'classical': 0.0, 'country': 0.0, 'disco': 0.0, 'pop': 0.0, 'hiphop': 0.0, 'metal': 0.0,
          'reggae': 0.0, 'rock': 0.0}
    audio = AudioSegment.from_file(songfile_path)
    length = audio.duration_seconds
    sound = AudioSegment.from_mp3(songfile_path)
    sound.export("music_file.wav", format="wav")
    p = 0
    for i in range(0, int(length), 3):
        extract_relevant("music_file.wav", i, i + 3)
        create_melspectrogram("extracted.wav")
        image_data = load_img('melspectrogram.png', color_mode='rgba', target_size=(288, 432))
        class_label, prediction = predict(image_data, model)
        cl[class_labels[class_label]] += max(prediction)
        p += 1

    for genre, value in cl.items():
        cl[genre] = (value / p) * 100

    sorted_dict = dict(sorted(cl.items(), key=lambda item: item[1]))
    os.remove("extracted.wav")
    os.remove("melspectrogram.png")
    os.remove("music_file.wav")

    return sorted_dict
