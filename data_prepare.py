import os
import random
import shutil

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment

# Жанры музыки
genres = 'ambient blues chanson classic country disco drum_n_bass hardcore hiphop hyperpop indie jazz pop rap reggae ' \
         'rock ' \
         'techno trance '
genres = genres.split()


def convert_mp3_to_wav(music_file, path_to_dir):
    sound = AudioSegment.from_mp3(music_file)
    wav_filename = os.path.splitext(os.path.basename(music_file))[0] + ".wav"
    sound.export(os.path.join(f"{path_to_dir}_wav", wav_filename), format="wav")


def convert_all_mp3_to_wav(path_to_dir):
    if not os.path.exists(f"{path_to_dir}_wav"):
        os.makedirs(f"{path_to_dir}_wav")

    # Получаем список всех mp3 файлов в папке
    mp3_files = [f for f in os.listdir(path_to_dir) if f.endswith(".mp3")]

    # Конвертируем каждый mp3 файл в wav и сохраняем в папку _wav
    for mp3_file in mp3_files:
        convert_mp3_to_wav(os.path.join(path_to_dir, mp3_file), path_to_dir)


def create_snippets(input_file, output_file):
    audio = AudioSegment.from_wav(input_file)
    midpoint = len(audio) // 2
    # Определяем начальную и конечную точки для сниппета (15 секунд перед серединой и 15 секунд после)
    start_point = max(0, midpoint - 15000)
    end_point = min(len(audio), midpoint + 15000)
    # Обрезаем
    snippet = audio[start_point:end_point]
    # Устанавливаем длительность сниппета в 30 секунд
    snippet = snippet[:30000]
    # Сохраняем сниппет в формате WAV
    snippet.export(output_file, format="wav")


def create_all_snippets(path_to_dir):
    # Получаем список всех WAV файлов в папке ambient_wav
    wav_files = [f for f in os.listdir(path_to_dir) if f.endswith(".wav")]

    # Создаем папку ambient_wav_30sec, если ее еще нет
    if not os.path.exists(f"{path_to_dir}_30sec"):
        os.makedirs(f"{path_to_dir}_30sec")

    # Создаем сниппеты для каждого аудиофайла
    for wav_file in wav_files:
        input_file = os.path.join(path_to_dir, wav_file)
        output_file = os.path.join(f"{path_to_dir}_30sec", wav_file)
        create_snippets(input_file, output_file)


def make_dirs():
    # Создаём папки для обработки датасета
    os.makedirs('spectres')
    os.makedirs('spectres/genres3sec')
    os.makedirs('spectres/spectrograms3sec')
    os.makedirs('spectres/spectrograms3sec/train')
    os.makedirs('spectres/spectrograms3sec/test')
    os.makedirs('spectres/spectrogramss3sec/not_separated')
    for g in genres:
        os.makedirs('spectres/genres3sec/' + g + "/")
        os.makedirs('spectres/spectrograms3sec/not_separated/' + g + "/")
        os.makedirs('spectres/spectrograms3sec/train/' + g + "/")
        os.makedirs('spectres/spectrograms3sec/test/' + g + "/")
    print('Были созданы папки для обработки датасета')


# Разбиваем каждый .wav файл датасета на 3-х секундный фрагмент
def split_snippet(snippet_file, output_dir):
    # Загружаем аудиофайл сниппета
    snippet = AudioSegment.from_wav(snippet_file)

    # Получаем длительность сниппета в миллисекундах
    snippet_duration = len(snippet)

    # Определяем количество частей сниппета по 3 секунды
    num_parts = snippet_duration // 3000

    # Разделяем сниппет на части по 3 секунды и сохраняем каждую часть в отдельный файл
    for i in range(num_parts):
        start_time = i * 3000
        end_time = start_time + 3000
        part = snippet[start_time:end_time]
        output_file = os.path.join(output_dir,
                                   os.path.splitext(os.path.basename(snippet_file))[0] + f"_part{i + 1}.wav")
        part.export(output_file, format="wav")


def split_all_snippets(input_dir, output_dir):
    # Получаем список всех файлов в папке сниппетов
    snippet_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

    # Разделяем каждый сниппет на части по 3 секунды
    for snippet_file in snippet_files:
        split_snippet(os.path.join(input_dir, snippet_file), output_dir)


def generate_spectrograms(g):
    j = 0
    print(f"Создание спектрограмм для: {g}")
    for filename in os.listdir(f'spectres//genres3sec//{g}//'):
        song = os.path.join(f'spectres//genres3sec//{g}//{filename}')
        j = j + 1
        y, sr = librosa.load(song, duration=3)
        mels = librosa.feature.melspectrogram(y=y, sr=sr)
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
        plt.savefig(f'spectres//spectograms3sec//not_separated//{g}//{g + str(j)}.png')


def copy_files(source_dir, target_train_dir, target_test_dir, genre, num_train, num_test):
    files = [f for f in os.listdir(os.path.join(source_dir, genre)) if
             os.path.isfile(os.path.join(source_dir, genre, f))]

    random.shuffle(files)

    train_files = files[:num_train]

    test_files = files[num_train:]

    for file in train_files:
        src_file = os.path.join(source_dir, genre, file)
        dest_file = os.path.join(target_train_dir, genre, file)
        shutil.copy(src_file, dest_file)

    for file in test_files:
        src_file = os.path.join(source_dir, genre, file)
        dest_file = os.path.join(target_test_dir, genre, file)
        shutil.copy(src_file, dest_file)


if __name__ == '__main__':
    print('Запущена обработка данных.')

    # Вызываем функцию для конвертации всех mp3 файлов в wav
    # for g in genres:
    #   convert_all_mp3_to_wav(f"new_dataset//{g}")

    # Создание 30 секундных сниппетов из песен
    # for g in genres:
    #   create_all_snippets(f'new_dataset//{g}')

    # Создание папок для спектрограм
    # make_dirs()

    # Из 30 секундных сниппетов делаем кусочки по 3 секунды
    # for g in genres:
    #    split_all_snippets(f"new_dataset//{g}", f"spectres//genres3sec//{g}")

    # Генерация спектрограмм для 3-х секундных кусочков
    # for g in genres:
    # generate_spectrograms(g)

    # Разбиваем данные на тренировочные и проверочные
    # for g in genres:
    # copy_files("spectres//spectograms3sec//not_separated", "spectres//spectograms3sec//train", "spectres//spectograms3sec//test", g, 900, 100)
