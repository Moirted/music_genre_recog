import logging
import os

import cnn_core
import cnn_core_old
import aiohttp
import asyncio
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

API_TOKEN = 'putputput'

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
# Путь к папке, куда будут сохраняться аудиофайлы
AUDIO_SAVE_PATH = "audio_files/"

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
# Выбор модели
new_model = True

genre_names = {
    'ambient': "Амбиент",
    'blues': "Блюз",
    'chanson': "Шансон",
    'classic': "Классика",
    'country': "Кантри",
    'disco': "Диско",
    'drum_n_bass': "Драм-н-басс",
    'hardcore': "Хардкор",
    'hiphop': "Хип-хоп",
    'hyperpop': "Гиперпоп",
    'indie': "Инди",
    'jazz': "Джаз",
    'pop': "Поп",
    'reggae': "Регги",
    'rock': "Рок",
    'techno': "Техно",
    'trance': "Транс",
    'rap': "Рэп",
    'metal': "Метал"
}


# Функция для скачивания аудиофайла на сервер
async def download_audio(session, file_url, file_name):
    async with session.get(file_url) as response:
        if response.status == 200:
            with open(AUDIO_SAVE_PATH + file_name, 'wb') as f:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    f.write(chunk)
            return True
        else:
            return False


# Обработчик команды /start
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Я бот для классификации музыки по жанрам, отправь мне mp3 или запиши голосовое "
                        "сообщение с песней и я тебе помогу. У меня есть две модели для распознования "
                        "произведений, для выбора введи команду - /model. Изначально будет выбрана современная "
                        "модель.")


# Функция для создания клавиатуры с кнопками
def create_keyboard():
    keyboard = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True, one_time_keyboard=True)
    button1 = KeyboardButton('NewSM')
    button2 = KeyboardButton('OldSM')
    keyboard.add(button1, button2)
    return keyboard


# Обработчик команды /model
@dp.message_handler(commands=['model'])
async def send_welcome(message: types.Message):
    keyboard = create_keyboard()
    await message.reply("Выберите модель для распознавания жанра. Доступны две: OldSM, NewSM\nOldSM - содежит в себе "
                        "9 жанров, обучена на старом датасете GTZAN\nNewSM - модель с 18 жанрами, обучена на "
                        "новом датасете", reply_markup=keyboard)


# Обработчик нажатия на кнопки
@dp.message_handler(lambda message: message.text in ['NewSM', 'OldSM'])
async def process_model_choice(message: types.Message):
    global new_model
    if message.text == 'NewSM':
        new_model = True
        await message.answer("Вы выбрали модель NewSM")
    elif message.text == 'OldSM':
        new_model = False
        await message.answer("Вы выбрали модель OldSM")


# Обработчик всех текстовых сообщений
@dp.message_handler()
async def echo(message: types.Message):
    await message.answer("Не знаю что тебе ответить, отправь мне аудио или голосовое и я определю жанр.")


# Обработчик документов
@dp.message_handler(content_types=['document'])
async def handle_document(message: types.Message):
    document = message.document
    await message.answer(f"Вы отправили документ, а не музыку. Я принимаю только формат mp3 или голосовые сообщения.")


# Обработчик голосовых
@dp.message_handler(content_types=['voice'])
async def handle_voice(message: types.Message):
    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    voice_file_name = os.path.join("audio_files", f"{message.voice.file_id}.mp3")
    await bot.download_file(file_path, voice_file_name)
    if not new_model:
        res = cnn_core_old.show_output(voice_file_name)
    else:
        res = cnn_core.show_output(voice_file_name)
    # os.remove(file_path)
    g1, value1 = res.popitem()
    g2, value2 = res.popitem()
    g3, value3 = res.popitem()

    await message.answer(
        f"Жанр этой песни {genre_names[g1]}, вес: {value1:.3f}\r\nТакже похоже на {genre_names[g2]} с весом {value2:.3f} и {genre_names[g3]} с весом {value3:.3f}")


# Обработчик аудиофайлов
@dp.message_handler(content_types=['audio'])
async def handle_audio(message: types.Message):
    file_id = message.audio.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    audio_file_name = os.path.join("audio_files", message.audio.file_name)
    await bot.download_file(file_path, audio_file_name)
    if not new_model:
        res = cnn_core_old.show_output(audio_file_name)
    else:
        res = cnn_core.show_output(audio_file_name)
    # os.remove(file_path)
    g1, value1 = res.popitem()
    g2, value2 = res.popitem()
    g3, value3 = res.popitem()

    await message.answer(
        f"Жанр этой песни {genre_names[g1]}, вес: {value1:.3f}\r\nТакже похоже на {genre_names[g2]} с весом {value2:.3f} и {genre_names[g3]} с весом {value3:.3f}")


# Запуск бота
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
