from fastapi import FastAPI, HTTPException
import requests
import asyncio
import os
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import torch
from transformers import pipeline
import html
import mimetypes

app = FastAPI()

print("Загрузка моделей, это может занять некоторое время...")

# Инициализация моделей для перевода, суммаризации и пунктуации
translator_ru_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")
translator_en_ru = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
model, example_texts, languages, punct, apply_te = torch.hub.load(
    repo_or_dir='snakers4/silero-models', model='silero_te'
)

# Скачивание файла с Яндекс.Диска
@app.get("/download")
async def download(path: str):
    url = f"https://cloud-api.yandex.net/v1/disk/resources/download?path={path}"
    headers = {"Authorization": f"OAuth {YANDEX_DISK_TOKEN}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Получаем ссылку на скачивание
        data = response.json()
        url = data['href']
        start = url.find('&filename=') + 10
        end = url.rfind('&disposition')
        filename = url[start:end]
        folder = os.getenv('APPDATA')
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        response = requests.get(url, stream=True)

        # Сохраняем файл
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Файл {filename} успешно скачан в папку {folder}")

        # Проверка типа файла
        mimetype, _ = mimetypes.guess_type(filepath)
        if mimetype and mimetype.startswith('video'):
            # Извлекаем аудио из видео для распознавания речи
            video = VideoFileClip(filepath)
            audio = video.audio
            filepath2 = os.path.join(folder, "audio.wav")
            audio.write_audiofile(filepath2)
            audio_file = filepath2

            max_duration = 61  # Максимальная длина сегмента аудио в секундах
            audio = AudioSegment.from_file(audio_file)
            pr_text = ""
            r = sr.Recognizer()
            text = ""
            fragments = []
            start_time = 0
            end_time = 0

            # Чтение и обработка аудиофайла
            while end_time < len(audio):
                end_time = min(start_time + max_duration * 1000, len(audio))
                audio_fragment = audio[start_time:end_time]

                with sr.AudioFile(audio_fragment.export("temp.wav", format="wav")) as source:
                    audio_data = r.record(source)
                    try:
                        pr_text = r.recognize_google(audio_data, language="ru-RU")
                        last_space_index = pr_text.rfind("  ")
                        if last_space_index != -1:
                            end_time = start_time + (last_space_index + 1) * 1000 / len(pr_text) * len(audio_fragment)
                        end_time = min(end_time, len(audio))
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        print(f"Ошибка распознавания речи: {e}")

                text += pr_text + " "
                fragment_name = f"{audio_file[:-4]}_{start_time // 1000}_{end_time // 1000}.wav"
                audio_fragment.export(fragment_name, format="wav")
                fragments.append(fragment_name)

                if end_time < len(audio):
                    start_time = end_time

            # Разделение текста на фрагменты и добавление пунктуации
            max_len = 500
            text_fragments = []
            start = 0
            while start < len(text):
                end = min(start + max_len, len(text))
                text_fragment = text[start:end]

                if end < len(text) and text_fragment[-1] != ' ':
                    last_space = text_fragment.rfind(' ')
                    if last_space != -1:
                        end = start + last_space
                        text_fragment = text[start:end]

                text_fragments.append(text_fragment)
                start = end

            processed_fragments = []
            for tfragment in text_fragments:
                if tfragment.strip():
                    try:
                        punctuated_text = apply_te(tfragment, lan="ru")
                        processed_fragments.append(punctuated_text)
                    except IndexError:
                        processed_fragments.append(tfragment)

            text = " ".join(processed_fragments)
            text = html.unescape(text)
            text_chunks = chunk_text_by_sentence(text, max_chunk_size=500)

            # Перевод текста на английский, суммаризация и обратный перевод
            translated_text = await translate_to_english(text_chunks)
            summary_chunks = chunk_text_by_sentence(translated_text, max_chunk_size=1024)
            summary_en = await summarize_text(summary_chunks)
            final_summary_chunks = chunk_text_by_sentence(summary_en, max_chunk_size=500)
            final_summary = await translate_to_russian(final_summary_chunks)
            final_summary = html.unescape(final_summary)

            print("Cуммаризация:")
            print(final_summary)
        else:
            print("Скачанный файл не является видео.")
    else:
        print(f"Ошибка скачивания файла. Код статуса: {response.status_code}")

# Список элементов в директории Яндекс.Диска
@app.get("/list_items")
async def get_list_items(path: str):
    url = f"https://cloud-api.yandex.net/v1/disk/resources?path={path}&limit=1000"
    headers = {"Authorization": f"OAuth {YANDEX_DISK_TOKEN}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        items = data["_embedded"]["items"]
        names = []
        types = []

        for item in items:
            names.append(item["name"])
        for i, name in enumerate(names, start=1):
            print(f"{i}. {name}")

        num = int(input('Впишите номер нужного элемента: ')) - 1

        for item in items:
            types.append(item["type"])

        if types[num] == 'dir':
            operations_dir = ['Открыть', 'Скачать', 'Удалить']
            for i, oper_dir in enumerate(operations_dir, start=1):
                print(f"{i}. {oper_dir}")
            num_op = int(input('Впишите номер нужной операции: '))

            if num_op == 1:
                path = path + names[num] + '/'
                await get_list_items(path)
            if num_op == 2:
                await download(path + names[num])
            if num_op == 3:
                await delete(path + names[num])
        else:
            operations_files = ['Скачать и суммаризировать (если видео)', 'Удалить']
            for j, oper_file in enumerate(operations_files, start=1):
                print(f"{j}. {oper_file}")
            num_op = int(input('Впишите номер нужной операции: '))

            if num_op == 1:
                await download(path + names[num])
            if num_op == 2:
                await delete(path + names[num])

# Удаление элемента
@app.delete("/delete")
async def delete(path: str):
    url = f"https://cloud-api.yandex.net/v1/disk/resources?path={path}&permanently=true"
    headers = {"Authorization": f"OAuth {YANDEX_DISK_TOKEN}"}
    response = requests.delete(url, headers=headers)

    if response.status_code == 204:
        print(f"Успешное удаление.")
    else:
        print(f"Ошибка при удалении: {response.status_code}")
        print(response.text)

# Получение URL для загрузки файла
@app.get("/get_upload_url")
async def upload_url():
    file_path = input('Укажите локальный путь к файлу: ')
    file_name = file_path.rsplit("\\", 1)[-1]
    to_path = input('Введите путь для загрузки на Диск (без имени файла): ')
    if not to_path.endswith('/'):
        to_path += '/'
    url = f"https://cloud-api.yandex.net/v1/disk/resources/upload?path={to_path+file_name}"

    headers = {"Authorization": f"OAuth {YANDEX_DISK_TOKEN}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        upload_url = data['href']
        print(upload_url)
        await upload(upload_url, file_path)

# Загрузка файла
@app.put("/upload")
async def upload(upload_url: str, file_path: str):
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.put(upload_url, files=files)
        response.raise_for_status()
    print(f"Файл {os.path.basename(file_path)} успешно загружен.")

# Функция для разбиения текста на фрагменты
def chunk_text_by_sentence(text, max_chunk_size=500):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        end = text.rfind('.', start, end) + 1
        if end == 0:
            end = start + max_chunk_size
        chunks.append(text[start:end].strip())
        start = end
    return chunks

# Асинхронные функции для перевода и суммаризации текста
async def translate_to_english(text_chunks):
    tasks = [asyncio.to_thread(translator_ru_en, chunk, max_length=512) for chunk in text_chunks]
    translations = await asyncio.gather(*tasks)
    return " ".join([t[0]['translation_text'] for t in translations])

async def summarize_text(summary_chunks):
    summarized_chunks = []
    for chunk in summary_chunks:
        input_length = len(chunk.split())
        max_summary_length = max(30, int(input_length * 0.5))
        summary = await asyncio.to_thread(summarizer, chunk, max_length=max_summary_length, min_length=15, do_sample=False)
        summarized_chunks.append(summary[0]['summary_text'])
    return " ".join(summarized_chunks)

async def translate_to_russian(text_chunks):
    tasks = [asyncio.to_thread(translator_en_ru, chunk, max_length=512) for chunk in text_chunks]
    translations = await asyncio.gather(*tasks)
    return " ".join([t[0]['translation_text'] for t in translations])

if __name__ == "__main__":
    YANDEX_DISK_TOKEN = input('Введите токен (без OAuth): ')
    print ('Выберите операцию:')
    print ('1. Загрузить файл на Диск')
    print ('2. Открыть корневую директорию')
    oper_num = int(input('Введите номер нужной операции: '))

    if oper_num == 1:
        asyncio.run(upload_url())
    if oper_num == 2:
        path = '/'
        asyncio.run(get_list_items(path))
