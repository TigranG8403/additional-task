# Яндекс.Диск API + Суммаризация видео

## Важное объявление: из-за технической ошибки в презентацию не прикрепились ссылки на макет и прототип. Прикладываем их здесь: https://www.figma.com/design/2oZFpNdu1qjJxFL9fFcUQt/Untitled?node-id=0-1&t=YgpK1eevKslr5EAe-1 (макет); https://github.com/TigranG8403/education-assistant/ (прототип). Приносим извинения за предоставленные неудобства!

## Этот проект представляет собой консольное приложение, которое позволяет:
1. Скачивать и удалять содержимое Вашего Яндекс.Диска и загружать туда свои файлы через Yandex API.
2. Выполнять суммаризацию текста из видеофайлов с использованием модели `facebook/bart-large-cnn` (*перевод текста перед суммаризацией и после неё осуществляется с помощью моделей `Helsinki-NLP/opus-mt-ru-en` и `Helsinki-NLP/opus-mt-en-ru`*)

## Структура файлов
- **main/...**: главная папка, содержащая весь код приложения и системные файлы, необходимые для работы;
- **requirements.txt**: список зависимостей, необходимых для работы проекта

## Установка и настройка

### № Шаг 1: Скачивание, импорт проекта и установка зависимостей
Импортируйте разархивированную скачанную папку (всего репозитория) в среду разработки (PyCharm, VS Code), установите в неё виртуальное окружение Python 3.11.
В терминале выполните команду: `pip install fastapi uvicorn requests torch transformers moviepy pydub SpeechRecognition` или переместите requirements.txt в папку виртуального окружения и выполните команду: `pip install -r requirements.txt`.

### № Шаг 2: Получение OAuth-токена для Yandex API
Ознакомиться с документацией для получения OAuth-токенов можете [здесь](https://yandex.ru/dev/disk-api/doc/ru/concepts/quickstart).

### № Шаг 3: Запуск программы
Запустите через среду разработки `main.py`.

### № Шаг 4: Ввод QAuth-токена *(без "OAuth")*
После запуска программы вам будет предложено ввести ваш OAuth-токен. Введите токен и нажмите Enter. Программа использует предоставленный токен для авторизации и доступа к вашему Яндекс.Диску.

### № Шаг 5: Выбор типа операции

1. **Загрузить файл** - эта опция позволяет загрузить файл с Вашего устройства на Яндекс.Диск;
2. **Открыть корневую директорию** - позволяет вам просмотреть содержимое корневой папки на вашем Яндекс.Диске.

Выберите нужный Вам вариант, введя соответствующий номер операции и нажав Enter. 

### № Шаг 6: Выполнение операции и выбор доп. функции

- **Если Вы выбрали загрузку файла**: программа запросит у вас локальный путь к файлу, который вы хотите загрузить. После указания пути, файл будет загружен на ваш Яндекс.Диск в указанную папку.
  
- **Если Вы выбрали открытие корневой директории**: программа загрузит список файлов и папок в корневой директории и отобразит их на экране. Вы сможете выбрать один из элементов для выполнения дополнительных операций (например, скачивание или удаление).

### № Шаг 7: Получение ответа :)
