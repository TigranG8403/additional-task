# Яндекс.Диск API + Суммаризация текста

## Этот проект представляет собой консольное приложение, которое позволяет:
1. Скачивать файлы с Яндекс.Диска через Yandex API;
2. Выполнять суммаризацию текста из .mp4 файла с использованием модели `facebook/bart-large-cnn` (*перевод текста перед суммаризацией и после неё осуществляется с помощью моделей `Helsinki-NLP/opus-mt-ru-en` и `Helsinki-NLP/opus-mt-en-ru`*)

## Структура файлов
- **main/...**: папка, содержащая весь код приложения и системные файлы, необходимые для работы;
- **program.exe**: исполняемый файл, который пользователь может скачать для работы с приложением без необходимости устанавливать Python и зависимости.

# Установка и настройка

## Шаг 1: Скачивание исполняемого файла
Скачайте исполняемый файл `program.exe`, который находится в корневой директории репозитория.

## Шаг 2: Получение QAuth-токена
Ознакомиться с документацией для получения QAuth-токенов можете [здесь](https://yandex.ru/dev/disk-api/doc/ru/concepts/quickstart)

## Шаг 3: Запуск программы
Запустите недавно Вами скачанный исполняемый файл `program.exe`

## Шаг 4: Ввод QAuth-токена (без "QAuth")
После запуска программы вам будет предложено ввести ваш OAuth-токен. Введите токен, который вы получили на предыдущих шагах, и нажмите Enter. Программа использует предоставленный токен для авторизации и доступа к вашему Яндекс Диску.

## Шаг 5: Выбор типа операции
После успешного ввода QAuth-токена, программа предложит вам выбрать одну из доступных операций. Вы сможете выбрать из следующих вариантов:

1. **Загрузить файл** - эта опция позволяет вам загрузить локальный файл на ваш Яндекс Диск;
2. **Открыть корневую директорию** - позволяет вам просмотреть содержимое корневой папки на вашем Яндекс Диске.

Выберите нужный Вам вариант, введя соответствующий номер операции и нажав Enter. Программа продолжит выполнение в зависимости от вашего выбора.

## Шаг 6: Выполнение операции

- **Если вы выбрали загрузку файла**: программа запросит у вас локальный путь к файлу, который вы хотите загрузить. После указания пути, файл будет загружен на ваш Яндекс Диск в указанную папку.
  
- **Если вы выбрали открытие корневой директории**: программа загрузит список файлов и папок в корневой директории и отобразит их на экране. Вы сможете выбрать один из элементов для выполнения дополнительных операций (например, скачивание или удаление).

## Шаг 7: Работа с элементами

После выбора элемента из корневой директории программа предложит вам выполнить одну из следующих операций:

- **Скачать** - 
- **Удалить** - удалить выбранный файл или папку с вашего Яндекс Диска.

Выберите нужную Вам операцию, введя номер действия, и программа выполнит выбранное действие.

## Шаг 8: Получение ответа в виде готовой суммаризации
