# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import os
import re

from colorama import Fore, Style, init

from tools.core.constants import INCORRECT_CHOICE
from tools.core.utils import wait_for_enter_to_choose_opt


def remove_links(text):
    """Удаляет ссылки и форматирует текст."""

    # Удаление ссылок (любой последовательности, начинающейся с http или www)
    text_without_links = re.sub(r'\(https?://\S+|www\.\S+\)', '', text)

    # Удаление простых сносок на литературу, например, [1]
    text_without_simple_references = re.sub(r'\s*\[\d+\]\s*', '', text_without_links)

    # Удаление сложных сносок на литературу, например, [1, 2], [1-3], [1; 2]
    text_without_complex_references_square_br = re.sub(r'\s*\[\d+((,\s*\d+)|([–-]\d+)|;\s*\d+)+\]\s*', '',
                                                       text_without_simple_references)

    # Удаление сложных сносок на литературу, например, (1, 2), (1-3), (1; 2)
    text_without_complex_references_round_br = re.sub(r'\s*\(\d+((,\s*\d+)|([–-]\d+)|;\s*\d+)+\)\s*', '',
                                                      text_without_complex_references_square_br)

    # Удаление ссылок, начинающихся на ( и содержащих год (четыре цифры) и оканчивающихся на )
    text_without_round_bracket_references = re.sub(r'\([^\)]*\b\d{4}\b[^\)]*\)', '',
                                                   text_without_complex_references_round_br)

    # Удаление ссылок, начинающихся на [ и содержащих год (четыре цифры) и оканчивающихся на ]
    text_without_square_bracket_references = re.sub(r'\[[^\)]*\b\d{4}\b[^\)]*\]', '',
                                                    text_without_round_bracket_references)

    # Удаление ссылок, начинающихся на ( или [ и содержащих 'с.' и заканчивающихся на ) или ]
    text_without_references_with_rus_page_1 = re.sub(r'[\[\(][^\]\)]+[сc]\.[^\]\)]+[\]\)]', '',
                                                     text_without_square_bracket_references)

    text_without_references_with_rus_page_2 = re.sub(r'\[\s*с\.\s*\d+\s*\]|\(\s*с\.\s*\d+\s*\)', '',
                                                     text_without_references_with_rus_page_1)

    # Замена двух и более пробелов на одинарный
    text_without_double_spaces = re.sub(r'\s{2,}', ' ', text_without_references_with_rus_page_2)

    # Удаляем красные строки (абзацы)
    text_without_newlines = text_without_double_spaces.replace('\n', ' ')

    # Удаление лишних пробелов в начале и конце текста
    text_without_extraendspaces = text_without_newlines.strip()

    # Удаление лишних пробелов перед пунктуацией (например, перед точкой)
    text_without_spaces_before_punct = re.sub(r'\s([?.!,":;])', r'\1', text_without_extraendspaces)

    # Удаляем повторяющиеся пунктуационные знаки
    text_without_repeated_punct = re.sub(r'(\W\s*)\1+', r'\1', text_without_spaces_before_punct)

    # Замена дефиса, окруженного пробелами, на длинное тире, окруженное пробелами
    text_without_hyphens = re.sub(r' \- ', ' — ', text_without_repeated_punct)

    print(Fore.LIGHTGREEN_EX + Style.BRIGHT + '\nОчищенный текст:\n' + Fore.RESET)
    print(text_without_hyphens)

    return text_without_hyphens


def process_text_from_file(file_path, init_dir):
    """Обрабатывает текст из файла и сохраняет результат."""

    # Чтение текста из файла
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Очистка текста
    cleaned_text = remove_links(text)
    save_directory = ""

    # Выбор места сохранения
    if init_dir == '1':
        save_directory = "../auth_ready/"
    if init_dir == '2':
        save_directory = "../mt_ready/"
    if init_dir == '3':
        save_directory = "../ht_ready/"

    # Создание папки для сохранения, если её нет
    os.makedirs(save_directory, exist_ok=True)

    # Формирование нового имени файла
    dir_name, base_name = os.path.split(file_path)
    name, extension = os.path.splitext(base_name)
    new_file_name = f"{name}_processed{extension}"
    new_file_path = os.path.join(save_directory, new_file_name)

    # Запись очищенного текста в новый файл
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

    print(
        Fore.LIGHTGREEN_EX + Style.BRIGHT + f"\nРезультат успешно сохранен! Вы можете найти его по пути: {new_file_path}." + Fore.RESET)
    print(
        Fore.LIGHTGREEN_EX + Style.BRIGHT + f"Для дальнейшего анализа текста скопируйте текст из файла. При необходимости отредактируйте его вручную." + Fore.RESET)
    wait_for_enter_to_choose_opt()


def main():
    while True:
        # Выбор режима ввода
        mode = input(
            Fore.LIGHTGREEN_EX + Style.BRIGHT + "Введите 'f' для обработки файла или 't' для ввода текста вручную: " + Fore.RESET).strip().lower()

        if mode.lower().strip() == 'f':
            while True:
                print(
                    Fore.GREEN + Style.BRIGHT + "\nВыберите директорию с Вашим текстовым файлом (.txt)." + Fore.RESET)
                print(Fore.BLACK + "1. Директория с аутентичными текстами /auth_texts/")
                print(Fore.BLACK + "2. Директория с машинными переводами /mt_texts/")
                print(Fore.BLACK + "3. Директория с переводами, сделанными человеком /ht_texts/")
                dir_choice = input(
                    Fore.GREEN + Style.BRIGHT + "Введите номер директории: " + Fore.RESET).strip()
                directory = ''
                if dir_choice == '1':
                    directory = "../auth_texts/"
                elif dir_choice == '2':
                    directory = "../mt_texts/"
                elif dir_choice == '3':
                    directory = "../ht_texts/"
                else:
                    print(INCORRECT_CHOICE)
                    continue

                file_name = input(
                    Fore.LIGHTGREEN_EX +
                    "Введите название файла в"
                    " выбранной директории "
                    "(только файлы .txt): "
                ).strip()
                file_path = directory + file_name + '.txt'
                if os.path.isfile(file_path):
                    process_text_from_file(file_path, dir_choice)
                else:
                    print(
                        Fore.LIGHTRED_EX + Style.BRIGHT +
                        "Файл не найден. Проверьте путь "
                        "и попробуйте снова.")
                continue

        elif mode.lower().strip() == 't':
            while True:
                print(Fore.GREEN + Style.BRIGHT +
                      "\nВыберите тип текста.")
                print(Fore.BLACK + "1. Аутентичный текст")
                print(Fore.BLACK + "2. Машинный перевод")
                print(Fore.BLACK + "3. Перевод, сделанный человеком")
                text_type_choice = input(
                    Fore.GREEN + Style.BRIGHT +
                    "Введите номер типа текста: ").strip()

                # Выбор места сохранения
                if text_type_choice == '1':
                    save_directory = "../auth_ready/"
                elif text_type_choice == '2':
                    save_directory = "../mt_ready/"
                elif text_type_choice == '3':
                    save_directory = "../ht_ready/"
                else:
                    print(INCORRECT_CHOICE)
                    continue

                # Ввод текста вручную
                text = get_full_input()
                # Формирование имени файла для сохранения
                while True:
                    file_name = input(
                        Fore.LIGHTGREEN_EX + Style.BRIGHT + "Введите имя файла для сохранения результата (без расширения): " + Fore.RESET).strip()

                    if file_name:  # Проверяем, что имя файла не пустое
                        break
                    else:
                        print(
                            Fore.LIGHTRED_EX + Style.BRIGHT + "Имя файла не может быть пустым. Пожалуйста, введите имя файла." + Fore.RESET)

                # Создание папки для сохранения, если её нет
                os.makedirs(save_directory, exist_ok=True)

                file_path = os.path.join(save_directory, f"{file_name}_processed.txt")

                # Очистка текста
                cleaned_text = remove_links(text)
                # Запись очищенного текста в файл
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(cleaned_text)

                print(
                    Fore.LIGHTGREEN_EX + Style.BRIGHT + f"\nРезультат успешно сохранен! "
                                                        f"Вы можете найти его по пути: '{file_path}'." + Fore.RESET)
                print(
                    Fore.LIGHTGREEN_EX + Style.BRIGHT + f"Для дальнейшего анализа текста скопируйте текст из файла. "
                                                        f"При необходимости отредактируйте его вручную." + Fore.RESET)

                break
            wait_for_enter_to_choose_opt()
            break
        else:
            print(INCORRECT_CHOICE)


def get_full_input():
    print(Fore.LIGHTGREEN_EX + Style.BRIGHT + "\nВведите текст для анализа (по окончанию ввода с красной строки"
                                              " напечатайте 'r' и нажмите 'Enter'):" + Fore.RESET)
    input_lines = []
    while True:
        line = input()
        if line.lower().strip() == "r":
            break
        input_lines.append(line)
    return '\n'.join(input_lines)


if __name__ == "__main__":
    main()
