# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import re

from colorama import Fore, Style, init
from nltk.tokenize import word_tokenize

from rich.console import Console
from rich.table import Table

from tools.core.utils import wait_for_enter_to_analyze

console = Console()
init(autoreset=True)


def mean_word_length_char(text, show_analysis=True):
    """
    Вычисляет среднюю длину слов в тексте в символах.

    :param text: Текст для анализа.
    :param show_analysis: Если True,
    выводит результат в виде таблицы.
    :return float: Средняя длина слов в символах.
    """
    patterns = r"[^А-Яа-яёЁa-zA-Z\-]+"
    text = re.sub(patterns, ' ', text)

    # Разбиваем фильтрованный текст на слова, создаем их список
    tokens = word_tokenize(text, language="russian")

    # Проверка на наличие слов
    if len(tokens) == 0:
        if show_analysis:
            print(Fore.RED + Style.BRIGHT +
                  "\n* Во введенном тексте "
                  "слова отсутствуют.")
        return 0
    total_chars_num = sum(map(len, tokens))
    mean_word_length_in_chars = round(total_chars_num / len(tokens), 3)

    if show_analysis:

        print(Fore.GREEN + Style.BRIGHT +
              "\n         СРЕДНЯЯ ДЛИНА СЛОВ В СИМВОЛАХ")
        table = Table()
        table.add_column("Параметр", justify="left",
                         no_wrap=True, min_width=30, style="bold")
        table.add_column("Значение", justify="center", min_width=10)

        table.add_row("Длина в символах",
                      str(mean_word_length_in_chars))
        console.print(table)
        wait_for_enter_to_analyze()

    return mean_word_length_in_chars


def count_syllables(text):
    """
    Подсчитывает общее количество слогов в тексте.

    :param text: Текст для подсчета слогов.
    :return int: Общее количество слогов в тексте.
    """
    syllables_count = 0
    syllables = set("аяуюоеёэиы")
    for syllable in text:
        if syllable in syllables:
            syllables_count += 1
    return syllables_count


def mean_word_length_syllab(
        text, show_in_console=True
):
    """
    Вычисляет показатель соотношения слогов к словам в тексте.

    :param text: Текст для анализа.
    :param show_in_console: Если True, выводит результат в виде таблицы.
    :return: Кортеж из двух значений: соотношение слогов к словам
    и общее количество слогов.
    """
    patterns = r"[^А-Яа-яёЁ\-]+"
    alpha_text = re.sub(patterns, ' ', text.lower())
    # Разбиваем фильтрованный текст на слова, создаем их список
    alpha_tokens = word_tokenize(alpha_text, language="russian")
    if len(alpha_tokens) == 0:
        if show_in_console:
            print_syllable_ratio_table(0, 0)
            wait_for_enter_to_analyze()
        return 0, 0

    combined_words_str = ' '.join(alpha_tokens)
    syllables_count = count_syllables(combined_words_str)
    syllable_ratio = round((syllables_count / len(alpha_tokens)), 3)
    if show_in_console:
        print_syllable_ratio_table(syllable_ratio, syllables_count)
        wait_for_enter_to_analyze()

    return syllable_ratio, syllables_count


def print_syllable_ratio_table(syllable_ratio, syllables_count):
    """
       Создает и выводит таблицу для соотношения слогов.

       :param syllable_ratio: Соотношение слогов к словам.
       :param syllables_count: Общее количество слогов.
       """
    print(Fore.GREEN + Style.BRIGHT +
          "\n        СРЕДНЯЯ ДЛИНА СЛОВ В СЛОГАХ")
    print(Fore.RED + Style.BRIGHT +
          "Внимание! Учитываются только "
          "слова на кириллице.")

    table = Table()
    table.add_column("Параметр", justify="left",
                     no_wrap=True, min_width=30, style="bold")
    table.add_column("Значение", justify="center", min_width=10)

    table.add_row("Длина в слогах", str(syllable_ratio))
    table.add_row("Общее количество слогов", str(syllables_count))
    console.print(table)


if __name__ == "__main__":
    # Текст для примера (сгенерирован ИИ)
    text = """
    Осенний ветер за окном напоминал о скором приходе холодов. Листья деревьев медленно кружились в воздухе, постепенно 
    покрывая землю золотым ковром. В парке гуляли немногочисленные прохожие, наслаждаясь последними тёплыми днями. Вдоль 
    аллеи бежала собака, радостно виляя хвостом. Маленький мальчик с интересом наблюдал за ней, крепко держа за руку свою 
    маму. Она говорила ему о том, как важно сохранять природу и уважать окружающий мир. Вдалеке был виден силуэт 
    человека, сидящего на лавочке с книгой. Он не спешил никуда, погружённый в чтение. Вокруг царила атмосфера 
    умиротворённости и спокойствия. Солнце постепенно уходило за горизонт, окутывая парк мягким оранжевым светом. Небо 
    меняло свой цвет, переходя от светло-голубого к насыщенному розовому. Птицы готовились к ночи, прячась в ветвях 
    деревьев. Где-то рядом слышался тихий плеск воды из фонтана. Люди начинали расходиться по домам, постепенно покидая 
    парк. И вот, когда город погрузился в вечерние сумерки, наступила долгожданная тишина.
    """

    mean_word_length_char(text)
    mean_word_length_syllab(text)
