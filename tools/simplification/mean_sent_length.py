# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import re

from colorama import Fore, Style, init
from nltk.tokenize import word_tokenize
from rich.console import Console
from rich.table import Table

from tools.core.utils import wait_for_enter_to_analyze

console = Console()
init(autoreset=True)


def mean_sentence_length_in_tokens(
        sent_list, show_analysis=True
):
    """
    Вычисляет среднюю длину предложения в токенах
    (включая знаки препинания).

    :param sent_list: Список предложений текста.
    :param show_analysis: Если True,
    выводит результат в виде таблицы.
    :return float: Средняя длина предложений в токенах.
    """
    sentence_lengths = []

    # Подсчитываем токены в каждом предложении
    for sentence in sent_list:
        tokens = word_tokenize(sentence, language="russian")
        sentence_lengths.append(len(tokens))

    mean_sent_length = round(
        (sum(sentence_lengths) / len(sentence_lengths)), 3
    ) if sentence_lengths else 0

    if show_analysis:
        # Создаем таблицу для вывода
        print(Fore.GREEN + Style.BRIGHT +
              "\n     СРЕДНЯЯ ДЛИНА ПРЕДЛОЖЕНИЙ В ТОКЕНАХ")
        print(Fore.RED + Style.BRIGHT +
              "Внимание! Знаки препинания "
              "считаются как токены.")
        table = Table()
        table.add_column("Параметр", justify="left",
                         no_wrap=True, min_width=30, style="bold")
        table.add_column("Значение", justify="center", min_width=10)

        table.add_row("Ср. длина в токенах", str(mean_sent_length))
        table.add_row("Всего токенов", str(sum(sentence_lengths)))
        table.add_row("Всего предложений", str(len(sentence_lengths)))
        console.print(table)

        wait_for_enter_to_analyze()
    return mean_sent_length


def mean_sentence_length_in_chars(sent_list: list, show_analysis=True):
    """
    Вычисляет среднюю длину предложения в символах (без учета пробелов).

    :param sent_list: Список предложений текста.
    :param show_analysis: Если True, выводит результат в виде таблицы.
    :return float:  Средняя длина предложений в символах.
    """
    sentence_lengths = []
    # Подсчитываем количество символов в каждом предложении
    for sentence in sent_list:
        cleaned_sentence = re.sub(r'\s+', '', sentence)
        length = len(cleaned_sentence)
        sentence_lengths.append(length)
    print()

    mean_sent_length = round(
        (sum(sentence_lengths) / len(sentence_lengths)
         ), 3) if (
        sentence_lengths
    ) else 0

    if show_analysis:
        print(Fore.GREEN + Style.BRIGHT +
              "     СРЕДНЯЯ ДЛИНА ПРЕДЛОЖЕНИЙ В СИМВОЛАХ")
        table = Table()
        table.add_column("Параметр", justify="left",
                         no_wrap=True, min_width=30, style="bold")
        table.add_column("Значение", justify="center", min_width=10)

        table.add_row("Ср. длина в символах", str(mean_sent_length))
        table.add_row("Всего символов", str(sum(sentence_lengths)))
        table.add_row("Всего предложений", str(len(sentence_lengths)))
        console.print(table)
        wait_for_enter_to_analyze()

    return mean_sent_length


if __name__ == "__main__":
    # Список предложений для примера:
    text = ['Осенний ветер за окном напоминал о скором приходе холодов.',
            'Листья деревьев медленно кружились в воздухе.']
    mean_sentence_length_in_tokens(text)
    mean_sentence_length_in_chars(text)
