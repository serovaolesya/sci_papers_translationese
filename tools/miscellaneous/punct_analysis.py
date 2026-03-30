# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import json
from collections import Counter

from colorama import Fore, Style, init
from nltk.tokenize import word_tokenize
from rich.console import Console
from rich.table import Table

from tools.core.utils import wait_for_enter_to_analyze

init(autoreset=True)

PUNCT_MARKS_PATTERN = r"[.,!?;:…—–-()\"'«»‘’“”/\[\]{}\\]"

# Список знаков препинания, которые
# мы будем учитывать в анализе
punctuation_marks = set(PUNCT_MARKS_PATTERN)


def analyze_punctuation(text, show_analysis=True):
    """
    Анализирует использование знаков препинания
     в заданном фрагменте текста.

    :param text: Текстовый фрагмент для анализа.
    :param show_analysis: Определяет, выводить
     ли анализ в терминал.
    :return: Два словаря в формате JSON:
        - normalized_frequency: частоты
         знаков препинания относительно
          общего числа токенов.
        - punct_to_all_punct_frequency:
        частоты знаков препинания
        относительно всех знаков
        препинания.
    """
    # Токенизация текста для подсчета слов
    # и знаков препинания
    tokens = word_tokenize(
        text,
        language="russian"
    )
    total_tokens = len(tokens)
    punctuation_counts = Counter(
        token for token in tokens
        if token in punctuation_marks
    )
    total_punctuation_marks = sum(punctuation_counts.values())
    normalized_frequency = {}
    punct_to_all_punct_frequency = {}

    for punct in punctuation_marks:
        n = punctuation_counts.get(punct, 0)

        if total_tokens > 0:
            normalized_freq = round((n / total_tokens) * 100, 3)
        else:
            normalized_freq = 0

        if total_punctuation_marks > 0:
            punct_freq = round((n / total_punctuation_marks) * 100, 3)
        else:
            punct_freq = 0
        if normalized_freq > 0 or punct_freq > 0:
            normalized_frequency[punct] = normalized_freq
            punct_to_all_punct_frequency[punct] = punct_freq

    normalized_frequency = dict(
        sorted(normalized_frequency.items(),
               key=lambda item: item[1],
               reverse=True))

    punct_to_all_punct_frequency = dict(
        sorted(punct_to_all_punct_frequency.items(),
               key=lambda item: item[1],
               reverse=True))

    punctuation_counts = dict(sorted(punctuation_counts.items(),
                                     key=lambda item: item[1],
                                     reverse=True))
    punctuation_counts = {k: v for k, v
                          in punctuation_counts.items()
                          if v > 0}

    normalized_frequency_json = json.dumps(
        normalized_frequency,
        ensure_ascii=False
    )
    punct_to_all_punct_frequency_json = json.dumps(
        punct_to_all_punct_frequency,
        ensure_ascii=False
    )
    punctuation_counts_json = json.dumps(
        punctuation_counts,
        ensure_ascii=False
    )

    if show_analysis:
        display_punctuation_analysis(
            normalized_frequency_json,
            punct_to_all_punct_frequency_json,
            punctuation_counts_json)
        wait_for_enter_to_analyze()
    return (normalized_frequency_json,
            punct_to_all_punct_frequency_json,
            punctuation_counts_json)


def display_punctuation_analysis(
        normalized_frequency_json,
        punct_to_all_punct_frequency_json,
        punctuation_counts_json
):
    """
    Выводит результаты анализа знаков препинания
    в консоль, принимая их в формате JSON.

    :param normalized_frequency_json:
    Частоты знаков препинания относительно
     общего числа токенов (в формате JSON).
    :param punct_to_all_punct_frequency_json:
     Частоты знаков препинания относительно
     всех знаков препинания (в формате JSON).
    :param punctuation_counts_json:
    Счетчики количества знаков препинания.
    """
    normalized_frequency = json.loads(
        normalized_frequency_json
    )
    punct_to_all_punct_frequency = json.loads(
        punct_to_all_punct_frequency_json
    )
    punctuation_counts = json.loads(
        punctuation_counts_json
    )

    console = Console()
    print(Fore.GREEN + Style.BRIGHT +
          "\n                     "
          "ЧАСТОТЫ ЗНАКОВ ПРЕПИНАНИЯ")

    table = Table()

    table.add_column("Знак\n", justify="center",
                     style="bold", no_wrap=True)
    table.add_column("Абсолютная\nчастота",
                     justify="center")
    table.add_column("Относительно всех"
                     "\nтокенов в тексте (%)",
                     justify="center")
    table.add_column("Относительно всех"
                     "\nзнаков препинания (%)",
                     justify="center")

    for punct in normalized_frequency:
        if normalized_frequency[punct] > 0:
            absolute_count = punctuation_counts.get(punct, 0)
            normalized_value = f"{normalized_frequency[punct]:.3f}%"
            punct_to_all_punct_freq = f"{punct_to_all_punct_frequency[punct]:.3f}%"
            table.add_row(
                punct,
                str(absolute_count),
                normalized_value,
                punct_to_all_punct_freq
            )

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
    (norm_freq, punct_freq,
     punctuation_counts) = analyze_punctuation(text)
