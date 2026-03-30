# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import math
import statistics

import nltk
from colorama import Fore, Style, init
from rich.console import Console
from rich.table import Table

from tools.core.utils import wait_for_enter_to_analyze

console = Console()
init(autoreset=True)


def calculate_sttr(
        lemmatized_words: list,
        chunk_size: int = 1000
) -> float:
    """
    Вычисляет стандартизированное соотношение типов/токенов
    (STTR) по методу Скотта (2004):
    - делит текст на подряд идущие чанки по `chunk_size`;
    - считает TTR на каждом чанке;
    - возвращает среднее значение (в долях, не в процентах);
    - если текст короче `chunk_size`, возвращает 0.0.

    :param lemmatized_words: Список лемм текста
    :param chunk_size: Размер чанка (по умолчанию 1000)
    :return: Средний TTR по чанкам (0..1)
    """
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size должен быть положительным числом!"
        )

    total_tokens = len(lemmatized_words)

    if total_tokens == 0:
        return 0.0

    chunks = [
        lemmatized_words[i:i + chunk_size]
        for i in range(0, total_tokens, chunk_size)
        if len(lemmatized_words[i:i + chunk_size]) == chunk_size
    ]

    if not chunks:
        return 0.0

    chunk_ttrs = [
        len(set(chunk)) / len(chunk)
        for chunk in chunks
    ]
    return round(statistics.mean(chunk_ttrs), 4)


def lexical_variety(
        lemmatized_words: list,
        show_analysis=True,
        chunk_size: int = 1000
):
    """
     Рассчитывает лексическую вариативность текста:

    1. **TTR (Type-Token Ratio)**: Соотношение типов
    (уникальных слов) к токенам (общему количеству слов)
    в процентах.

    2. **Log-TTR**: Логарифмическое соотношение типов к
     токенам, основанное на логарифмах их количества,
     в процентах.

    3. **Modified TTR**: Модифицированное соотношение
    типов к токенам, которое учитывает уникальные типы,
     встречающиеся только один раз в тексте.

    4. **STTR**: Стандартный TTR по последовательным
     чанкам фиксированного размера (по умолчанию 1000 токенов).

    :param lemmatized_words: Список лемм текста
    :param show_analysis: Если True, выводит результат
     и пояснение в виде таблицы.
    :param chunk_size: Размер чанка для STTR (по умолчанию 1000).

    :return tuple(float, float, float, float)
        Кортеж из четырех значений:
        - TTR (округленное до 3 знаков после запятой)
        - Log-TTR (округленное до 3 знаков после запятой)
        - Modified TTR (округленное до 3 знаков после запятой)
        - STTR (округленное до 3 знаков после запятой)
    """
    tokens_number = len(lemmatized_words)

    if tokens_number == 0:
        if show_analysis:
            print(
                Fore.LIGHTRED_EX +
                "В тексте нет слов для анализа "
                "лексического разнообразия."
            )
        return 0, 0, 0

    freq_dist = nltk.FreqDist(lemmatized_words)

    types = set(lemmatized_words)
    types_number = len(types)

    types_occur_once = sum(1 for count in freq_dist.values() if count == 1)

    # 1. TTR = V / N
    ttr = round(
        (types_number / tokens_number) * 100, 3
    ) if tokens_number > 0 else 0

    # 2. Log-TTR = log(V) / log(N)
    if tokens_number > 0 and types_number > 0:
        try:
            log_ttr = round(
                (math.log(types_number)
                 / math.log(tokens_number)
                 ) * 100, 3)
        except ZeroDivisionError:
            log_ttr = 0
    else:
        log_ttr = 0

    # 3. Modified TTR = 100 × log(N) / (1 - V1 / V)
    if types_number > 0:
        denominator = 1 - (types_occur_once / types_number)
        if denominator > 0:
            try:
                modified_ttr = round(
                    (100 * math.log(tokens_number)
                     ) / denominator, 3)
            except ZeroDivisionError:
                modified_ttr = 0
        else:
            modified_ttr = 0
    else:
        modified_ttr = 0

    # 4. STTR (Scott, 2004) — средний TTR по чанкам размера chunk_size
    sttr_ratio = calculate_sttr(lemmatized_words, chunk_size=chunk_size)
    sttr = round(sttr_ratio * 100, 3)  # в процентах для консоли

    if show_analysis:
        print(Fore.GREEN + Style.BRIGHT +
              "\n       ЛЕКСИЧЕСКАЯ ВАРИАТИВНОСТЬ ТЕКСТА")
        print(
            Fore.RED +
            "Внимание! Значение показателя `Стандартизированный TTR` "
            "\nподсчитывается только при условии, что длина текста"
            "\nсоставляет минимум 1000 токенов. ")
        table = Table()
        table.add_column("Параметр", justify="left",
                         no_wrap=True, min_width=30, style="bold")
        table.add_column("Значение", justify="center", min_width=10)

        table.add_row("Лексическая вариативность (TTR)", f"{ttr}%")
        table.add_row("Логарифмический TTR", f"{log_ttr}%")
        table.add_row("TTR на основе уникальных типов", f"{modified_ttr}")
        table.add_row("Стандартизированный TTR", f"{sttr}%")
        table.add_row("\nКоличество типов", '\n' + str(types_number))
        table.add_row("Количество токенов", str(tokens_number))

        console.print(table)
        wait_for_enter_to_analyze()

        print(Fore.GREEN + Style.BRIGHT +
              "\n                        ЧАСТОТНОЕ "
              "РАСПРЕДЕЛЕНИЕ ТИПОВ" + Fore.RESET)
        wait_for_enter_to_analyze()
        frequency_dict = {}
        for word, freq in freq_dist.items():
            if freq not in frequency_dict:
                frequency_dict[freq] = []
            frequency_dict[freq].append(word)

        table = Table()
        table.add_column("Частота", justify="center", width=10)
        table.add_column("Тип", max_width=80)

        for freq in sorted(frequency_dict.keys(), reverse=True):
            words = ', '.join(frequency_dict[freq])
            table.add_row(str(freq), words)
        console.print(table)
        wait_for_enter_to_analyze()

    return ttr, log_ttr, modified_ttr, sttr


if __name__ == "__main__":
    # Текст для примера (сгенерирован ИИ)
    text = """ Осенний ветер за окном напоминал о скором приходе холодов.
    Листья деревьев медленно кружились в воздухе, постепенно покрывая землю золотым ковром. В парке гуляли
    немногочисленные прохожие, наслаждаясь последними тёплыми днями. Вдоль аллеи бежала собака, радостно виляя
    хвостом. Маленький мальчик с интересом наблюдал за ней, крепко держа за руку свою маму. Она говорила ему о том,
    как важно сохранять природу и уважать окружающий мир. Вдалеке был виден силуэт человека, сидящего на лавочке с
    книгой. Он не спешил никуда, погружённый в чтение. Вокруг царила атмосфера умиротворённости и спокойствия.
    Солнце постепенно уходило за горизонт, окутывая парк мягким оранжевым светом. Небо меняло свой цвет,
    переходя от светло-голубого к насыщенному розовому. Птицы готовились к ночи, прячась в ветвях деревьев. Где-то
    рядом слышался тихий плеск воды из фонтана. Люди начинали расходиться по домам, постепенно покидая парк. И вот,
    когда город погрузился в вечерние сумерки, наступила долгожданная тишина. """

    from tools.core.lemmatizators import lemmatize_words
    text = input('Введите текст:')
    parsed_text = lemmatize_words(text)
    text_lemmas = [token.normal_form for token in parsed_text]

    ttr, log_ttr, modified_ttr, sttr = lexical_variety(text_lemmas)
