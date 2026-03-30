# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import json
import re
from collections import Counter
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pymorphy2.analyzer"
)

import pymorphy2
from colorama import Fore, Style
from nltk import ngrams
from rich.console import Console
from rich.table import Table


from tools.core.utils import (
    display_grammemes,
    wait_for_enter_to_analyze
)

console = Console()


def pos_ngrams(
        n_grams,
        lemmas,
        n_values=(1, 2, 3),
        show_analysis=True
):
    """
    Вычисляет n-граммы частей речи для русского
    текста и возвращает результаты в виде строк JSON.

    :param n_grams: Список частеречных тегов текста.
    :param lemmas: Список лемматизированных слов текста.
    :param n_values: Размеры n-грамм, которые нужно
     вычислить (по умолчанию (1, 2, 3)).
    :param show_analysis: Флаг для отображения анализа
    (по умолчанию True).

    :return tuple: Строки JSON с абсолютными и
     нормализованными частотами униграмм, биграмм
     и триграмм.
    """
    if show_analysis:
        display_grammemes()
        wait_for_enter_to_analyze()
        print(Fore.GREEN + Style.BRIGHT +
              "\n                    "
              "ЧАСТОТЫ ЧАСТЕРЕЧНЫХ"
              " N-ГРАММ")
        print(Fore.LIGHTGREEN_EX +
              "По умолчанию в консоль "
              "выводятся n-граммы с "
              "абсолютной частотой >20.")
    # print(n_grams)
    # print(lemmas)
    # print('!!!!!')

    # Счётчики для запрошенных n
    counters = {n: Counter() for n in n_values}
    for n in n_values:
        if n >= 1:
            counters[n].update(ngrams(n_grams, n))

    def normalize(counter: Counter) -> dict:
        total = sum(counter.values()) or 1
        return {k: round((v / total) * 100, 3)
                for k, v in counter.items()}

    def to_sorted_json(counter: Counter) -> str:
        # сортируем по убыванию значений и
        # конвертируем ключи в строки
        items = sorted(counter.items(),
                       key=lambda kv: kv[1],
                       reverse=True)
        return json.dumps({str(k): v for k, v in items},
                          ensure_ascii=False)

    def to_sorted_json_from_dict(d: dict) -> str:
        items = sorted(d.items(),
                       key=lambda kv: kv[1],
                       reverse=True)
        return json.dumps({str(k): v for k, v in items},
                          ensure_ascii=False)

    # Готовим выходные значения только для тех n, что запрошены
    uni_c = counters.get(1, Counter())
    bi_c = counters.get(2, Counter())
    tri_c = counters.get(3, Counter())

    uni_abs_json = to_sorted_json(uni_c)
    uni_rel_json = to_sorted_json_from_dict(normalize(uni_c))

    bi_abs_json = to_sorted_json(bi_c)
    bi_rel_json = to_sorted_json_from_dict(normalize(bi_c))

    tri_abs_json = to_sorted_json(tri_c)
    tri_rel_json = to_sorted_json_from_dict(normalize(tri_c))

    if show_analysis:
        display_ngrams_summary(
            uni_abs_json, uni_rel_json, bi_abs_json,
            bi_rel_json, tri_abs_json, tri_rel_json
        )
    return (
        uni_abs_json, uni_rel_json,
        bi_abs_json, bi_rel_json,
        tri_abs_json, tri_rel_json,
    )


def character_ngrams(
        text,
        n_values=(1, 2, 3, 4, 5),
        show_analysis=True
):
    """
    Вычисляет символьные n-граммы для текста
     и возвращает результаты в виде строк JSON.

    :param text: Входной текст.
    :param n_values: Размеры n-грамм, которые нужно
    вычислить (по умолчанию (1, 2, 3, 4, 5)).
    :param show_analysis: Флаг для отображения
    анализа (по умолчанию True).

    :return tuple: Строки JSON с абсолютными и
    нормализованными частотами униграмм,
     биграмм и триграмм.
    """
    if show_analysis:
        print(Fore.GREEN + Style.BRIGHT +
              "\n                      "
              "ЧАСТОТЫ БУКВЕННЫХ N-ГРАММ")
        print(
            Fore.LIGHTRED_EX + Style.BRIGHT +
            "Внимание! N-граммы '<' и '>' используются "
            "для обозначения начала и конца \nслов"
            " соответственно.\n")
        wait_for_enter_to_analyze()

    # Оставляем только буквы и пробелы, нижний регистр
    text = re.sub(r'[^а-яА-ЯёЁ\s]', '', text).lower()

    # Разделение текста на слова и добавление специальных
    # токенов начала и конца слова
    words = text.split()
    processed_text = ''.join(f'<{w}>' for w in words)

    # Подсчёты для запрошенных n
    counters = {n: Counter() for n in n_values}
    for n in n_values:
        if n >= 1:
            counters[n].update(ngrams(processed_text, n))

    total_len = len(processed_text)

    def to_sorted_json(
            counter: Counter
    ) -> str:
        items = sorted(counter.items(),
                       key=lambda kv: kv[1],
                       reverse=True)
        return json.dumps({str(k): v for k, v in items},
                          ensure_ascii=False)

    def norm(
            counter: Counter, n: int
    ) -> dict:
        all_pos = max(total_len - n + 1, 0) or 1
        return {k: round((v / all_pos) * 100, 3)
                for k, v in counter.items()}

    def to_sorted_json_from_dict(d: dict) -> str:
        items = sorted(d.items(),
                       key=lambda kv: kv[1],
                       reverse=True)
        return json.dumps({str(k): v for k, v in items},
                          ensure_ascii=False)

    # Достаём счётчики по умолчанию
    uni_c = counters.get(1, Counter())
    bi_c = counters.get(2, Counter())
    tri_c = counters.get(3, Counter())
    four_c = counters.get(4, Counter())
    five_c = counters.get(5, Counter())

    # Абсолютные
    unigram_counts = to_sorted_json(uni_c)
    bigram_counts = to_sorted_json(bi_c)
    trigram_counts = to_sorted_json(tri_c)
    fourgram_counts = to_sorted_json(four_c)
    fivegram_counts = to_sorted_json(five_c)

    # Нормализованные (в %)
    unigram_normalized_frequencies = (
        to_sorted_json_from_dict(norm(uni_c, 1)))
    bigram_normalized_frequencies = (
        to_sorted_json_from_dict(norm(bi_c, 2)))
    trigram_normalized_frequencies = (
        to_sorted_json_from_dict(norm(tri_c, 3)))
    fourgram_normalized_frequencies = (
        to_sorted_json_from_dict(norm(four_c, 4)))
    fivegram_normalized_frequencies = (
        to_sorted_json_from_dict(norm(five_c, 5)))

    if show_analysis:
        display_ngrams_summary(
            unigram_counts, unigram_normalized_frequencies,
            bigram_counts, bigram_normalized_frequencies,
            trigram_counts, trigram_normalized_frequencies,
            fourgram_counts, fourgram_normalized_frequencies,
            fivegram_counts, fivegram_normalized_frequencies
        )
    return (
        unigram_counts,
        unigram_normalized_frequencies,
        bigram_counts,
        bigram_normalized_frequencies,
        trigram_counts,
        trigram_normalized_frequencies,
        fourgram_counts,
        fourgram_normalized_frequencies,
        fivegram_counts,
        fivegram_normalized_frequencies
    )


def display_ngrams(
        title,
        ngram_counts,
        normalized_counts,
        min_frequency=20
):
    """
    Отображает таблицы с частотами n-грамм,
    фильтруя по минимальной частоте.

    :param title: Заголовок таблицы.
    :param ngram_counts: JSON-строка с абсолютными
    частотами n-грамм.
    :param normalized_counts: JSON-строка с
    нормализованными частотами n-грамм.
    :param min_frequency: Минимальная частота
    для отображения (по умолчанию 20).
    """
    # Нечего показывать — вызывают с необязательными слотами (None)
    if ngram_counts is None or normalized_counts is None:
        return

    ngram_counts = json.loads(ngram_counts)
    normalized_counts = json.loads(normalized_counts)
    print(Fore.GREEN + Style.BRIGHT +
          f"                          "
          f"{title.upper()}")
    wait_for_enter_to_analyze()
    table = Table()

    table.add_column("N-грамма\n", justify="center",
                     no_wrap=True)
    table.add_column("Абсолютная частота\n",
                     justify="center")
    table.add_column("Нормализованная частота \n(%)",
                     justify="center")

    for ngram, count in ngram_counts.items():
        if count >= min_frequency:
            normalized_freq = normalized_counts.get(ngram, 0)
            table.add_row(ngram,
                          str(count),
                          str(normalized_freq) + "%")

    if len(table.rows) > 0:
        console.print(table)
        wait_for_enter_to_analyze()
    else:
        print(Fore.LIGHTRED_EX +
              f"Отсутствуют {title.lower()} "
              f"n-граммы с частотой > "
              f"или = {min_frequency}.")
        wait_for_enter_to_analyze()


def display_ngrams_summary(
        unigram_counts, unigram_freq, bigram_counts,
        bigram_freq, trigram_counts, trigram_freq,
        fourgram_counts=None, fourgram_freq=None,
        fivegram_counts=None, fivegram_freq=None,
):
    """
    Запрашивает минимальную частоту и отображает
    результаты для униграмм, биграмм и триграмм.

    :param unigram_counts: JSON-строка с абсолютными
     частотами униграмм.
    :param unigram_freq: JSON-строка с нормализованными
    частотами униграмм.
    :param bigram_counts: JSON-строка с абсолютными
    частотами биграмм.
    :param bigram_freq: JSON-строка с нормализованными
    частотами биграмм.
    :param trigram_counts: JSON-строка с абсолютными
    частотами триграмм.
    :param trigram_freq: JSON-строка с нормализованными
    частотами триграмм.
    """
    print(
        Fore.LIGHTGREEN_EX + Style.BRIGHT +
        "Введите интересующую Вас минимальную "
        "частоту n-грамм или просто нажмите "
        "'Enter' \n(по умолчанию значение=20):")
    while True:
        min_frequency = input()
        if not min_frequency:
            min_frequency = 20
            break
        try:
            min_frequency = int(min_frequency)
            if min_frequency > 0:
                break
            else:
                print(Fore.LIGHTRED_EX +
                      "Ошибка: Введите "
                      "значение больше "
                      "нуля.\n")
        except ValueError:
            print(Fore.LIGHTRED_EX +
                  "Ошибка: Введите "
                  "числовое значение "
                  "или нажмите "
                  "'Enter'.\n")
    display_ngrams("Униграммы", unigram_counts, unigram_freq, min_frequency)
    display_ngrams("Биграммы",  bigram_counts,  bigram_freq,  min_frequency)
    display_ngrams("Триграммы", trigram_counts, trigram_freq, min_frequency)

    # Печатать 4- и 5-граммы только если они переданы
    if fourgram_counts is not None and fourgram_freq is not None:
        display_ngrams("Четырехграммы",
                       fourgram_counts,
                       fourgram_freq,
                       min_frequency)
    if fivegram_counts is not None and fivegram_freq is not None:
        display_ngrams("Пятиграммы",
                       fivegram_counts,
                       fivegram_freq,
                       min_frequency)


if __name__ == "__main__":
    # Текст для примера (сгенерирован ИИ)
    text = """
    Космонавт Алексей всегда мечтал о звёздах. С детства он читал книги о космосе и представлял себя на
    борту космического корабля. После долгих лет учёбы и тренировок, его мечта стала реальностью. В 2024 году Алексей
    был выбран для участия в международной миссии на Марс. Экипаж состоял из учёных и инженеров разных стран,
    и все они работали как единое целое. Путешествие длилось шесть месяцев, и каждый день приносил новые вызовы. На
    борту Алексей отвечал за поддержание систем жизнеобеспечения. Технологии, используемые в полёте,
    были новаторскими и требовали постоянного контроля. В свободное время он смотрел в иллюминатор на бесконечный
    космос, размышляя о своём месте во Вселенной. По прибытию на Марс, команда начала исследования поверхности
    планеты. Алексей был первым человеком, ступившим на красную пыль марсианской пустыни. Он взял пробы грунта и
    отправил их на анализ в корабль. Возвращение на Землю прошло успешно, и Алексей стал национальным героем. Его
    истории вдохновляли новое поколение детей мечтать о космосе. Алексей продолжил работать в космической программе,
    передавая свой опыт молодым космонавтам. В каждом его слове чувствовалась страсть к исследованиям и вера в
    будущее человечества среди звёзд.
    """
    from colorama import init

    from tools.core.natasha_pymorphy_pos_tagger import (
        pos_tagger
    )
    init(autoreset=True)

    n_grams, lemmas = pos_tagger(text)

    # (unigram_counts, unigram_normalized_frequencies,
    #  bigram_counts, bigram_normalized_frequencies,
    #  trigram_counts, trigram_normalized_frequencies,
    #  fourgram_counts, fourgram_normalized_frequencies,
    #  fivegram_counts, fivegram_normalized_frequencies
    #  ) = character_ngrams(
    #     text, show_analysis=True)

    (unigram_counts, unigram_normalized_frequencies,
     bigram_counts, bigram_normalized_frequencies,
     trigram_counts, trigram_normalized_frequencies) = pos_ngrams(
        n_grams, lemmas, show_analysis=True)
