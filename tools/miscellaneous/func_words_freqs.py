# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import json
import re

from collections import defaultdict

import pymorphy2
from colorama import Fore, Style
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rich.console import Console
from rich.table import Table

from tools.core.data import pronouns, prepositions, particles, conjunctions
from tools.core.lemmatizators import parse_cached
from tools.core.utils import wait_for_enter_to_analyze

console = Console()

# Загрузка стоп-слов
nltk_stopwords_ru = stopwords.words("russian")

# 1. Объединение всех списков в один
all_stopwords = set(conjunctions.conjunctions_list + prepositions.prepositions_list
                    + particles.particles_list + pronouns.pronouns_list + nltk_stopwords_ru)

# 2. Сортировка по длине по убыванию
all_stopwords_sorted = sorted(list(all_stopwords), key=len, reverse=True)

morph = pymorphy2.MorphAnalyzer()


def compute_function_word_frequencies(text, show_analysis=True):
    """
    Рассчитывает нормализованные частоты функциональных слов в тексте и их абсолютные частоты.

    :param text: str - Входной текст для анализа.
    :param show_analysis: Bool - Флаг для отображения анализа (по умолчанию True).

    :return: tuple - Кортеж из двух элементов:
        - str: JSON-строка с нормализованными частотами функциональных слов.
        - str: JSON-строка с абсолютными частотами функциональных слов.
    """
    text = re.sub(r'[^а-яА-ЯËёa-zA-Z\-]', ' ', text)
    # Замена многотокенных стоп-слов на уникальные маркеры
    for stopword in all_stopwords_sorted:
        stopword_tokens = stopword.split()
        if len(stopword_tokens) > 1:  # Если стоп-слово состоит из нескольких токенов
            pattern = r'\b' + r'\s+'.join(map(re.escape, stopword_tokens)) + r'\b'
            text = re.sub(pattern, stopword.replace(' ', '_'), text)

    function_word_counts = defaultdict(int)

    tokens = word_tokenize(text.lower(), language="russian")
    total_tokens = len(tokens)

    for token in tokens:
        # Учитываем, что составные стоп-слова уже заменены на уникальные маркеры
        token = re.sub(r'_', ' ', token)
        token = parse_cached(token)
        lemmatized_token = token.normal_form
        if lemmatized_token in all_stopwords_sorted:
            function_word_counts[lemmatized_token] += 1

    normalized_frequencies = {word: round((count / total_tokens) * 100, 3) for word, count in
                              function_word_counts.items()}

    sorted_frequencies = dict(sorted(normalized_frequencies.items(), key=lambda item: item[1], reverse=True))
    freqs = json.dumps(sorted_frequencies, ensure_ascii=False)

    sorted_counts = dict(sorted(function_word_counts.items(), key=lambda item: item[1], reverse=True))
    counts = json.dumps(sorted_counts, ensure_ascii=False)

    if show_analysis:
        print_func_w__frequencies(freqs, counts)
        wait_for_enter_to_analyze()

    return freqs, counts


def print_func_w__frequencies(frequencies, counts, for_corpus=False):
    """
    Печатает частоты функциональных слов в виде таблицы.

    :param frequencies: str - JSON-строка с нормализованными частотами функциональных слов.
    :param counts: str - JSON-строка с абсолютными частотами функциональных слов.
    :param for_corpus: Bool - Флаг для отображения данных для корпуса (по умолчанию False).
    """
    if not for_corpus:
        frequencies = json.loads(frequencies)
        counts = json.loads(counts)

    print(Fore.GREEN + Style.BRIGHT + "\n                     ЧАСТОТЫ ФУНКЦИОНАЛЬНЫХ СЛОВ" + Fore.RESET)
    wait_for_enter_to_analyze()
    table = Table()
    table.add_column("Токен\n", justify="center", style="bold", min_width=20)
    table.add_column("Абсолютная частота\n", justify="center", min_width=20)
    table.add_column("Нормализованная частота\n(%)", justify="center", min_width=20)

    for token in frequencies.keys():
        count = counts[token]
        frequency = frequencies[token]
        table.add_row(token, str(count), f"{frequency:.3f}%")

    console.print(table)


# Пример использования
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
    compute_function_word_frequencies(text)
