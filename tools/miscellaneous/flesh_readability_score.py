# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import re

from colorama import Fore, Style
from nltk.tokenize import word_tokenize
from rich.console import Console
from rich.table import Table

from tools.core.custom_punkt_tokenizer import (
    sent_tokenize_with_abbr
)
from tools.core.utils import wait_for_enter_to_analyze

console = Console()


def mean_sentence_length_in_tokens(text):
    """
    Рассчитывает среднюю длину предложения в токенах.

    :param text: str - Входной текст для анализа.
    :return: float - Средняя длина предложения в
    токенах, округленная до трех знаков после запятой.
    """
    sentence_lengths = []
    sentences = sent_tokenize_with_abbr(text)

    for sentence in sentences:
        sentence = re.sub(r'[^а-яА-ЯёЁ\s]', '', sentence)
        tokens = word_tokenize(sentence, language="russian")
        sentence_lengths.append(len(tokens))

    mean_sent_length = round(
        (sum(sentence_lengths) / len(sentence_lengths)
         ), 3) if sentence_lengths else 0
    return mean_sent_length


def flesh_readability_index_for_rus(
        text,
        aver_syll_per_word,
        show_analysis=True
):
    """
    Рассчитывает индекс удобочитаемости Флеша для РЯ.

    :param text: str - Входной текст для анализа.
    :param show_analysis:  Флаг для отображения
    анализа (по умолчанию True).

    :return: float - Индекс удобочитаемости
    текста, округленный до трех знаков
    после запятой.
    """
    # Получаем среднюю длину предложения (aver_sent_len)
    aver_sent_len = (
        mean_sentence_length_in_tokens(text)
    )


    # Рассчитываем индекс удобочитаемости
    # по формуле И. В. Оборневой
    flesh_idx = round(
        206.836 - 1.3 * aver_sent_len -
        60.1 * aver_syll_per_word, 3
    )
    if show_analysis:
        display_readability_index(flesh_idx)
        wait_for_enter_to_analyze()

    return flesh_idx


def display_readability_index(flesh_idx):
    """
    Отображает индекс удобочитаемости в виде таблицы.

    :param flesh_idx: float -
    Индекс удобочитаемости текста.
    """
    print(Fore.GREEN + Style.BRIGHT +
          "\n                   ИНДЕКС"
          " УДОБОЧИТАЕМОСТИ")
    table = Table()
    table.add_column("Показатель",
                     style="bold",
                     justify="center",
                     min_width=30)
    table.add_column("Значение",
                     justify="center",
                     min_width=20)

    table.add_row("Индекс удобочитаемости",
                  str(flesh_idx))
    console.print(table)

    print(
        Fore.LIGHTGREEN_EX + Style.BRIGHT +
        "Чем ниже значение индекса, "
        "тем более сложен текст для"
        " прочтения.\n"
    )


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
    from tools.simplification.mean_word_length import (
        mean_word_length_syllab
    )

    syllable_ratio, total_syllables_count = (
        mean_word_length_syllab(text, show_in_console=False)
    )

    readability_index = flesh_readability_index_for_rus(
        text, syllable_ratio
    )
