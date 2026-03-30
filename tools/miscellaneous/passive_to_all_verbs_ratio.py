# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import re

from colorama import Fore, Style
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    Doc,
)
from pymorphy2 import MorphAnalyzer
from rich.console import Console
from rich.table import Table

from tools.core.utils import wait_for_enter_to_analyze

console = Console()

morph = MorphAnalyzer()

patterns = r"[^А-Яа-яёЁ\-]+"  # Оставляем только кириллицу и дефис


def calculate_passive_verbs_ratio(text, show_analysis=True,
                                   segmenter=None, morph_tagger=None):
    """
    Подсчитывает соотношение глаголов в пассивном залоге ко всем глаголам в тексте.

    :param text: str - Текст для анализа.
    :param show_analysis: Bool - Флаг для отображения анализа (по умолчанию True).
    :param segmenter: Segmenter - внешний сегментатор (если None — создаётся внутри).
    :param morph_tagger: NewsMorphTagger - внешний теггер (если None — создаётся внутри).

    :return: tuple - Кортеж из пяти элементов:
        - float: Соотношение пассивных глаголов к общему количеству глаголов в процентах.
        - str: Список пассивных глаголов в виде строки.
        - int: Количество пассивных глаголов.
        - str: Список всех глаголов в виде строки.
        - int: Общее количество глаголов.
    """
    if segmenter is None:
        segmenter = Segmenter()
    if morph_tagger is None:
        _emb = NewsEmbedding()
        morph_tagger = NewsMorphTagger(_emb)

    passive_verbs_count = 0
    passive_verbs_list = []
    all_verbs_count = 0
    all_verbs_list = []

    text = re.sub(patterns, ' ', text)
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        natasha_pos = token.pos
        token_pymorphy_info = morph.parse(token.text)[0]
        if (token_pymorphy_info.tag.POS in {
            'VERB', 'INFN', 'PRTF', 'PRTS', 'GRND'
        } and natasha_pos != 'AUX'):  # Если это глагол/ его форма
            all_verbs_count += 1
            all_verbs_list.append(token_pymorphy_info.word)
            # Определяем залог
            voice_py = token_pymorphy_info.tag.voice

            if voice_py == 'pssv' or (token.feats and token.feats.get('Voice') == 'Pass'):
                passive_verbs_count += 1
                passive_verbs_list.append(token_pymorphy_info.word)

    if all_verbs_count > 0:
        passive_to_all_v_ratio = round((passive_verbs_count / all_verbs_count) * 100, 3)
    else:
        passive_to_all_v_ratio = 0
    passive_verbs_str = str(passive_verbs_list)
    all_verbs_str = str(all_verbs_list)

    if show_analysis:
        print_passive_verbs_ratio(passive_to_all_v_ratio, passive_verbs_str, passive_verbs_count, all_verbs_str,
                                  all_verbs_count)
        wait_for_enter_to_analyze()
    return passive_to_all_v_ratio, passive_verbs_str, passive_verbs_count, all_verbs_str, all_verbs_count


def print_passive_verbs_ratio(ratio, passive_verbs_str, passive_verbs_count, all_verbs_str, all_verbs_count):
    """
    Печатает соотношение пассивных глаголов к общему количеству глаголов и списки глаголов.

    :param ratio: float - Соотношение пассивных глаголов к общему количеству глаголов в процентах.
    :param passive_verbs_str: str - Список пассивных глаголов в виде строки.
    :param passive_verbs_count: int - Количество пассивных глаголов.
    :param all_verbs_str: str - Список всех глаголов в виде строки.
    :param all_verbs_count: int - Общее количество глаголов.
    """
    print(
        Fore.GREEN + Style.BRIGHT + "\n      СООТНОШЕНИЕ ПАССИВНЫХ ГЛАГОЛОВ КО ВСЕМ ГЛАГОЛАМ В ТЕКСТЕ" + Fore.RESET)
    wait_for_enter_to_analyze()
    table = Table()
    table.add_column("Показатель", justify="left", style="bold")
    table.add_column("Значение", justify="left")

    table.add_row("Соотношение (%)", f"{ratio:.2f}%")
    table.add_row("Количество пассивных глаголов", str(passive_verbs_count))
    table.add_row("Всего глаголов", str(all_verbs_count))
    table.add_row("Список пассивных глаголов", passive_verbs_str)
    table.add_row("Список всех глаголов", all_verbs_str)

    console.print(table)


if __name__ == "__main__":
    # Текст для примера (сгенерирован ИИ)
    text = """В старом доме на окраине города жила семья, которая была известна своей, гостеприимностью. Дом был 
    построен много лет назад и был окружен большим садом, который был засажен цветами и деревьями. Семья была любима 
    всеми соседями, и к ней часто приходили гости. Семья была большая, и в ней было много детей, которые всегда 
    играли во дворе. Дети были веселыми и любопытными, и они всегда находили что-то интересное, чтобы сделать. Они 
    были окружены любящими родителями, которые всегда заботились о них. Отец семьи был известным художником, 
    и он часто писал картины, которые были выставлены в местных галереях. Мать была отличной поварихой, и она всегда 
    готовила вкусные блюда, которые были любимы всеми. Дети были учениками местной школы, и они всегда получали 
    хорошие оценки. В доме часто проводились вечеринки, на которые приходили друзья и соседи. Вечеринки были всегда 
    веселыми, и все гости всегда уходили с улыбками на лицах. Семья была счастлива и гармонична, и все члены семьи 
    любили друг друга. Дом был полон книг, и все члены семьи любили читать. Они часто сидели в библиотеке и читали 
    книги, которые были написаны известными авторами. Семья была любима всеми, и к ней всегда приходили гости."""

    calculate_passive_verbs_ratio(text)
