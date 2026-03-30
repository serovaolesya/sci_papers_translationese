# -*- coding: utf-8 -*-  # Языковая кодировка UTF-8
import json

from colorama import Fore, Style
from rich.console import Console
from rich.table import Table

from tools.core.utils import (
    wait_for_enter_to_analyze, display_position_explanation
)

console = Console()


FINAL_PUNCT_MAP = {
    '.': 'FULL_STOP',
    '!': 'EXCLM_MARK',
    '?': 'Q_MARK',
    '…': 'ELLIPSIS'
}


def extract_positions(
        pos_tags_seq,
        tokens_seq,
        show_analysis: bool = True
):
    """
    Извлекает позиции токенов и их POS из уже
    размеченных последовательностей.

    Аргументы:
        pos_tags_seq (list[str]):
        список POS/маркерных тегов той же длины, что и tokens_seq.
        tokens_seq   (list[str]):
        список токенов той же длины (на местах 'S_START' — такой же маркер,
        на местах 'S_END' — финальный пунктуационный токен предложения).
        show_analysis (bool):
        печатать ли таблицу с позициями (по умолчанию True).

    Правила:
      • Предложение начинается на 'S_START'
       и заканчивается на 'S_END'.
      • В предложение НЕ входит маркер 'S_START'.
      • Финальный токен, идущий рядом с 'S_END',
      ВХОДИТ в предложение.
      • В вывод попадают только предложения длиной ≥ 5
      токенов (включая финальный знак).
      • Для каждой подходящей фразы выводятся позиции:
      first, second, antepenultimate, penultimate, last.

    Возвращает:
        str: JSON-строка с массивом объектов по предложениям,
        где у каждой позиции хранится пара (token, pos_tag).
    """
    if (not isinstance(pos_tags_seq, list) or
            not isinstance(tokens_seq, list)):
        raise TypeError("extract_positions ожидает "
                        "два списка: (pos_tags_seq,"
                        " tokens_seq)")

    if len(pos_tags_seq) != len(tokens_seq):
        raise ValueError(
            f"Длины не совпадают: "
            f"POS={len(pos_tags_seq)} и "
            f"Tokens={len(tokens_seq)}. "
            "Списки должны быть синхронны "
            "и одинаковой длины."
        )

    token_positions_in_sent = []
    inside = False
    sent_tokens: list[str] = []
    sent_pos: list[str] = []
    sent_idx = 0

    for tag, tok in zip(pos_tags_seq, tokens_seq):
        # Открываем новое предложение —
        # сам 'S_START' не включаем
        if tag == 'S_START':
            inside = True
            sent_tokens = []
            sent_pos = []
            continue

        # Закрываем предложение — ВКЛЮЧАЕМ
        # финальный токен и тег 'S_END'
        if tag == 'S_END':
            if inside:
                sent_tokens.append(tok)
                sent_pos.append(
                    FINAL_PUNCT_MAP.get(tok, 'FINAL_PUNCT')
                )
                if len(sent_tokens) >= 5:
                    sent_idx += 1
                    positions_with_pos = {
                        'first': (sent_tokens[0],
                                  sent_pos[0]),
                        'second': (sent_tokens[1],
                                   sent_pos[1]),
                        'antepenultimate': (sent_tokens[-3],
                                            sent_pos[-3]),
                        'penultimate': (sent_tokens[-2],
                                        sent_pos[-2]),
                        'last': (sent_tokens[-1],
                                 sent_pos[-1]),
                    }
                    token_positions_in_sent.append(
                        {'SENTENCE_NUMBER': sent_idx,
                         'positions': positions_with_pos}
                    )
                inside = False
                sent_tokens = []
                sent_pos = []
            continue

        # Обычный токен внутри предложения
        if inside:
            sent_tokens.append(tok)
            sent_pos.append(tag)

    positions_contexts = json.dumps(
        token_positions_in_sent,
        ensure_ascii=False
    )
    if show_analysis:
        print_positions(positions_contexts)
    return positions_contexts


def print_positions(sentence_data):
    """
    Выводит позиционные контексты слов
     и их частей речи.

    :param sentence_data:
    str - JSON-строка с данными о позициях
    слов, их частях речи.
    """
    sentence_data = json.loads(sentence_data)

    print(
        Fore.GREEN + Style.BRIGHT +
        "\n             ТОКЕНЫ И ИХ "
        " ЧАСТИ РЕЧИ НА РАЗНЫХ ПОЗИЦИЯХ "
        "В ПРЕДЛОЖЕНИИ"
    )
    print(
        Fore.RED +
        "Внимание! Выводятся "
        "контексты предложений,"
        " длиннее 5 токенов.\n")

    display_position_explanation()

    table = Table()

    table.add_column("Sent", justify="center",
                     style="bold", no_wrap=True, width=21)
    table.add_column("First", justify="center",
                     no_wrap=True, width=30)
    table.add_column("Second", justify="center",
                     no_wrap=True, width=30)
    table.add_column("Antepenultimate",
                     justify="center",
                     no_wrap=True, width=30)
    table.add_column("Penultimate",
                     justify="center",
                     no_wrap=True, width=30)
    table.add_column("Last", justify="center",
                     no_wrap=True, width=30)

    for data in sentence_data:
        sentence_number = data['SENTENCE_NUMBER']
        positions = data['positions']

        tokens_row = [f"{sentence_number}"]
        # Пустая ячейка для выравнивания
        # под номером предложения
        pos_row = [""]

        for position in ['first', 'second',
                         'antepenultimate',
                         'penultimate', 'last']:
            token, pos = positions[position]
            tokens_row.append(token)
            pos_row.append(pos)

        # Добавляем строки в таблицу:
        # первая строка — токены,
        # вторая строка — части речи
        table.add_row(*tokens_row)
        table.add_row(*pos_row,
                      end_section=True)

    console.print(table)
    wait_for_enter_to_analyze()


if __name__ == "__main__":
    text = """
    Осенний ветер за окном напоминал о скором приходе холодов. Листья деревьев медленно кружились в воздухе, постепенно 
    покрывая землю золотым ковром. В парке гуляли немногочисленные прохожие, наслаждаясь последними тёплыми днями. Вдоль 
    аллеи бежала собака, радостно виляя хвостом. Маленький мальчик с интересом наблюдал за ней, крепко держа за руку свою 
    маму. Она говорила ему о том, как важно сохранять природу и уважать окружающий мир. Вдалеке был виден силуэт 
    человека, сидящего на лавочке с книгой. Он не спешил никуда, погружённый в чтение. Вокруг царила атмосфера 
    умиротворённости и спокойствия. Солнце постепенно уходило за горизонт, окутывая парк мягким оранжевым светом. Небо 
    меняло свой цвет, переходя от светло-голубого к насыщенному розовому. Птицы готовились к ночи, прячась в ветвях 
    деревьев. Где-то рядом слышался тихий плеск воды из фонтана. Люди начинали расходиться по домам, постепенно покидая 
    парк. И вот, когда город погрузился в вечерние сумерки, наступила долгожданная тишина…
    """

    from colorama import init

    from tools.core.natasha_pymorphy_pos_tagger import (
        pos_tagger
    )

    init(autoreset=True)

    n_grams, lemmas = pos_tagger(text)

    sentence_data = extract_positions(n_grams, lemmas)
    print(sentence_data)
