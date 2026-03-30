# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import json

from colorama import Fore, Style, init
from rich.console import Console
from rich.table import Table

from tools.core.utils import wait_for_enter_to_analyze

console = Console()
init(autoreset=False)


def find_n_most_frequent_words(
        total_tokens_count: int,
        content_word_counts_json: str,
        values=(5, 10, 50,),
        show_analysis=True
):
    """
    Выводит наиболее частотные слова в тексте с
    нормализованными частотами.

    :param total_tokens_count: Общее число токенов в
    тексте (для нормализации).
    :param content_word_counts_json: Абсолютные
    частоты знаменательных слов — JSON-строка
    :param values: Кортеж размеров топ-списков
     (По умолчанию: 5, 10, 50).
    :param show_analysis: Если True,
    выводит результаты в виде таблицы.
    :return: JSON-строка с нормализованными
    частотами знаменательных слов относительно всех
    словарных токенов в тексте (%), ключи — леммы.
    """
    content_word_counts: dict[str, int] = (
        json.loads(content_word_counts_json)
    )

    normalized_frequencies = {
        word: round((count / total_tokens_count) * 100, 3)
        for word, count in content_word_counts.items()
    }

    if show_analysis:
        for top_n in values:
            top_items = sorted(
                normalized_frequencies.items(),
                key=lambda item: item[1],
                reverse=True
            )[:top_n]

            print(Fore.GREEN + Style.BRIGHT +
                  f"\n {top_n} НАИБОЛЕЕ ЧАСТОТНЫХ"
                  f" ЗНАМЕНАТЕЛЬНЫХ СЛОВ")
            wait_for_enter_to_analyze()

            table = Table()
            table.add_column("№", style="bold",
                             justify="center")
            table.add_column("Слово\n", style="bold",
                             justify="center")
            table.add_column("Нормализованная\nчастота (%)",
                             justify="center")

            for idx, (word, freq) in enumerate(top_items, start=1):
                table.add_row(str(idx), word, f"{freq}%")

            console.print(table)
            wait_for_enter_to_analyze()

    sorted_normalized_frequencies = dict(
        sorted(
            normalized_frequencies.items(),
            # по частоте ↓, при равенстве — по алфавиту
            key=lambda kv: (-kv[1], kv[0])
        )
    )
    return json.dumps(
        sorted_normalized_frequencies, ensure_ascii=False, indent=4
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
    import re

    from tools.core.utils import count_types_in_text

    total_alpha_tokens_count = len(re.findall(r'\b\w+\b', text))

    json_content_word_counts = (
        count_types_in_text(text.lower())
    )
    a = find_n_most_frequent_words(
        total_alpha_tokens_count,
        json_content_word_counts
    )
    print(a)
