# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
from colorama import Fore, Style, init
from rich.console import Console
from rich.table import Table

from tools.core.data.lang_word_frequencies import words_by_freq
from tools.core.utils import wait_for_enter_to_analyze
from tools.core.lemmatizators import lemmatize_words

console = Console()
init(autoreset=True)

DEFAULT_RANK = 6000


def calculate_mean_word_rank(
        lemmatized_words: list,
        show_analysis=True
):
    """
    Рассчитывает средний ранг слов в тексте двумя
    способами и выводит результаты в виде таблицы.

    :param lemmatized_words: Список лемм текста
    :param show_analysis: Если True, выводит
    результат и пояснение в виде таблицы.
    :return: Кортеж из двух значений:
             - Средний ранг слов (метрика 1), где слова,
             не найденные в частотном списке, получают ранг 6000.
             - Средний ранг слов (метрика 2), где слова,
             не найденные в частотном списке, игнорируются.
    """
    total_rank_1 = 0
    total_rank_2 = 0
    word_count_1 = 0
    word_count_2 = 0

    for word in lemmatized_words:
        rank = words_by_freq.get(word)
        if rank:
            total_rank_1 += rank
            total_rank_2 += rank
            word_count_2 += 1
        else:
            total_rank_1 += DEFAULT_RANK
        word_count_1 += 1

    if word_count_1 == 0:
        if show_analysis:
            print(
                Fore.RED + Style.BRIGHT +
                "Текст не содержит лексических"
                " единиц для анализа."
            )
            wait_for_enter_to_analyze()
        return 0, 0

    mean_word_rank_1 = int(
        round(total_rank_1 / word_count_1)
    ) if word_count_1 else 0

    mean_word_rank_2 = int(
        round(total_rank_2 / word_count_2)
    ) if word_count_2 else 0

    if show_analysis:
        print(Fore.GREEN + Style.BRIGHT +
              "\n       СРЕДНИЕ РАНГИ СЛОВ ТЕКСТА")
        table = Table()
        table.add_column("Параметр", justify="left",
                         no_wrap=True, min_width=30, style="bold")
        table.add_column("Значение", justify="center", min_width=10)

        table.add_row("Средний ранг (1)", str(mean_word_rank_1))
        table.add_row("Средний ранг (2)", str(mean_word_rank_2))
        console.print(table)

        print(Fore.RED + Style.BRIGHT + " ПОЯСНЕНИЕ:" + Fore.RESET)
        print(Fore.BLUE + Style.BRIGHT +
              " Средний ранг слов рассчитывается двумя способами:"
              + Fore.RESET)
        print(
            "  1." + Fore.BLUE + "mean_word_rank_1" + Fore.RESET +
            ": Все слова, которых нет в частотном\n    списке слов РЯ, "
            "получают высший ранг 6000.\n" +
            "  2." + Fore.BLUE + "mean_word_rank_2" + Fore.RESET +
            ": Все слова, которых нет в частотном\n    списке слов РЯ, "
            "игнорируются при подсчете.\n"
        )
        wait_for_enter_to_analyze()

    return mean_word_rank_1, mean_word_rank_2


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

    parsed_text = lemmatize_words(text)
    text_lemmas = [token.normal_form for token in parsed_text]
    calculate_mean_word_rank(text_lemmas)
