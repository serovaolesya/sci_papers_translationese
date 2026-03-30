# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
from colorama import Fore, Style, init
from rich.console import Console
from rich.table import Table

from tools.core.utils import wait_for_enter_to_analyze

console = Console()
init(autoreset=True)


def calculate_lexical_density(
        total_alpha_tokens_count: int,
        lemmatized_content_words: list,
        show_analysis=True
):
    """
    Рассчитывает лексическую плотность текста, определяя соотношение
    знаменательных слов (содержательных частей речи, а именно глаголов,
    существительных, прилагательных и наречий) к общему числу
    слов в тексте.

    :param total_alpha_tokens_count: Общее количество словарных токенов.
    :param lemmatized_content_words: Список лемм знаменательных слов.
    :param show_analysis: Boolean.
    :return float: лексическая плотность текста, выраженная в процентах.
    """
    if len(lemmatized_content_words) == 0:
        if show_analysis:
            print(Fore.GREEN + Style.BRIGHT +
                  "\n        ЛЕКСИЧЕСКАЯ ПЛОТНОСТЬ ТЕКСТА")
            print(Fore.LIGHTRED_EX +
                  "В тексте нет слов для анализа лексической плотности.")
        return 0
    lexical_density = round(
        len(lemmatized_content_words) / total_alpha_tokens_count * 100, 3
    )

    if show_analysis:
        print(Fore.GREEN + Style.BRIGHT +
              "\n        ЛЕКСИЧЕСКАЯ ПЛОТНОСТЬ ТЕКСТА")

        print(
            Fore.RED +
            "Внимание! При подсчете показателя лексической "
            "\nплотности знаменательными словами считаются имена "
            "\nсуществительные, прилагательные, глаголы и наречия.")

        table = Table()
        table.add_column("Параметр", justify="left",
                         no_wrap=True, min_width=30, style="bold")
        table.add_column("Значение", justify="center", min_width=10)

        table.add_row("Лексическая плотность (%)",
                      f"{lexical_density:.2f}")
        table.add_row("\nВсего слов в тексте",
                      '\n' + str(total_alpha_tokens_count))
        table.add_row("Знаменательных слов (на рус. яз)",
                      str(len(lemmatized_content_words)))

        console.print(table)
        wait_for_enter_to_analyze()

    return lexical_density


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
    парк. И вот, когда город погрузился в вечерние сумерки, наступила долгожданная тишина за вычетом на основании.
    """
    import re

    from tools.core.lemmatizators import lemmatize_words_without_stopwords

    parsed_text_without_stopwords, removed_stop_w_count = (
        lemmatize_words_without_stopwords(text)
    )
    # Список лемм знаменательных слов (Cyrillic и Latin)
    lemmatized_content_words = [
        token.normal_form for token in parsed_text_without_stopwords
    ]
    total_alpha_tokens_count = (
        len(re.findall(r'\b\w+\b', text))
    )
    calculate_lexical_density(
        total_alpha_tokens_count, lemmatized_content_words
    )

