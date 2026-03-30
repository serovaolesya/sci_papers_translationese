# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import json
from collections import defaultdict

from colorama import Fore, Style, init
from rich.console import Console
from rich.table import Table

from tools.core.utils import wait_for_enter_to_analyze

console = Console()
init(autoreset=True)


def calculate_repetition(
        total_tokens_count: int,
        content_word_counts_json: str,
        show_analysis=True
):
    """
    Рассчитывает повторяемость знаменательных слов по формуле:
    число типов с частотой ≥ 2 делится на общее число словарных
    токенов в тексте.

    :param total_tokens_count: Количество словарных токенов в тексте.
    :param content_word_counts_json: Абсолютные
    частоты знаменательных слов — JSON-строка
    :param show_analysis: bool, Если True,
    промежуточные результаты и таблицы будут отображены.
    :return: tuple (float, str, int, int)
        - repetition: float, Процент повторяемости.
        - sorted_content_word_counts_json: str, JSON-строка
         с количеством вхождений знаменательных слов.
        - repeated_types_count: int, Число типов знаменательных
        слов с частотой ≥ 2.
        - total_word_tokens_count: int, Общее количество
        словарных токенов.
    """
    if show_analysis:
        print(Fore.GREEN + Style.BRIGHT +
              "\n                "
              "ПОВТОРЯЕМОСТЬ ЗНАМЕНАТЕЛЬНЫХ СЛОВ")
        print(
            Fore.LIGHTGREEN_EX +
            "В зависимости от размера текста "
            "подсчет повторяемости может \nзанять какое-то время. "
            "Пожалуйста, будьте готовы подождать.\n")
        print(
            Fore.RED +
            "Внимание! При подсчете повторяемости учитывается повторяемость "
            "\nтолько знаменательных слов. Служебные слова, включая составные"
            "\n(например, предлог `в связи с`), из подсчета исключаются."
            "\nТо есть слово `связь` в данном случае не будет считаться"
            "\nзнаменательным.\n")

        wait_for_enter_to_analyze()

    content_word_counts: dict[str, int] = (
        json.loads(content_word_counts_json)
    )
    # Подсчитываем число типов со встречаемостью ≥ 2
    repeated_types_count = sum(
        1 for count in content_word_counts.values() if count >= 2
    )

    if total_tokens_count > 0:
        repetition = round(
            repeated_types_count / total_tokens_count * 100, 3
        )
        if show_analysis:
            table = Table()

            table.add_column("Параметр", justify="left",
                             no_wrap=True, min_width=30, style="bold")
            table.add_column("Значение", justify="center", min_width=10)

            table.add_row("Повторяемость", str(repetition) + '%')
            table.add_row("Кол-во типов с частотой ≥ 2",
                          str(repeated_types_count))
            table.add_row("Общее количество словарных токенов",
                          str(total_tokens_count))

            console.print(table)
            wait_for_enter_to_analyze()

        # Сортируем типы знаменательных
        # слов по убыванию частоты.
        sorted_content_word_counts = dict(sorted(
            content_word_counts.items(),
            key=lambda item: item[1], reverse=True))

        sorted_content_word_counts_json = (
            json.dumps(sorted_content_word_counts,
                       ensure_ascii=False, indent=4))
        if show_analysis:
            print_word_occurrences_table(
                sorted_content_word_counts_json
            )

        return (
            repetition, sorted_content_word_counts_json,
            repeated_types_count, total_tokens_count
        )

    else:
        if show_analysis:
            print(Fore.LIGHTRED_EX +
                  "\nВо введенном тексте"
                  " отсутствуют буквенные "
                  "токены.")

        return 0, json.dumps({}, ensure_ascii=False, indent=4), 0, 0


def print_word_occurrences_table(
        sorted_content_word_counts_json,
        min_occurrences=2
):
    """
    Выводит таблицу с количеством вхождений знаменательных слов,
    начиная с заданного минимального значения.

    :param sorted_content_word_counts_json: str, JSON-строка
    с количеством вхождений знаменательных слов,
    :param min_occurrences: int, Минимальное количество вхождений
     для отображения в таблице. Значение по умолчанию 2.
    """
    # Преобразование строки JSON обратно в словарь
    print(Fore.GREEN + Style.BRIGHT +
          "\n              КОЛИЧЕСТВА "
          "ВХОЖДЕНИЙ ЗНАМЕНАТЕЛЬНЫХ СЛОВ")
    while True:
        try:
            user_input = input(
                Fore.BLUE +
                "Введите минимальное значение "
                "интересующей Вас частоты"
                " или нажмите \n'Enter', "
                "чтобы пропустить выбор "
                "(по умолчанию значение=2):"
            ).strip()

            if not user_input:
                # Значение по умолчанию, если
                # пользователь ничего не ввел
                break
            min_occurrences = int(user_input)
            if min_occurrences > 0:
                break
            else:
                print(Fore.LIGHTRED_EX +
                      "\nМинимальное количество вхождений "
                      "должно быть больше нуля.\n")
        except ValueError:
            print(Fore.LIGHTRED_EX +
                  "\nПожалуйста, введите целое число.")

    try:
        sorted_content_word_counts = (
            json.loads(sorted_content_word_counts_json)
        )
    except json.JSONDecodeError as e:
        print(Fore.LIGHTRED_EX + f"Ошибка парсинга JSON: {e}")
        return

    word_table = Table()

    word_table.add_column("Абс. частота", style="bold",
                          justify="center", max_width=7)
    word_table.add_column("Знаменательные слова",  max_width=50)

    grouped_word_counts = defaultdict(list)
    for word, count in sorted_content_word_counts.items():
        if count >= min_occurrences:
            # Кладём лемму в список, соответствующий её частоте.
            grouped_word_counts[count].append(word)

    if not grouped_word_counts:
        print(
            Fore.LIGHTRED_EX +
            "\nНет слов с количеством вхождений"
            " > или = указанному минимуму.")
        return

    for count, words in sorted(grouped_word_counts.items(), reverse=True):
        word_table.add_row(str(count), ", ".join(words))

    console.print(word_table)
    wait_for_enter_to_analyze()


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
    import re

    from tools.core.utils import count_types_in_text

    total_alpha_tokens_count = len(re.findall(r'\b\w+\b', text))
    json_content_word_counts = (
        count_types_in_text(text.lower())
    )

    (repetition, content_word_count,
     repeated_words, total_tokens) = (
        calculate_repetition(
            total_alpha_tokens_count,
            json_content_word_counts
        )
    )
    # print(repetition)
    # print(content_word_count)
    # print(repeated_words)
    # print(total_tokens)
