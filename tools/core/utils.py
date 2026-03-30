import os

from colorama import init
from natasha import (Segmenter,
                     NewsEmbedding,
                     NewsMorphTagger,
                     NewsSyntaxParser,
                     Doc
                     )
from prettytable import PrettyTable
from rich.console import Console
from rich.table import Table

from tools.core.constants import (
    GRAMMEMES_MORPH_ANNOTATION_GRAM_CATEGORIES, GRAMMEMES_NGRAMS,
    GRAMMEMES_MORPH_ANNOTATION, NON_TRANSLATED_DB_NAME,
    MACHINE_TRANSLATED_DB_NAME, HUMAN_TRANSLATED_DB_NAME,
    CONTENT_POS, EXCLUDED_LEMMAS, INCORRECT_CHOICE
)

console = Console()
init(autoreset=True)


def wait_for_enter_to_analyze():
    """Функция, которая ждет нажатия Enter, чтобы продолжить анализ."""
    while True:
        pause = input(Fore.LIGHTBLUE_EX + Style.BRIGHT
                      + "Нажмите 'Enter', чтобы продолжить.")
        if pause.strip() == '':
            # если нажали только
            # Enter (строка пустая)
            break


def wait_for_enter_to_choose_opt():
    """
     Ожидает нажатия клавиши Enter от пользователя, чтобы вернуться к выбору опции.
     """
    while True:
        pause = input(
            Fore.LIGHTBLUE_EX + Style.BRIGHT +
            "Нажмите 'Enter', чтобы вернуться к выбору опции.")
        if pause.strip() == '':  # если нажали только Enter (строка пустая)
            break


def display_grammemes(for_n_grams=True):
    """
    Отображает справочную информацию о
    грамматических категориях и
    их обозначениях.

    :param for_n_grams: Если True, отображает
    информацию о частеречных n-граммах.
    Если False, отображает информацию
    о морфологических аннотациях.
    """
    if for_n_grams:
        print(
            Fore.LIGHTGREEN_EX + Style.BRIGHT +
            "\nДАЛЕЕ БУДЕТ ВЫВЕДЕН АНАЛИЗ ЧАСТЕРЕЧНЫХ "
            "N-ГРАММ И СПЕЦИАЛЬНЫХ ОБОЗНАЧЕНИЙ")
        print(
            Fore.LIGHTRED_EX + Style.BRIGHT +
            "\nВНИМАТЕЛЬНО ПОСМОТРИТЕ НА ОБОЗНАЧЕНИЯ"
            " ТЕГОВ ПЕРЕД ТЕМ, КАК ПРОДОЛЖИТЬ")
    else:
        print(
            Fore.LIGHTRED_EX + Style.BRIGHT +
            "\nВНИМАТЕЛЬНО ПОСМОТРИТЕ НА ОБОЗНАЧЕНИЯ "
            "ЧАСТЕЙ РЕЧИ И ИХ КАТЕГОРИЙ ПЕРЕД\n"
            "ТЕМ, КАК ПРОДОЛЖИТЬ")

        print(
            Fore.BLUE + Style.BRIGHT +
            "\nРазметка производится при помощи библиотеки "
            "pymorphy2, теги приведены\nв соответствии с ней")

    table = Table()
    table.add_column("POS", no_wrap=True,
                     justify="center", style="bold")
    table.add_column("Значение")
    table.add_column("Примеры")
    if for_n_grams:
        for grammeme, (
                description,
                examples
        ) in GRAMMEMES_NGRAMS.items():
            table.add_row(grammeme,
                          description,
                          examples)
    else:
        for grammeme, (
                description,
                examples
        ) in GRAMMEMES_MORPH_ANNOTATION.items():
            table.add_row(grammeme,
                          description,
                          examples)

    console.print(table)


def format_morphological_features(tag):
    """
    Преобразует морфологические признаки в формат feature=value.

    :param tag: Морфологическая метка (тег) для преобразования.
    :return: Строка с форматированными морфологическими признаками.
    """
    features = []
    # Добавляем морфологические при
    if 'anim' in tag:
        features.append("Animacy=Animate")
    if 'inan' in tag:
        features.append("Animacy=Inanimate")
    if 'nomn' in tag:
        features.append("Case=Nominative")
    if 'gent' in tag:
        features.append("Case=Genetive")
    if 'datv' in tag:
        features.append("Case=Dative")
    if 'accs' in tag:
        features.append("Case=Accusative")
    if 'ablt' in tag:
        features.append("Case=Instrumental")
    if 'loct' in tag:
        features.append("Case=Locative")
    if 'sing' in tag:
        features.append("Number=Singular")
    if 'plur' in tag:
        features.append("Number=Plural")
    if 'masc' in tag:
        features.append("Gender=Masculine")
    if 'femn' in tag:
        features.append("Gender=Feminine")
    if 'neut' in tag:
        features.append("Gender=Neuter")

    if '1per' in tag:
        features.append("Person=1st")
    if '2per' in tag:
        features.append("Person=2nd")
    if '3per' in tag:
        features.append("Person=3d")

    # Признаки для глаголов
    if 'perf' in tag:
        features.append("Aspect=Perfect")
    if 'impf' in tag:
        features.append("Aspect=Imperfect")
    if 'pres' in tag:
        features.append("Tense=Present")
    if 'past' in tag:
        features.append("Tense=Past")
    if 'futr' in tag:
        features.append("Tense=Future")
    if 'tran' in tag:
        features.append("Transitivity=Transitive")
    if 'intr' in tag:
        features.append("Transitivity=Intransitive")
    if 'indc' in tag:
        features.append("Mood=Indicative")
    if 'impr' in tag:
        features.append("Mood=Imperative")
    if 'actv' in tag:
        features.append("Voice=Active")
    if 'pssv' in tag:
        features.append("Voice=Passive")

    return ', '.join(features)


def display_morphological_annotation(sentences_info):
    """
    Отображает морфологическую разметку текста в виде таблиц по предложениям.

    :param sentences_info: Список словарей, содержащих информацию о лексемах, их леммах,
     частях речи и морфологических характеристиках.
    """
    print("\n" + Fore.LIGHTWHITE_EX + "*" * 80)
    print(
        Fore.GREEN + Style.BRIGHT +
        "                                   "
        "МОРФОЛОГИЧЕСКАЯ РАЗМЕТКА")
    print("" + Fore.LIGHTWHITE_EX + "*" * 80)

    def show_annot_info():
        display_grammemes(False)
        wait_for_enter_to_analyze()
        display_gr_categories()

    while True:
        user_input = input(
            Fore.LIGHTGREEN_EX + Style.BRIGHT +
            "Отобразить справку об используемых "
            "обозначениях (y/n)?\n").strip().lower()
        if user_input.lower() in ["y", "н"]:
            show_annot_info()
            break
        elif user_input.lower() in ["n", "т"]:
            break
        else:
            print(
                Fore.LIGHTRED_EX +
                "\nНеверный ввод. Пожалуйста,"
                " выберите один из возможных "
                "вариантов (y/n).")
            continue

    if sentences_info:
        total_sentences = len(sentences_info)
        start_index = 0
        print(
            Fore.LIGHTGREEN_EX + Style.BRIGHT +
            "\nДалее на экран будет выводиться"
            " по 5 предложений текста.")
        wait_for_enter_to_analyze()

        while start_index < total_sentences:
            end_index = min(start_index + 5, total_sentences)
            for index in range(start_index, end_index):
                sentence_info = sentences_info[index]
                # Создаем таблицу для каждого предложения
                table = PrettyTable()
                table.field_names = [
                    Fore.BLUE + Style.BRIGHT +
                    "Словоформа", "Лемма", "Часть речи",
                    "Морфологические характеристики"
                    + Fore.RESET]
                for info in sentence_info.values():
                    table.add_row([info["token"], info["lemma"],
                                   info["POS"], info["morph_features"]])

                print("\n" + Fore.LIGHTBLUE_EX + Style.BRIGHT
                      + "*" * 150)
                print(Fore.GREEN + Style.BRIGHT
                      + f"Предложение {index + 1}:")
                print(table)

            start_index = end_index
            print(
                Fore.LIGHTWHITE_EX + Style.BRIGHT +
                f"Отображено {start_index} предложений. "
                f"Всего предложений: {total_sentences} ")
            if start_index < total_sentences:
                while True:
                    user_input = input(
                        Fore.GREEN + Style.BRIGHT +
                        "Отобразить следующие "
                        "5 предложений? (y/n)\n"
                    ).strip().lower()
                    if user_input in ["y", "n", "т", "н"]:
                        break
                    print(
                        Fore.LIGHTRED_EX +
                        "\nНеверный ввод. "
                        "Пожалуйста, выберите "
                        "один из возможных "
                        "вариантов (y/n).")

                if user_input in ["n", "т"]:
                    break

    else:
        print("Морфологический анализ"
              " еще не был выполнен.")


def display_gr_categories():
    """
    Отображает таблицу с видами
    грамматических категорий и их подкатегориями.
    """
    console = Console()

    for category, subcategories in (
            GRAMMEMES_MORPH_ANNOTATION_GRAM_CATEGORIES.items()):
        print(
            Fore.GREEN + Style.BRIGHT + f"* {category.upper()}")

        table = Table()
        table.add_column("Подкатегория", width=20, justify="center")
        table.add_column("Описание", justify="center")

        for subcategory, description in subcategories.items():
            table.add_row(subcategory, description)

        console.print(table)
        wait_for_enter_to_analyze()


def display_position_explanation():
    """
     Отображает объяснение позиций токенов в предложениях.
     """
    print(Fore.GREEN + Style.BRIGHT +
          "Анализируются следующие позиции "
          "в предложениях:")
    print(Fore.BLACK +
          " - first - "
          "Первый токен")
    print(Fore.BLACK +
          " - second - "
          "Второй токен")
    print(Fore.BLACK +
          " - antepenultimate - "
          "Третий токен с конца")
    print(Fore.BLACK +
          " - penultimate - "
          "Предпоследний токен")
    print(Fore.BLACK +
          " - last -"
          " Последний токен")
    wait_for_enter_to_analyze()


def get_syntactic_annotation(text):
    """Метод для выполнения синтаксической
    разметки текста с использованием Natasha."""
    doc = Doc(text)
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    # Сегментация на предложения для дальнейшего анализа
    doc.segment(segmenter)
    # Морфологический анализ для дальнейшего синтаксического анализа
    doc.tag_morph(morph_tagger)
    # Анализ синтаксиса
    doc.parse_syntax(syntax_parser)
    # Собираем синтаксическую разметку в строку
    tokens_info = []
    i = 0

    for token in doc.tokens:
        # Получаем информацию о каждом токене
        token_info = {
            f"TOKEN {i + 1}": token.text,
            "id": token.id,
            "head_id": token.head_id,
            "pos": token.pos,
            "dependency": token.rel,
            "features": token.feats
        }
        tokens_info.append(token_info)
        i += 1

    return doc


def display_syntactic_annotation(doc):
    """Метод для отображения синтаксической разметки в виде древовидной структуры, аналогичной Natasha."""
    print("\n" + Fore.LIGHTWHITE_EX + "*" * 80)
    print(
        Fore.GREEN + Style.BRIGHT +
        "                         "
        "          СИНТАКСИЧЕСКАЯ РАЗМЕТКА")
    print("" + Fore.LIGHTWHITE_EX + "*" * 80)
    print(Fore.GREEN + Style.BRIGHT +
          "В  программе используется разметка"
          " синтаксических зависимостей в "
          "формате (UD) Universal Dependencies.\n"
          "Cинтаксический анализ будет представлен "
          "в виде древовидной структуры. "
          "На на экран будет выводиться по"
          "\n5 предложений текста.\n")
    wait_for_enter_to_analyze()
    total_sentences = len(doc.sents)
    start = 0
    batch_size = 5

    while start < total_sentences:
        end = min(start + batch_size, total_sentences)

        for i in range(start, end):
            print(Fore.GREEN + Style.BRIGHT + f'\nПРЕДЛОЖЕНИЕ {i + 1}:\n')
            doc.sents[i].syntax.print()
            print("\n" + Fore.LIGHTBLUE_EX + Style.BRIGHT + "*" * 150)

        start = end  # Обновляем значение start до фактического конца отображенных предложений
        print(
            Fore.LIGHTGREEN_EX + Style.BRIGHT +
            f"Отображено {start} предложений. "
            f"Всего предложений: {total_sentences}")

        if start < total_sentences:
            user_input = input(
                Fore.GREEN + Style.BRIGHT +
                "Отобразить следующие 5 предложений (y/n)?\n"
            ).strip().lower()

            while user_input not in ["y", "n", "т", "н"]:
                user_input = input(
                    Fore.LIGHTRED_EX +
                    "Неверный ввод. Пожалуйста, "
                    "выберите один из возможных "
                    "вариантов (y/n):\n"
                ).strip().lower()

            if user_input in ["n", "т"]:
                break


def check_db_exists(db_name):
    """
    Проверяет, существует ли база данных SQLite с заданным именем.

    :param db_name Имя базы данных (путь к файлу базы данных).
    :return True, если база данных существует, иначе False.
    """
    return os.path.exists(db_name)


def choose_universal():
    while True:
        print(Fore.GREEN + Style.BRIGHT + "\n Выберите опцию: ")
        print(Fore.GREEN + Style.BRIGHT
              + "  1."
              + Style.NORMAL + Fore.BLACK +
              " Средние показатели индикаторов "
              "характеристики Simplification")
        print(Fore.GREEN + Style.BRIGHT
              + "  2."
              + Style.NORMAL + Fore.BLACK +
              " Средние показатели индикаторов "
              "характеристики Normalisation")
        print(Fore.GREEN + Style.BRIGHT
              + "  3."
              + Style.NORMAL + Fore.BLACK +
              " Средние показатели индикаторов "
              "характеристики Explicitation")
        print(Fore.GREEN + Style.BRIGHT
              + "  4."
              + Style.NORMAL + Fore.BLACK +
              " Средние показатели индикаторов "
              "характеристики Interference")
        print(Fore.GREEN + Style.BRIGHT
              + "  5."
              + Style.NORMAL + Fore.BLACK +
              " Средние показатели "
              "остальных индикаторов")
        print(Fore.BLACK + Style.BRIGHT
              + "  6."
              + Style.NORMAL + Fore.LIGHTBLACK_EX +
              " Выйти в главное меню")

        choice = input(
            Fore.GREEN + Style.BRIGHT +
            "Введите номер опции.\n ")
        try:
            choice = int(choice)
            if 1 <= choice <= 6:
                return str(choice)
            else:
                print(INCORRECT_CHOICE)

        except ValueError:
            print(
                Fore.RED + Style.BRIGHT +
                "Пожалуйста, введите "
                "числовое значение.")
            continue

    return choice


def choose_db():
    while True:
        print(Fore.GREEN + Style.BRIGHT +
              "Выберите базу данных:")
        print(Fore.GREEN + Style.BRIGHT + "1."
              + Style.NORMAL + Fore.BLACK +
              " База непереводных текстов")
        print(Fore.GREEN + Style.BRIGHT
              + "2." + Style.NORMAL + Fore.BLACK +
              " База машинных переводов")
        print(Fore.GREEN + Style.BRIGHT
              + "3." + Style.NORMAL + Fore.BLACK +
              " База ручных переводов")
        print(Fore.LIGHTBLACK_EX + Style.BRIGHT
              + "4." + Style.NORMAL + Fore.LIGHTBLACK_EX +
              " Вернуться в главное меню")
        db_choice = input(Fore.GREEN + Style.BRIGHT +
                          "Введите номер базы данных:\n")
        if db_choice == "1":
            db = NON_TRANSLATED_DB_NAME
            break
        elif db_choice == "2":
            db = MACHINE_TRANSLATED_DB_NAME
            break
        elif db_choice == "3":
            db = HUMAN_TRANSLATED_DB_NAME
            break
        elif db_choice == "4":
            return
        else:
            print(INCORRECT_CHOICE)
    return db


def count_content_words(tokens):
    """
    Подсчитывает абсолютные частоты лемм только
    для знаменательных частей речи.
    :param tokens: Список объектов Parse (pymorphy2)
    :return: dict {лемма: частота}
    """
    from collections import defaultdict
    counts = defaultdict(int)
    for token in tokens:
        if getattr(token.tag, "POS", None) in CONTENT_POS:
            lemma = token.normal_form
            if lemma not in EXCLUDED_LEMMAS:
                counts[lemma] += 1
    return dict(counts)


def count_types_in_text(text):
    """
    Лемматизирует текст, подсчитывает количество знаменательных
    частей речи (по леммам) и возвращает результат в формате JSON.
    """
    import json
    from tools.core.lemmatizators import lemmatize_words_without_stopwords
    words, _ = lemmatize_words_without_stopwords(text.lower())
    content_words_count = count_content_words(words)
    sorted_content_words_counts = dict(
        sorted(
            content_words_count.items(),
            # по частоте ↓, при равенстве — по алфавиту
            key=lambda kv: (-kv[1], kv[0])
        )
    )
    return json.dumps(
        sorted_content_words_counts, ensure_ascii=False, indent=4
    )


from colorama import Fore, Style, init
import re

# Вспомогательное: аккуратно центрируем строку с ANSI‑цветами
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _center_ansi(s: str, width: int) -> str:
    vis = _ANSI_RE.sub("", s)
    pad = max(0, (width - len(vis)) // 2)
    return " " * pad + s


def print_characteristic_header(
        word: str,
        *,
        label: str = "ИНДИКАТОРЫ ХАРАКТЕРИСТИКИ",
        width: int = 80,
        center: bool = True
) -> None:
    """
    Печатает шапку типа:
    **************************************************
         ИНДИКАТОРЫ ХАРАКТЕРИСТИКИ EXPLICITATION
    **************************************************
    """
    line = Fore.LIGHTWHITE_EX + "*" * width + Fore.RESET
    title = (
            Fore.GREEN + Style.BRIGHT + f"{label} " +
            Fore.LIGHTGREEN_EX + Style.BRIGHT + word.upper() +
            Fore.RESET
    )
    print("\n" + line)
    print(_center_ansi(title, width) if center else title)
    print(line)
    wait_for_enter_to_analyze()


def print_greeting():
    print(
        Fore.GREEN + Style.BRIGHT + '\nАНАЛИЗАТОР ФЕНОМЕНА' +
        Fore.LIGHTGREEN_EX + Style.BRIGHT + ' TRANSLATIONESE' +
        Fore.GREEN + Style.BRIGHT + ' ЗАПУЩЕН\n')


def print_annotation_ready(ann_type: str):
    print(
        Fore.GREEN + Style.BRIGHT +
        f"\n{ann_type.upper()} РАЗМЕТКА "
        "СОЗДАНА И СОХРАНЕНА "
        "В БАЗУ ДАННЫХ"
    )
    print(
        Fore.LIGHTRED_EX +
        "Внимание! Разметка может занять"
        " много места на экране."
    )


def print_analysis_ready():
    print(
        Fore.GREEN + Style.BRIGHT +
        "\nАНАЛИЗ ИНДИКАТОРОВ ФЕНОМЕНА"
        + Fore.BLUE + Style.BRIGHT +
        " TRANSLATIONESE "
        + Fore.GREEN + Style.BRIGHT +
        "УСПЕШНО ЗАВЕРШЕН!"
    )
    print(Fore.RED + Style.BRIGHT +
          "РЕЗУЛЬТАТЫ АНАЛИЗА СОХРАНЕНЫ "
          "В БАЗУ ДАННЫХ.\n")
    wait_for_enter_to_analyze()


def print_synt_annot_display():
    print("\n" + Fore.LIGHTWHITE_EX + "*" * 80)
    print(
        Fore.GREEN + Style.BRIGHT +
        "                           "
        "        СИНТАКСИЧЕСКАЯ РАЗМЕТКА")
    print("" + Fore.LIGHTWHITE_EX + "*" * 80)

    print(Fore.GREEN + Style.BRIGHT +
          "В  программе используется разметка "
          "синтаксических зависимостей в формате "
          "Universal Dependencies (UD).\n"
          "Cинтаксический анализ будет представлен"
          " в виде древовидной структуры. "
          "На экран будет выводиться по"
          "\n5 предложений текста.\n")
    wait_for_enter_to_analyze()


def print_context_func_words_text():
    print(
        Fore.GREEN + Style.BRIGHT +
        "\n           ЧАСТОТЫ ЧАСТЕРЕЧНЫХ "
        "ТРИГРАММ С ФУНКЦИОНАЛЬНЫМИ СЛОВАМИ")
    print(Fore.LIGHTRED_EX + Style.BRIGHT +
          "Внимание! В зависимости от размера"
          " текста подсчет может занять какое-то время."
          "\nПожалуйста, будьте готовы подождать.\n")
    wait_for_enter_to_analyze()


def print_main_menu():
    print(Fore.GREEN + Style.BRIGHT + "Выберите действие: ")
    print(Fore.GREEN + Style.BRIGHT + "1." + Style.NORMAL +
          Fore.BLACK + " Проанализировать новый текст")
    print(
        Fore.GREEN + Style.BRIGHT + "2." + Style.NORMAL +
        Fore.BLACK + " Проанализировать несколько текстов за раз")
    print(
        Fore.GREEN + Style.BRIGHT + "3." + Style.NORMAL + Fore.BLACK +
        " Отобразить информацию о выбранном тексте в корпусе")
    print(
        Fore.GREEN + Style.BRIGHT + "4." + Style.NORMAL + Fore.BLACK +
        " Отобразить информацию о выбранном корпусе")
    print(
        Fore.GREEN + Style.BRIGHT + "5." + Style.NORMAL + Fore.BLACK +
        " Отобразить средние показатели по всем корпусам")
    print(
        Fore.GREEN + Style.BRIGHT + "6." + Style.NORMAL + Fore.BLACK
        + " Подготовить текст к анализу "
          "(удаление ссылок, выравнивание текста)")
    print(Fore.LIGHTBLACK_EX + Style.BRIGHT + "7." + Style.NORMAL
          + Fore.LIGHTBLACK_EX + " Выйти из программы")

    return input(Fore.GREEN + Style.BRIGHT +
                 "Введите номер действия: \n")


def not_positive_int_error(n: int):
    print(Fore.RED + Style.BRIGHT +
          f"Ошибка! Введите целое число > {n} или"
          " оставьте поле пустым для\nзначения"
          " по умолчанию.")


def be_ready_to_wait():
    print(
        Fore.RED + Style.BRIGHT
        + "Пожалуйста, после начала "
          "анализа будьте готовы "
          "подождать.\n"
    )
    wait_for_enter_to_analyze()


def display_mi_explanation():
    """
    Короткое объяснение PMI / Modified MI,
    а также Average MI и Threshold MI
    (в двух вариантах: по типам и взвешенно).
    """
    print(Fore.GREEN + Style.BRIGHT + "Что такое PMI и Modified MI?")
    print(Fore.BLACK +
          "- PMI (Pointwise Mutual Information, поточечная взаимная информация) —\n"
          "  мера ассоциативной связности пары слов.\n"
          "  Формула:\n"
          "   PMI(x,y) = log2( (N * f(x,y)) / (f(x) * f(y) * v) ),\n"
          "    где N — число токенов в корпусе,\n"
          "        f(x), f(y) — индивидуальные частоты лемм,\n"
          "        f(x, y) — число совместных появлений в окне,\n"
          "        v — средняя фактическая длина анализируемого окна.\n"
          "  Интерпретация: чем выше PMI, тем выше степень ассоциативной связанности\n"
          "  пары слов.")
    print(
        Fore.RED +
        "  ВНИМАНИЕ! Точность подсчета PMI"
        " зависит от размера корпуса. "
        "\n  При небольшом размере корпуса "
        "высока вероятность получить "
        "искусственно\n  завышенные "
        "значения показателя PMI:"
        " случайные редкие словосочетания\n"
        "  могут иметь высокие значения PMI,"
        " что будет говорить не о сильной\n"
        "  ассоциативной связи между словами, "
        "а о недостаточно большом размере\n  корпуса."
    )
    wait_for_enter_to_analyze()
    print(Fore.BLACK +
          "- Modified MI (Modified Mutual Information, модифицированная версия MI) —\n"
          "  взвешенная версия PMI, снижающая влияние редких случайных совпадений:\n"
          "  Формула:\n"
          "   Modified MI = f(x,y) * PMI.\n"
          "    где f(x, y) — число совместных появлений в окне,\n"
          "        PMI(x,y) — значение, полученное с помощью предыдущей формулы."
          )
    wait_for_enter_to_analyze()
    print(Fore.GREEN + Style.BRIGHT + "\nЧто подсчитывается?")
    print(Fore.GREEN + Style.BRIGHT + "  Average MI")
    print(Fore.BLACK +
          "  Среднее значение метрики по всем уникальным парам слов.")
    print()

    print(Fore.GREEN + Style.BRIGHT + "  Threshold MI")
    print(Fore.BLACK +
          "  Доля пар, у которых значение метрики превышает заданный порог T\n"
          "  (по умолчанию T = 0).")
    wait_for_enter_to_analyze()
    print(Fore.GREEN + Style.BRIGHT + "\nНастройки подсчёта")
    print(Fore.BLACK +
          "- window_size — ширина контекстного окна\n"
          "- direction — 'forward' (учитываются слова справа от анализируемого)\n"
          "  или 'sym' (±k, учитываются слова с обеих сторон)\n"
          "- min_cooc — минимальное f(x,y) для попадания пары в таблицу")
    wait_for_enter_to_analyze()


def process_text_from_file(file_path):
    """Обрабатывает текст из файла и сохраняет результат."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text
# display_mi_explanation()