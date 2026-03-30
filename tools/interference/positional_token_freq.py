# -*- coding: utf-8 -*-  # Языковая кодировка UTF-8
import json
from collections import Counter, defaultdict

from colorama import Fore, Style
from rich.console import Console
from rich.table import Table


from tools.core.utils import (
    wait_for_enter_to_analyze,
    display_position_explanation
)

console = Console()


def calculate_position_frequencies(
        pos_tags_seq,
        tokens_seq,
        show_analysis=True
):
    """
    Вычисляет нормализованные частоты появления токенов
    на позициях ('first', 'second', 'antepenultimate',
    'penultimate', 'last') на основе двух
    последовательностей: POS‑тегов и самих токенов.

    Ожидается, что в POS‑последовательности
    присутствуют маркеры 'S_START' и 'S_END'
    для сегментации предложений. Сегменты в
    tokens_seq должны соответствовать
    тем же позициям.

    :param pos_tags_seq:
    list[str] — последовательность POS‑тегов
    (включая 'S_START'/'S_END').
    :param tokens_seq:
    list[str] — последовательность токенов
    той же длины, что и pos_tags_seq.
    :param show_analysis: bool
    печать шапки и таблиц.

    :return: (frequencies_json, raw_counts_json)
    """
    # ========== ШАГ 1: ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ ==========
    if show_analysis:
        print(Fore.GREEN + Style.BRIGHT +
              "\n             ПОЗИЦИОННАЯ "
              "ЧАСТОТА ТОКЕНОВ")
        print(
            Fore.RED +
            "Внимание! При подсчете позиционной частоты "
            "токенов\nпредложения короче 5 токенов"
            " игнорируются.\n")
    # Проверка типов данных: оба аргумента должны быть итерируемыми
    if (not isinstance(pos_tags_seq, (list, tuple))
            or not isinstance(tokens_seq, (list, tuple))):
        raise TypeError("pos_tags_seq и tokens_seq "
                        "должны быть списками/кортежами")
    # Проверка синхронности: каждому тегу должен соответствовать токен
    if len(pos_tags_seq) != len(tokens_seq):
        raise ValueError("pos_tags_seq и tokens_seq "
                         "должны быть одинаковой длины")

    # ========== ШАГ 2: ИНИЦИАЛИЗАЦИЯ СТРУКТУР ДАННЫХ ==========
    # defaultdict(Counter) создает словарь, где каждый ключ (позиция)
    # автоматически инициализируется объектом Counter для подсчета токенов
    # Структура: {'first': Counter({'токен1': 5, 'токен2': 3}), 'second': ...}
    position_counts = defaultdict(Counter)
    # Счетчик предложений (только >= 5 токенов) для нормализации
    total_sentences = 0

    # ========== ШАГ 3: СЕГМЕНТАЦИЯ ТЕКСТА НА ПРЕДЛОЖЕНИЯ ==========

    # Список для накопления токенов текущего предложения.
    # Собираем предложения по маркерам S_START / S_END
    sent_tokens = []
    # Флаг: находимся ли мы внутри предложения
    # (между S_START и S_END)
    inside = False

    # Проходим по параллельным последовательностям тегов и токенов
    for tag, tok in zip(pos_tags_seq, tokens_seq):

        # --- ОБРАБОТКА НАЧАЛА ПРЕДЛОЖЕНИЯ ---
        if tag == 'S_START':
            # Обнуляем список токенов для нового предложения
            sent_tokens = []
            # Активируем флаг: теперь находимся внутри предложения
            inside = True
            # Переходим к следующей итерации (S_START не добавляется в токены)
            continue

        if tag == 'S_END':
            if inside:  # Проверяем, что предложение было открыто

                # КРИТИЧНО: добавляем финальный знак препинания
                # tok здесь содержит '.', '!', '?' и т.д.
                sent_tokens.append(tok)

                # ========== ШАГ 4: ПОДСЧЕТ ПОЗИЦИОННЫХ ЧАСТОТ ==========

                # Обрабатываем только предложения >= 5 токенов
                # (чтобы все 5 позиций были валидны)
                if len(sent_tokens) >= 5:

                    # Увеличиваем счетчик валидных предложений
                    total_sentences += 1

                    # Извлекаем токены на ключевых позициях
                    # sent_tokens[0]  = первый токен (first)
                    # sent_tokens[1]  = второй токен (second)
                    # sent_tokens[-3] = третий с конца (antepenultimate)
                    # sent_tokens[-2] = предпоследний (penultimate)
                    # sent_tokens[-1] = последний (last) — это знак препинания

                    positions = {
                        'first': sent_tokens[0],
                        'second': sent_tokens[1],
                        'antepenultimate': sent_tokens[-3],
                        'penultimate': sent_tokens[-2],
                        'last': sent_tokens[-1],
                    }
                    # Для каждой позиции увеличиваем счетчик соответствующего токена
                    # position_counts['first']['The'] += 1
                    # position_counts['last']['.'] += 1
                    for position, token in positions.items():
                        position_counts[position][token] += 1
            # Обнуляем список токенов (предложение завершено)
            sent_tokens = []
            # Деактивируем флаг (вышли из предложения)
            inside = False
            # Переходим к следующей итерации
            continue
        # --- ОБРАБОТКА ОБЫЧНЫХ ТОКЕНОВ ---
        # Если мы внутри предложения (между S_START и S_END),
        # накапливаем токены
        if inside:
            sent_tokens.append(tok)

    # Создаем словарь для нормализованных частот
    normalized_frequencies = {}

    # Для каждой позиции и её Counter-объекта
    for position, counter in position_counts.items():
        # Преобразуем абсолютные частоты в относительные (%)
        # Формула: (count / total_sentences) * 100
        # Пример: токен встретился 15 раз на позиции 'first' в 100 предложениях
        # → 15/100 * 100 = 15.0%
        normalized_frequencies[position] = dict(
            sorted(
                {token: round(
                    count / (total_sentences or 1) * 100, 3)
                    # 'or 1' предотвращает деление на ноль
                    for token, count in counter.items()}.items(),
                key=lambda item: item[1],  # Сортируем по частоте
                reverse=True  # От большего к меньшему
            )
        )

    # ========== ШАГ 6: СОРТИРОВКА АБСОЛЮТНЫХ ЧАСТОТ ==========

    # Создаем отсортированный словарь абсолютных частот
    # (для вывода в таблице рядом с нормализованными)
    sorted_raw_counts = {}
    for position, counter in position_counts.items():
        sorted_raw_counts[position] = dict(
            sorted(counter.items(),
                   key=lambda item: item[1],
                   reverse=True)
        )

    # Конвертируем словари в JSON-строки
    # ensure_ascii=False сохраняет кириллицу в читаемом виде
    frequencies = json.dumps(normalized_frequencies,
                             ensure_ascii=False)
    raw_counts = json.dumps(sorted_raw_counts,
                            ensure_ascii=False)

    # ========== ШАГ 8: ВЫВОД РЕЗУЛЬТАТОВ ==========

    if show_analysis:
        print_frequencies(frequencies, raw_counts)

    # Возвращаем обе JSON-строки
    return frequencies, raw_counts


def min_count_choice_for_posiitions():
    """
    Запрашивает у пользователя минимальное
    значение для средней частоты токенов
    на позициях или использует значение
    по умолчанию (1), если пользователь
    ничего не вводит.

    Возвращает:
    float: Минимальное значение
    для средней частоты токенов.
    """
    while True:
        try:
            min_count_input = input(
                Fore.GREEN + Style.BRIGHT +
                f"\nВведите минимальное "
                f"значение для средней "
                f"частоты\nвыводимых "
                f"токенов на той или иной"
                f"позиции или\nпросто"
                f" нажмите 'Enter', "
                f"чтобы продолжить\n"
                f"(по умолчанию "
                f"значение=1):\n"
            ).strip()

            if not min_count_input:
                min_count = 1
                break
            else:
                min_count = float(min_count_input)
                if min_count <= 0:
                    print(Fore.LIGHTRED_EX +
                          Style.BRIGHT +
                          "Ошибка! "
                          "Значение "
                          "не может быть "
                          "< или = 0.")
                    continue
                break
        except ValueError:
            print(Fore.LIGHTRED_EX +
                  Style.BRIGHT +
                  "Ошибка! Введите "
                  "числовое значение.")
    return min_count


def print_frequencies(
        frequencies,
        raw_counts,
):
    """
    Выводит таблицы с частотами токенов
    на разных позициях.

    :param frequencies: JSON-строка
    с нормализованными частотами токенов.
    :param raw_counts: JSON-строка с абсолютными
    частотами токенов.
    """
    display_position_explanation()
    frequencies = json.loads(frequencies)
    raw_counts = json.loads(raw_counts)

    min_count = min_count_choice_for_posiitions()

    for position in ['first', 'second',
                     'antepenultimate',
                     'penultimate', 'last']:
        print(Fore.GREEN + Style.BRIGHT +
              f"             Токены на "
              f"позиции: {position.capitalize()}")
        wait_for_enter_to_analyze()

        table = Table()
        table.add_column("Токен\n", no_wrap=True,
                         min_width=15)
        table.add_column("Абсолютная частота",
                         justify="center", width=15)
        table.add_column("Нормализованная частота (%)",
                         justify="center", width=15)

        freq_data = frequencies.get(position, {})
        raw_data = raw_counts.get(position, {})

        sorted_tokens = sorted(freq_data.items(),
                               key=lambda item: item[1],
                               reverse=True)

        for token, freq in sorted_tokens:
            count = raw_data.get(token, 0)
            if count >= min_count:
                table.add_row(token,
                              str(count),
                              f"{freq:.2f}%")

        console.print(table)
        wait_for_enter_to_analyze()


if __name__ == "__main__":
    # Текст для примера (сгенерирован ИИ)
    text = """
    Осенний ветер за окном напоминал.  Листья деревьев медленно кружились в воздухе, постепенно 
    покрывая землю золотым ковром. В парке гуляли немногочисленные прохожие, наслаждаясь последними тёплыми днями. Вдоль 
    аллеи бежала собака, радостно виляя хвостом. Маленький мальчик с интересом наблюдал за ней, крепко держа за руку свою 
    маму. Она говорила ему о том, как важно сохранять природу и уважать окружающий мир. Вдалеке был виден силуэт 
    человека, сидящего на лавочке с книгой. Он не спешил никуда, погружённый в чтение. Вокруг царила атмосфера 
    умиротворённости и спокойствия. Солнце постепенно уходило за горизонт, окутывая парк мягким оранжевым светом. Небо 
    меняло свой цвет, переходя от светло-голубого к насыщенному розовому. Птицы готовились к ночи, прячась в ветвях 
    деревьев. Где-то рядом слышался тихий плеск воды из фонтана. Люди начинали расходиться по домам, постепенно покидая 
    парк. И вот, когда город погрузился в вечерние сумерки, наступила долгожданная тишина.
    """
    from colorama import init

    from tools.core.natasha_pymorphy_pos_tagger import (
        pos_tagger
    )

    init(autoreset=True)
    n_grams, lemmas = pos_tagger(text)

    frequencies, counts = (
        calculate_position_frequencies(
            n_grams, lemmas)
    )
    print(frequencies)
    print(type(frequencies))
    print(counts)
    print(type(counts))

