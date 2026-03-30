# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import json
import re
from collections import defaultdict, Counter

from colorama import Fore, Style
from nltk.corpus import stopwords
from rich.console import Console
from rich.table import Table
from functools import lru_cache
from tools.core.natasha_pymorphy_pos_tagger import (
    pos_tagger as custom_pos_tagger
)

from tools.core.data import (
    pronouns, prepositions,
    particles, conjunctions
)
from tools.core.utils import (
    wait_for_enter_to_analyze,
    print_context_func_words_text
)

console = Console()

# Загрузка стоп-слов
nltk_stopwords_ru = stopwords.words("russian")

# 1. Объединение всех списков в один
all_stopwords = set(conjunctions.conjunctions_list +
                    prepositions.prepositions_list
                    + particles.particles_list +
                    pronouns.pronouns_list +
                    nltk_stopwords_ru)

# 2. Сортировка по длине по убыванию
all_stopwords_sorted = sorted(list(all_stopwords), key=len, reverse=True)

# --- Precompiled helpers for speed ---
# 1) Regexes compiled once
PUNCT_SPACER_RE = re.compile(r'([.,!?;–])')
CLEAN_RE = re.compile(r'[^а-яА-ЯёЁA-Za-z0-9\s.,!?;\-–]')

# 2) Multiword stopwords: build a normalized map
# and a single alternation regex
_multiword_stopwords = [s for s in all_stopwords_sorted if ' ' in s]
# Map with normalized single spaces as keys -> underscored replacement
_MULTIWORD_MAP = {
    ' '.join(s.split()): s.replace(' ', '_')
    for s in _multiword_stopwords
}

# Build alternation with flexible whitespace between parts (\s+)
# Example: "не только" -> pattern "не\s+только"
_multiword_patterns = []
for s in _multiword_stopwords:
    parts = [re.escape(p) for p in s.split()]
    _multiword_patterns.append(r'(?:' + r'\s+'.join(parts) + r')')
if _multiword_patterns:
    MULTIWORD_STOPWORDS_RE = re.compile(
        r'\b(?:' + '|'.join(_multiword_patterns) + r')\b'
    )
else:
    # Fallback that never matches
    MULTIWORD_STOPWORDS_RE = re.compile(r'$(?!)')


# Кэшируем POS-тег для одиночного
# токена через кастомный теггер
@lru_cache(maxsize=10000)
def get_pos_custom(token: str) -> str:
    # Вызываем кастомный теггер
    # на одиночном токене
    try:
        tags, toks = custom_pos_tagger(token)
        # Возвращаем первый реальный тег (не S_START)
        for t in tags:
            if t != 'S_START':
                return t
        return 'N/A'
    except Exception:
        return 'N/A'


def contextual_function_words_in_trigrams(
        list_of_sentences,
        show_analysis=True):
    """
    Анализирует текст и находит триграммы, содержащие
     функциональные слова, а также их контексты и частоты.

    :param list_of_sentences: Список предложений.
    :param show_analysis: Если True,
    выводит результаты анализа.

    :return tuple: Состоящий из трех JSON-строк:
        - Нормализованные частоты триграмм с
        функциональными словами.
        - Абсолютные частоты триграмм с
        функциональными словами.
        - Контексты триграмм с
        функциональными словами.
    """
    if show_analysis:
        print_context_func_words_text()

    func_words_contexts_with_tokens_or_pos = defaultdict(list)
    token_trigram_counts = Counter()
    pos_trigram_counts = {
        'one_function_word': Counter(),
        'two_function_words': Counter(),
        'three_function_words': Counter()
    }
    # Счётчик не нужен — считаем только общее число
    total_all_trigrams = 0

    for sent in list_of_sentences:
        # Разделение слов и знаков препинания пробелами
        # (быстрее за счёт precompiled regex)
        sent = PUNCT_SPACER_RE.sub(r' \1 ', sent)

        sent = CLEAN_RE.sub(' ', sent)
        # Единая замена многословных стоп-слов: одно прохождение по строке
        if _MULTIWORD_MAP:
            def _mw_sub(m):
                key = ' '.join(m.group(0).split())  # нормализуем пробелы
                return _MULTIWORD_MAP.get(key, m.group(0))
            sent = MULTIWORD_STOPWORDS_RE.sub(_mw_sub, sent)

        # Разделение предложения на токены
        tokens = sent.strip().split()
        n = len(tokens)
        if n < 3:
            continue

        # Предвычисляем «очищенные» токены и признаки функциональности
        clean_tokens = [t.replace('_', ' ') for t in tokens]
        is_func = [ct in all_stopwords for ct in clean_tokens]

        # Предвычисляем POS/функциональные метки для каждого токена в предложении
        pos_or_func = [ (ct if is_func[i] else get_pos_custom(ct)) for i, ct in enumerate(clean_tokens) ]

        # Счётчик всех возможных триграмм увеличиваем сразу (без отдельного Counter)
        total_all_trigrams += (n - 2)

        # Скользящее окно по триграммам
        for i in range(n - 2):
            # Токены в виде слов
            clean_trigram = clean_tokens[i:i + 3]
            token_trigram_counts[tuple(clean_trigram)] += 1

            # Подсчёт количества функциональных слов в триграмме
            swc = int(is_func[i]) + int(is_func[i + 1]) + int(is_func[i + 2])
            if swc == 0:
                # нет функциональных слов — пропускаем накопление POS-триграмм
                continue

            trigram_pos = pos_or_func[i:i + 3]

            # Накапливаем частоты POS-триграмм (обрабатываем только 1..3)
            if swc == 1:
                pos_trigram_counts['one_function_word'][tuple(trigram_pos)] += 1
                if show_analysis:
                    func_words_contexts_with_tokens_or_pos['one_function_word'].append((clean_trigram, trigram_pos))
            elif swc == 2:
                pos_trigram_counts['two_function_words'][tuple(trigram_pos)] += 1
                if show_analysis:
                    func_words_contexts_with_tokens_or_pos['two_function_words'].append((clean_trigram, trigram_pos))
            elif swc == 3:
                pos_trigram_counts['three_function_words'][tuple(trigram_pos)] += 1
                if show_analysis:
                    func_words_contexts_with_tokens_or_pos['three_function_words'].append((clean_trigram, trigram_pos))
    if total_all_trigrams == 0:
        total_all_trigrams = 1  # защита от деления на ноль при пустом входе
    normalized_freqs = {
        'one_function_word': {
            trigram: round(count / total_all_trigrams * 100, 3)
            for trigram, count in pos_trigram_counts['one_function_word'].items()
        },
        'two_function_words': {
            trigram: round(count / total_all_trigrams * 100, 3)
            for trigram, count in pos_trigram_counts['two_function_words'].items()
        },
        'three_function_words': {
            trigram: round(count / total_all_trigrams * 100, 3)
            for trigram, count in pos_trigram_counts['three_function_words'].items()
        }
    }

    sorted_normalized_freqs = {
        k: dict(sorted(v.items(), key=lambda item: item[1], reverse=True))
        for k, v in normalized_freqs.items()
    }

    pos_trigram_counts = {
        'one_function_word': dict(
            sorted(pos_trigram_counts['one_function_word'].items(), key=lambda item: item[1], reverse=True)),
        'two_function_words': dict(
            sorted(pos_trigram_counts['two_function_words'].items(), key=lambda item: item[1], reverse=True)),
        'three_function_words': dict(
            sorted(pos_trigram_counts['three_function_words'].items(), key=lambda item: item[1], reverse=True))
    }

    func_words_contexts_with_tokens_or_pos = dict(func_words_contexts_with_tokens_or_pos)

    def convert_keys_to_str(d):
        """
        Рекурсивно преобразует ключи словаря в строки.

        :param d: (dict или list): Словарь или список для преобразования.

        :return dict или list: Преобразованный словарь или список.
        """
        if isinstance(d, dict):
            return {str(k): convert_keys_to_str(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_keys_to_str(i) for i in d]
        else:
            return d

    sorted_normalized_freqs = convert_keys_to_str(sorted_normalized_freqs)
    pos_trigram_counts = convert_keys_to_str(pos_trigram_counts)
    func_words_contexts_with_tokens_or_pos = convert_keys_to_str(func_words_contexts_with_tokens_or_pos)
    sorted_normalized_freqs_json = json.dumps(sorted_normalized_freqs, ensure_ascii=False)
    pos_trigram_counts_json = json.dumps(pos_trigram_counts, ensure_ascii=False)
    func_words_contexts_with_tokens_or_pos_json = json.dumps(func_words_contexts_with_tokens_or_pos, ensure_ascii=False)

    if show_analysis:
        print_trigram_tables_with_func_w(
            sorted_normalized_freqs_json,
            pos_trigram_counts_json,
            func_words_contexts_with_tokens_or_pos_json)
    return (
        sorted_normalized_freqs_json,
        pos_trigram_counts_json,
        func_words_contexts_with_tokens_or_pos_json
    )


def print_trigram_tables_with_func_w(
        sorted_normalized_freqs,
        pos_trigram_counts_json,
        func_words_contexts_with_tokens_or_pos_json,
        for_corpus=False
):
    """
    Выводит таблицы с данными о триграммах, содержащих
    функциональные слова, на основе предоставленных JSON-строк.

    :param sorted_normalized_freqs: JSON-строка, содержащая
    нормализованные частоты триграмм с функциональными словами.
    :param pos_trigram_counts_json: JSON-строка, содержащая
     абсолютные частоты триграмм с функциональными словами.
    :param func_words_contexts_with_tokens_or_pos_json: JSON-строка,
     содержащая контексты триграмм с функциональными словами.
    :param for_corpus: Если True, контексты не будут выводиться.
    По умолчанию False.

    Выводит таблицы с нормализованными частотами и
    абсолютными частотами триграмм в зависимости от
    минимальной частоты, указанной пользователем.
    Также запрашивает у пользователя, нужно ли вывести
    полные контексты триграмм с функциональными словами.
        """
    if for_corpus:
        print(
            Fore.GREEN + Style.BRIGHT +
            "\n           ЧАСТОТЫ ЧАСТЕРЕЧНЫХ "
            "ТРИГРАММ С 1,2,3  ФУНКЦИОНАЛЬНЫМИ "
            "СЛОВАМИ")
        wait_for_enter_to_analyze()
    # Принимаем как JSON-строки, так и уже-готовые dict
    if isinstance(sorted_normalized_freqs, str):
        sorted_normalized_freqs = json.loads(sorted_normalized_freqs)
    if isinstance(pos_trigram_counts_json, str):
        pos_trigram_counts = json.loads(pos_trigram_counts_json)
    else:
        pos_trigram_counts = pos_trigram_counts_json
    if isinstance(func_words_contexts_with_tokens_or_pos_json, str):
        func_words_contexts_with_tokens_or_pos = json.loads(
            func_words_contexts_with_tokens_or_pos_json)
    else:
        func_words_contexts_with_tokens_or_pos = (
            func_words_contexts_with_tokens_or_pos_json
        )

    print(
        Fore.LIGHTGREEN_EX + Style.BRIGHT +
        "Введите интересующую Вас минимальную "
        "частоту триграмм или просто нажмите "
        "'Enter' \n(по умолчанию значение=1):")
    while True:
        min_frequency_input = input()
        if not min_frequency_input:
            min_frequency = 1
            break
        try:
            min_frequency = int(min_frequency_input)
            if min_frequency > 0:
                break
            else:
                print(Fore.LIGHTRED_EX +
                      "Ошибка: Значение должно "
                      "быть больше 0.")
        except ValueError:
            print(Fore.LIGHTRED_EX +
                  "Ошибка: Введите числовое"
                  " значение или нажмите "
                  "'Enter'.\n")

    for category in [
        'one_function_word',
        'two_function_words',
        'three_function_words'
    ]:
        if category == 'one_function_word':
            show_text = ('ТРИГРАММЫ С ОДНИМ ФУНКЦИОНАЛЬНЫМ '
                         'СЛОВОМ И ДВУМЯ МАРКЕРАМИ POS В СОСТАВЕ')
        elif category == 'two_function_words':
            show_text = ('ТРИГРАММЫ С ДВУМЯ ФУНКЦИОНАЛЬНЫМИ '
                         'СЛОВАМИ И ОДНИМ МАРКЕРОМ POS В СОСТАВЕ')
        elif category == 'three_function_words':
            show_text = ('ТРИГРАММЫ С ТРЕМЯ ФУНКЦИОНАЛЬНЫМИ '
                         'СЛОВАМИ В СОСТАВЕ')
        print(Fore.GREEN + Style.BRIGHT + "\n* " + show_text)
        wait_for_enter_to_analyze()
        table = Table()
        table.add_column("Триграмм\n", no_wrap=True,
                         style="bold", max_width=40)
        table.add_column("Абсолютная частота\n",
                         justify="center", min_width=15)
        table.add_column("Нормализованная частота\n(%)",
                         justify="center", min_width=15)

        # Объединяем данные из двух словарей
        normalized_freqs = sorted_normalized_freqs.get(category, {})
        pos_counts = pos_trigram_counts.get(category, {})
        sorted_data = dict(sorted(pos_counts.items(),
                                  key=lambda item: item[1],
                                  reverse=True))

        for trigram, count in sorted_data.items():
            if count >= min_frequency:
                freq = normalized_freqs.get(trigram, 0)
                table.add_row(str(trigram),
                              str(count),
                              f"{freq:.3f}%")

        console.print(table)
        wait_for_enter_to_analyze()

    if not for_corpus:
        print(
            Fore.LIGHTGREEN_EX + Style.BRIGHT +
            "\nВывести полные контексты триграмм"
            " с функциональными словами (y/n)? ")
        print(Fore.LIGHTRED_EX + Style.BRIGHT +
              "Внимание! Контексты могут занять"
              " много места на экране.")
        while True:
            choice = input().strip().lower()

            if choice == 'y':
                break
            elif choice == 'n':
                print(Fore.LIGHTRED_EX + Style.BRIGHT
                      + "Вывод контекстов пропущен.")
                return
            else:
                print(Fore.LIGHTRED_EX +
                      "\nНеверный ввод. "
                      "Пожалуйста, введите "
                      "'y' или 'n'.")

        print(
            Fore.GREEN + Style.BRIGHT +
            "\n            КОНТЕКСТЫ ЧАСТЕРЕЧНЫХ "
            "ТРИГРАММ С ФУНКЦИОНАЛЬНЫМИ СЛОВАМИ")
        wait_for_enter_to_analyze()
        for category in [
            'one_function_word',
            'two_function_words',
            'three_function_words']:
            if category == 'one_function_word':
                show_text = ('ТРИГРАММ С ОДНИМ ФУНКЦИОНАЛЬНЫМ '
                             'СЛОВОМ И ДВУМЯ МАРКЕРАМИ POS В СОСТАВЕ')
            elif category == 'two_function_words':
                show_text = ('ТРИГРАММ С ДВУМЯ ФУНКЦИОНАЛЬНЫМИ СЛОВАМИ'
                             ' И ОДНИМ МАРКЕРОМ POS В СОСТАВЕ')
            elif category == 'three_function_words':
                show_text = ('ТРИГРАММ С ТРЕМЯ ФУНКЦИОНАЛЬНЫМИ'
                             ' СЛОВАМИ В СОСТАВЕ')
            print(Fore.GREEN + Style.BRIGHT
                  + "\n* КОНТЕКСТЫ " + show_text)
            wait_for_enter_to_analyze()
            table = Table()
            table.add_column("С POS-тегами", no_wrap=True, width=40)
            table.add_column("Полные контексты в токенах",
                             no_wrap=True, width=52)

            contexts = func_words_contexts_with_tokens_or_pos.get(category, [])
            for tokens, pos_tags in contexts:
                table.add_row(str(pos_tags), str(tokens))

            console.print(table)
            wait_for_enter_to_analyze()


if __name__ == "__main__":
    # Текст для примера (сгенерирован ИИ)
    text = """
    В огромном городе, где небоскрёбы касались облаков, жила девушка по имени Лиза. Она работала в 
       необычной организации под названием "Луч Света". Эта организация ставила перед собой амбициозную цель — помочь 
       каждому человеку на Земле и сделать мир добрее. Каждое утро Лиза приходила в светлый офис, наполненный зелёными 
       растениями и солнечными лучами, проникающими через огромные окна. Она садилась за свой стол и начинала день с 
       того, что проверяла письма и сообщения от людей со всего мира. Кто-то нуждался в помощи с оплатой медицинских 
       счетов, кто-то искал поддержку в трудную минуту, а кто-то просто хотел поделиться своей радостью. Однажды Лиза 
       получила письмо от маленькой девочки из далёкой деревни. Девочка писала, что её мама тяжело больна, 
       и она не знает, как ей помочь. Лиза немедленно связалась с коллегами, и вскоре команда врачей отправилась в ту 
       самую деревню, чтобы оказать необходимую помощь. Через несколько недель пришло радостное сообщение: мама девочки 
       пошла на поправку. Но "Луч Света" занимался не только экстренной помощью. Они организовывали образовательные 
       программы для детей из бедных семей, строили парки и школы, проводили акции по защите окружающей среды. Лиза 
       особенно гордилась проектом, который они запустили в Африке: благодаря усилиям команды, в нескольких деревнях 
       появились чистая вода и солнечные батареи. Однажды к ним в офис пришёл пожилой человек. Он выглядел растерянным и 
       усталым. Лиза подошла к нему и узнала, что его зовут Иван. Иван потерял всё: дом, семью, работу. Он чувствовал 
       себя одиноким и беспомощным. Лиза пригласила его в свой кабинет, выслушала его историю и пообещала помочь. В 
       течение нескольких дней команда "Луча Света" нашла для Ивана временное жильё, помогла восстановить документы и 
       устроить его на работу. Иван был бесконечно благодарен и часто заходил в офис, чтобы поделиться своими успехами. 
       Каждый вечер Лиза возвращалась домой с чувством выполненного долга. Она знала, что её работа имеет значение, 
       что люди, которым она помогла, снова верят в добро и справедливость. И хотя задачи, стоящие перед "Лучом Света", 
       казались порой непосильными, Лиза и её коллеги не сдавались. Они верили, что капля добра способна вызвать 
       настоящий океан перемен. Так проходили дни, месяцы и годы. Лиза продолжала работать в "Луче Света", 
       оставаясь верной своей мечте — сделать мир лучше. Она знала, что впереди ещё много вызовов, но с каждым добрым 
       делом, с каждой спасённой жизнью, с каждой улыбкой на лице благодарного человека мир становился чуть светлее и 
       добрее.
       
       """

    from colorama import init

    from tools.core.custom_punkt_tokenizer import (
        sent_tokenize_with_abbr
    )

    init(autoreset=True)
    list_of_sentences = sent_tokenize_with_abbr(text)

    (sorted_normalized_freqs, pos_trigram_counts,
     func_words_contexts_with_tokens_or_pos) = (
        contextual_function_words_in_trigrams(
            list_of_sentences)
    )
    # print(sorted_normalized_freqs)
    # print()
    #
    # print(pos_trigram_counts)
    # print()
    #
    # print(func_words_contexts_with_tokens_or_pos)
