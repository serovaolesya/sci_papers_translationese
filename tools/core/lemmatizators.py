# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import re
from functools import lru_cache

from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer

from tools.core.data import pronouns, prepositions, particles, conjunctions

nltk_stopwords_ru = stopwords.words("russian")

# 1. Объединение всех списков в один
all_stopwords = set(
    conjunctions.conjunctions_list + prepositions.prepositions_list
    + particles.particles_list + pronouns.pronouns_list + nltk_stopwords_ru
)

# 2. Сортировка по длине по убыванию: длинные фразы (например,
#    «несмотря на то что») должны проверяться раньше более коротких
#    («несмотря на», «на»).
all_stopwords_sorted = sorted(list(all_stopwords), key=len, reverse=True)

# Единый комбинированный паттерн вместо цикла из N отдельных.
# Было: N вызовов findall+sub (N ≈ 1760) на каждый текст.
# Стало: один проход по тексту — на порядок быстрее.
# Порядок альтернатив в | соответствует all_stopwords_sorted (длина ↓),
# что гарантирует жадный захват более длинных фраз первыми.
_COMBINED_STOPWORDS_PATTERN = re.compile(
    r'(?<!-)\b(?:' +
    '|'.join(re.escape(w) for w in all_stopwords_sorted) +
    r')\b(?!-)',
    re.IGNORECASE
)

# Разрешаем: кириллицу, латиницу и дефис `-
_ALLOWED_CHARS = re.compile(r"[^А-Яа-яЁёA-Za-z\-]+")


def remove_stopwords_and_filter(
        text: str
) -> tuple[str, int]:
    """
    Удаляет стоп-слова, считает их количество,
    затем оставляет в тексте только кириллицу,
    латиницу и дефис '-'.
    Возвращает очищенный текст и число удалённых стоп-слов.

    Важно:
    - Границы слов учитываются, чтобы не удалять подстроки.
    - Стоп-слова внутри сложных слов с дефисом сохраняются
      (например, 'по-моему' не затрагивается).

    :param text: Исходный текст
    :return: (очищенный_текст, количество удаленных стоп-слов)
    """
    if not text:
        return "", 0

    # 1) Удаляем все стоп-слова за один проход
    removed_count = len(_COMBINED_STOPWORDS_PATTERN.findall(text))
    tmp = _COMBINED_STOPWORDS_PATTERN.sub('', text)

    # 2) Фильтруем допустимые символы: кириллица/латиница/дефис '-'
    tmp = _ALLOWED_CHARS.sub(' ', tmp)

    # Нормализуем пробелы
    cleaned_text = re.sub(r'\s+', ' ', tmp).strip()
    return cleaned_text, removed_count


# patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"”“]
# patterns = r"[^А-Яа-яёЁ\-]+"  # Оставляем только кириллицу и дефис
patterns = r"[^А-Яа-яёЁA-Za-z\-]+"  # оставляем кириллицу + латиницу + дефис

morph = MorphAnalyzer()


@lru_cache(maxsize=200_000)
def parse_cached(token: str):
    """Кэшируем лучший разбор токена.
    Ключ кэша – нижний регистр токена."""
    token = token.strip()
    return morph.parse(token)[0]


@lru_cache(maxsize=1000)
def lemmatize_words_without_stopwords(
        text: str
) -> tuple[list, int]:
    """
    Лемматизирует слова в тексте после удаления кастомных стоп-слов.

    Результат кэшируется по тексту целиком: повторный вызов с тем же
    текстом (в рамках одного сеанса) возвращает сохранённый результат
    без повторного запуска регулярок и лемматизации.
    Возвращённый список объектов Parse не следует изменять —
    он разделяется между всеми вызовами с одинаковым ключом.

    :param text: Входной текст на русском языке.
    :return: Кортеж (tokens_full_info, stop_w_count), где
             - tokens_full_info: список объектов Parse по каждому токену,
             которое не является стоп-словом.
             - stop_w_count: количество удалённых стоп-слов.
    """
    cleaned_text, stop_w_count = remove_stopwords_and_filter(text)
    tokens_full_info = []
    for token in cleaned_text.split():
        token = token.strip()
        token = parse_cached(token)
        tokens_full_info.append(token)
    return tokens_full_info, stop_w_count


def lemmatize_words(text):
    """
    Лемматизирует слова в тексте, оставляя
    только кириллицу, латиницу и дефис.
    Используется в:
    - calculate_lexical_density;


    :param text: Входной текст на русском языке.
    :return: Список объектов Parse, содержащих информацию о каждом токене.
    """
    # Оставляем только буквенные символы
    text = re.sub(patterns, ' ', text)

    tokens_full_info = []
    for token in text.split():
        token = token.strip()
        token = parse_cached(token)
        tokens_full_info.append(token)
    return tokens_full_info


if __name__ == "__main__":
    # Текст для примера (сгенерирован ИИ)
    text = """В старом доме на окраине города жила семья, которая была известна своей гостеприимностью. Дом был 
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

    import warnings

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="pymorphy2.analyzer"
    )

    a = lemmatize_words_without_stopwords(text)
    b = lemmatize_words(text)

    print(a)
    # print()

    print(b)
