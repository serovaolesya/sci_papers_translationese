# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import re

from nltk.corpus import stopwords

from tools.core.data import pronouns, prepositions, particles, conjunctions
from tools.core.data.discource_markers import final_sci_dm_list

nltk_stopwords_ru = stopwords.words("russian")

# 1. Объединение всех списков stopwords в один
all_stopwords = set(
    conjunctions.conjunctions_list + prepositions.prepositions_list
    + particles.particles_list + pronouns.pronouns_list + nltk_stopwords_ru
)
# 2. Сортировка stopwords по длине по убыванию
all_stopwords_sorted = sorted(list(all_stopwords), key=len, reverse=True)


def count_custom_stopwords(text):
    """
    Подсчитывает количество стоп-слов в тексте.
    Args:
        text (str): Текст для анализа.
    Returns:
        tuple: Общее количество найденных стоп-слов,
        словарь, где ключи - это стоп-слова,
        а значения - количество их вхождений в текст,
        и общее количество вхождений всех стоп-слов в тексте.
    """
    stopword_counts = {
        stopword: 0 for stopword
        in all_stopwords_sorted
    }
    text_to_clean = text.lower()

    for stopword in all_stopwords_sorted:
        count = len(re.findall(r'\b' + re.escape(stopword) + r'\b', text_to_clean))
        if count > 0:
            stopword_counts[stopword] += count
            text_to_clean = re.sub(r'\b' + re.escape(stopword) + r'\b', '', text_to_clean)

    # Фильтруем только те стоп-слова, которые были найдены в тексте
    found_stopwords = {word: count for word, count in stopword_counts.items() if count > 0}
    sorted_stopwords = dict(sorted(found_stopwords.items(), key=lambda item: item[1], reverse=True))
    stopwords_total_count_with_rep = sum(found_stopwords.values())
    unique_stopwords = len(found_stopwords)

    # print(text_to_clean)
    # print(f"\nОбщее количество найденных стоп-слов (с учетом их повторений): {stopwords_total_count_with_rep}")
    # print(f"\nОбщее количество различных стоп-слов в тексте: {unique_stopwords}")
    # print("\nКоличество вхождений каждого найденного стоп-слова:\n")
    #
    # for stopword, count in sorted_stopwords.items():
    #     print(f"'{stopword}': {count} раз(а)")

    return (
        stopwords_total_count_with_rep,
        sorted_stopwords,
        unique_stopwords
    )


_DM_LIST_CLEANED = []
for _dm in final_sci_dm_list:
    _dm_clean = _dm.strip().lower()
    _dm_clean = re.sub(r'[\,\.;:!?]+$', '', _dm_clean)
    if _dm_clean:
        _DM_LIST_CLEANED.append(_dm_clean)


_DM_LIST_CLEANED = sorted(set(_DM_LIST_CLEANED), key=len, reverse=True)
_DM_ALTERNATION = "|".join(map(re.escape, _DM_LIST_CLEANED))
_DM_PATTERN = re.compile(
    rf'(?<!\w)(?:{_DM_ALTERNATION})(?!\w)[\s]*[\,\.;:!?—–-]*',
    re.IGNORECASE
)


def rebuild_dm_pattern():
    """Rebuild the global DM regex after
    external updates to final_sci_dm_list."""
    global _DM_LIST_CLEANED, _DM_ALTERNATION, _DM_PATTERN
    _DM_LIST_CLEANED = []
    for _dm in final_sci_dm_list:
        _dm_clean = _dm.strip().lower()
        _dm_clean = re.sub(r'[\,\.;:!?]+$', '', _dm_clean)
        if _dm_clean:
            _DM_LIST_CLEANED.append(_dm_clean)
    _DM_LIST_CLEANED = sorted(set(_DM_LIST_CLEANED), key=len, reverse=True)
    _DM_ALTERNATION = "|".join(map(re.escape, _DM_LIST_CLEANED))
    _DM_PATTERN = re.compile(
        rf'(?<!\w)(?:{_DM_ALTERNATION})(?!\w)[\s]*[\,\.;:!?—–-]*',
        re.IGNORECASE
    )

def remove_dm(text):
    """
    Функция удаляет дискурсивные маркеры из текста
    и считает их количество.

    Параметры:
    text (str): Входной текст.

    Возвращает:
    tuple: (обновленный текст без маркеров,
    количество удаленных маркеров)
    """

    # Один проход замены + счёт с
    # уже предкомпилированным шаблоном
    text_no_dm, deleted_dms_num = _DM_PATTERN.subn('', text)

    # Подчистка пробелов и небуквенных
    text_no_dm = re.sub(r'\s+', ' ', text_no_dm).strip()
    text_cleaned = re.sub(r'[^а-яА-Яa-zA-Z\s-]', '', text_no_dm)

    return text_cleaned, deleted_dms_num


if __name__ == "__main__":
    # Текст для примера (сгенерирован ИИ)
    text = """
    Таким образом, анализ проведенных исследований показал, что внедрение новых технологий значительно повысило эффективность работы компании. Однако, несмотря на положительные результаты, остались некоторые области, требующие дальнейшего улучшения. Подводя итог, можно сказать, что проект был успешным, но есть еще над чем работать. Подводя итоги, мы можем отметить, что команда справилась с поставленными задачами на высоком уровне    """

    a = remove_dm(text)
    print(a)
