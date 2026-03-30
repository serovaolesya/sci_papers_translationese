# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import ast
import json
import re
from collections import Counter, defaultdict

from colorama import Style, Fore, init
from rich.console import Console
from rich.table import Table

from tools.core.data.discource_markers import (
    final_sci_dm_list, topic_intro_dm, info_sequence, illustration_dm,
    material_sequence, conclusion_dm, intro_new_addit_info,
    info_explanation_or_repetition, contrast_dm, examples_introduction_dm,
    author_opinion, author_attitude, high_certainty_modal_words,
    moderate_certainty_modal_words, uncertainty_modal_words,
    call_to_action_dm, joint_action, putting_emphasis_dm,
    refer_to_background_knowledge, cause_effect_dm,
    purpose_statement_dm
)
from tools.core.utils import wait_for_enter_to_analyze
from tools.core.stop_words_extraction_removal import remove_dm

console = Console()
init(autoreset=False)


def sort_by_length(lst):
    """Сортирует список строк по убыванию
    длины строки."""
    return sorted(lst, key=len, reverse=True)


# Нормализация маркеров: снимаем завершающую
# пунктуацию и пробелы, понижаем регистр
def _normalize_marker(s: str) -> str:
    # убираем пробелы в конце + финальную пунктуацию ",.;:!?…"
    return re.sub(r"[\s\.,;:!?…]+$", "", s.strip().lower())


def _normalize_list(coll) -> list:
    return sort_by_length([_normalize_marker(x) for x in coll])


# Реестр всех групп ДМ (по именам переменных)
_DM_NAMES = [
    "topic_intro_dm",
    "info_sequence",
    "illustration_dm",
    "material_sequence",
    "conclusion_dm",
    "intro_new_addit_info",
    "info_explanation_or_repetition",
    "contrast_dm",
    "examples_introduction_dm",
    "author_opinion",
    "author_attitude",
    "high_certainty_modal_words",
    "moderate_certainty_modal_words",
    "uncertainty_modal_words",
    "call_to_action_dm",
    "joint_action",
    "putting_emphasis_dm",
    "refer_to_background_knowledge",
    "cause_effect_dm",
    "purpose_statement_dm",
]

# Человеко‑читабельные имена категорий (для вывода)
CATEGORY_LABELS = {
    "topic_intro_dm": "Введение в тему",
    "info_sequence": "Порядок следования информации",
    "illustration_dm": "Иллюстративный материал",
    "material_sequence": "Порядок расположения материала",
    "conclusion_dm": "Выводы/заключение",
    "intro_new_addit_info": "Введение новой/доп. информации",
    "info_explanation_or_repetition":
        "Повтор/перефразирование/конкретизация информации",
    "contrast_dm": "Противопоставление",
    "examples_introduction_dm": "Введение примеров",
    "author_opinion": "Мнение автора",
    "author_attitude": "Отношение автора",
    "high_certainty_modal_words": "Высокая степень уверенности",
    "moderate_certainty_modal_words": "Средняя степень уверенности",
    "uncertainty_modal_words": "Низкая степень уверенности",
    "call_to_action_dm": "Призыв к действию",
    "joint_action": "Совместное с читателем действие",
    "putting_emphasis_dm": "Акцентирование внимания",
    "refer_to_background_knowledge":
        "Отсылка к фоновым знаниям читателя",
    "cause_effect_dm": "Причинно-следственные отношения",
    "purpose_statement_dm": "Постановка цели"
}

# Нормализация каждой коллекции по имени,
# сеты становятся отсортированными списками
for _dm_name in _DM_NAMES:
    globals()[_dm_name] = _normalize_list(globals()[_dm_name])

# Единый список нормализованных маркеров
# с сохранением порядка и без дублей
_ALL_MARKERS_NORM = list(dict.fromkeys(
    m for _name in _DM_NAMES for m in globals()[_name]
))

# Компилируем общий шаблон один раз на весь модуль
_PATTERN = re.compile(
    r'(?<!-)\b(?:' + '|'.join(map(re.escape, _ALL_MARKERS_NORM)) +
    r')(?=[\s\)\]\}\.,;:!?…]|$)'
)

# Строим индекс: маркер -> список имён категорий,
# к которым принадлежит маркер
_DM_TO_CATS = {}
for _cat_name in _DM_NAMES:
    for _m in globals()[_cat_name]:
        _DM_TO_CATS.setdefault(_m, []).append(_cat_name)


def sci_dm_search(text, show_analysis=True):
    """
    Ищет дискурсивные маркеры (ДМ)
    научного текста в тексте и анализирует их.

    :param text: Текст для анализа.
    :param show_analysis: Флаг, указывающий,
    нужно ли выводить результаты анализа.

    :return: Кортеж с результатами анализа,
    включающий количество и частоту различных типов ДМ.
    """
    lowercase_text = text.lower()
    only_alpha_text_without_stopwords, found_dm_num = remove_dm(lowercase_text)
    found_sci_dms = [m.group() for m in _PATTERN.finditer(lowercase_text)]
    found_sci_dms_str = '; '.join(found_sci_dms)
    total_num_dms = len(found_sci_dms)
    marker_counts = Counter(found_sci_dms)
    marker_counts_json = json.dumps(dict(marker_counts), ensure_ascii=False)
    # Базовые токены (без пунктуации/стоп‑слов)
    tokens = only_alpha_text_without_stopwords.split()
    base_tokens_count = len(tokens)

    # Сколько слов занимают найденные ДМ в исходном тексте
    # (каждое вхождение ДМ может быть многословным)
    dm_tokens_total = sum(
        len(_normalize_marker(dm).split())
        for dm in found_sci_dms
    )
    # Удаляем вклад слов ДМ из базового числа токенов и
    # учитываем сами ДМ как 1 токен
    # Пример: "таким образом, исследования показывают"
    # → базово 4 слова;
    # ДМ "таким образом" = 2 слова → 4 - 2 + 1 = 3
    non_dm_tokens = max(0, base_tokens_count - dm_tokens_total)
    total_tokens_with_dms = non_dm_tokens + total_num_dms

    if total_tokens_with_dms == 0:
        return (0, '', '', 0, '', 0, 0, '', 0, 0, '', 0, 0,
                '', 0, 0, '', 0, 0, '', 0, 0, '', 0, 0, '',
                0, 0, '', 0, 0, '', 0, 0, '', 0, 0, '', 0, 0,
                '', 0, 0, '', 0, 0, '', 0, 0, '', 0, 0, '', 0)

    # Analyze discourse marker types using per-category counters
    dm_type_counts = defaultdict(Counter)

    for dm in found_sci_dms:
        for cat in _DM_TO_CATS.get(dm, ()):
            dm_type_counts[cat][dm] += 1

    # Оптимизация: формируем агрегаты
    # по всем категориям в одном цикле
    category_results = {}
    for category in _DM_NAMES:
        elems = list(dm_type_counts[category].elements())
        elements_str = "; ".join(elems)
        count = sum(dm_type_counts[category].values())
        freq = round(count / total_tokens_with_dms * 100, 3) if (
            count) else 0.0
        category_results[category] = (elements_str, count, freq)

    # Распаковка по прежним именам (совместимость с выводом/return)
    (topic_intro_dm_str,
     topic_intro_dm_count,
     topic_intro_dm_freq) = category_results["topic_intro_dm"]
    (info_sequence_str,
     info_sequence_count,
     info_sequence_freq) = category_results["info_sequence"]
    (illustration_dm_str,
     illustration_dm_count,
     illustration_dm_freq) = category_results["illustration_dm"]
    (material_sequence_str,
     material_sequence_count,
     material_sequence_freq) = category_results["material_sequence"]
    (conclusion_dm_str,
     conclusion_dm_count,
     conclusion_dm_freq) = category_results["conclusion_dm"]
    (intro_new_addit_info_str,
     intro_new_addit_info_count,
     intro_new_addit_info_freq) = category_results["intro_new_addit_info"]
    (info_explanation_or_repetition_str,
     info_explanation_or_repetition_count,
     info_explanation_or_repetition_freq) = (
        category_results)["info_explanation_or_repetition"]
    (contrast_dm_str,
     contrast_dm_count,
     contrast_dm_freq) = category_results["contrast_dm"]
    (examples_introduction_dm_str,
     examples_introduction_dm_count,
     examples_introduction_dm_freq) = (
        category_results)["examples_introduction_dm"]
    (author_opinion_str,
     author_opinion_count,
     author_opinion_freq) = category_results["author_opinion"]
    (author_attitude_str,
     author_attitude_count,
     author_attitude_freq) = category_results["author_attitude"]
    (high_certainty_modal_words_dm_str,
     high_certainty_modal_words_dm_count,
     high_certainty_modal_words_dm_freq) = (
        category_results)["high_certainty_modal_words"]
    (moderate_certainty_modal_words_dm_str,
     moderate_certainty_modal_words_dm_count,
     moderate_certainty_modal_words_dm_freq) = (
        category_results)["moderate_certainty_modal_words"]
    (uncertainty_modal_words_dm_str,
     uncertainty_modal_words_dm_count,
     uncertainty_modal_words_dm_freq) = (
        category_results)["uncertainty_modal_words"]
    (call_to_action_dm_str,
     call_to_action_dm_count,
     call_to_action_dm_freq) = category_results["call_to_action_dm"]
    (joint_action_str,
     joint_action_count,
     joint_action_freq) = category_results["joint_action"]
    (putting_emphasis_dm_str,
     putting_emphasis_dm_count,
     putting_emphasis_dm_freq) = category_results["putting_emphasis_dm"]
    (refer_to_background_knowledge_str,
     refer_to_background_knowledge_count,
     refer_to_background_knowledge_freq) = (
        category_results)["refer_to_background_knowledge"]
    (cause_effect_dm_str, cause_effect_dm_count,
     cause_effect_dm_freq) = category_results["cause_effect_dm"]
    (purpose_statement_dm_str, purpose_statement_dm_count,
     purpose_statement_dm_freq) = category_results["purpose_statement_dm"]

    if show_analysis:
        print_dm_analysis_results(
            total_num_dms,
            dict(marker_counts),
            category_results,
        )

    return (total_tokens_with_dms, total_num_dms,
            found_sci_dms_str, marker_counts_json,
            topic_intro_dm_count,
            topic_intro_dm_str,
            topic_intro_dm_freq,
            info_sequence_count,
            info_sequence_str,
            info_sequence_freq,
            illustration_dm_count,
            illustration_dm_str,
            illustration_dm_freq,
            material_sequence_count,
            material_sequence_str,
            material_sequence_freq,
            conclusion_dm_count,
            conclusion_dm_str,
            conclusion_dm_freq,
            intro_new_addit_info_count,
            intro_new_addit_info_str,
            intro_new_addit_info_freq,
            info_explanation_or_repetition_count,
            info_explanation_or_repetition_str,
            info_explanation_or_repetition_freq,
            contrast_dm_count, contrast_dm_str,
            contrast_dm_freq,
            examples_introduction_dm_count,
            examples_introduction_dm_str,
            examples_introduction_dm_freq,
            author_opinion_count,
            author_opinion_str,
            author_opinion_freq,
            author_attitude_count,
            author_attitude_str,
            author_attitude_freq,
            high_certainty_modal_words_dm_count,
            high_certainty_modal_words_dm_str,
            high_certainty_modal_words_dm_freq,
            moderate_certainty_modal_words_dm_count,
            moderate_certainty_modal_words_dm_str,
            moderate_certainty_modal_words_dm_freq,
            uncertainty_modal_words_dm_count,
            uncertainty_modal_words_dm_str,
            uncertainty_modal_words_dm_freq,
            call_to_action_dm_count,
            call_to_action_dm_str,
            call_to_action_dm_freq,
            joint_action_count,
            joint_action_str,
            joint_action_freq,
            putting_emphasis_dm_count,
            putting_emphasis_dm_str,
            putting_emphasis_dm_freq,
            refer_to_background_knowledge_count,
            refer_to_background_knowledge_str,
            refer_to_background_knowledge_freq,
            cause_effect_dm_count,
            cause_effect_dm_str,
            cause_effect_dm_freq,
            purpose_statement_dm_count,
            purpose_statement_dm_str,
            purpose_statement_dm_freq
            )


def print_dm_analysis_results(
        total_num_dms,
        marker_counts,
        category_results):
    """
        Выводит результаты анализа дискурсивных маркеров.

    :param total_num_dms: Общее количество найденных ДМ.
    :param marker_counts: dict:
    {маркер -> абсолютная частота вхождений}.
    :param category_results: dict:
    {internal_category_name -> (elements_str, count, freq)}.
    """
    # --- Normalize inputs coming from DB (may arrive as JSON strings/floats) ---
    # Coerce marker_counts from JSON string or repr to dict[str,int]
    if isinstance(marker_counts, str):
        try:
            marker_counts = json.loads(marker_counts)
        except Exception:
            try:
                marker_counts = ast.literal_eval(marker_counts)
            except Exception:
                marker_counts = {}
    if not isinstance(marker_counts, dict):
        marker_counts = {}
    # Coerce total_num_dms to int for pretty printing
    try:
        total_num_dms = int(total_num_dms)
    except Exception:
        pass
    print(
        Fore.GREEN + Style.BRIGHT +
        "\n                       НАЙДЕННЫЕ ДИСКУРСИВНЫЕ МАРКЕРЫ"
    )
    print(
        Fore.BLUE + Style.BRIGHT +
        f"Всего найдено {int(total_num_dms) if isinstance(total_num_dms, (int, float)) else total_num_dms} ДМ"
    )

    # ----- Таблица 1: маркеры по категориям -----
    table1 = Table()
    table1.add_column("Категория ДМ", justify="left")
    table1.add_column("Маркеры (кол-во вхождений)", justify="left")

    for cat_key in _DM_NAMES:
        label = CATEGORY_LABELS.get(cat_key, cat_key)
        elements_str, count, _freq = category_results.get(
            cat_key, ("", 0, 0.0)
        )
        if count <= 0:
            continue
        # Сохраняем порядок появления и убираем дубликаты
        raw = [m for m in elements_str.split("; ") if m]
        unique_in_order = list(dict.fromkeys(raw))
        markers_with_count = [
            f"{dm} ({marker_counts.get(dm, 0)})"
            for dm in unique_in_order
        ]
        table1.add_row(label,
                       ", ".join(markers_with_count))
    console.print(table1)
    wait_for_enter_to_analyze()

    # ----- Таблица 2: частоты по категориям -----
    print(
        Fore.GREEN + Style.BRIGHT +
        "\n              ЧАСТОТЫ "
        "НАЙДЕННЫХ ДИСКУРСИВНЫХ "
        "МАРКЕРОВ ПО КАТЕГОРИЯМ"
    )
    table2 = Table()
    table2.add_column("Категория ДМ\n", justify="left")
    table2.add_column("Абсолютная частота\n", justify="center")
    table2.add_column("Нормализованная частота\n (%)",
                      justify="center")

    rows = []
    for cat_key in _DM_NAMES:
        label = CATEGORY_LABELS.get(cat_key, cat_key)
        _elements_str, count, freq = category_results.get(
            cat_key, ("", 0, 0.0)
        )
        if count > 0:
            rows.append((label, count, freq))

    for label, count, freq in rows:
        table2.add_row(label, str(count), f"{freq:.3f}")

    console.print(table2)
    wait_for_enter_to_analyze()


if __name__ == "__main__":
    # Текст для примера (сгенерирован ИИ)
    text = """
    Таким образом, исследования показывают, что с увеличением температуры многие виды растений начинают цветение на 
    несколько недель раньше, чем это было зафиксировано в прошлом веке. Это явление особенно заметно в умеренных 
    широтах, где сезонные изменения температуры наиболее выражены. Например, анализ данных за последние 50 лет 
    показал, что средняя дата начала цветения для ряда видов флоры Северной Америки и Европы сместилась на 10–15 дней 
    раньше. Такие сдвиги могут нарушить синхронизацию между растениями и их опылителями, что потенциально угрожает 
    успешному размножению. Смещение ареалов обитания растений является одной из наиболее очевидных реакций на 
    изменение климата. По мере повышения температуры виды стремятся мигрировать в более прохладные регионы, 
    такие как горные районы или более высокие широты. Однако скорость изменений климата зачастую превышает 
    способности видов к миграции, что приводит к сокращению их популяций и даже к локальному или глобальному 
    вымиранию. Наше моделирование показало, что при сценарии повышения глобальной температуры на 2°C до конца XXI 
    века, порядка 20% изученных видов растений будут вынуждены сместиться на север или в более высокие горные районы. 
    В частности, виды, обитающие на низких широтах и в равнинных регионах, окажутся под наибольшим давлением. Виды, 
    обладающие узкой экологической нишей и ограниченными возможностями для миграции, как, например, эндемики горных 
    экосистем, наиболее уязвимы к этим изменениям. отсюда следует вывод объясняется это тем, что потому, что
    """
    init(autoreset=True)

    a = sci_dm_search(text)
    print(a)
