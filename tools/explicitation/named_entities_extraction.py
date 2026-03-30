# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import json

from colorama import Style, Fore, init
from natasha import (
    NewsNERTagger,
    NewsEmbedding,
    Doc, Segmenter
)
from rich.console import Console
from rich.table import Table

from tools.core.data.false_named_entities import (
    false_named_entities
)
from tools.core.utils import wait_for_enter_to_analyze

console = Console()
init(autoreset=False)

segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

# Добавьте сюда свои кастомные сущности и типы,
# если NER неправильно их распознает
CUSTOM_NAMED_ENTITIES = {
    "Минздрава России": "ORG",
    "ИПО Красноярского ГМУ им. проф. В. Ф. Войно-Ясенецкого": "ORG",
    "Красноярского ГМУ им. проф. В. Ф. Войно-Ясенецкого": "ORG",
    "ФГБОУ ВО СамГМУ": "ORG",
    'Дж. Г. Мида': 'PER',
    "Протестантская этика и дух капитализма": "WORK",
    "European Atherosclerosis Society, EAS": "ORG",
    "Российской Федерации (РФ)": "LOC",
    "Фрамингемском исследовании": "STUDY",
    "Скандинавском исследовании": "STUDY",
    "ФГБУ “НМИЦ ТПМ”": "ORG",
    "Научно-клинического и образовательного центра "
    "«Кардиология» Санкт-Петербургского государственного университета":
        "ORG",
    "Общества исследований гормона роста": "ORG",
    "клинику им. Е. М. Тареева": "ORG",
}


def preprocess_text(text, custom_entities):
    """
    Обрабатывает текст, оборачивая кастомные сущности в квадратные скобки.
    Выполняет замены кумулятивно и сохраняет все вхождения.
    """
    text_with_custom_entities = text
    for entity, entity_type in custom_entities.items():
        if entity in text_with_custom_entities:
            text_with_custom_entities = text_with_custom_entities.replace(entity, f"[{entity}]")
    return text_with_custom_entities


def extract_entities(
        text,
        show_entities_in_console=True
):
    """
    Извлекает именованные сущности и их типы из текста.

    :param text: Входной текст для анализа на наличие
     именованных сущностей.
    :param show_entities_in_console: Флаг, определяющий,
    выводить ли найденные сущности в консоль
    (по умолчанию True).

    :return: Кортеж, содержащий:
        - Строку в JSON-подобном формате с
        найденными сущностями и их типами.
        - Количество найденных сущностей.
    """
    processed_text = preprocess_text(
        text,
        CUSTOM_NAMED_ENTITIES
    )
    # print(processed_text)

    # Создаем объект Doc с текстом
    doc = Doc(processed_text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    # Создаем список кортежей (сущность, тип)
    found_named_entities = [
        (doc.text[span.start:span.stop], span.type)
        for span in doc.ner.spans
        if doc.text[span.start:span.stop]
           not in false_named_entities
    ]

    import re
    # Добавляем кастомные сущности, помеченные в тексте квадратными скобками
    custom_marked = re.findall(r"\[([^\]]+)\]", processed_text)
    for ent in custom_marked:
        ent_type = CUSTOM_NAMED_ENTITIES.get(ent)
        if ent_type and ent not in false_named_entities:
            found_named_entities.append((ent, ent_type))

    found_entities_count = len(found_named_entities)

    found_named_entities_str = json.dumps(
        found_named_entities, ensure_ascii=False
    )

    if show_entities_in_console:
        display_entities(
            found_named_entities_str,
            found_entities_count
        )
        wait_for_enter_to_analyze()

    return found_named_entities_str, found_entities_count


def display_entities(
        found_named_entities_str,
        found_entities_count
):
    """
    Отображает найденные именованные сущности и
     их типы в консоли.

    :param found_named_entities_str: Строка в формате JSON,
     содержащая найденные сущности и их типы.
    :param found_entities_count: Количество найденных
    сущностей.
    """
    try:
        found_named_entities = json.loads(
            found_named_entities_str.replace("'", '"')
        )
    except json.JSONDecodeError as e:
        print(Fore.LIGHTRED_EX + f"Ошибка парсинга JSON: {e}")
        return

    print(Fore.GREEN + Style.BRIGHT +
          "\n              НАЙДЕННЫЕ ИМЕНОВАННЫЕ"
          " СУЩНОСТИ И ИХ ТИПЫ")
    print(Fore.RED + Style.BRIGHT +
          f"ВНИМАНИЕ! Иногда при распознавании и категоризации "
          f"именованных сущностей\nмогут возникать ошибки."
          f"\nПожалуйста, будьте внимательны и учитывайте"
          f" это при анализе результатов.")
    wait_for_enter_to_analyze()
    print(Fore.BLUE + Style.BRIGHT +
          f"\nВсего найдено {found_entities_count} сущности(ей)")

    table = Table()

    table.add_column("Именованная сущность", min_width=20)
    table.add_column("Тип", justify="center", min_width=10)

    for entity, entity_type in found_named_entities:
        table.add_row(entity, entity_type)

    console.print(table)


if __name__ == "__main__":
    # Текст для примера (сгенерирован ИИ)
    text = """

    В огромном городе, где небоскрёбы касались облаков, жила девушка по имени Лиза. Она работала в необычной
    организации под названием "Луч Света". Эта организация ставила перед собой амбициозную цель — помочь каждому
    человеку на Земле и сделать мир добрее. Каждое утро Лиза приходила в светлый офис, наполненный зелёными
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

    # text = """
    #
    #
    # """
    entities_str, entities_count = extract_entities(text)

    # print(entities_str)
    # print(entities_count)
