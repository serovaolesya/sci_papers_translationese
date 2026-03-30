# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
from colorama import Fore, Style, init
from rich.console import Console
from rich.table import Table

from tools.core.data.pronouns import (
    pers_possessive_pronouns_analysis_list
)
from tools.core.utils import wait_for_enter_to_analyze
from tools.explicitation.named_entities_extraction import (
    extract_entities
)

console = Console()
init(autoreset=False)


def calculate_explicit_naming_ratio(
        text,
        parsed_words,
        show_analysis=True
):
    """
    Рассчитывает показатель эксплицитного называния (отношение личных
    и притяжательных местоимений к именам собственным в тексте).

    :param text: Входной текст на русском языке.
    :param parsed_words: Список объектов Parse,
    содержащих информацию о каждом токене.
    :param show_analysis: Если True, отображает
    результаты анализа в виде таблицы (по умолчанию True).

    :return: Отношение личных местоимений к именам собственным в процентах.
    Если в тексте нет имен собственных, возвращается 0.
    И список с информации о найденных именованных сущностях.
    """
    found_entities, entities_count = extract_entities(
        text, False
    )

    # Считаем личные и притяжательные местоимения
    pronouns_count = sum(
        1 for w in parsed_words
        if getattr(w, "normal_form", None)
        in pers_possessive_pronouns_analysis_list
    )

    if entities_count > 0:
        ratio = round((pronouns_count / entities_count) * 100, 3)
    else:
        ratio = 0

    if show_analysis:
        print(Fore.GREEN + Style.BRIGHT +
              "\nОТНОШЕНИЕ ЛИЧНЫХ МЕСТОИМЕНИЙ "
              "К ИМЕНАМ СОБСТВЕННЫМ\n"
              "              "
              "(EXPLICIT NAMING)")

        table = Table()
        table.add_column("Параметр", justify="left",
                         no_wrap=True, min_width=30, style="bold")
        table.add_column("Значение", justify="center", min_width=10)

        table.add_row(
            "Отношение местоимений к именам"
            "\nсобственным (%)", f"{ratio:.2f}%"
        )
        table.add_row(
            "Количество личных/притяжательных "
            "\nместоимений", str(pronouns_count)
        )
        table.add_row("Количество имен собственных",
                      str(entities_count))

        console.print(table)
        wait_for_enter_to_analyze()

    return ratio, found_entities, entities_count


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
    отправил их на анализ в корабль. Возвращение на Землю прошло успешно, и Алексей стал героем. Его 
    истории вдохновляли новое поколение детей мечтать о космосе. Алексей продолжил работать в космической программе, 
    передавая свой опыт молодым космонавтам. В каждом его слове чувствовалась страсть к исследованиям и вера в 
    будущее человечества среди звёзд.
    """
    from tools.core.lemmatizators import lemmatize_words

    parsed_words = lemmatize_words(text)
    a, b, c = calculate_explicit_naming_ratio(text, parsed_words)
    print(a)
    print(b)
    print(c)

