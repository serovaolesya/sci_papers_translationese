# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import os
import glob
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pymorphy2.analyzer"
)

from datetime import datetime
from pathlib import Path

from colorama import Style, Fore
import pandas as pd

from rich.console import Console

from tools.core.utils import (wait_for_enter_to_analyze, not_positive_int_error,
                               be_ready_to_wait, display_mi_explanation)
from tools.normalisation.mutual_info import PMICalculator, PMIConfig

console = Console()


def ask_user_cfg(
        comparison: bool = False
) -> PMIConfig:
    """
    Интерактивно запрашивает параметры конфигурации PMI:
    - window_size (окно)
    - min_cooc (минимальное число совместных появлений)
    - direction (направление: forward/sym)

    Пустой ввод оставляет значение по умолчанию.
    Возвращает PMIConfig.
    """
    if not comparison:
        print("\n" + Fore.LIGHTWHITE_EX + "*" * 80)
        print(
            Fore.GREEN + Style.BRIGHT +
            "                        АНАЛИЗ " +
            Fore.LIGHTGREEN_EX + Style.BRIGHT +
            "ПОКАЗАТЕЛЕЙ MI" + Fore.GREEN +
            Style.BRIGHT + " ДЛЯ КОРПУСА"
        )
        print(Fore.LIGHTWHITE_EX + "*" * 80)

        print(
            Fore.BLUE + Style.BRIGHT +
            "\nДалее будет проведен "
            "подсчет показателей поточечной взаимной информации "
            "для\nлемматизированных пар слов.")
        print(
            Fore.RED +
            "\nВНИМАНИЕ! При большом размере корпуса подсчет показателей"
            " может занять некоторое\nвремя. Будьте готовы подождать.\n"
        )
        ans = input(
            Fore.GREEN + Style.BRIGHT +
            "Показать справку по показателям PMI и Modified MI? (y/N):\n"
        ).strip().lower()
        if ans in {"y", "yes", "д", "да"}:
            display_mi_explanation()

    else:
        print("\n" + Fore.LIGHTWHITE_EX + "*" * 80)
        print(
            Fore.GREEN + Style.BRIGHT +
            "                        АНАЛИЗ " +
            Fore.LIGHTGREEN_EX + Style.BRIGHT +
            "ПОКАЗАТЕЛЕЙ MI" + Fore.GREEN +
            Style.BRIGHT + " ПО КОРПУСАМ"
        )
        print(Fore.LIGHTWHITE_EX + "*" * 80)

        print(
            Fore.GREEN + Style.BRIGHT +
            "\nДалее будет проведен "
            "подсчет показателей поточечной взаимной информации "
            "для\nлемматизированных пар слов.")
        print(
            Fore.RED +
            "\nВНИМАНИЕ! При большом размере корпуса подсчет показателей"
            " может занять некоторое\nвремя. Будьте готовы подождать.\n"
        )
        ans = input(
            Fore.GREEN + Style.BRIGHT +
            "Показать справку по показателям PMI и Modified MI? (y/n):\n"
        ).strip().lower()
        if ans in {"y", "yes", "д", "да"}:
            display_mi_explanation()

    default_window = 5
    default_min_cooc = 1
    default_direction = "forward"  # или "sym"

    # Окно
    while True:
        raw = input(
            Fore.GREEN + Style.BRIGHT +
            f"Введите интересующий Вас размер "
            f"окна (по умолчанию {default_window}):\n"
        ).strip()
        if raw == "":
            window_size = default_window
            break
        try:
            value = int(raw)
            if value <= 0:
                not_positive_int_error(0)
                continue
            window_size = value
            break
        except ValueError:
            not_positive_int_error(0)

    # Минимальное количество совместных появлений
    while True:
        raw = input(
            Fore.GREEN + Style.BRIGHT +
            f"Введите минимальное значение "
            f"интересующей Вас совместной"
            f"\nвстречаемости слов "
            f"(по умолчанию {default_min_cooc}):\n"
        ).strip()
        if raw == "":
            min_cooc = default_min_cooc
            break
        try:
            value = int(raw)
            if value <= 0:
                not_positive_int_error(0)
                continue
            min_cooc = value
            break
        except ValueError:
            not_positive_int_error(0)

    # Направление
    while True:
        raw = input(
            Fore.GREEN + Style.BRIGHT +
            f"Введите направление анализа "
            f"(f=forward, s=symmetrical)\n"
            f"(по умолчанию "
            f"{default_direction}):\n"
        ).strip().lower()
        if raw == "":
            direction = default_direction
            break
        if raw in {"forward", "f"}:
            direction = "forward"
            break
        if raw in {"symmetrical", "s"}:
            direction = "sym"
            break
        print(Fore.LIGHTRED_EX +
              "Недопустимое значение. "
              "Введите 'f' (forward) "
              "или 's' (sym).")

    # Сформируем конфиг
    cfg = PMIConfig(
        window_size=window_size,
        direction=direction,
        cross_sentences=False,
        include_stopwords=True,
        min_cooc=min_cooc,
        lemmatize=True,
    )

    print(
        Fore.RED + Style.BRIGHT +
        f"Выбранные параметры:\n"
        f"- window_size={cfg.window_size}, "
        f"\n- min_cooc={cfg.min_cooc},"
        f"\n- direction={cfg.direction}\n"
    )
    be_ready_to_wait()
    return cfg


def read_texts_from_directory(directory):
    """
    Считывает все текстовые файлы из указанной директории
    и возвращает их содержимое в виде списка строк.

    :param directory: Путь к директории,
    содержащей текстовые файлы.
    :return: Множество строк, каждая из которых представляет
     содержимое одного текстового файла.
    """
    corpus = set()
    # Ищем все файлы с расширением .txt в указанной директории
    for filename in glob.glob(os.path.join(directory, '*.txt')):
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
            corpus.add(text)
    return corpus


def calculate_pmi(
        corpus_directory=None,
        corpus_set: set = None,
        comparison: bool = False,
        cfg: PMIConfig = None
):
    """
    Расчёт MI (PMI/Modified MI) с использованием
    движка PMICalculator.

    :param comparison: True, если сравниваем
    :param corpus_directory: путь к директории с .txt файлами
    (если используем директорию)
    :param corpus_set: набор текстов
    (если передаём готовый set[str])
    :param cfg: конфигурация PMIConfig; если None —
    запросим у пользователя (ask_user_cfg).
    :return: tuple:
    (DataFrame со столбцами x,y,fxy,fx,fy,PMI,Modifie dMI,
     dict с Threshold MI (долей пар > 0) по каждой метрике,
     PMICalculator instance)
    """
    # 1) Собираем корпус
    if corpus_directory:
        corpus = read_texts_from_directory(corpus_directory)
    else:
        corpus = corpus_set or set()

    # Запросим у пользователя конфиг, если не передан
    if cfg is None:
        cfg: PMIConfig = ask_user_cfg(comparison=comparison)
    if cfg is None:
        cfg = PMIConfig(
            window_size=5,
            direction="forward",
            cross_sentences=False,
            include_stopwords=True,
            min_cooc=1,
            lemmatize=True,
        )

    # 2) Считаем таблицу метрик
    calc_mi = PMICalculator(cfg).fit(corpus)
    data_frame = calc_mi.compute_scores()

    # 3) Сразу посчитаем долю строк выше порога 0 для всех метрик
    shares = {}
    for metric in ("PMI", "Mod. MI"):
        shares[metric] = calc_mi.threshold_share(
            data_frame,
            metric=metric,
            threshold=0.0,
            weighted=False
        )

    return data_frame, shares, calc_mi


def display_mi_table(
        df: pd.DataFrame,
        calc_mi: PMICalculator
) -> None:
    """
    Интерактивный вывод таблицы с метриками PMI/Modified MI.
    — Спрашиваем у пользователя
    по какой метрике сортировать.
    — Предлагаем задать порог (threshold)
    для расчёта доли пар > threshold.
    — Выводим Average MI, Threshold MI и топ-N строк
    по выбранной метрике.
    """
    # --- Summary across all metrics
    # (using class methods) ---
    metrics_all = ["PMI", "Mod. MI"]

    print(Fore.GREEN + Style.BRIGHT +
          "\nAverage MI:")
    for m in metrics_all:
        avg_type = calc_mi._avg_type(df, m)
        print(f"  {m}: {avg_type:.3f}")

    print(Fore.GREEN + Style.BRIGHT +
          "\nThreshold MI (доля пар с MI > 0):")
    for m in metrics_all:
        share_type0 = calc_mi.threshold_share(
            df, metric=m, threshold=0.0, weighted=False
        )
        print(f"  {m}: {share_type0:.3f}")

    wait_for_enter_to_analyze()

    # 1) Выбор метрики сортировки
    while True:
        metric = input(
            Fore.GREEN + Style.BRIGHT +
            "\nВыберите метрику для сортировки таблицы "
            "(1 для PMI, 2 для Modified MI)\n"
            "или нажмите Enter для сортировки"
            " по Modified MI (рекомендовано):\n"
        ).strip()
        metric_map = {"1": "PMI", "2": "Mod. MI"}
        if metric == "":
            metric = "2"
        if metric in metric_map:
            metric = metric_map[metric]
            break
        print(Fore.LIGHTRED_EX +
              "Недопустимая метрика. "
              "Повторите ввод.\n")

    # 2) Выбор направления сортировки
    while True:
        order_input = input(
            Fore.GREEN + Style.BRIGHT +
            "Сортировать по "
            "убыванию значений (y/n)?\n"
        ).strip().lower()
        if order_input == "":
            descending = True
            break
        elif order_input in {"y", "yes"}:
            descending = True
            break
        elif order_input in {"n", "no"}:
            descending = False
            break
        else:
            print(Fore.LIGHTRED_EX +
                  "Ошибка! Введите 'y', "
                  "'n' или нажмите Enter.")

    # 3) Порог для доли > threshold
    while True:
        thr_raw = input(
            Fore.GREEN + Style.BRIGHT +
            "Введите пороговое значения "
            "для вывода (по умолчанию 0):\n"
        ).strip()
        if thr_raw == "":
            threshold = 0.0
            break
        try:
            threshold = float(thr_raw)
            break
        except ValueError:
            print(Fore.LIGHTRED_EX +
                  "Ошибка! Введите "
                  "числовое значение.")

    # 4) Сколько строк показывать
    while True:
        top_raw = input(
            Fore.GREEN + Style.BRIGHT +
            "Сколько верхних строк "
            "показать "
            "(по умолчанию 30)?\n"
        ).strip()
        if top_raw == "":
            top_n = 30
            break
        try:
            top_n = max(1, int(top_raw))
            break
        except ValueError:
            not_positive_int_error(0)

    # 5) Расчёт доли пар выше порога
    if metric not in df.columns:
        print(Fore.LIGHTRED_EX + Style.BRIGHT +
              f"В таблице нет столбца '{metric}'. "
              f"Доступные: {list(df.columns)}")
        return

    mask = df[metric] > threshold
    threshold_mi = float(mask.mean()) if not df.empty else float("nan")

    print(Fore.GREEN + Style.BRIGHT +
          f"\nThreshold MI {metric} > {threshold}: "
          f"{threshold_mi:.3f} "
          f"({threshold_mi * 100:.1f}% пар)")

    # 6) Сортировка и отображение топа
    cols = ["x", "y", "f(x,y)", "f(x)", "f(y)", "PMI", "Mod. MI"]
    present_cols = [c for c in cols if c in df.columns]
    df_sorted = df.sort_values(metric, ascending=not descending)
    to_show = df_sorted[present_cols].head(top_n)

    print(Fore.GREEN + Style.BRIGHT +
          f"Топ-{top_n}, отсортированный по {metric} "
          f"(descending={descending}):\n")
    pd.options.display.float_format = '{:.3f}'.format
    col_space = {c: 3 for c in to_show.columns}
    col_space.update({
        "x": 12,
        "y": 12,
        "f(x,y)": 6,
        "f(x)": 6,
        "f(y)": 6,
        "PMI": 6,
        "Mod. MI": 6,
    })
    print(to_show.to_string(index=False, col_space=col_space))
    wait_for_enter_to_analyze()


if __name__ == "__main__":
    # Пример того, как передавать директорию с Вашими текстами в метод calculate_pmi. Раскомментируйте код ниже.
    # directory = '../your_directory'  # (должна быть в головной директории)
    # example_1 = calculate_pmi(corpus_directory=directory)
    # display_pmi_table(example_1)

    # Список текстов для примера (сгенерированы ИИ)
    text_1 = """
    Осенний Космонавт Алексей ветер за окном напоминал о скором приходе холодов. Листья деревьев медленно кружились в воздухе, постепенно 
    покрывая землю золотым ковром. В парке гуляли немногочисленные прохожие, наслаждаясь последними тёплыми днями. Вдоль 
    аллеи бежала собака, радостно виляя хвостом. Маленький мальчик с интересом наблюдал за ней, крепко держа за руку свою 
    маму. Она говорила ему о том, как важно сохранять природу и уважать окружающий мир. Вдалеке был виден силуэт 
    человека, сидящего на лавочке с книгой. Он не спешил никуда, погружённый в чтение. Вокруг царила атмосфера 
    умиротворённости и спокойствия. Солнце постепенно уходило за горизонт, окутывая парк мягким оранжевым светом. Небо 
    меняло свой цвет, переходя от светло-голубого к насыщенному розовому. Птицы готовились к ночи, прячась в ветвях 
    деревьев. Где-то рядом слышался тихий плеск воды из фонтана. Люди начинали расходиться по домам, постепенно покидая 
    парк. И вот, когда город погрузился в вечерние сумерки, наступила долгожданная тишина.
    """

    text_2 = """
    Космонавт Алексей всегда мечтал о звёздах ветер за окном. С детства он читал книги о космосе и представлял себя на 
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

    from colorama import init

    texts_set = set()
    init(autoreset=True)

    # Добавляем тексты в set
    texts_set.update([text_1, text_2])

    # Передаём выбранный конфиг в расчёт
    df, shares, calc_mi = calculate_pmi(
        corpus_set=texts_set, comparison=False
    )

    display_mi_table(df, calc_mi)
