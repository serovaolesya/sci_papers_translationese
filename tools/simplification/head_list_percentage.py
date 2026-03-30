# -*- coding: utf-8 -*-
from collections import Counter
from typing import Optional

from colorama import Fore, Style
from rich.console import Console
from rich.table import Table

from tools.core.utils import wait_for_enter_to_analyze, not_positive_int_error

console = Console()

# Леммы, которые исключаем из анализа LHR (можно расширять)
DEFAULT_EXCLUDED_LEMMAS = {
    "быть", "являться", "находиться", "иметь", "делать", "сделать",
    "мочь", "хотеть", "стать", "становиться", "оказаться", "казаться"
}



def ask_yes_no(
        prompt: str,
        default: bool = True
) -> bool:
    """
    Универсальный запрос да/нет.
    Enter -> значение по умолчанию.
    """
    default_hint = "Y/n" if default else "y/N"

    while True:
        user_input = input(
            f"{prompt} ({default_hint}): \n"
        ).strip().lower()

        if not user_input:
            return default

        if user_input in {"y", "yes", "д", "да"}:
            return True
        if user_input in {"n", "no", "н", "нет"}:
            return False

        print(Fore.RED + Style.BRIGHT +
              "Выберите один из допустимых вариантов.\n")


def ask_non_negative_int(
        prompt: str,
        default: int,
        min_value: int = 0
) -> int:
    """
    Запрашивает целое число >= min_value.
    Enter -> значение по умолчанию.
    """
    while True:
        user_input = input(
            f"{prompt} (по умолчанию "
            f"{default}):\n"
        ).strip()

        if not user_input:
            return default

        if user_input.isdigit():
            value = int(user_input)
            if value >= min_value:
                return value

        not_positive_int_error(min_value)


def calculate_lhrt(
        corpus_lemmas: list[list[str]],
        coverage_threshold: float = 0.1,
        total_tokens_per_text: Optional[list[int]] = None,
        exclude_lemmas: Optional[list[str]] = None,
        adjust_denominator: bool = False,
) -> dict:
    """
    Вычисляет List Head Ratio (LHRT) для корпуса текстов.

    Args:
        corpus_lemmas: Список списков лемм знаменательных слов для каждого текста
                       [[лемма1, лемма2, ...], [лемма1, лемма3, ...], ...]
        coverage_threshold: Пороговое значение индивидуального покрытия для
                           включения леммы в list head (в процентах, по умолчанию 0.1%).
                           Лемма включается, если
                           (частота_леммы / размер_корпуса) * 100 >= coverage_threshold
        total_tokens_per_text: (опционально) список общих количеств словарных токенов
                               по каждому тексту (включая незнаменательные). Если
                               задан, используется как знаменатель LHRT и для
                               расчёта общего числа токенов корпуса.
        exclude_lemmas: (опционально) список лемм, которые полностью
                        исключаются из анализа (не попадают ни в частоты,
                        ни в покрытие топ-лемм). По умолчанию — None.
        adjust_denominator: если True, то исключаемые леммы также вычитаются
                            из знаменателя (N) при расчёте LHR.

    Returns:
        Словарь с результатами анализа:
        - 'corpus_frequency': Counter с частотностью всех лемм (по corpus_lemmas)
        - 'list_head_lemmas': Список лемм, попавших в list head (покрытие >= threshold)
        - 'text_lhrt': Список значений LHRT для каждого текста
        - 'mean_lhrt': Среднее значение LHRT по корпусу
        - 'total_corpus_tokens': Общее количество токенов в корпусе
        - 'list_head_coverage': Сколько токенов покрывают леммы list head
        - 'n_list_head': Количество лемм в list head (определено эмпирически)
        - 'coverage_threshold': Использованный порог покрытия
    """

    # Валидация входа
    if (total_tokens_per_text is not None
            and len(total_tokens_per_text) != len(corpus_lemmas)):
        raise ValueError(
            "Длина total_tokens_per_text должна совпадать "
            "с количеством текстов в corpus_lemmas"
        )

    # Подготовка множества исключаемых лемм
    exclude_set = set(exclude_lemmas) if exclude_lemmas else set()

    # 1. Фильтруем леммы и считаем частоты корпуса без исключаемых лемм
    filtered_corpus_lemmas = []
    excluded_counts_per_text = []

    for text_lemmas in corpus_lemmas:
        filtered = [lemma for lemma in text_lemmas if lemma not in exclude_set]
        filtered_corpus_lemmas.append(filtered)
        excluded_counts_per_text.append(len(text_lemmas) - len(filtered))

    all_lemmas = []
    for filtered in filtered_corpus_lemmas:
        all_lemmas.extend(filtered)

    corpus_frequency = Counter(all_lemmas)

    # Размер корпуса (в токенах): учитываем вычитание
    # исключаемых лемм из N при необходимости
    if total_tokens_per_text is not None:
        base_total = int(sum(max(0, int(n)) for n in total_tokens_per_text))
        if adjust_denominator:
            base_total -= sum(excluded_counts_per_text)
        # Избегаем деления на 0
        total_corpus_tokens = max(1, base_total)
    else:
        total_corpus_tokens = max(1, len(all_lemmas))

    # 2. Определяем list head по критерию индивидуального покрытия
    # отбираем леммы, для которых
    # (частота / размер_корпуса) * 100 >= coverage_threshold
    list_head_lemmas = []
    for lemma, freq in corpus_frequency.items():
        individual_coverage = (freq / total_corpus_tokens) * 100
        if individual_coverage >= coverage_threshold:
            list_head_lemmas.append(lemma)

    # Сортируем по убыванию частоты для удобства
    list_head_lemmas.sort(key=lambda x: corpus_frequency[x], reverse=True)
    list_head_set = set(list_head_lemmas)
    n_list_head = len(list_head_lemmas)

    # Общее количество токенов, покрытых list head во всём корпусе
    list_head_coverage = sum(corpus_frequency[lemma] for lemma in list_head_lemmas)

    # 3. Вычисляем LHRT для каждого текста
    texts_lhrt_values = []
    denominators = []

    for idx, text_lemmas in enumerate(corpus_lemmas):
        filtered = filtered_corpus_lemmas[idx]

        if total_tokens_per_text is not None:
            denom = int(total_tokens_per_text[idx])
            if adjust_denominator:
                denom = max(0, denom - excluded_counts_per_text[idx])
        else:
            denom = len(filtered)

        denominators.append(denom)

        if denom <= 0:
            texts_lhrt_values.append(0.0)
            continue

        # Числитель считаем по ОТФИЛЬТРОВАННЫМ леммам
        list_head_tokens_in_text = sum(
            1 for lemma in filtered if lemma in list_head_set
        )
        # LHRT по формуле: (V_n / N) * 100
        text_lhrt = (list_head_tokens_in_text / denom) * 100
        texts_lhrt_values.append(text_lhrt)

    # 4. Вычисляем средние значения LHR: невзвешенное и взвешенное
    if texts_lhrt_values:
        mean_lhrt_unweighted = sum(texts_lhrt_values) / len(texts_lhrt_values)
        total_weight = sum(denominators)

        if total_weight > 0:
            mean_lhrt_weighted = sum(
                v * w for v, w in zip(texts_lhrt_values, denominators)
            ) / total_weight
        else:
            mean_lhrt_weighted = mean_lhrt_unweighted
    else:
        mean_lhrt_unweighted = 0.0
        mean_lhrt_weighted = 0.0

    # По умолчанию возвращаемое 'mean_lhrt' делаем взвешенным
    mean_lhrt = mean_lhrt_weighted

    return {
        'corpus_frequency': corpus_frequency,
        'list_head_lemmas': list_head_lemmas,
        'text_lhrt': texts_lhrt_values,
        'mean_lhrt_weighted': mean_lhrt_weighted,
        'mean_lhrt_unweighted': mean_lhrt_unweighted,
        'mean_lhrt': mean_lhrt,
        'total_corpus_tokens': total_corpus_tokens,
        'list_head_coverage': list_head_coverage,
        'n_list_head': n_list_head,
        'coverage_threshold': coverage_threshold
    }


def print_lhrt_results(
        results: dict,
        show_top_lemmas: Optional[bool] = None,
        max_lemmas_display: int = 20,
        show_per_text: Optional[bool] = None
):
    """
    Выводит результаты анализа LHR в удобочитаемом формате.

    Args:
        results: Результаты функции calculate_lhrt()
        show_top_lemmas: Показывать ли список лемм list head.
                         Если None, спросить у пользователя.
        max_lemmas_display: Максимальное количество лемм для отображения.
        show_per_text: Показывать ли LHR по каждому тексту.
                       Если None, спросить у пользователя.
    """
    total_tokens = results.get('total_corpus_tokens', 0)
    unique_lemmas = len(results.get('corpus_frequency', {}))
    n_list_head = results.get('n_list_head', 0)
    coverage = results.get('list_head_coverage', 0)
    threshold = results.get('coverage_threshold', 0.1)

    # Сводная таблица
    summary = Table()
    summary.add_column("Показатель", justify="left")
    summary.add_column("Значение", justify="center")

    pct_coverage = (coverage / total_tokens * 100) if total_tokens else 0.0

    summary.add_row(f"Минимальный порог покрытия для list head", f"{threshold}%")
    summary.add_row("Общее число токенов в корпусе", f"{total_tokens:,}")
    summary.add_row("Число уникальных лемм", f"{unique_lemmas:,}")
    summary.add_row(f"Число лемм в list head (n)", f"{n_list_head}")
    summary.add_row(f"Покрытие list head (токены)", f"{coverage:,}")
    summary.add_row(f"Покрытие list head (%)", f"{pct_coverage:.2f}%")
    summary.add_row("LHR среднее (взвешенное)",
                    f"{results['mean_lhrt_weighted']:.2f}%")
    summary.add_row("LHR среднее (невзвешенное)",
                    f"{results['mean_lhrt_unweighted']:.2f}%")
    summary.add_row("Количество текстов",
                    f"{len(results.get('text_lhrt', []))}")

    console.print(summary)
    wait_for_enter_to_analyze()

    # --- вывод list head по желанию ---
    if not show_top_lemmas:
        show_top_lemmas = ask_yes_no(
            Fore.BLUE + Style.BRIGHT +
            "Вывести леммы из list head на экран?",
            default=True
        )

    if show_top_lemmas:
        to_show_n = ask_non_negative_int(
            Fore.BLUE + Style.BRIGHT +
            "Сколько лемм из list head вывести на экран?",
            default=max_lemmas_display,
            min_value=1
        )

        if to_show_n > 0:
            print(
                Fore.GREEN + Style.BRIGHT +
                f"\n         ЛЕММЫ В LIST HEAD "
                f"(показано {min(to_show_n, len(results['list_head_lemmas']))} "
                f"из {n_list_head})"
            )

            table_top = Table()
            table_top.add_column("№", justify="center")
            table_top.add_column("Лемма", justify="center")
            table_top.add_column("Частота", justify="center")
            table_top.add_column("Покрытие, %", justify="center")

            lemmas_to_show = results['list_head_lemmas'][:to_show_n]
            freq = results['corpus_frequency']
            total_tok = results['total_corpus_tokens']

            for i, lemma in enumerate(lemmas_to_show, 1):
                f = freq[lemma]
                pct = (f / total_tok * 100) if total_tok else 0.0
                table_top.add_row(f"{i}", lemma, f"{f:,}", f"{pct:.3f}")

            console.print(table_top)

            if len(results['list_head_lemmas']) > to_show_n:
                print(
                    Fore.GREEN +
                    f"... и ещё {len(results['list_head_lemmas']) - to_show_n} лемм"
                )

            wait_for_enter_to_analyze()

    # --- вывод LHR по каждому тексту по желанию ---
    if show_per_text is None:
        show_per_text = ask_yes_no(
            Fore.GREEN + Style.BRIGHT +
            "Вывести LHR по текстам корпуса?",
            default=True
        )

    if show_per_text:
        print(
            Fore.GREEN + Style.BRIGHT +
            "\n      LHR ДЛЯ КАЖДОГО ТЕКСТА"
            "\n   (% ОТ ВСЕХ СЛОВАРНЫХ ТОКЕНОВ)"
        )

        table_per_text = Table()
        table_per_text.add_column("Текст", justify="center", min_width=15)
        table_per_text.add_column("LHR, %", justify="center", min_width=15)

        for i, value in enumerate(results.get('text_lhrt', []), 1):
            table_per_text.add_row(f"{i}", f"{value:.2f}")

        console.print(table_per_text)
        wait_for_enter_to_analyze()