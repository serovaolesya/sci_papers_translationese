from datetime import datetime
import re

from colorama import Fore


def validate_gender(gender_string):
    """Метод для валидации пола автора(ов). Допускаются только 'м' или 'ж' через запятую."""
    while True:
        gender_list = [g.strip().lower() for g in gender_string.split(",")]
        if all(g in ['м', 'ж'] for g in gender_list) or all(g in ['муж', 'жен'] for g in gender_list):
            return ", ".join(gender_list)
        elif gender_string == "":
            return ""
        else:
            gender_string = input(Fore.LIGHTRED_EX + "Ошибка: введите только 'м' и/или 'ж' (через запятую): " + Fore.RESET).strip()


def validate_years(
        year_string,
        field_name,
        single_year=False
):
    """Метод для валидации годов рождения и публикации."""
    current_year = datetime.now().year
    year_pattern = r"\b(19[0-9]{2}|20[0-9]{2})\b"
    years = re.findall(year_pattern, year_string)
    # Преобразуем к числовому типу
    # и проверяем диапазон
    valid_years = [
        int(year) for year in years
        if 1900 <= int(year) <= current_year
    ]

    if single_year and len(valid_years) != 1:
        raise ValueError(
            f"Проверьте правильность ввода {field_name}, "
            f"он должен быть в диапазоне "
            f"1900-{current_year} гг."
        )
    elif not valid_years:
        raise ValueError(
            f"Проверьте правильность ввода. "
            f"Выберите {field_name} в диапазоне "
            f"1900-{current_year} гг.")

    return ", ".join(map(str, valid_years))


