# -*- coding: utf-8 -*-
from collections import Counter
from dataclasses import dataclass
from math import log2
from typing import Iterable, Optional
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pymorphy2.analyzer"
)

from tools.core.lemmatizators import parse_cached

from razdel import sentenize, tokenize
import pandas as pd
from tqdm import tqdm
import pymorphy2

# Инициализация морфологического анализатора
morph = pymorphy2.MorphAnalyzer()


@dataclass
class PMIConfig:
    # размер окна (k)
    window_size: int = 5
    # "sym" (±k, неупорядоченно) или
    # "forward" (только вправо, упорядоченно)
    direction: str = "forward"
    # Можно ли пересекать предложения
    cross_sentences: bool = False
    lowercase: bool = True
    lemmatize: bool = True
    include_stopwords: bool = True  # если False, будут отфильтрованы стоп-слова (список можно передать)
    stopwords: Optional[set] = None
    # отсечка по совместным появлениям для стабильности PMI
    min_cooc: int = 5
    # True — удаляем токены без букв (цифры/знаки)
    keep_alpha_only: bool = True


class PMICalculator:
    def __init__(self, cfg: PMIConfig):
        self.cfg = cfg
        # Для лемматизации
        self.morph = morph
        # Тут создаём кэш лемм (чтобы повторно
        # не лемматизировать одно и то же слово):
        # токен → лемма
        self._lemma_cache: dict[str, str] = {}

        # Счётчики: N — общее число токенов в тексте
        self.N_tokens: int = 0

        # Хранит суммарное число соседних позиций,
        # которые мы фактически учли при обходе
        # текста окнами (для расчёта v).
        # Длина окна - количество слов справа /
        # вокруг анализируемого слова
        # (само слово не считается как элемент окна)
        self.total_window_positions: int = 0

        # Хранит для каждой леммы, сколько раз
        # она встретилась в корпусе после предобработки
        # (регистры, фильтры, лемматизация).
        self.f_token: Counter = Counter()  # f(x)
        # сколько раз пара слов попала в одно окно
        self.f_pair: Counter = Counter()  # f(x,y)
        # Использовался ли avg_window при расчёте PMI
        self.avg_window_used: bool = False

    def _lemma(self, token: str) -> str:
        # Получаем лемму слова или извлекаем
        # её из кэша, чтобы не делать лишнюю
        # работу и сократить время обработки
        if token in self._lemma_cache:
            return self._lemma_cache[token]
        # normal_form уже даёт лемму
        lemma = parse_cached(token).normal_form
        self._lemma_cache[token] = lemma
        return lemma

    def _normalize_token(
            self, token: str
    ) -> Optional[str]:

        if self.cfg.lowercase:
            token = token.lower()
        # убираем пунктуацию
        if (self.cfg.keep_alpha_only
                and not any(char.isalpha() for char in token)):
            return None
        if self.cfg.lemmatize:
            token = self._lemma(token)
        if (not self.cfg.include_stopwords
                and self.cfg.stopwords
                and token in self.cfg.stopwords):
            return None
        return token

    def _text_to_sentences(
            self, text: str
    ) -> list[list[str]]:
        sents: list[list[str]] = []
        for sent in sentenize(text):
            tokens = [
                token.text for token
                in tokenize(sent.text)
            ]
            normalised_tokens = []
            for tok in tokens:
                norm_token = self._normalize_token(tok)
                if norm_token:
                    normalised_tokens.append(norm_token)
            if normalised_tokens:
                sents.append(normalised_tokens)
        return sents

    # ---------- основной подсчёт ----------
    def fit(self, texts: Iterable[str]):
        """
        Подсчитывает f(x), f(x, y), N, v,
        согласно выбранной конфигурации.
        """
        # Очистка внутренних счетчиков
        self.N_tokens = 0
        self.total_window_positions = 0
        self.f_token.clear()
        self.f_pair.clear()

        window_size = self.cfg.window_size
        mode = self.cfg.direction
        cross = self.cfg.cross_sentences

        for text in tqdm(texts, desc="Processing texts"):
            sents = self._text_to_sentences(text)

            # Задаем последовательности слов
            if cross:
                # Склеиваем все предложения
                # в один поток токенов
                # (один список для
                # всего корпуса)
                seq = [w for sent in sents
                       for w in sent]
                sequences = [seq]
            else:
                sequences = sents

            for seq in sequences:
                # Вычисляем длину
                # последовательности,
                # чтобы определять
                # границы окон.
                seq_len = len(seq)
                if seq_len == 0:
                    continue
                # Определяем частоты токенов и N
                for w in seq:
                    self.f_token[w] += 1
                self.N_tokens += seq_len

                if window_size <= 0:
                    continue

                if mode == "forward":
                    # упорядоченные пары:
                    # (x_token -> y_token),
                    # окно только вправо
                    for x_token_index, x_token in enumerate(seq):
                        # Добавляем единицу, так окно
                        # начинается со следующего слова
                        start = x_token_index + 1
                        # Предотвращаем выход за
                        # границы предложения
                        end = min(seq_len, x_token_index + 1 + window_size)
                        # Длина фактического правого окна
                        # (сколько позиций (слов) действительно вошло
                        # в окно)
                        self.total_window_positions += (end - start)
                        for y_token_index in range(start, end):
                            y_token = seq[y_token_index]
                            self.f_pair[(x_token, y_token)] += 1

                elif mode == "sym":
                    # симметричное окно ±k, пары неупорядоченные;
                    # считаем только j>w_index, чтобы не дублировать
                    for x_token_index, x_token in enumerate(seq):
                        # Левая граница окна (включительно),
                        # правая — ИСКЛЮЧИТЕЛЬНО (для range)
                        # Задаем левую границу окна
                        # (индекс самого левого слова в левом окне)
                        left_bound_idx = max(0, x_token_index - window_size)
                        # Задаем правую границу окна, добавляем +1,
                        # так как range сам по себе не включает
                        # последний индекс
                        right_bound_exclusive_idx = min(
                            seq_len, x_token_index + window_size + 1
                        )
                        left_positions_in_window = (
                                x_token_index - left_bound_idx
                        )
                        right_positions_in_window = (
                                right_bound_exclusive_idx -
                                x_token_index - 1
                        )
                        self.total_window_positions += (
                                left_positions_in_window +
                                right_positions_in_window)
                        # Почему range(x_token_index + 1,
                        # right_bound_exclusive_idx)
                        # и «куда делось левое окно?»
                        # В режиме sym пары считаются
                        # неупорядоченно (без направления).
                        # Чтобы не задвоить одну и ту же пару,
                        # мы берём только соседей справа
                        # от текущего слова:
                        # •	для центра x_token_index берём
                        # y_token_index в диапазоне (x+1 … right);
                        # •	левые соседи (индексы < x) не теряются:
                        # они будут посчитаны, когда сам левый
                        # сосед станет центром на своей итерации
                        # внешнего цикла.
                        for y_token_index in range(
                                x_token_index + 1,
                                right_bound_exclusive_idx
                        ):
                            y_token = seq[y_token_index]
                            # Канонизируем ключ неупорядоченной пары:
                            # ставим лексикографически меньший токен первым.
                            # Так ('хлеб','масло') и ('масло','хлеб')
                            # считаются одной парой ('масло','хлеб').
                            # Сравнение строк (x_token <= y_token) —
                            # обычное лексикографическое сравнение
                            # в Python (по Unicode).
                            # После канонизации увеличиваем счётчик
                            # совместных вхождений этой пары на 1.
                            pair = ((x_token, y_token)
                                    if x_token <= y_token
                                    else (y_token, x_token))
                            self.f_pair[pair] += 1
                else:
                    raise ValueError('Направление должно быть '
                                     '"sym" или "forward"')

        # Возвращаем экземпляр PMICalculator (как self) со всеми
        # заполненными полями.
        # К этому моменту всё необходимое уже посчитано и сохранено
        # внутри экземпляра:
        # •	self.cfg — конфигурация (window_size, direction, и т.д.).
        # •	self._lemma_cache — кэш лемм (слово → лемма).
        # •	self.N_tokens — общее число токенов в корпусе.
        # •	self.total_window_positions — суммарное число реально
        #   учтённых «позиций окна» (для расчёта среднего v).
        # •	self.f_token: Counter — частоты отдельных токенов f(x).
        # •	self.f_pair: Counter — частоты пар f(x, y)
        #   (уже с учётом выбранного режима forward/sym).
        # •	self.avg_window_used: bool — флаг, использовался ли v
        #   при расчёте PMI (дальше читает compute_scores()).
        # •	Плюс методы класса: _avg_window(), compute_scores(),
        # collocates_of(), explain_pair() и т.д.
        return self

    def _calc_avg_window(self) -> float:
        # avg_window — средняя фактическая длина окна
        # (сколько соседей учли на один токен)
        # Как считаем среднее v?
        # •	Мы хотим знать сколько соседей в среднем реально
        # попадает в окно у одного центрального слова.
        # •	Для каждого центра мы добавляем в счётчик столько
        #   позиций, сколько действительно рассмотрели
        #   (в sym: L+R, в forward: end-start).
        # •	Если сложить это по всем центрам, получим общее
        #   число реально просмотренных соседних позиций:
        #   total_window_positions.
        # •	Делим на количество центров (то есть на общее число
        # токенов N_tokens) и получаем среднее число соседей
        # на один токен — ровно то, что нам нужно как
        # эмпирическая средняя длина окна v.
        if self.N_tokens == 0:
            return 0.0
        return self.total_window_positions / self.N_tokens

    def compute_scores(self) -> pd.DataFrame:
        """
        Возвращает DataFrame со столбцами:
        x, y, f(x,y), fx, fy, PMI, Modified MI.
        Отсечка по min_cooc из конфигурации.
        """
        avg_window = self._calc_avg_window()
        N = self.N_tokens
        # Минимальный порог по совместным
        # появлениям (co-occurrences).
        # Зачем он нужен:
        # •	Когда пара слов встретилась только 1–2 раза,
        #   PMI может взлетать очень высоко
        #   (такое свойство метрики:
        #    редкие события дают огромный PMI).
        # •	Но это «шум», статистически неустойчивые пары.
        # •	Поэтому в конфигурации (PMIConfig) задаётся
        #   min_cooc - минимальное количество совместных
        #   вхождений, которое нужно для того, чтобы пара
        #   вообще попала в таблицу.
        min_c = self.cfg.min_cooc

        # Подготавливаем список строк rows, куда будем
        # складывать рассчитанные метрики по парам (x, y),
        rows = []

        for (x_token, y_token), f_x_token_y_token in self.f_pair.items():
            # Отбрасываем «слишком редкие»
            # пары (меньше порога min_cooc),
            # чтобы стабилизировать PMI.
            if f_x_token_y_token < min_c:
                continue
            # Берем маргинальные (одномерные) частоты,
            # т.е частоты каждого слова по отдельности
            f_x_token = self.f_token[x_token]
            f_y_token = self.f_token[y_token]

            # Если окно > 1, учитываем среднюю длину окна (avg_window),
            # т.к. увеличивается общее число позиций для co-occurrence.
            # Формула со всеми произведенными сокращениями:
            # PMI = log2( (N * f(x,y)) / (fx * fy * avg_window) ). Проверена!
            # Это уточнённая версия, эквивалентная log2(p(x,y)/p(x)p(y)).
            use_avg_window = self.cfg.window_size > 1
            # Произведение маргинальных частот: f(x) * f(y).
            # На англ. это product.
            fx_fy_product = f_x_token * f_y_token
            if use_avg_window and avg_window > 0:
                # Полный знаменатель для формулы с avg_window:
                # f(x)*f(y)*avg_window
                pmi = log2(
                    (N * f_x_token_y_token) / (fx_fy_product * avg_window)
                )
            else:
                pmi = log2((N * f_x_token_y_token) / fx_fy_product)
            self.avg_window_used = use_avg_window

            # В отличие от «чистого» PMI, Modified MI не переоценивает
            # редкие, случайные совпадения: если f_{xy} крошечная,
            # то Modified MI остаётся небольшой, даже при высоком PMI.
            modified_mi = f_x_token_y_token * pmi

            # Складываем в список кортежи с готовыми метриками для каждой пары:
            # — сами леммы x, y,
            # — их частоты fx, fy,
            # — совместная частота f(x,y),
            # — вычисленные PMI, Modified MI.
            rows.append((x_token, y_token, f_x_token_y_token,
                         f_x_token, f_y_token, pmi, modified_mi))

        # Превращаем список кортежей rows в табличную структуру
        # pandas DataFrame.
        data_frame = pd.DataFrame(rows,
                                  columns=["x", "y",
                                           "f(x,y)", "f(x)", "f(y)",
                                           "PMI", "Mod. MI"]
                                  )
        data_frame.sort_values(["Mod. MI", "PMI"], ascending=False, inplace=True)
        return data_frame

    # Вспомогательная выборка коллокатов заданной леммы
    # lemma: str — какая лемма нас интересует (например, "масло").
    # table: pd.DataFrame — таблица с результатами compute_scores()
    # (там столбцы x, y, f(x,y), fx, fy, PMI, Modified MI).
    # Метод ничего не пересчитывает, он просто фильтрует
    # и сортирует уже готовую таблицу.
    # topn: int = 30 — сколько лучших строк вернуть в конце
    # (top-N результатов после сортировки).
    def collocates_of(
            self, lemma: str,
            table: pd.DataFrame,
            topn: int = 30,
            side: str = "both"  # "right", "left", "both"
    ) -> pd.DataFrame:
        if side == "right":
            mask = (table["x"] == lemma)
        elif side == "left":
            mask = (table["y"] == lemma)
        elif side == "both":
            mask = (table["x"] == lemma) | (table["y"] == lemma)
        # table[m] — это срез DataFrame, содержащий
        # только релевантные строки.
        # .copy() делает отдельную копию, чтобы потом
        # безопасно добавлять новые столбцы
        # (без предупреждений Pandas).
        table_subset = table[mask].copy()

        # Добавляем столбец "collocate", чтобы было понятно,
        # какой второй коллокат к интересующему нас слову
        table_subset["collocate_word"] = table_subset.apply(
            lambda row: row["y"]
            if row["x"] == lemma
            else row["x"], axis=1
        )
        # Берём table_subset (где уже только коллокации интересующей леммы).
        # Сортируем строки:
        # сначала по Modified MI (чем выше, тем лучше),
        # при равенстве — по PMI,
        # ascending=False значит, что сортировка
        # по убыванию (сначала бОльшие значения).
        return table_subset.sort_values(
            ["Mod. MI", "PMI"], ascending=False
        ).head(topn)

    def explain_pair(
            self,
            x_token: str,
            y_token: str,
            normalize_input: bool = True
    ) -> dict:
        """
        Пошаговое объяснение расчёта PMI для пары (x, y)
        с учётом текущих настроек и счётчиков.
        Возвращает словарь с промежуточными величинами
        и печатает человекочитаемый отчёт.
        Если normalize_input=True, к входам применяется тот
        же препроцесс, что и к корпусу
        (нижний регистр, лемматизация, фильтры).
        Если после нормализации токен уходит в фильтр,
        он станет None и будет отмечен в отчёте.
        """
        # 1) Нормализация введенных токенов
        #    или оставление их "как есть"
        x_token_norm = (self._normalize_token(x_token)
                        if normalize_input
                        else x_token)
        y_token_norm = (self._normalize_token(y_token)
                        if normalize_input
                        else y_token)

        # Если что-то отфильтровалось
        if x_token_norm is None or y_token_norm is None:
            print("[explain_pair] Один из токенов был "
                  "отфильтрован при нормализации:",
                  x_token, y_token, "->",
                  x_token_norm, y_token_norm)
            return {
                "x_raw": x_token, "y_raw": y_token,
                "x": x_token_norm, "y": y_token_norm,
                "note": "Один из токенов отфильтрован "
                        "(например, небуквенный или"
                        " в списке стоп-слов)."
            }

        # 2) Выбор ключа пары с учётом режима.
        #    Ключи нужны для того, чтобы правильно достать
        #    из f_pair совместную частоту пары.
        if self.cfg.direction == "sym":
            # Кладём лексикографически “меньшее” слово первым.
            key = ((x_token_norm, y_token_norm)
                   if x_token_norm <= y_token_norm
                   else (y_token_norm, x_token_norm))
        else:
            key = (x_token_norm, y_token_norm)

        # 3) Извлекаем счётчики
        N = self.N_tokens
        use_avg_window = self.cfg.window_size > 1
        if use_avg_window:
            avg_window = self._calc_avg_window()
        else:
            avg_window = 0.0

        fx = self.f_token.get(x_token_norm, 0)
        fy = self.f_token.get(y_token_norm, 0)
        fxy = self.f_pair.get(key, 0)

        # 4) Вероятности по определению
        px = fx / N if N else 0.0
        py = fy / N if N else 0.0
        pxy = fxy / N if N else 0.0

        # 5) Отношение и PMI
        # PMI = log2( (N * f(x,y)) / (fx * fy * avg_window) ). Проверена!
        if use_avg_window and avg_window > 0 and N > 0:
            denominator = fx * fy * avg_window
        else:
            denominator = fx * fy

        numerator = N * fxy

        # Guard against zero denominator (e.g., unseen tokens or v == 0)
        if denominator == 0:
            ratio = 0.0
        else:
            ratio = numerator / denominator

        if ratio > 0:
            pmi = log2(ratio)
            modified_mi = fxy * pmi
        else:
            # пара не встречалась или нулевые маргиналы
            pmi = float("-inf")
            modified_mi = float("-inf") if fxy > 0 else 0.0

        # 6) Печать отчёта
        print("\nОТЧЁТ О ВЫБРАННОЙ ПАРЕ\n")
        print(f"Режим: {self.cfg.direction},\n"
              f"окно={self.cfg.window_size},\n"
              f"пересекать предложения={self.cfg.cross_sentences}")
        print(f"Токен x='{x_token}' → '{x_token_norm}',\n"
              f"токен y='{y_token}' → '{y_token_norm}'")
        print(f"Всего токенов (N) = {N}")
        print(f"Средняя длина окна (v) = {avg_window} "
              f"(Использована: {'да' if use_avg_window else 'нет'})")
        print(f"Частота x (f(x)) = {fx}, \n"
              f"Частота y (f(y)) = {fy}, \n"
              f"Совместная частота (f(x,y)) = {fxy}\n")

        if use_avg_window and avg_window > 0:
            # PMI с оконной поправкой
            print(
                "PMI = log2((N * f(x,y)) / (f(x) * f(y) * v)) = "
                f"log2(({N} * {fxy}) / ({fx} * {fy} * {avg_window:.6f}))"
                f" = {pmi:.3f}"
            )
        else:
            # Классический PMI без v
            print(
                "PMI = log2((N * f(x,y)) / (f(x) * f(y))) = "
                f"log2(({N} * {fxy}) / ({fx} * {fy})) "
                f"= {pmi:.3f}"
            )

        print(f"Modified MI = f(x,y) * PMI = {fxy} * {pmi:.3f} = {modified_mi:.3f}")

        return {
            "x_raw": x_token, "y_raw": y_token,
            "x_norm": x_token_norm, "y_norm": y_token_norm,
            "N": N, "avg_window": avg_window,
            "f(x)": fx, "f(y)": fy, "f(x,y)": fxy,
            "p(x)": px, "p(y)": py, "p(x,y)": pxy,
            "ratio": ratio,
            "PMI": pmi, "Mod. MI": modified_mi,
            "use_avg_window": use_avg_window
        }

    # ---------- Averages (macro/type vs micro/token-weighted) ----------
    def _filter_subset(
            self,
            table: pd.DataFrame,
            lemma: Optional[str],
            side: str
    ) -> pd.DataFrame:
        """Внутренний помощник: фильтрация по лемме и стороне."""
        if lemma is None:
            return table
        if side == "right":
            mask = (table["x"] == lemma)
        elif side == "left":
            mask = (table["y"] == lemma)
        else:  # "both"
            mask = (table["x"] == lemma) | (table["y"] == lemma)
        return table[mask]

    def _avg_type(
            self,
            table: pd.DataFrame,
            metric: str,  # PMI / Modified MI
            lemma: Optional[str] = None,
            side: str = "both",
            threshold: Optional[float] = None
    ) -> float:
        """
        Macro / type-average:Простое арифметическое среднее
        по всем уникальным парам
        Каждая пара имеет равный вес (1 голос = 1 пара)
        threshold: если задан, считаем среднее только
        по строкам, где metric > threshold.
        """
        subset = self._filter_subset(table, lemma, side)
        if threshold is not None:
            subset = subset[subset[metric] > threshold]
        return float(
            subset[metric].mean()
        ) if not subset.empty else float("nan")

    def _avg_weighted(
            self,
            table: pd.DataFrame,
            metric: str,  # PMI / Modified MI
            lemma: Optional[str] = None,
            side: str = "both",
            threshold: Optional[float] = None
    ) -> float:
        """
        Micro / token-weighted: Взвешенное среднее,
        где вес = частота совместной встречаемости (f(xy))
        Чаще встречающиеся пары имеют больший вес
        fxy (каждое совместное вхождение — один «голос»).
        """
        subset = self._filter_subset(table, lemma, side)
        if threshold is not None:
            # Удаляем строки со значением метрики < threshold
            subset = subset[subset[metric] > threshold]
        # Если подтаблица пуста, то нет строк,
        # удовлетворяющим условиям, возвращаем nan
        if subset.empty:
            return float("nan")
        # Подсчитываем средние взвешенные значений.
        # Извлекаем столбец с совместными появлениями
        w = subset["f(x,y)"]
        w_sum = float(w.sum())
        if w_sum <= 0:
            return float("nan")
        # subset[metric] * w — поэлементное умножение двух Series
        # ⇒ получаем Series из «взвешенных значений метрики»:
        # metric_i * fxy_i.
        return float((subset[metric] * w).sum() / w_sum)

    # ---- PMI ----
    def avg_pmi_type(
            self,
            table: pd.DataFrame,
            lemma: Optional[str] = None,
            side: str = "both",
            threshold: Optional[float] = None) -> float:
        return self._avg_type(table, "PMI", lemma, side, threshold)

    def avg_pmi_weighted(self, table: pd.DataFrame,
                         lemma: Optional[str] = None, side: str = "both",
                         threshold: Optional[float] = None) -> float:
        return self._avg_weighted(table, "PMI", lemma, side, threshold)

    # ---- Modified MI ----
    def avg_modified_mi_type(self, table: pd.DataFrame,
                              lemma: Optional[str] = None, side: str = "both",
                              threshold: Optional[float] = None) -> float:
        return self._avg_type(table, "Mod. MI", lemma, side, threshold)

    def avg_modified_mi_weighted(self, table: pd.DataFrame,
                                  lemma: Optional[str] = None, side: str = "both",
                                  threshold: Optional[float] = None) -> float:
        return self._avg_weighted(table, "Mod. MI", lemma, side, threshold)

    def threshold_share(
            self,
            table: pd.DataFrame,
            metric: str,
            lemma: Optional[str] = None,
            side: str = "both",
            threshold: float = 0.0,
            weighted: bool = False,
    ) -> float:
        """
        Доля пар, у которых выбранная метрика > threshold.
        По умолчанию — по типам (каждая уникальная пара = 1 голос).
        Если weighted=True, считаем токен-взвешенную долю (вес = fxy).
        """
        # Проверим, что метрика есть в таблице
        if metric not in table.columns:
            raise ValueError(f"Значение '{metric}' "
                             f"отсутствует в таблице: "
                             f"{list(table.columns)}")

        subset = self._filter_subset(table, lemma, side)
        if subset.empty:
            return float("nan")

        mask = subset[metric] > threshold

        if not weighted:
            # Type-based: простая доля строк,
            # где metric > threshold
            return float(mask.mean())
        else:
            # Token-weighted: доля по сумме весов
            # fxy в строках, где metric > threshold
            w = subset["f(x,y)"]
            w_total = float(w.sum())
            if w_total <= 0:
                return float("nan")
            return float(w[mask].sum() / w_total)


if __name__ == "__main__":
    # Пример: корпус как список строк (каждая строка = документ)
    # Замените на свою загрузку (читать файлы, БД и т.п.)
    corpus = [
        """
        Осенний Космонавт Алексей ветер за окном напоминал о скором приходе холодов. Листья деревьев медленно кружились в воздухе, постепенно 
        покрывая землю золотым ковром. В парке гуляли немногочисленные прохожие, наслаждаясь последними тёплыми днями. Вдоль 
        аллеи бежала собака, радостно виляя хвостом. Маленький мальчик с интересом наблюдал за ней, крепко держа за руку свою 
        маму. Она говорила ему о том, как важно сохранять природу и уважать окружающий мир. Вдалеке был виден силуэт 
        человека, сидящего на лавочке с книгой. Он не спешил никуда, погружённый в чтение. Вокруг царила атмосфера 
        умиротворённости и спокойствия. Солнце постепенно уходило за горизонт, окутывая парк мягким оранжевым светом. Небо 
        меняло свой цвет, переходя от светло-голубого к насыщенному розовому. Птицы готовились к ночи, прячась в ветвях 
        деревьев. Где-то рядом слышался тихий плеск воды из фонтана. Люди начинали расходиться по домам, постепенно покидая 
        парк. И вот, когда город погрузился в вечерние сумерки, наступила долгожданная тишина.
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
        """]

    cfg = PMIConfig(
        window_size=5,
        direction="forward",
        cross_sentences=False,
        include_stopwords=True,
        min_cooc=1,
        lemmatize=True
    )

    calculate_pmi = PMICalculator(cfg).fit(corpus)
    data_frame = calculate_pmi.compute_scores()

    # print(data_frame)

    # --- Full, no-ellipsis DataFrame printing ---
    import pandas as pd

    # Показывать все строки целиком
    pd.set_option('display.max_rows', None)
    # Показывать все столбцы
    pd.set_option('display.max_columns', None)
    # Отключает фиксированную ширину вывода.
    # Pandas не будет насильно переносить строки
    # под ширину терминала.
    pd.set_option('display.width', None)
    # Не обрезать содержимое ячеек
    pd.set_option('display.max_colwidth', None)
    # Разрешить многострочный вывод таблицы,
    # чтобы избежать ... и печатать по нескольку
    # строк шириной — более читабельно для
    # широких таблиц.
    pd.set_option('display.expand_frame_repr', True)
    # Глобальный формат чисел с плавающей запятой —
    # тут до 2 знаков после запятой
    pd.options.display.float_format = '{:.2f}'.format

    # Печать топа
    print(data_frame.to_string(
        index=False,
        max_rows=None,
        max_cols=None)
    )

    # Пример: коллокаты для леммы
    print("\nКоллокаты для выбранной леммы':")
    res = calculate_pmi.collocates_of("хлеб", data_frame, topn=10)
    print(res.to_string(index=False, max_rows=None, max_cols=None))

    # --- Пояснение по шагам пары слов ---
    print("\n* Разбор по шагам разных MI для выбранной пары слов':")
    a = calculate_pmi.explain_pair("хлеб", "масло")

    # Просто среднее арифметическое всех значений
    # столбца PMI в твоей таблице:
    # avg_windowed_pmi = data_frame["PMI"].mean()
    # print(avg_windowed_pmi)
    # Или вот так посчитать (с методом)
    print('Выводим avg_pmi по типам:')
    avg_pmi_macro = calculate_pmi.avg_pmi_type(data_frame)
    print(avg_pmi_macro)

    print('Выводим avg_modified_mi по типам:')
    avg_modified_mi_macro = calculate_pmi.avg_modified_mi_type(data_frame)
    print(avg_modified_mi_macro)

    print('Выводим avg_pmi взвешенное:')
    avg_pmi_micro = calculate_pmi.avg_pmi_weighted(data_frame)
    print(avg_pmi_micro)

    print('Выводим avg_modified_mi взвешенное:')
    avg_modified_mi_micro = calculate_pmi.avg_modified_mi_weighted(data_frame)
    print(avg_modified_mi_micro)

    # # только для леммы "масло", обе стороны
    # avg_modified_mi_macro_maslo = calculate_pmi.avg_modified_mi_type(
    #     data_frame, lemma="слово", side="both"
    # )

    # Примеры расчёта доли пар выше порога
    # (threshold share) для PMI
    print('Выводим threshold_share pmi по типам:')
    share_pos_pmi = calculate_pmi.threshold_share(
        data_frame, metric="PMI", threshold=0
    )
    print(share_pos_pmi)
    print('Выводим threshold_share pmi взвешенного:')
    share_pos_pmi_weighted = calculate_pmi.threshold_share(
        data_frame, metric="PMI", threshold=0, weighted=True
    )
    print(share_pos_pmi_weighted)

    # Примеры расчёта доли пар выше порога
    # (threshold share) для Modified MI
    print('Выводим threshold_share modified_mi по типам:')
    share_pos_modified_mi = calculate_pmi.threshold_share(
        data_frame, metric="Mod. MI", threshold=0
    )
    print(share_pos_modified_mi)
    print('Выводим threshold_share modified_mi взвешенного:')
    share_pos_modified_mi_weighted = calculate_pmi.threshold_share(
        data_frame, metric="Mod. MI", threshold=0, weighted=True
    )
    print(share_pos_modified_mi_weighted)


