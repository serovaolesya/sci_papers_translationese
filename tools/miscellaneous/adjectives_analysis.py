# -*- coding: utf-8 -*-
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from colorama import Fore, Style
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)

from tools.core.utils import wait_for_enter_to_analyze

# ---- Глобальные (одноразовая инициализация для скорости) -----------------
_SEGMENTER = Segmenter()
_EMB = NewsEmbedding()
_MORPH = NewsMorphTagger(_EMB)
_MORPH_VOCAB = MorphVocab()


class RuAdjectiveAnalyzer:
    """
    Расчёт трёх показателей по прилагательным
    для русского текста (один проход по токенам).

    Индикаторы:
      - Доля прилагательных в тексте (adjectives_ratio)
      - Доля кратких прилагательных (short_adjectives_ratio)
      - Доля прилагательных в сравнительной степени (comparative_adjectives_ratio)
      - Доля прилагательных в превосходной степени (superlative_adjectives_ratio)

    Определения:
      - Прилагательное: token.pos == 'ADJ'
      - Краткое прилагательное: 'Short' ∈ feats
      - Степени сравнения: 'Cmp' (сравнительная), 'Sup' (превосходная),
        а также морфологические/аналитические эвристики (самый, наи-, -ейший/-айший)
    """

    def __init__(self):
        self.total_tokens: int = 0
        self.total_adjectives: int = 0
        self.total_short: int = 0
        self.total_comparative: int = 0
        self.total_superlative: int = 0

    @staticmethod
    def _feats_set(token) -> set:
        """
        Возвращает нормализованный набор морфологических
        меток Natasha/UD.
        """
        feats_obj = getattr(token, 'feats', None)
        result: set = set()
        if feats_obj is None:
            return result

        # 1) Попытка извлечь из типичных атрибутов контейнера
        for attr in ('values', 'grammemes', 'tags'):
            vals = getattr(feats_obj, attr, None)
            if vals:
                try:
                    for v in vals:
                        if isinstance(v, str):
                            result.add(v)
                except TypeError:
                    pass

        # 2) Если объект выглядит как dict‑like
        if hasattr(feats_obj, 'items'):
            try:
                for k, v in feats_obj.items():
                    if isinstance(k, str):
                        result.add(k)
                    if isinstance(v, str):
                        result.add(v)
            except Exception:
                pass

        # 3) Fallback: парсинг строкового представления
        if not result:
            s = str(feats_obj)
            candidates = re.findall(r"[A-Za-z][A-Za-z:]+", s)
            KNOWN = {
                'Pos', 'Cmp', 'Sup', 'Short',
                'Sing', 'Plur', 'Masc', 'Fem', 'Neut',
                'Nom', 'Gen', 'Dat', 'Acc', 'Ins', 'Loc',
                'Anim', 'Inan', 'Variant', 'Yes', 'No', 'Neg'
            }
            for c in candidates:
                if c in KNOWN:
                    result.add(c)

        return result

    @staticmethod
    def _get_degree(feats: set) -> str:
        """Определение степени сравнения по морфологическим меткам."""
        if 'Cmp' in feats:
            return 'comparative'
        elif 'Sup' in feats:
            return 'superlative'
        return 'positive'

    def analyze(self, text: str) -> None:
        """
        Размечает текст и заполняет агрегаты.
        Повторный вызов перезапишет агрегаты новыми значениями.
        """
        # Сброс счётчиков
        self.__init__()

        doc = Doc(text)
        doc.segment(_SEGMENTER)
        doc.tag_morph(_MORPH)

        # Лемматизация (нужна для эвристик степеней сравнения)
        for token in doc.tokens:
            token.lemmatize(_MORPH_VOCAB)

        # Все словарные токены
        self.total_tokens = sum(
            1 for t in doc.tokens if t.text.isalpha()
        )

        prev_token = None

        # Анализ прилагательных
        for token in doc.tokens:
            prev = prev_token

            if token.pos != 'ADJ':
                prev_token = token
                continue

            # «самый»/«сам» — маркер аналитической превосходной степени,
            # сам по себе не считается прилагательным, но сохраняется
            # как prev_token для распознавания суперлатива у следующего ADJ.
            lemma_lower_tmp = (
                getattr(token, 'lemma', None) or token.text
            ).lower()
            if lemma_lower_tmp in ['самый', 'сам']:
                prev_token = token
                continue

            self.total_adjectives += 1
            feats = self._feats_set(token)

            # Краткая форма
            if 'Short' in feats:
                self.total_short += 1

            # Степень сравнения (морфологические метки)
            degree = self._get_degree(feats)

            # Дополнительные эвристики для положительной степени
            if degree == 'positive':
                lemma_lower = (
                    getattr(token, 'lemma', None) or token.text
                ).lower()
                text_lower = token.text.lower()
                prev_lemma = (
                    getattr(prev, 'lemma', None) or
                    getattr(prev, 'text', '')
                ).lower() if prev is not None else ''

                # 1) Синтетический суперлатив: -ейш-/-айш- в ФОРМЕ слова
                #    (работает для всех падежей: высочайшими, красивейшего и т.д.)
                if re.search(r'(ейш|айш)', text_lower):
                    degree = 'superlative'
                # 2) Суплетивные формы превосходной степени
                elif text_lower in {
                    'лучший', 'наилучший', 'худший', 'наихудший'
                }:
                    degree = 'superlative'
                # 3) Префикс «наи-» (наибольший, наименьший, наилучший…)
                elif text_lower.startswith('наи'):
                    degree = 'superlative'
                # 4) Аналитический суперлатив: «самый»/«наиболее» перед прил.
                elif prev_lemma in {'самый', 'сам', 'наиболее'}:
                    degree = 'superlative'
                # 5) Аналитическая сравнительная: «более»/«менее» перед прил.
                elif prev_lemma in {'более', 'менее'}:
                    degree = 'comparative'

            if degree == 'comparative':
                self.total_comparative += 1
            elif degree == 'superlative':
                self.total_superlative += 1

            prev_token = token

    # ----------------------- Метрики (соотношения) -------------------------

    def ratio_adjectives_to_tokens(self) -> float:
        """Доля прилагательных среди всех словарных токенов."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_adjectives / self.total_tokens

    def ratio_short_to_adjectives(self) -> float:
        """Доля кратких прилагательных среди всех прилагательных."""
        if self.total_adjectives == 0:
            return 0.0
        return self.total_short / self.total_adjectives

    def ratio_comparative_to_adjectives(self) -> float:
        """Доля прилагательных в сравнительной степени."""
        if self.total_adjectives == 0:
            return 0.0
        return self.total_comparative / self.total_adjectives

    def ratio_superlative_to_adjectives(self) -> float:
        """Доля прилагательных в превосходной степени."""
        if self.total_adjectives == 0:
            return 0.0
        return self.total_superlative / self.total_adjectives

    # ----------------------- Утилиты форматирования ------------------------

    @staticmethod
    def pct(x: float) -> str:
        return f"{x * 100:.2f}"

    def print_analysis(self):
        print(Fore.GREEN + Style.BRIGHT +
              "      РЕЗУЛЬТАТЫ АНАЛИЗА ПРИЛАГАТЕЛЬНЫХ")
        print(Fore.BLACK + Style.NORMAL +
              "Всего токенов:", self.total_tokens)
        print("Всего прилагательных:", self.total_adjectives)
        print("— краткие:", self.total_short)
        print("— сравнительная степень:", self.total_comparative)
        print("— превосходная степень:", self.total_superlative)
        print("Доля прилагательных к словарным токенам (%):",
              self.pct(self.ratio_adjectives_to_tokens()))
        print("Доля кратких прилагательных ко всем прил. (%):",
              self.pct(self.ratio_short_to_adjectives()))
        print("Доля прил. в сравнительной степени ко всем прил. (%):",
              self.pct(self.ratio_comparative_to_adjectives()))
        print("Доля прил. в превосходной степени ко всем прил. (%):",
              self.pct(self.ratio_superlative_to_adjectives()))
        wait_for_enter_to_analyze()


if __name__ == "__main__":
    analyzer = RuAdjectiveAnalyzer()
    text = input("Введите текст для анализа: ").strip()
    analyzer.analyze(text)
    analyzer.print_analysis()
