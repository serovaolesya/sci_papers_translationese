# -*- coding: utf-8 -*-
from typing import Dict
import re

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


class RuNounAnalyzer:
    """
    Расчёт четырёх показателей по существительным
    для русского текста (один проход по токенам).

    Индикаторы:
      - Доля существительных в тексте (nouns_ratio)
      - Доля существительных среднего рода (neuter_nouns_ratio)
      - Доля существительных в единственном числе (singular_nouns_ratio)
      - Доля абстрактных существительных (abstract_nouns_ratio)
    """

    def __init__(self):
        self.total_tokens: int = 0
        self.total_nouns: int = 0
        self.total_singular: int = 0
        self.total_neuter: int = 0
        self.total_abstract: int = 0

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
                'Sing', 'Plur', 'Masc', 'Fem', 'Neut',
                'Nom', 'Gen', 'Dat', 'Acc', 'Ins', 'Loc',
                'Anim', 'Inan', 'Pos', 'Cmp', 'Sup',
                'Neg', 'Yes', 'No', 'Variant', 'Short'
            }
            for c in candidates:
                if c in KNOWN:
                    result.add(c)

        return result

    @staticmethod
    def _is_abstract_noun(lemma: str) -> bool:
        """
        Эвристическое определение абстрактного существительного
        по характерным суффиксам.
        """
        abstract_suffixes = [
            'ость', 'ние', 'тие', 'ция', 'сия', 'изм', 'ика',
            'ство', 'енность', 'анность', 'ант', 'ент', 'ура',
            'логия', 'фия', 'метрия', 'есть', 'ствие',
            'ота', 'сть', 'номия', 'графия', 'патия',
        ]
        lemma_lower = lemma.lower()
        return any(lemma_lower.endswith(suffix)
                   for suffix in abstract_suffixes)

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

        # Лемматизация (нужна для определения абстрактности)
        for token in doc.tokens:
            token.lemmatize(_MORPH_VOCAB)

        # Все словарные токены
        self.total_tokens = sum(
            1 for t in doc.tokens if t.text.isalpha()
        )

        # Анализ существительных
        for token in doc.tokens:
            if token.pos != 'NOUN':
                continue

            self.total_nouns += 1
            feats = self._feats_set(token)

            # Единственное число
            if 'Sing' in feats:
                self.total_singular += 1

            # Средний род
            if 'Neut' in feats:
                self.total_neuter += 1

            # Абстрактность (по лемме)
            if hasattr(token, 'lemma') and token.lemma:
                if self._is_abstract_noun(token.lemma):
                    self.total_abstract += 1

    # ----------------------- Метрики (соотношения) -------------------------

    def ratio_nouns_to_tokens(self) -> float:
        """Доля существительных среди всех словарных токенов."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_nouns / self.total_tokens

    def ratio_singular_to_nouns(self) -> float:
        """Доля существительных в единственном числе."""
        if self.total_nouns == 0:
            return 0.0
        return self.total_singular / self.total_nouns

    def ratio_neuter_to_nouns(self) -> float:
        """Доля существительных среднего рода."""
        if self.total_nouns == 0:
            return 0.0
        return self.total_neuter / self.total_nouns

    def ratio_abstract_to_nouns(self) -> float:
        """Доля абстрактных существительных."""
        if self.total_nouns == 0:
            return 0.0
        return self.total_abstract / self.total_nouns

    # ----------------------- Утилиты форматирования ------------------------

    @staticmethod
    def pct(x: float) -> str:
        return f"{x * 100:.2f}"

    def print_analysis(self):
        print(Fore.GREEN + Style.BRIGHT +
              "      РЕЗУЛЬТАТЫ АНАЛИЗА СУЩЕСТВИТЕЛЬНЫХ")
        print(Fore.BLACK + Style.NORMAL +
              "Всего токенов:", self.total_tokens)
        print("Всего существительных:", self.total_nouns)
        print("— единственное число:", self.total_singular)
        print("— средний род:", self.total_neuter)
        print("— абстрактные:", self.total_abstract)
        print("Доля существительных к словарным токенам (%):",
              self.pct(self.ratio_nouns_to_tokens()))
        print("Доля сущ. среднего рода ко всем сущ. (%):",
              self.pct(self.ratio_neuter_to_nouns()))
        print("Доля сущ. в ед. числе ко всем сущ. (%):",
              self.pct(self.ratio_singular_to_nouns()))
        print("Доля абстрактных сущ. ко всем сущ. (%):",
              self.pct(self.ratio_abstract_to_nouns()))
        wait_for_enter_to_analyze()


if __name__ == "__main__":
    analyzer = RuNounAnalyzer()
    text = input("Введите текст для анализа: ").strip()
    analyzer.analyze(text)
    analyzer.print_analysis()
