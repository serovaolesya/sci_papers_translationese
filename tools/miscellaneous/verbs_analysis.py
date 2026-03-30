# -*- coding: utf-8 -*-
from typing import Dict, Tuple, Optional
import re
import csv

from colorama import Fore, Style
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc
)

from tools.core.utils import wait_for_enter_to_analyze

# ---- Глобальные (одноразовая инициализация для скорости) -----------------
_SEGMENTER = Segmenter()
_EMB = NewsEmbedding()
_MORPH = NewsMorphTagger(_EMB)
# Синтаксис подключаем по требованию
# (дороже по времени и памяти)
_SYNTAX = NewsSyntaxParser(_EMB)


class RuVerbAnalyzer():
    """
    Быстрый расчёт показателей по глагольным
    формам для русского текста.

    Все вычисления делаются за один
    проход по токенам.

    Определения:
      - Глагол: token.pos == 'VERB'
      - Причастие: 'Part' ∈ feats
          * Краткое причастие: 'Short' ∈ feats
          * Полное причастие: 'Part' ∈ feats и 'Short' ∉ feats
      - Деепричастие: 'Conv' ∈ feats
      - Инфинитив: 'Inf' ∈ feats
      - Фенитная форма: VERB, у которого нет Part/Conv/Inf
      - Вид: 'Imp' (несов.), 'Perf' (сов.)
    """

    def __init__(self, use_syntax: bool = False):
        self.use_syntax = use_syntax

        self.total_tokens: int = 0
        self.total_verbs_and_verbids: int = 0
        self.total_finite: int = 0
        self.total_participles: int = 0
        self.total_participles_short: int = 0
        self.total_participles_full: int = 0
        self.total_converbs: int = 0  # деепричастия
        self.total_infinitives: int = 0

        self.total_imp_all_verbs: int = 0
        self.total_perf_all_verbs_and_verbids: int = 0
        self.total_imp_finite: int = 0
        self.total_perf_finite: int = 0

        # Для эвристики препозиции причастий
        self.preposed_participles: int = 0
        self.non_preposed_participles: int = 0

        # Времена и наклонения — только для ФИНИТНЫХ форм
        self.total_finite_pres: int = 0
        self.total_finite_past: int = 0
        self.total_finite_fut: int = 0

        self.total_finite_mood_ind: int = 0

        # Времена — среди ВСЕХ глагольных форм
        self.total_all_pres: int = 0
        self.total_all_past: int = 0
        self.total_all_fut: int = 0

    @staticmethod
    def _feats_set(token) -> set:
        """
        Возвращает нормализованный набор морфологических
        меток Natasha/UD.
        Работает устойчиво с разными представлениями token.feats:
        - объект с атрибутами .values/.grammemes/.tags
        - dict‑like (ключи/значения)
        - строковое представление вида "<Imp,Ind,Sing,Past,Fin,Act>" или
          "{'VerbForm': 'Part', ...}"
          (в этом случае применяется regex‑парсинг)
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
            # Кандидаты: слова из латинских
            # букв (включая метки с двоеточием)
            candidates = re.findall(r"[A-Za-z][A-Za-z:]+", s)
            KNOWN = {
                # VerbForm & related
                'Part', 'Short', 'Conv', 'Inf', 'Fin',
                # Aspect / Tense / Mood / Voice
                'Imp', 'Perf', 'Past', 'Pres', 'Fut',
                'Ind', 'Cnd', 'Act', 'Pass', 'Mid',
                # Agreement / Case / Animacy
                'Masc', 'Fem', 'Neut', 'Sing', 'Plur',
                'Nom', 'Gen', 'Dat', 'Acc', 'Ins', 'Loc', 'Anim', 'Inan',
                # Degree/Polarity/Other common flags
                'Pos', 'Cmp', 'Sup', 'Neg', 'Yes', 'No', 'Variant'
            }
            for c in candidates:
                if c in KNOWN:
                    result.add(c)

        return result

    @staticmethod
    def _is_participle(feats: set) -> bool:
        return "Part" in feats

    @staticmethod
    def _is_short_participle(feats: set) -> bool:
        return "Part" in feats and "Short" in feats

    @staticmethod
    def _is_full_participle(feats: set) -> bool:
        return "Part" in feats and "Short" not in feats

    @staticmethod
    def _is_converb(feats: set) -> bool:
        return "Conv" in feats

    @staticmethod
    def _is_infinitive(feats: set) -> bool:
        return "Inf" in feats

    @staticmethod
    def _is_finite(feats: set) -> bool:
        # Фенитная форма: нет маркеров Part/Conv/Inf
        return (
                "Part" not in feats and
                "Conv" not in feats and
                "Inf" not in feats
        )

    @staticmethod
    def _is_imp(feats: set) -> bool:
        return "Imp" in feats

    @staticmethod
    def _is_perf(feats: set) -> bool:
        return "Perf" in feats

    def analyze(self, text: str) -> None:
        """
        Размечает текст и заполняет агрегаты.
        Повторный вызов перезапишет агрегаты
        новыми значениями.
        """
        # Сброс счётчиков
        self.__init__(use_syntax=self.use_syntax)

        doc = Doc(text)
        doc.segment(_SEGMENTER)
        doc.tag_morph(_MORPH)
        if self.use_syntax:
            doc.parse_syntax(_SYNTAX)

        self.total_tokens = sum(1 for t in doc.tokens
                                if t.text.isalpha())

        # Для эвристики препозиции:
        # нужен быстрый доступ по id -> token
        id2token = {}
        if self.use_syntax:
            for sent in doc.sents:
                for token in sent.tokens:
                    # id2token: Dict[str, DocToken]
                    # Ключ = f"{sent_index}_{token_index}" (строка),
                    # Значение = DocToken(text=..., id=..., head_id=...,
                    # rel=..., pos=..., feats=...)
                    # Нужно для быстрого доступа к "голове" и др.
                    # связям по head_id при синтаксическом разборе.
                    id2token[token.id] = token

        for token in doc.tokens:
            # Сохраняем feats для токенов
            feats = self._feats_set(token)

            # Считаем глагольными любые формы:
            # VERB, а также токены
            # с признаками Part/Conv/Inf

            # Это причастие?
            is_part = self._is_participle(feats)

            # Это деепричастие?
            is_conv = self._is_converb(feats)

            is_inf = self._is_infinitive(feats)

            is_verb_like = ((token.pos == "VERB")
                            or is_part
                            or is_conv
                            or is_inf)
            if not is_verb_like:
                continue

            self.total_verbs_and_verbids += 1

            # Вид среди всех глагольных форм
            if self._is_imp(feats):
                self.total_imp_all_verbs += 1
            if self._is_perf(feats):
                self.total_perf_all_verbs_and_verbids += 1

            # Времена для всех глагольных форм
            if "Pres" in feats:
                self.total_all_pres += 1
            if "Past" in feats:
                self.total_all_past += 1
            if "Fut" in feats:
                self.total_all_fut += 1
                # print(token)

            # Финитные формы считаем только среди
            # истинных VERB без Part/Conv/Inf
            is_fin = ((token.pos == "VERB")
                      and self._is_finite(feats))

            if is_fin:
                self.total_finite += 1
                if self._is_imp(feats):
                    self.total_imp_finite += 1
                if self._is_perf(feats):
                    self.total_perf_finite += 1
                # Времена для финитных форм
                if "Pres" in feats:
                    self.total_finite_pres += 1
                if "Past" in feats:
                    self.total_finite_past += 1
                if "Fut" in feats:
                    self.total_finite_fut += 1

                # Изъявительное наклонение для финитных форм
                if "Ind" in feats:
                    self.total_finite_mood_ind += 1

            if is_part:
                self.total_participles += 1
                if self._is_short_participle(feats):
                    self.total_participles_short += 1
                elif self._is_full_participle(feats):
                    self.total_participles_full += 1
                    # print(token)

                    # Эвристика "препозиция причастия" —
                    # считаем только для ПОЛНЫХ причастий
                    if self.use_syntax and token.head_id is not None:
                        head = id2token.get(token.head_id)
                        if head and head.pos == "NOUN":
                            # 1) Надёжнее сравнивать реальные позиции в тексте (start),
                            # а не строковые id вида "1_10" vs "1_2" (лексикографический баг).
                            if (getattr(token, 'start', None) is not None
                                    and getattr(head, 'start', None) is not None):
                                preposed = token.start < head.start
                            else:
                                # 2) Фоллбэк: пробуем разобрать id как пары целых
                                # (sent_idx, tok_idx)
                                try:
                                    t_sent, t_idx = (int(x) for x in token.id.split('_', 1))
                                    h_sent, h_idx = (int(x) for x in head.id.split('_', 1))
                                    preposed = (t_sent, t_idx) < (h_sent, h_idx)
                                except Exception:
                                    # 3) Самый грубый фоллбэк: строковое сравнение
                                    # (нежелательно, но лучше, чем падать)
                                    preposed = token.id < head.id

                            if preposed:
                                self.preposed_participles += 1
                                # print(f"[PREPOSED] причастие: {token.text!r} "
                                #       f"→ головное NOUN: {head.text!r}")
                            else:
                                self.non_preposed_participles += 1
                                # print(f"[POSTPOSED] причастие: {token.text!r} "
                                #       f"→ головное NOUN: {head.text!r}")
                        # else:
                        #     # Отладочная информация: нет головы или голова не NOUN
                        #     if head is None:
                        #         print(f"[SKIP] у {token.text!r} нет головы в "
                        #               f"id2token (head_id={token.head_id!r})")
                        #     else:
                        #         print(f"[SKIP] голова {head.text!r} имеет "
                        #               f"POS={head.pos!r}, ожидается 'NOUN'")

            if is_conv:
                self.total_converbs += 1

            if is_inf:
                self.total_infinitives += 1



    # ----------------------- Метрики (соотношения) -------------------------

    def ratio_preposed_participles(self) -> float:
        """Причастия в препозиции / все причастия."""
        total = self.total_participles
        if total == 0:
            return 0.0
        return self.preposed_participles / total

    def ratio_nonpreposed_participles(self) -> float:
        """Причастия в постпозиции / все причастия."""
        total = self.total_participles
        if total == 0:
            return 0.0
        return self.non_preposed_participles / total

    def ratio_verbs_to_tokens(self) -> float:
        """Любые глагольные формы / все токены."""
        if self.total_tokens == 0:
            return 0.0
        return (self.total_verbs_and_verbids /
                self.total_tokens)

    def ratio_participles_to_verbs(self) -> float:
        """Причастия / все глаголы."""
        if self.total_verbs_and_verbids == 0:
            return 0.0
        return (self.total_participles /
                self.total_verbs_and_verbids)

    def ratio_converbs_to_verbs(self) -> float:
        """Деепричастия / все глаголы."""
        if self.total_verbs_and_verbids == 0:
            return 0.0
        return (self.total_converbs /
                self.total_verbs_and_verbids)

    def ratio_short_part_to_part(self) -> float:
        """Краткие причастия / все причастия."""
        if self.total_participles == 0:
            return 0.0
        return (self.total_participles_short /
                self.total_participles)

    def ratio_full_part_to_part(self) -> float:
        """Полные причастия / все причастия."""
        if self.total_participles == 0:
            return 0.0
        return (self.total_participles_full /
                self.total_participles)

    def counts_participles(self) -> Tuple[int, int, int]:
        """(все причастия, полные, краткие)."""
        return (
            self.total_participles,
            self.total_participles_full,
            self.total_participles_short,
        )

    def ratio_imp_among_all_verbs(self) -> float:
        """Доля несовершенного вида среди ВСЕХ
        глагольных форм."""
        if self.total_verbs_and_verbids == 0:
            return 0.0
        return (self.total_imp_all_verbs /
                self.total_verbs_and_verbids)

    def ratio_imp_among_finite(self) -> float:
        """Доля несовершенного вида среди ФИНИТНЫХ
        глаголов."""
        if self.total_finite == 0:
            return 0.0
        return self.total_imp_finite / self.total_finite

    def tense_distribution_among_finite(self) -> Dict[str, float]:
        """Доли времен (Pres/Past/Fut) среди ФИНИТНЫХ глаголов."""
        if self.total_finite == 0:
            return {"Pres": 0.0, "Past": 0.0, "Fut": 0.0}
        return {
            "Pres": self.total_finite_pres / self.total_finite,
            "Past": self.total_finite_past / self.total_finite,
            "Fut": self.total_finite_fut / self.total_finite,
        }

    def tense_distribution_among_all_verbs(self) -> Dict[str, float]:
        """Доли времен (Pres/Past/Fut) среди ВСЕХ глагольных форм."""
        if self.total_verbs_and_verbids == 0:
            return {"Pres": 0.0, "Past": 0.0, "Fut": 0.0}
        return {
            "Pres": self.total_all_pres / self.total_verbs_and_verbids,
            "Past": self.total_all_past / self.total_verbs_and_verbids,
            "Fut": self.total_all_fut / self.total_verbs_and_verbids,
        }

    def indicative_share_among_finite(self) -> float:
        """Доля ИЗЪЯВИТЕЛЬНОГО наклонения от всех финитных глагольных форм.
        """
        if self.total_verbs_and_verbids == 0:
            return 0.0
        return self.total_finite_mood_ind / self.total_finite

    # ----------------------- Эвристики по препозиции -----------------------

    def counts_participle_position(self) -> Optional[Tuple[int, int]]:
        """
        Возвращает (препозиция, не препозиция) для причастий.
        Работает только если use_syntax=True.
        """
        if not self.use_syntax:
            return None
        return (self.preposed_participles,
                self.non_preposed_participles)

    # ----------------------- Утилиты форматирования ------------------------

    @staticmethod
    def pct(x: float) -> str:
        return f"{x * 100:.2f}"

    def print_analysis(self):
        print(Fore.GREEN + Style.BRIGHT +
              "      РЕЗУЛЬТАТЫ АНАЛИЗА ГЛАГОЛОВ И ГЛАГОЛЬНЫХ ФОРМ")
        print(Fore.BLACK + Style.NORMAL +"Всего токенов:", self.total_tokens)
        print("Всего глаголов и особых форм глагола:", self.total_verbs_and_verbids)
        print("— финитные:", self.total_finite)
        print("— причастия (всего/полных/кратких):", self.counts_participles())
        print("— деепричастия:", self.total_converbs)
        print("— инфинитивы:", self.total_infinitives)
        print("Доля глаголов и особых форм глагола ко всем словарным токенам:",
              self.pct(self.ratio_verbs_to_tokens()))
        print("Доля причастий к глаголам:", self.pct(self.ratio_participles_to_verbs()))
        print("Доля деепричастий к глаголам:", self.pct(self.ratio_converbs_to_verbs()))
        print("Краткие/все причастия:", self.pct(self.ratio_short_part_to_part()))
        print("Полные/все причастия:", self.pct(self.ratio_full_part_to_part()))
        print("Несовершенный вид среди ВСЕХ глагольных форм:", self.pct(self.ratio_imp_among_all_verbs()))
        print("Несовершенный вид среди ФИНИТНЫХ:", self.pct(self.ratio_imp_among_finite()))

        tense = self.tense_distribution_among_finite()
        print("Распределение времен среди ФИНИТНЫХ:",
              f"Pres={self.pct(tense['Pres'])}",
              f"Past={self.pct(tense['Past'])}",
              f"Fut={self.pct(tense['Fut'])}")

        tense_all = self.tense_distribution_among_all_verbs()
        print("Распределение времен среди ВСЕХ глагольных форм:",
              f"Pres={self.pct(tense_all['Pres'])}",
              f"Past={self.pct(tense_all['Past'])}",
              f"Fut={self.pct(tense_all['Fut'])}")

        print("Доля изъявительного среди ФИНИТНЫХ:", self.pct(self.indicative_share_among_finite()))

        pos_counts = self.counts_participle_position()
        if pos_counts:
            pre, nonpre = pos_counts
            print("Причастия: препозиция / не препозиция:", pre, "/", nonpre)
            print("Доля причастий в препозиции:", self.pct(self.ratio_preposed_participles()))
            print("Доля причастий в постпозиции:", self.pct(self.ratio_nonpreposed_participles()))
        wait_for_enter_to_analyze()


if __name__ == "__main__":
    analyzer = RuVerbAnalyzer(use_syntax=True)
    text = input("Введите текст для анализа: ").strip()
    analyzer.analyze(text)
    analyzer.print_analysis()