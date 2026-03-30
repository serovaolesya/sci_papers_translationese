# -*- coding: utf-8 -*-
"""
Показатели лексической когезии текста (Lexical Cohesion):

  - avg_token_pair_similarity : среднее косинусное сходство пар знаменательных слов
  - mean_adjacent_sentence_cosine : среднее косинусное сходство соседних предложений

Загрузка модели — один раз (lru_cache). Передаётся в analyze_text_cohesion().
"""
import os
from collections import Counter
from functools import lru_cache
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gensim.downloader as api
import numpy as np
import stanza
from colorama import Fore, Style
from gensim.models import KeyedVectors
from rich.console import Console
from rich.table import Table

from tools.core.utils import wait_for_enter_to_analyze

try:
    import torch
    _GPU_AVAILABLE = torch.cuda.is_available()
except Exception:
    _GPU_AVAILABLE = False

try:
    from nltk.corpus import stopwords
    _ = stopwords.words("russian")
except LookupError:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

from tools.core.constants import (
    INDEFINITE_NUMS_PRNS, NUMERAL_WDS, CONJ_WRDS_PRNS, CONJ_WRDS_ADVBS,
    PARENTH_WORDS, SUPRL_ADVBS, DEF_NOUN_ADJ_PRNS, DEF_ADVBS_PRNS,
    MODAL_WDS, COPULA_VERBS, INDEFINITE_NOUNS_PRNS, INDEFINITE_ADJS_PRNS,
    INDEFINITE_ADVBS_PRNS, NEGATIVE_NOUNS_PRNS, NEGATIVE_ADJS_PRNS,
    NEGATIVE_ADVBS_PRNS, NEGATIVE_NUMS_PRNS, DMSTR_PRNS, POSSESSIVE_PRNS,
    PER_PRONOUNS, COMP_ADVBS, EXTRA_STOP_LEMMAS,
)
console = Console()

BASE_DIR = Path(__file__).resolve().parent
MODEL_KV_PATH = BASE_DIR / "ruscorpora_upos_skipgram_300_5_2018.kv"
MODEL_BIN_PATH = BASE_DIR / "model.bin"

# Части речи, исключаемые при анализе
EXCLUDE_UPOS = frozenset({
    "ADP", "PUNCT", "PRON", "DET", "CCONJ", "SCONJ", "PART", "INTJ"
})

# Стоп-леммы (создаём один раз при загрузке модуля)
STOP_LEMMAS = frozenset(
    set(stopwords.words("russian")) |
    EXTRA_STOP_LEMMAS | PER_PRONOUNS | CONJ_WRDS_ADVBS |
    CONJ_WRDS_PRNS | POSSESSIVE_PRNS | DMSTR_PRNS |
    COPULA_VERBS | INDEFINITE_NOUNS_PRNS | INDEFINITE_ADJS_PRNS |
    INDEFINITE_NUMS_PRNS | INDEFINITE_ADVBS_PRNS |
    NEGATIVE_NOUNS_PRNS | NEGATIVE_ADJS_PRNS | NEGATIVE_ADVBS_PRNS |
    NEGATIVE_NUMS_PRNS | DEF_NOUN_ADJ_PRNS | DEF_ADVBS_PRNS |
    MODAL_WDS | COMP_ADVBS | SUPRL_ADVBS | PARENTH_WORDS | NUMERAL_WDS
)


# ─────────────────────────────────────────────────────────────────────────────
# Кеш векторов
# ─────────────────────────────────────────────────────────────────────────────

class OptimizedVectorCache:
    """LRU-кеш векторов с поддержкой batch-операций."""

    def __init__(self, model: KeyedVectors, cache_size: int = 10000):
        self.model = model
        self._cache: Dict[str, Optional[np.ndarray]] = {}
        self.cache_size = cache_size

    def get_batch(self, tokens: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """Возвращает {token: вектор | None} для списка токенов."""
        result: Dict[str, Optional[np.ndarray]] = {}
        missing: List[str] = []
        for token in tokens:
            if token in self._cache:
                result[token] = self._cache[token]
            else:
                missing.append(token)
        for token in missing:
            try:
                vec = self.model[token]
                if len(self._cache) < self.cache_size:
                    self._cache[token] = vec
                result[token] = vec
            except KeyError:
                result[token] = None
        return result

    def get(self, token: str) -> Optional[np.ndarray]:
        if token in self._cache:
            return self._cache[token]
        try:
            vec = self.model[token]
            if len(self._cache) < self.cache_size:
                self._cache[token] = vec
            return vec
        except KeyError:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Инициализация пайплайна и загрузка модели
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def init_pipeline(language: str = 'ru') -> stanza.Pipeline:
    """Инициализирует пайплайн Stanza с кешированием (создаётся один раз)."""
    return stanza.Pipeline(
        language,
        processors='tokenize,pos,lemma',
        use_gpu=_GPU_AVAILABLE,
        verbose=False,
    )

def ensure_cohesion_model() -> Path:
    """
    Проверяет наличие локальной модели лексической когезии.

    Приоритет:
      1. Если задан TAYGA_VEC_PATH и файл существует — используем его.
      2. Если рядом есть model.bin — используем его.
      3. Если рядом есть ruscorpora_upos_skipgram_300_5_2018.kv — используем его.
      4. Иначе скачиваем word2vec-ruscorpora-300 через gensim
         и сохраняем как ruscorpora_upos_skipgram_300_5_2018.kv.
    """
    env_path = os.environ.get("TAYGA_VEC_PATH")
    if env_path and os.path.isfile(env_path):
        return Path(env_path)

    if MODEL_BIN_PATH.is_file():
        return MODEL_BIN_PATH

    if MODEL_KV_PATH.is_file():
        return MODEL_KV_PATH

    model = api.load("word2vec-ruscorpora-300")
    model.save(str(MODEL_KV_PATH))

    return MODEL_KV_PATH

@lru_cache(maxsize=1)
def load_model() -> KeyedVectors:
    """
    Загружает Word2Vec-модель (кешируется: загружается ровно один раз).

    Если локальная модель отсутствует, она будет автоматически скачана
    через gensim как word2vec-ruscorpora-300 и сохранена в папку
    tools/explicitation/.
    """
    model_path = ensure_cohesion_model()

    # Если используется внешний/локальный бинарный файл Taiga
    if str(model_path).endswith(".bin") or str(model_path).endswith(".bin.gz"):
        return KeyedVectors.load_word2vec_format(str(model_path), binary=True)

    # Если используется сохранённый KeyedVectors-файл
    return KeyedVectors.load(str(model_path), mmap="r")


# ─────────────────────────────────────────────────────────────────────────────
# Предобработка текста
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_sentences(
        text: str,
        nlp: stanza.Pipeline,
) -> List[List[str]]:
    """
    Токенизирует и лемматизирует текст через Stanza.

    Возвращает список предложений; каждое предложение — список строк вида
    'лемма_UPOS'. Служебные части речи (EXCLUDE_UPOS) и стоп-леммы
    (STOP_LEMMAS) исключаются.
    """
    doc = nlp(text)
    sentences: List[List[str]] = []
    for sentence in doc.sentences:
        tokens: List[str] = []
        for word in sentence.words:
            lemma = (word.lemma or "").lower()
            upos = word.upos
            if (not lemma or not upos
                    or upos in EXCLUDE_UPOS
                    or lemma in STOP_LEMMAS):
                continue
            tokens.append(f"{lemma}_{upos}")
        if tokens:
            sentences.append(tokens)
    return sentences

# ─────────────────────────────────────────────────────────────────────────────
# Вычисление метрик
# ─────────────────────────────────────────────────────────────────────────────

def _sentence_vectors(
        sentences: List[List[str]],
        vec_cache: OptimizedVectorCache,
) -> List[Optional[np.ndarray]]:
    """Вычисляет усреднённый вектор для каждого предложения."""
    all_tokens: set = set()
    for sent in sentences:
        all_tokens.update(sent)
    token_vecs = vec_cache.get_batch(list(all_tokens))

    result: List[Optional[np.ndarray]] = []
    for sent_tokens in sentences:
        vecs = [token_vecs[t] for t in sent_tokens if token_vecs.get(t) is not None]
        result.append(np.mean(vecs, axis=0) if vecs else None)
    return result


def _mean_adjacent_cosine(vectors: List[Optional[np.ndarray]]) -> float:
    """Среднее арифметическое косинусного сходства соседних предложений."""
    sims: List[float] = []
    for i in range(len(vectors) - 1):
        v1, v2 = vectors[i], vectors[i + 1]
        if v1 is None or v2 is None:
            continue
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm > 0:
            sims.append(float(dot / norm))
    return float(np.mean(sims)) if sims else 0.0


def _pairwise_similarities(
        tokens: List[str],
        model: KeyedVectors,
        vec_cache: Optional['OptimizedVectorCache'] = None,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], int]]:
    """
    Возвращает два словаря:

    1. sim_dict[(t1, t2)] = косинусное сходство пары
    2. weight_dict[(t1, t2)] = вес пары с учётом повторов токенов в тексте

    Логика:
    - самопара (t, t) учитывается, если слово встречается >= 2 раз;
      её вес = C(freq, 2)
    - пара разных слов (t1, t2) учитывается один раз в sim_dict,
      но её вес = freq(t1) * freq(t2)

    Реализация: векторизованная через numpy (матричное умножение BLAS)
    вместо O(n²) вызовов model.similarity(). Если передан vec_cache,
    векторы берутся из него — повторные тексты в сессии не обращаются
    к модели заново для уже встречавшихся слов.
    """
    type_counts = Counter(tokens)

    # Фильтруем токены, присутствующие в модели
    valid = [t for t in type_counts if t in model]
    if not valid:
        return {}, {}

    # Загружаем векторы: из кэша сессии (если есть) или напрямую из модели
    if vec_cache is not None:
        token_vecs = vec_cache.get_batch(valid)
        valid = [t for t in valid if token_vecs.get(t) is not None]
        if not valid:
            return {}, {}
        raw_vecs = [token_vecs[t] for t in valid]
    else:
        raw_vecs = [model[t] for t in valid]

    vecs = np.array(raw_vecs, dtype=np.float32)  # (n, dim)

    # Нормируем строки для косинусного сходства
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms

    # Вся матрица косинусного сходства за одно умножение (BLAS, миллисекунды)
    sim_matrix = (vecs @ vecs.T).astype(float)  # (n, n)

    sim_dict: Dict[Tuple[str, str], float] = {}
    weight_dict: Dict[Tuple[str, str], int] = {}

    for i, t1 in enumerate(valid):
        freq1 = type_counts[t1]
        # Самопары
        if freq1 >= 2:
            sim_dict[(t1, t1)] = 1.0
            weight_dict[(t1, t1)] = comb(freq1, 2)
        # Кросс-пары (верхний треугольник)
        for j in range(i + 1, len(valid)):
            t2 = valid[j]
            sim_dict[(t1, t2)] = float(sim_matrix[i, j])
            weight_dict[(t1, t2)] = freq1 * type_counts[t2]

    return sim_dict, weight_dict


def _weighted_average_similarity(
        sim_dict: Dict[Tuple[str, str], float],
        weight_dict: Dict[Tuple[str, str], int],
) -> float:
    """
    Считает среднее сходство пар с учётом повторов слов в тексте.
    """
    if not sim_dict:
        return 0.0

    total_weight = sum(weight_dict.values())
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(
        sim_dict[pair] * weight_dict[pair]
        for pair in sim_dict
    )
    return float(weighted_sum / total_weight)

# ─────────────────────────────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────────────────────────────

def analyze_text_cohesion(
        text: str,
        model: KeyedVectors,
        nlp: stanza.Pipeline,
        vec_cache: OptimizedVectorCache,
        show_analysis: bool
) -> Dict[str, float]:
    """
    Вычисляет два показателя лексической когезии текста.

    Параметры
    ----------
    text : str
        Исходный текст.
    model : KeyedVectors
        Загруженная Word2Vec-модель (``load_model()``).
    nlp : stanza.Pipeline
        Инициализированный пайплайн Stanza (``init_pipeline()``).
    vec_cache : OptimizedVectorCache
        Кеш векторов, созданный на основе той же модели.

    Возвращает
    ----------
    dict с ключами:
      - ``avg_token_pair_similarity``  — среднее косинусное сходство пар
        знаменательных слов (arithmetic mean по всем уникальным парам);
      - ``mean_adjacent_sentence_cosine`` — среднее косинусное сходство
        между векторными представлениями последовательных предложений.

    При пустом или нераспознанном тексте возвращает нули.
    """
    sentences = preprocess_sentences(text, nlp)
    all_tokens = [tok for sent in sentences for tok in sent]

    if not sentences or not all_tokens:
        return {
            "avg_token_pair_similarity": 0.0,
            "mean_adjacent_sentence_cosine": 0.0,
        }

    sim_dict, weight_dict = _pairwise_similarities(all_tokens, model, vec_cache)
    avg_pair_sim = _weighted_average_similarity(sim_dict, weight_dict)

    sent_vecs = _sentence_vectors(sentences, vec_cache)
    mean_adj_cosine = _mean_adjacent_cosine(sent_vecs)

    if show_analysis:
        print(
            Fore.GREEN + Style.BRIGHT +
            "              СРЕДНИЕ ПОКАЗАТЕЛИ" +
            Fore.LIGHTGREEN_EX + " ЛЕКСИЧЕСКОЙ КОГЕЗИИ КОРПУСА"
        )
        _coh_table = Table()
        _coh_table.add_column("Индикатор", no_wrap=True, style="bold")
        _coh_table.add_column("Значение", max_width=10, justify="center")
        _coh_table.add_row(
            "Среднее косинусное сходство пар знаменательных слов",
            f"{float(avg_pair_sim):.4f}",
        )
        _coh_table.add_row(
            "Среднее косинусное сходство соседних предложений",
            f"{float(mean_adj_cosine):.4f}",
        )
        console.print(_coh_table)
        wait_for_enter_to_analyze()

    return {
        "avg_token_pair_similarity": float(avg_pair_sim),
        "mean_adjacent_sentence_cosine": float(mean_adj_cosine),
    }
