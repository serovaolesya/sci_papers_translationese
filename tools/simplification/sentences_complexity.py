# -*- coding: utf-8 -*-
from __future__ import annotations

from colorama import Fore, Style
from rich.console import Console
from rich.table import Table
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc
)

from tools.core.utils import wait_for_enter_to_analyze

console = Console()

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)

# ──────────────────────────────────────────────
# Константы: синтаксические отношения по типам
# ──────────────────────────────────────────────

# Финитные клаузы — головная вершина несёт финитную предикацию
FINITE_CLAUSE_RELS = {"root", "advcl", "ccomp", "acl:relcl"}

# Нефинитные — отдельные отношения + проверка VerbForm
NON_FINITE_RELS = {
    "acl",  # причастный оборот (VerbForm=Part)
    "csubj",  # инфинитивное подлежащее (VerbForm=Inf)
}

def _get_verb_form(tok) -> str:
    """Возвращает значение VerbForm из морфологических признаков токена Natasha."""
    feats = getattr(tok, 'feats', None) or {}
    return feats.get('VerbForm', '')


def _is_finite_verb(tok) -> bool:
    """Финитный глагол: POS=VERB и VerbForm ≠ Part/Conv/Inf (или VerbForm отсутствует)."""
    if getattr(tok, 'pos', None) != 'VERB':
        return False
    vf = _get_verb_form(tok)
    return vf not in ('Part', 'Conv', 'Inf')


def _is_converb(tok) -> bool:
    """Деепричастие: VerbForm=Conv."""
    return (getattr(tok, 'pos', None) == 'VERB'
            and _get_verb_form(tok) == 'Conv')


def _is_participle(tok) -> bool:
    """Причастие: VerbForm=Part."""
    return (getattr(tok, 'pos', None) == 'VERB'
            and _get_verb_form(tok) == 'Part')


def _is_infinitive(tok) -> bool:
    """Инфинитив: VerbForm=Inf."""
    return (getattr(tok, 'pos', None) == 'VERB'
            and _get_verb_form(tok) == 'Inf')


def _is_junk_root(tok) -> bool:
    """Служебный ROOT без реальной предикации (имя, символ)."""
    return (getattr(tok, 'rel', None) == 'root'
            and getattr(tok, 'pos', None) in {'X', 'PROPN', 'SYM'})


# ──────────────────────────────────────────────
# Функция классификации одного токена
# ──────────────────────────────────────────────

def _get_variant(tok) -> str:
    """Возвращает значение Variant из feats (Short / Long / '')."""
    feats = getattr(tok, 'feats', None) or {}
    return feats.get('Variant', '')


def _classify_token(tok) -> str | None:
    """
    Возвращает строку-тип клаузы для токена или None.

    Обрабатывает POS=VERB и POS=ADJ, поскольку Natasha/SynTagRus
    тегирует причастия и краткие предикативы как ADJ:
      — ADJ + Variant=Short + rel ∈ FINITE_CLAUSE_RELS → finite
        (предикативные краткие прилагательные/причастия: «значимы», «собраны»)
      — ADJ + полная форма + rel=acl → acl_part
        (атрибутивные причастия: «заимствованные», «опубликованная»)

    Возможные значения:
      'finite'      — финитная клауза
      'advcl_conv'  — деепричастный оборот
      'acl_part'    — причастный оборот
    """
    rel = getattr(tok, 'rel', None)
    pos = getattr(tok, 'pos', None)

    if rel is None or pos not in ('VERB', 'ADJ'):
        return None

    # ── ADJ-ветка ─────────────────────────────────────────────────────────
    # Natasha тегирует краткие предикативные формы и ряд причастий как ADJ
    if pos == 'ADJ':
        variant = _get_variant(tok)
        # Краткая форма в предикативной позиции → финитная клауза
        # Примеры: «результаты значимы» (ccomp), «данные собраны» (acl:relcl)
        if variant == 'Short' and rel in FINITE_CLAUSE_RELS:
            return 'finite'
        # Полная форма в атрибутивной позиции → причастный оборот
        # Примеры: «термины, заимствованные…» (acl), «статья, опубликованная…»
        if variant != 'Short' and rel == 'acl':
            return 'acl_part'
        return None

    # ── VERB-ветка ────────────────────────────────────────────────────────
    if _is_junk_root(tok):
        return None

    # ── ФИНИТНЫЕ ──────────────────────────────
    if rel in FINITE_CLAUSE_RELS:
        # advcl — либо финитное придаточное, либо деепричастный оборот
        if rel == 'advcl' and _is_converb(tok):
            return 'advcl_conv'
        return 'finite'

    if rel == 'parataxis' and _is_finite_verb(tok):
        return 'finite'

    # ── НЕФИНИТНЫЕ ────────────────────────────
    if rel == 'acl' and _is_participle(tok):
        return 'acl_part'
    return None


# ──────────────────────────────────────────────
# Основная функция подсчёта клауз
# ──────────────────────────────────────────────

def analyze_clause_structure(
        sentences: list,
        show_analysis: bool = False,
) -> dict:
    """
    Считает клаузы всех типов (финитные + нефинитные) для списка предложений.

    Параметры:
      show_analysis  — если True, выводит итоговую таблицу статистики клауз

    Возвращает словарь с ключами:
      total_clauses        — все клаузы
      finite_clauses       — финитные
      advcl_conv_clauses   — деепричастные обороты
      acl_part_clauses     — причастные обороты
      total_sentences
      simple_sentences     — 1 клауза (любого типа)
      complex_sentences    — >1 клаузы
      avg_clauses_per_sent
      avg_clauses_per_complex
      simple_ratio         — % простых предложений
      avg_tokens_per_clause
    """
    if not sentences:
        return _empty_result()

    counts: dict[str, int] = {
        'finite': 0,
        'advcl_conv': 0,
        'acl_part': 0,
    }

    total_clauses = 0
    simple_sents = 0
    clauses_in_complex = 0
    total_non_punct_tokens = 0

    for sent_idx, sentence in enumerate(sentences, 1):
        doc = Doc(sentence)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)

        # Токены без пунктуации
        non_punct = [t for t in doc.tokens if getattr(t, 'pos', None) != 'PUNCT']
        total_non_punct_tokens += len(non_punct)

        # Классифицируем токены для подсчёта клауз
        sent_counts: dict[str, int] = {k: 0 for k in counts}
        for tok in doc.tokens:
            ctype = _classify_token(tok)

            if ctype:
                sent_counts[ctype] += 1

        n = sum(sent_counts.values())
        effective_n = n if n > 0 else 1

        # Аккумулируем в общие счётчики
        for k in counts:
            counts[k] += sent_counts[k]

        total_clauses += effective_n

        # Простое / сложное определяется ТОЛЬКО по числу финитных клауз:
        # простое — ровно 1 финитная клауза (нефинитных может быть ≥0)
        # сложное — 2 и более финитных клаузы (несколько грамматических центров)
        finite_n = sent_counts['finite']
        finite_n = finite_n if finite_n > 0 else 1  # фоллбек: минимум 1
        if finite_n == 1:
            simple_sents += 1
        else:
            clauses_in_complex += effective_n

    total_sents = len(sentences)
    complex_sents = total_sents - simple_sents

    avg_per_sent = total_clauses / total_sents if total_sents else 0.0
    avg_per_complex = clauses_in_complex / complex_sents if complex_sents > 0 else 0.0
    simple_ratio = (simple_sents / total_sents * 100) if total_sents else 0.0
    avg_tok_per_clause = total_non_punct_tokens / total_clauses if total_clauses else 0.0

    result = {
        'total_clauses': total_clauses,
        'finite_clauses': counts['finite'],
        'advcl_conv_clauses': counts['advcl_conv'],
        'acl_part_clauses': counts['acl_part'],
        'total_sentences': total_sents,
        'simple_sentences': simple_sents,
        'complex_sentences': complex_sents,
        'avg_clauses_per_sent': avg_per_sent,
        'avg_clauses_per_complex': avg_per_complex,
        'simple_ratio': simple_ratio,
        'avg_tokens_per_clause': avg_tok_per_clause,
    }

    if show_analysis:
        print_clause_statistics(result)

    return result


def _empty_result() -> dict:
    return {k: 0 for k in [
        'total_clauses', 'finite_clauses', 'advcl_conv_clauses',
        'acl_part_clauses',
        'total_sentences', 'simple_sentences', 'complex_sentences',
    ]} | {k: 0.0 for k in [
        'avg_clauses_per_sent', 'avg_clauses_per_complex',
        'simple_ratio', 'avg_tokens_per_clause',
    ]}


def print_clause_statistics(m: dict):
    print(Fore.GREEN + Style.BRIGHT + "\n      АНАЛИЗ СТРУКТУРЫ ПРЕДЛОЖЕНИЙ")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Показатель", justify="left", style="bold")
    table.add_column("Значение", justify="right")

    total = m['total_clauses']

    def pct(v):
        return f"({v / total * 100:.2f}%)" if total else ""

    table.add_row("Всего клауз", str(total))
    table.add_row("  ├─ Финитные клаузы",
                  f"{m['finite_clauses']}  {pct(m['finite_clauses'])}")
    table.add_row("  ├─ Деепричастные обороты (advcl+Conv)",
                  f"{m['advcl_conv_clauses']}  {pct(m['advcl_conv_clauses'])}")
    table.add_row("  └─ Причастные обороты (acl+Part)",
                  f"{m['acl_part_clauses']}  {pct(m['acl_part_clauses'])}")
    table.add_section()
    table.add_row("Предложений всего", str(m['total_sentences']))
    table.add_row("  ├─ Простых (1 финитная клауза)", str(m['simple_sentences']))
    table.add_row("  └─ Сложных (≥2 финитных клаузы)", str(m['complex_sentences']))
    table.add_section()
    table.add_row("Процент простых предложений", f"{m['simple_ratio']:.2f}%")
    table.add_row("Среднее клауз на предложение (все)", f"{m['avg_clauses_per_sent']:.2f}")
    table.add_row("Среднее клауз на сложное предложение", f"{m['avg_clauses_per_complex']:.2f}")
    table.add_row("Средняя длина клаузы (токены)", f"{m['avg_tokens_per_clause']:.2f}")
    console.print(table)
    wait_for_enter_to_analyze()
