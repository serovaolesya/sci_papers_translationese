# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
from typing import Union, Any

import warnings

from tools.core.constants import (
    COMPOSITE_PREPS_1,
    COMPOSITE_PREPS_2,
    ONE_WORD_PREPS,
    PUNCT_MAPPING,
    INDEFINITE_NUMS_PRNS,
    NUMERAL_WDS,
    CONJ_WRDS_PRNS,
    CONJ_WRDS_ADVBS,
    PARENTH_WORDS,
    SUPRL_ADVBS,
    DEF_NOUN_ADJ_PRNS,
    DEF_ADVBS_PRNS,
    MODAL_WDS, COPULA_VERBS,
    INDEFINITE_NOUNS_PRNS,
    INDEFINITE_ADJS_PRNS,
    INDEFINITE_ADVBS_PRNS,
    NEGATIVE_NOUNS_PRNS,
    NEGATIVE_ADJS_PRNS,
    NEGATIVE_ADVBS_PRNS,
    NEGATIVE_NUMS_PRNS,
    DMSTR_PRNS,
    POSSESSIVE_PRNS,
    PER_PRONOUNS,
    COMP_ADVBS,
    END_MARKERS,
    QUOTES, CONJ_WRDS_NUMS, CONTRACTED_WORDS_WITH_DOT
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pymorphy2.analyzer"
)

from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    NewsSyntaxParser,
    Doc,
    MorphVocab
)
from pymorphy2 import MorphAnalyzer

morph = MorphAnalyzer()
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
ner_tagger = NewsNERTagger(emb)
morph_vocab = MorphVocab()
syntax_parser = NewsSyntaxParser(emb)


def is_abbreviation_dot(tokens, dot_index):
    """Проверяет, является ли точка частью сокращения"""
    if dot_index == 0 or tokens[dot_index].text != '.':
        return False
    prev_token = tokens[dot_index - 1].text.lower()
    return prev_token in CONTRACTED_WORDS_WITH_DOT

def pymorphy_for_one_token(word):
    """
    Эта функция анализирует одно слово с помощью pymorphy2
    и возвращает его часть речи

    :param word: Слово, которое нужно проанализировать.
    """
    parse = morph.parse(word)[0]
    # print(parse)

    return parse.tag.POS


def pos_tagger(
        text: str
) -> tuple[list[Union[str, Any]], list[Any]]:
    doc = Doc(text)
    # Текст разбивается на токены,
    doc.segment(segmenter)
    # выполняется морфологический
    doc.tag_morph(morph_tagger)
    # и синтаксический анализ.
    doc.parse_syntax(syntax_parser)

    tokens_analyzed = []
    pos_tags = []

    def is_initial_dot(tokens, dot_index):
        """Проверяет, является ли точка на
        указанной позиции частью инициала"""
        return (dot_index > 0 and
                tokens[dot_index].text == '.' and
                dot_index - 1 >= 0 and
                tokens[dot_index - 1].text.isalpha() and
                len(tokens[dot_index - 1].text) == 1)

    for i, token in enumerate(doc.tokens):
        token_text = token.text

        # Определяем, является ли предыдущая точка частью инициала
        # (если да, то не ставим S_START)
        # Устанавливаем флаг is_prev_initial_dot = False
        # для каждого нового токена
        is_prev_initial_dot = False
        if i > 0 and doc.tokens[i - 1].text == '.':
            is_prev_initial_dot = (is_initial_dot(doc.tokens, i - 1) or
                                   is_abbreviation_dot(doc.tokens, i - 1))

        # Если это первый токен или если предыдущий токен - конец предложения
        # (но не точка после инициала)
        if i == 0 or (
                i > 0 and doc.tokens[i - 1].text in '.?!…'
                and not is_prev_initial_dot
        ):
            if token_text != ',':
                pos_tags.append('S_START')
                tokens_analyzed.append('S_START')

        # Обрабатываем пунктуацию только
        # если в токене нет букв/цифр.
        # Natasha иногда помечает как
        # PUNCT слова,
        # поэтому добавляем явную проверку
        # на состав токена.
        if ((token.pos == 'PUNCT' and
             not any(ch.isalpha() or ch.isdigit()
                     for ch in token_text))
                or token_text == '}'):
            # print('Проверка')
            # Простые знаки пунктуации - прямой поиск в словаре
            if token_text in PUNCT_MAPPING:
                pos_tags.append(PUNCT_MAPPING[token_text])
            # Обработка кавычек - прямая проверка множества
            elif token_text in QUOTES:
                pos_tags.append('QUOTE')
            # Обработка знаков конца предложения
            # с кэшированием проверки инициала
            elif token_text in END_MARKERS:
                if ((token_text == '.' and
                     (is_initial_dot(doc.tokens, i) or
                      is_abbreviation_dot(doc.tokens, i)))
                        or (i + 1 < len(doc.tokens) and
                            doc.tokens[i + 1].text == ',')):
                    pos_tags.append('PUNCT')
                else:
                    pos_tags.append('S_END')
            else:
                pos_tags.append('PUNCT')
        else:
            # Лемматизируем токены
            token.lemmatize(morph_vocab)
            # Извлекаем тег из библиотеки Pymorphy
            pymorphy_pos = pymorphy_for_one_token(token_text)
            # Извлекаем тег из библиотеки Natasha
            natasha_pos = token.pos
            # print(pymorphy_pos)
            # print(natasha_pos)
            # print(token.lemma.lower())
            # print(f"Морфологические признаки: {token.feats}")
            # print(token.rel)

            if token_text.lower() in ['воспитанные', 'существующие']:
                pos_tags.append('PRTF')
            elif token_text.lower() in ['популяризировал']:
                pos_tags.append('VERB')

            # Проверяем, является ли слово союзным словом-числит
            # до проверки на числит
            elif (token_text.lower()
                  in CONJ_WRDS_NUMS and
                  i > 0
                  and doc.tokens[i - 1].text == ','):
                pos_tags.append('NUM_CONJ_W')

            # Если токен стоит в начале предложения и
            # является вопросительным словом.
            # Надо как-то учесть, что это вопрос.!!!!!!
            elif (token.lemma and token.lemma.lower()
                  in CONJ_WRDS_NUMS and
                  ((i > 0 and doc.tokens[i - 1].text
                    in '.?!…') or
                   (i == 0 and token_text[0].isupper()))
            ):
                pos_tags.append('NUM_QW')

            # Если слово является неопределённым
            # местоимением–числительным
            elif (token.lemma.lower()
                  in INDEFINITE_NUMS_PRNS
            ):
                pos_tags.append('INDF_NUM_PRN')

            # 1. Проверяем, является ли
            # токен числом в виде цифр
            elif token_text.isdigit():
                pos_tags.append('NUM')
            elif ((token.lemma.lower() or token_text)
                  in NUMERAL_WDS):
                # Если слово есть в словаре числительных
                pos_tags.append('NUM')
            elif ((',' in token_text or '.' in token_text)
                  and token_text.replace(',', '.', 1).replace(
                        '.', '', 1).isdigit()):
                # Десятичное число (3.14 или 3,14)
                pos_tags.append('NUM')
            elif natasha_pos == 'NUM' or pymorphy_pos == 'NUMR':
                # Определение числительного с помощью библиотек
                pos_tags.append('NUM')
            # Проверяем, является ли токен вводным словом

            elif token_text.lower() in PARENTH_WORDS and (
                    # Случай 1: В начале предложения И после него запятая
                    ((i == 0 or (i > 0 and pos_tags
                                 and pos_tags[-1] == 'S_START'))
                     and i + 1 < len(doc.tokens)
                     and doc.tokens[i + 1].text == ',') or
                    # Случай 2: Обособлено запятыми с двух сторон
                    (0 < i < len(doc.tokens) - 1 and
                     doc.tokens[i - 1].text == ',' and
                     doc.tokens[i + 1].text == ',')
            ):
                pos_tags.append('PARENTH')

            elif token_text.lower() in ['например', ] and (
                    (i == 0
                     and i + 1 < len(doc.tokens)
                     and doc.tokens[i + 1].text == ','
                    )
            ):
                pos_tags.append('PARENTH')

            elif token_text.lower() in ['напротив', 'правда'] and (
                    (0 < i < len(doc.tokens) - 1 and
                     doc.tokens[i - 1].text
                     in [',', '(', ';'])
            ):
                pos_tags.append('PARENTH')

            # Проверяем, является ли текущий токен
            # словом "потому, оттого" и следующий - "что"
            elif (token_text.lower() in ['потому', 'оттого'] and
                  i + 1 < len(doc.tokens) and
                  doc.tokens[i + 1].text
                  and doc.tokens[i + 1].text.lower() == 'что'):
                pos_tags.append('SCONJ')

            # Проверяем, является ли текущий токен словом "что"
            # и предыдущий - "потому, оттого"
            elif (token_text.lower() == 'что' and
                  i > 0 and
                  doc.tokens[i - 1].text
                  and doc.tokens[i - 1].text.lower()
                  in ['потому', 'оттого']):
                pos_tags.append('SCONJ_PRT')

            # Проверяем, является ли текущий токен
            # словом "так" и следующий - "как, что"
            elif (token_text.lower() == 'так' and
                  i + 1 < len(doc.tokens) and
                  doc.tokens[i + 1].text
                  and doc.tokens[i + 1].text.lower()
                  in ['как', 'что']):
                pos_tags.append('SCONJ')

            # Проверяем, является ли текущий токен словом "как, что"
            # и предыдущий - "так"
            elif (token_text.lower() in ['как', 'что'] and
                  i > 0 and
                  doc.tokens[i - 1].text
                  and doc.tokens[i - 1].text.lower() == 'так'):
                pos_tags.append('SCONJ_PRT')

            elif (token_text.lower() == 'т' and
                  i + 1 < len(doc.tokens) and
                  doc.tokens[i + 1].text
                  and doc.tokens[i + 1].text == '.'
                  and doc.tokens[i + 2].text.lower() == 'к'

            ):
                pos_tags.append('SCONJ')

            elif (token_text.lower() == 'к' and
                  i > 0 and
                  doc.tokens[i - 1].text
                  and doc.tokens[i - 1].text == '.'
                  and doc.tokens[i - 2].text.lower() == 'т'

            ):
                pos_tags.append('SCONJ_PRT')

            elif (token_text.lower() == 'прежде' and
                  i + 1 < len(doc.tokens) and
                  doc.tokens[i + 1].text
                  and doc.tokens[i + 1].text.lower() in ['чем', ]):
                pos_tags.append('SCONJ')

            elif (token_text.lower() in ['чем', ] and
                  i > 0 and
                  doc.tokens[i - 1].text
                  and doc.tokens[i - 1].text.lower() == 'прежде'):
                pos_tags.append('SCONJ_PRT')

            # СОСТАВНЫЕ ПРЕДЛОГИ (унифицированная обработка)
            # Первая часть = PREP, вторая = PREP_PRT
            elif (
                    i + 1 < len(doc.tokens)
                    and (token_text.lower(),
                         doc.tokens[i + 1].text.lower())
                    in COMPOSITE_PREPS_1
            ):
                pos_tags.append('PREP')

            elif (
                    i > 0
                    and (doc.tokens[i - 1].text.lower(),
                         token_text.lower())
                    in COMPOSITE_PREPS_1
            ):
                pos_tags.append('PREP_PRT')

            # Первая часть = PREP_PRT, вторая = PREP
            elif (
                    i + 1 < len(doc.tokens)
                    and (token_text.lower(),
                         doc.tokens[i + 1].text.lower())
                    in COMPOSITE_PREPS_2
            ):
                pos_tags.append('PREP_PRT')

            elif (
                    i > 0
                    and (doc.tokens[i - 1].text.lower(),
                         token_text.lower())
                    in COMPOSITE_PREPS_2
            ):
                pos_tags.append('PREP')

            elif token_text.lower() in ONE_WORD_PREPS:
                pos_tags.append('PREP')

            # Проверяем, является ли текущий токен
            # словом "как" и следующий - "будто"
            elif (token_text.lower() == 'как' and
                  i + 1 < len(doc.tokens) and
                  doc.tokens[i + 1].text
                  and doc.tokens[i + 1].text.lower() == 'будто'):
                pos_tags.append('SCONJ')

            # Проверяем, является ли текущий токен словом "будто"
            # и предыдущий - "как"
            elif (token_text.lower() == 'будто' and
                  i > 0 and
                  doc.tokens[i - 1].text
                  and doc.tokens[i - 1].text.lower() == 'как'):
                pos_tags.append('SCONJ_PRT')

            elif (token_text.lower() == 'т' and
                  i + 1 < len(doc.tokens) and
                  doc.tokens[i + 1].text
                  and doc.tokens[i + 1].text == '.'
                  and doc.tokens[i + 2].text.lower() == 'е'):
                pos_tags.append('CCONJ')

            elif (token_text.lower() == 'е' and
                  i > 0 and
                  doc.tokens[i - 1].text
                  and doc.tokens[i - 1].text == '.'
                  and doc.tokens[i - 2].text.lower() == 'т'):
                pos_tags.append('CCONJ_PRT')

            elif (token_text.lower() == 'то' and
                  i + 1 < len(doc.tokens) and
                  doc.tokens[i + 1].text
                  and doc.tokens[i + 1].text.lower() == 'есть'):
                pos_tags.append('CCONJ')

            elif (token_text.lower() == 'есть' and
                  i > 0 and
                  doc.tokens[i - 1].text
                  and doc.tokens[i - 1].text.lower() == 'то'):
                pos_tags.append('CCONJ_PRT')

            elif (token_text.lower() == 'а' and
                  i + 1 < len(doc.tokens) and
                  doc.tokens[i + 1].text
                  and doc.tokens[i + 1].text.lower() == 'именно'):
                pos_tags.append('CCONJ')

            elif (token_text.lower() == 'именно' and
                  i > 0 and
                  doc.tokens[i - 1].text
                  and doc.tokens[i - 1].text.lower() == 'а'):
                pos_tags.append('CCONJ_PRT')

            elif (token_text.lower() == 'друг' and
                  i + 1 < len(doc.tokens) and
                  doc.tokens[i + 1].text
                  and doc.tokens[i + 1].text.lower() == 'друга'):
                pos_tags.append('REC_PRN')

            elif (token_text.lower() == 'друга' and
                  i > 0 and
                  doc.tokens[i - 1].text
                  and doc.tokens[i - 1].text.lower() == 'друг'):
                pos_tags.append('REC_PRN_PART')

            elif (token_text.lower() == 'друг' and
                  i + 1 < len(doc.tokens) and
                  doc.tokens[i + 1].text
                  and doc.tokens[i + 1].text
                  in ['от', 'с', 'по', 'на', 'к', 'без', 'о']
                  and doc.tokens[i + 2].text.lower()
                  in ['друга', 'другом', 'другу', 'друге']):
                pos_tags.append('REC_PRN')

            elif (token_text.lower()
                  in ['друга', 'другом', 'другу', 'друге']
                  and i > 0 and
                  doc.tokens[i - 1].text
                  and doc.tokens[i - 1].text
                  in ['от', 'с', 'по', 'на', 'к', 'без', 'о']
                  and doc.tokens[i - 2].text.lower() == 'друг'):
                pos_tags.append('REC_PRN_PART')

            # Тут идут односоставные сочинительные союзы,
            # с которыми мб проблемы
            elif (pymorphy_pos == 'CONJ' and
                  token_text.lower() in ['зато', 'однако']):
                pos_tags.append('CCONJ')

            # Тут идут односоставные подчинительные союзы (однозначно)
            elif ((pymorphy_pos == 'CONJ' and natasha_pos == 'SCONJ' and
                   token_text.lower() in
                   ['словно', 'чтобы', 'будто', 'если', 'ибо']) or
                  token_text.lower() == 'словно'):
                pos_tags.append('SCONJ')

            # Обрабатываем модификаторы превосходной степени,
            # только если дальше идёт прилагательное
            elif (token.lemma and token.lemma.lower()
                  in SUPRL_ADVBS
                  and i + 1 < len(doc.tokens)
                  and (doc.tokens[i + 1].pos == 'ADJ'
                       or pymorphy_for_one_token(doc.tokens[i + 1].text)
                       in ('ADJF', 'ADJS'))):
                pos_tags.append('SPRL_MOD')

            # Обрабатываем прилагательные в превосходной степени
            elif (pymorphy_pos == 'ADJF'
                  and 'Supr'
                  in morph.parse(token_text.lower())[0].tag
            ):
                pos_tags.append('SPRL')

            # Обрабатываем определённые
            # местоимения–существительные
            # или местоимения–прилагательные
            elif (token.lemma and token.lemma.lower()
                  in DEF_NOUN_ADJ_PRNS):
                pos_tags.append('DF_ADJ_N_PRN')

            elif (token_text.lower() in
                  ['прочий', 'прочего', 'прочее',
                   'прочим', 'прочем', 'прочему']
            ):
                pos_tags.append('DF_ADJ_N_PRN')

            # Обрабатываем определённые местоимения-наречия
            elif (token.lemma and token.lemma.lower()
                  in DEF_ADVBS_PRNS):
                pos_tags.append('DF_ADV_PRN')

            elif (natasha_pos == 'PRON'
                  and token.lemma.lower() == 'себя'):
                pos_tags.append('RFLX_PRN')

            # Если слово является неопределённым
            # местоимением–прилагательным
            elif (token.lemma.lower()
                  in INDEFINITE_ADJS_PRNS):
                pos_tags.append('INDF_ADJ_PRN')

            # Если слово является отрицательным
            # местоимением–прилагательным
            elif (token.lemma.lower()
                  in NEGATIVE_ADJS_PRNS):
                pos_tags.append('NEG_ADJ_PRN')

            # Если токен стоит в начале предложения и
            # является вопросительным словом.
            # Надо как-то учесть, что это вопрос.!!!!!!
            elif (token.lemma and token.lemma.lower()
                  in CONJ_WRDS_PRNS and
                  ((i > 0 and doc.tokens[i - 1].text
                    in '.?!…') or
                   (i == 0 and token_text[0].isupper()))
            ):
                pos_tags.append('PRN_QW')

            # Если токен относится к группе местоимений,
            # используемых в качестве союзных слов
            elif (token.lemma and token.lemma.lower()
                  in CONJ_WRDS_PRNS):
                # 1. Стандартный случай: местоимение сразу после запятой
                if i > 0 and doc.tokens[i - 1].text in ',(':
                    pos_tags.append('PRN_CONJ_W')
                # 2. Проверяем случаи с предлогами или существительными
                else:
                    # Ищем запятую в предшествующих токенах
                    # (ограничиваем поиск 40 токенами назад)
                    found_comma = False
                    found_prep_after_comma = False
                    found_noun_after_comma = False
                    found_conj_after_comma = False
                    comma_idx = -1

                    # Поиск запятой
                    for j in range(i - 1, max(0, i - 40) - 1, -1):
                        if doc.tokens[j].text == ',':
                            found_comma = True
                            comma_idx = j
                            break

                    # Если запятая найдена, ищем предлог
                    # или существительное между ней и текущим местоимением
                    if found_comma:
                        for k in range(comma_idx + 1, i):
                            # Получаем теги для текущего токена в цикле
                            curr_token = doc.tokens[k]
                            curr_pos = curr_token.pos
                            curr_pymorphy_pos = pymorphy_for_one_token(
                                curr_token.text)

                            # Проверяем, является ли токен предлогом
                            if curr_pos == 'ADP' or curr_pymorphy_pos == 'PREP':
                                found_prep_after_comma = True
                                break

                            # Проверяем, является ли токен существительным
                            elif (curr_pos == 'NOUN' or curr_pos == 'PROPN'
                                  or curr_pymorphy_pos == 'NOUN'):
                                found_noun_after_comma = True
                                break
                            # Проверяем, является ли токен
                            # соединительным союзом
                            elif (curr_pos == 'CCONJ'
                                  or curr_pymorphy_pos == 'CONJ'):
                                found_conj_after_comma = True
                                break

                        # Если после запятой найден предлог
                        # или существительное или соед. союз,
                        # маркируем текущее местоимение
                        # как союзное слово
                        if (found_prep_after_comma
                                or found_noun_after_comma
                                or found_conj_after_comma):
                            pos_tags.append('PRN_CONJ_W')
                        else:
                            pos_tags.append('PRON')
                    else:
                        pos_tags.append('PRON')

            # Если токен стоит в начале предложения или после S_START,
            # и является вопросительным словом-наречием.
            # Надо как-то учесть, что это вопрос.!!!!!!
            elif (
                    token.lemma and token.lemma.lower() in CONJ_WRDS_ADVBS
                    and (
                            # Вопросительное наречие в начале нового предложения
                            (i == 0) or
                            # или предыдущий токен — знак конца предложения
                            (i > 0 and doc.tokens[i - 1].text in '.?!…') or
                            # или мы уже эмитировали S_START перед текущим токеном
                            (pos_tags and pos_tags[-1] == 'S_START')
                    )
            ):
                pos_tags.append('ADV_QW')

            # Если слово является неопределённым
            # местоимением–существительным
            elif (token.lemma.lower()
                  in INDEFINITE_NOUNS_PRNS):
                pos_tags.append('INDF_NOUN_PRN')
            # Обрабатываем указательные и
            # притяжательные местоимения
            elif (natasha_pos == 'DET'
                  and (pymorphy_pos == 'ADJF'
                       or pymorphy_pos == 'NPRO')
                  or token.lemma.lower()
                  in DMSTR_PRNS):
                # Если слово является притяжательным местоимением
                if (token.lemma.lower()
                        in POSSESSIVE_PRNS):
                    pos_tags.append('POSSV_PRN')

                # Если слово является указательным местоимением
                elif token.lemma.lower() in DMSTR_PRNS:
                    pos_tags.append('DMSTR_PRN')

                elif token.lemma.lower() == 'один':
                    pos_tags.append('NUM')

                elif token.lemma.lower() == 'больший':
                    pos_tags.append('ADJF')

                elif token.lemma.lower() == 'повышенный':
                    pos_tags.append('PRTF')
                elif (token.lemma.lower() in
                      ['взаимосвязанный', 'простой',
                       'зримый', 'обратный', 'ровный']):
                    pos_tags.append('ADJF')
                elif (token.lemma.lower()
                      in ['устаревший', 'существующий', 'связанный']):
                    pos_tags.append('PRTF')
                elif (token_text.lower()
                      in ['само']):
                    pos_tags.append('DEF_NOUN_ADJ_PRNS')
                elif (token_text.lower()
                      in ['никоим']):
                    pos_tags.append('NEG_ADJ_PRN')
                else:
                    pos_tags.append('N/A')

            # Обрабатываем местоимения в функции дополнения
            # (предложные и прямые)
            elif ((natasha_pos == 'PRON'
                   and doc.tokens[i - 1].pos == 'ADP'
                   and (token.feats
                        and token.feats.get('Case')
                        in {'Gen', 'Dat', 'Ins', 'Loc'}))
                  or (natasha_pos == 'PRON'
                      and (token.feats
                           and token.feats.get('Case')
                           in {'Acc', 'Dat', 'Ins', 'Loc'})
                  )):
                pos_tags.append('OBJ_PRN')

            # Если токен входит в группу личных местоимений
            elif (natasha_pos == 'PRON'
                  and token.lemma.lower()
                  in PER_PRONOUNS):
                pos_tags.append('PER_PRN')
            elif (pymorphy_pos == 'NPRO'
                  and token_text.lower() == 'неё'):
                pos_tags.append('PER_PRN')

            # Обрабатываем наречия, используемые для формирования
            # аналитической сравнительной степени
            elif (natasha_pos == 'ADV' and
                  (token.feats
                   and token.feats.get('Degree') == 'Cmp') and
                  token_text.lower()
                  in COMP_ADVBS):
                pos_tags.append('COMP_MOD')

            elif ((natasha_pos == "ADJ"
                   and pymorphy_pos == 'COMP') or
                  (natasha_pos == "ADV"
                   and pymorphy_pos == 'COMP') or
                  ((natasha_pos == "ADJ"
                    or natasha_pos == "ADV"
                    and pymorphy_pos == 'COMP') and
                   token.feats
                   and token.feats.get('Degree') == 'Cmp')):
                pos_tags.append('COMP')

            elif (token_text.lower()
                  in MODAL_WDS):
                pos_tags.append('MDL_WRD')

            elif (token.lemma.lower()
                  in ['мочь', 'хотеть', 'желать']):
                pos_tags.append('MDL_VRB')

            elif (token.lemma.lower()
                  in ['следовать', 'стоить']
                  and ((i + 1 < len(doc.tokens)
                        and pymorphy_for_one_token(doc.tokens[i + 1].text)
                        == 'INFN')
                       or (i + 2 < len(doc.tokens) and
                           pymorphy_for_one_token(doc.tokens[i + 2].text)
                           == 'INFN')
                  )):
                pos_tags.append('MDL_VRB')

            # Проверяем, является ли слово союзным словом-наречием
            # до проверки на наречие
            elif (token_text.lower()
                  in CONJ_WRDS_ADVBS and
                  i > 0
                  and doc.tokens[i - 1].text == ','):
                pos_tags.append('ADV_CONJ_W')


            elif (
                    token.lemma and token.lemma.lower() == 'быть'
                    and i + 1 < len(doc.tokens)
                    and (doc.tokens[i + 1].pos == 'VERB' or
                         pymorphy_for_one_token(doc.tokens[i + 1].text)
                         in {'VERB', 'INFN'}
                    )
            ):
                pos_tags.append('AUX')

            elif (token.lemma and token.lemma.lower()
                  in COPULA_VERBS
                  and (natasha_pos in {'VERB', 'AUX'}
                       or pymorphy_pos == 'VERB')
                  and token.rel == 'cop'
            ):
                pos_tags.append('COPULA')

            # Если токен является инфинитивом
            elif pymorphy_pos == 'INFN':
                pos_tags.append('INFN')
            # Обрабатываем полные причастия
            elif (
                    (pymorphy_pos == 'ADJF'
                     and natasha_pos == 'VERB'
                     and token.feats
                     and token.feats.get('Voice') == 'Pass')
                    or (token.lemma
                        and token.lemma.lower() == 'связанный')
            ):
                pos_tags.append('PRTF')

            # Если слово является неопределённым
            # местоимением–наречием
            elif (token.lemma.lower()
                  in INDEFINITE_ADVBS_PRNS):
                pos_tags.append('INDF_ADV_PRN')

            # Если слово является отрицательным
            # местоимением–существительным
            elif (token.lemma.lower()
                  in NEGATIVE_NOUNS_PRNS):
                pos_tags.append('NEG_N_PRN')

            # Если слово является отрицательным
            # местоимением–наречием
            elif (token.lemma.lower()
                  in NEGATIVE_ADVBS_PRNS):
                pos_tags.append('NEG_ADV_PRN')

            # Если слово является отрицательным
            # местоимением–числительным
            elif (token.lemma.lower()
                  in NEGATIVE_NUMS_PRNS):
                pos_tags.append('NEG_NUM_PRN')

            # Обрабатываем имена собственные
            # как существительные
            elif ((natasha_pos == 'PROPN')
                  or (natasha_pos == 'PROPN' and
                      (pymorphy_pos is None
                       or pymorphy_pos == 'NOUN'))):
                pos_tags.append('NOUN')

            elif ((natasha_pos == 'ADJ'
                   or natasha_pos == 'ADV')
                  and pymorphy_pos == 'ADVB'
            ):
                pos_tags.append('ADVB')

            # Переопределяем название тега для предлога
            elif ((natasha_pos == 'ADP'
                   and pymorphy_pos == 'PREP')
                  or (natasha_pos == 'ADP'
                      and pymorphy_pos is None)
            ):
                pos_tags.append('PREP')

            # Переопределяем название тега
            # для прилагательного полного
            elif (natasha_pos == 'ADJ'
                  and pymorphy_pos == 'ADJF'):
                pos_tags.append('ADJF')

            # Переопределяем название тега
            # для прилагательного краткого
            elif (natasha_pos == 'ADJ'
                  and pymorphy_pos == 'ADJS'
            ):
                pos_tags.append('ADJS')

            # Обрабатываем краткие причастия
            elif pymorphy_pos == 'PRTS':
                pos_tags.append('PRTS')

            # Обрабатываем полные причастия
            elif pymorphy_pos == 'PRTF':
                pos_tags.append('PRTF')

            # Обрабатываем частичку "бы"
            elif (natasha_pos == 'AUX'
                  and pymorphy_pos == 'PRCL'
                  and token.lemma.lower()
                  in ['бы', 'б']):
                pos_tags.append('SBJ_PRCL')

            # ????
            elif (pymorphy_pos == 'NOUN'
                  and natasha_pos == 'ADV'
            ):
                pos_tags.append('NOUN')

            # Обрабатываем деепричастие
            elif ((pymorphy_pos == 'GRND'
                   and natasha_pos == 'VERB')
                  or token_text.lower() == 'высвобождая'):
                pos_tags.append('GRND')

            # Обрабатываем отрицательные частицы
            elif ((natasha_pos == 'PART'
                   or pymorphy_pos == 'PRED')
                  and token.lemma.lower()
                  in ['не', 'ни', 'нет']
            ):
                pos_tags.append('NEG_PART')

            elif natasha_pos == 'X':
                pos_tags.append('FRN_W')
            elif (pymorphy_pos == 'NOUN'
                  and natasha_pos == 'ADJ'):
                pos_tags.append('NOUN')
            elif (pymorphy_pos == 'VERB'
                  and natasha_pos == 'ADJ'):
                pos_tags.append('VERB')
            elif (pymorphy_pos == 'VERB'
                  and natasha_pos == 'ADV'):
                pos_tags.append('VERB')
            elif token_text.lower() == 'нед':
                pos_tags.append('NOUN')
            elif (pymorphy_pos == 'ADJS'
                  and natasha_pos == 'ADV'):
                pos_tags.append('ADVB')
            elif (token_text.lower()
                  in ['также', 'тоже']):
                pos_tags.append('CCONJ')
            elif (token_text.lower()
                  in ['скорее', 'значимо']):
                pos_tags.append('ADVB')
            elif (token_text.lower()
                  in ['известно', 'возможно']):
                pos_tags.append('ADVB')
            elif (pymorphy_pos == 'CONJ'
                  and natasha_pos == 'ADV'):
                pos_tags.append('SCONJ')
            # !валовым! национальным продуктом)
            elif (pymorphy_pos == 'ADJF'
                  and natasha_pos == 'ADV'):
                pos_tags.append('ADJF')
            # стадии !относительно! велико
            elif (pymorphy_pos == 'PREP'
                  and natasha_pos == 'ADV'):
                pos_tags.append('ADVB')
            elif (pymorphy_pos == 'PRCL'
                  and natasha_pos == 'ADV'):
                pos_tags.append('ADVB')
            elif (pymorphy_pos == 'PRED'
                  and natasha_pos == 'ADV'):
                pos_tags.append('ADVB')
            elif (pymorphy_pos == 'PRED'
                  and natasha_pos == 'ADV'):
                pos_tags.append('ADVB')
            # либо
            elif (pymorphy_pos == 'CONJ'
                  and natasha_pos == 'ADJ'):
                pos_tags.append('CCONJ')
            elif (pymorphy_pos == 'CONJ'
                  and natasha_pos == 'PRON'):
                pos_tags.append('SCONJ')
            elif ((pymorphy_pos is None
                   and natasha_pos == 'ADJ')
                  or token_text == 'Ха'
                  or token_text == 'р'):
                pos_tags.append('N/A')
            elif token_text.lower() == 'устаревшие':
                pos_tags.append('PRTF')

            else:
                # В остальных случаях доверяем
                # выбор библиотеки Natasha
                pos_tags.append(
                    natasha_pos
                    if natasha_pos
                    else 'N/A'
                )

        tokens_analyzed.append(token_text)

    return pos_tags, tokens_analyzed


from itertools import zip_longest


def _print_alignment(tags: list[str], toks: list[str]) -> None:
    """Построчно печатает индекс, токен и тег; отмечает первое расхождение."""
    n_tags, n_toks = len(tags), len(toks)
    print(f"TAGS: {n_tags} | TOKENS: {n_toks}")
    print("-" * 80)
    first_mismatch_idx = None
    for idx, (tok, tag) in enumerate(zip_longest(toks, tags, fillvalue=None)):
        mismatch = (tok is None) or (tag is None)
        mark = " <-- MISMATCH" if mismatch and first_mismatch_idx is None else ""
        if mismatch and first_mismatch_idx is None:
            first_mismatch_idx = idx
        print(f"{idx:05d}  {repr(tok):>20}  |  {repr(tag):<15}{mark}")
    if first_mismatch_idx is not None:
        print("-" * 80)
        print(f"Первое расхождение на позиции: {first_mismatch_idx}")


def debug_pos_tagger(text: str) -> None:
    tags, toks = pos_tagger(text)
    _print_alignment(tags, toks)


if __name__ == "__main__":
    text = input('Введите текст:')
    tags, tokens = pos_tagger(text)
    _print_alignment(tags, tokens)
