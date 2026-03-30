# -*- coding: utf-8 -*- # Языковая кодировка UTF-8
import json
import os
import re
import warnings

import nltk

from tools.core.natasha_pymorphy_pos_tagger import pos_tagger
from tools.miscellaneous.verbs_analysis import RuVerbAnalyzer
from tools.miscellaneous.nouns_analysis import RuNounAnalyzer
from tools.miscellaneous.adjectives_analysis import RuAdjectiveAnalyzer
from tools.simplification.sentences_complexity import analyze_clause_structure

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pymorphy2.analyzer"
)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt_tab/russian')
except LookupError:
    nltk.download('punkt_tab')

from natasha import (
    Segmenter, MorphVocab, NewsEmbedding,
    NewsMorphTagger, NewsSyntaxParser, Doc
)
from colorama import Fore, Style, init
from rich.console import Console
from rich.table import Table

from tools.core.text_preparation import TextPreProcessor
from tools.core import preprocess_text
from tools.core.constants import (
    NON_TRANSLATED_DB_NAME,
    MACHINE_TRANSLATED_DB_NAME,
    HUMAN_TRANSLATED_DB_NAME,
    RETURN_TO_MENU,
    INCORRECT_CHOICE
)
from tools.core.tokens_counter import count_tokens
from tools.core.custom_punkt_tokenizer import sent_tokenize_with_abbr
from tools.work_with_db import SaveToDatabase
from tools.core.utils import (
    wait_for_enter_to_analyze,
    display_morphological_annotation,
    format_morphological_features,
    wait_for_enter_to_choose_opt,
    check_db_exists, choose_db,
    count_types_in_text,
    print_characteristic_header,
    print_greeting,
    print_annotation_ready,
    print_analysis_ready,
    print_synt_annot_display,
    print_main_menu,
    process_text_from_file
)
from tools.core.validators import (
    validate_gender,
    validate_years
)

from tools.core.data.pronouns import (
    pers_possessive_pronouns_analysis_list,
    reflexive_pronoun_list, demonstrative_pronouns_list,
    defining_pronouns_list, relative_pronouns_list,
    indefinite_pronouns_list, negative_pronouns_list
)

# Импорты для "Simplification_features"
from tools.simplification.lexical_density import calculate_lexical_density
from tools.simplification.lexical_variety import lexical_variety
from tools.simplification.mean_word_length import (
    mean_word_length_char, mean_word_length_syllab
)
from tools.simplification.mean_sent_length import (
    mean_sentence_length_in_tokens, mean_sentence_length_in_chars
)
from tools.simplification.mean_word_rank import calculate_mean_word_rank
from tools.simplification.most_frequent_words import (
    find_n_most_frequent_words
)
# Импорты для "Normalisation_features"
from tools.normalisation.repetition import calculate_repetition

# Импорты для "Explicitation_features"
from tools.explicitation.explicit_naming import (
    calculate_explicit_naming_ratio
)
from tools.explicitation.single_naming import single_naming_frequency
from tools.explicitation.mean_multiple_naming import (
    calculate_mean_multiple_naming
)
from tools.explicitation.sci_dm_analysis import sci_dm_search
from tools.explicitation.lexical_cohesion import (
    load_model as load_cohesion_model,
    init_pipeline as init_cohesion_pipeline,
    OptimizedVectorCache,
    analyze_text_cohesion,
)
# Импорты для "Interference_features"
from tools.interference.n_grams_analyzer import (
    pos_ngrams,
    character_ngrams
)
from tools.interference.positional_token_freq import (
    calculate_position_frequencies
)
from tools.interference.positional_tokens_contexts_by_sent import (
    extract_positions,
)
from tools.interference.contextual_func_words import (
    contextual_function_words_in_trigrams
)

# Импорты для "Miscellaneous_features"
from tools.miscellaneous.func_words_freqs import (
    compute_function_word_frequencies
)
from tools.miscellaneous.pronouns_freq import compute_pronoun_frequencies
from tools.miscellaneous.punct_analysis import analyze_punctuation
from tools.miscellaneous.passive_to_all_verbs_ratio import (
    calculate_passive_verbs_ratio
)
from tools.miscellaneous.flesh_readability_score import (
    flesh_readability_index_for_rus
)

# Импорт для лемматизации
from tools.core.lemmatizators import (
    lemmatize_words,
    lemmatize_words_without_stopwords, parse_cached
)

console = Console()
init(autoreset=True)


class CorpusText:
    _shared_segmenter = None
    _shared_morph_vocab = None
    _shared_embedding = None
    _shared_morph_tagger = None
    _shared_syntax_parser = None

    _shared_cohesion_model = None
    _shared_cohesion_nlp = None
    _shared_cohesion_vec_cache = None

    _shared_verb_analyzer = None
    _shared_noun_analyzer = None
    _shared_adjective_analyzer = None

    def __init__(self, db=None, text=None):
        self.db = db
        self.text = text
        self.title = ""
        self.subject_area = ""
        self.keywords = ""
        self.publication_year = ""
        self.published_in = ""
        self.authors = ""
        self.author_gender = ""
        self.author_birth_year = ""

        self.segmenter = None
        self.morph_vocab = None
        self.embedding = None
        self.morph_tagger = None
        self.syntax_parser = None

        self._cohesion_model = None
        self._cohesion_nlp = None
        self._cohesion_vec_cache = None

        self.verb_analyzer = None
        self.noun_analyzer = None
        self.adjective_analyzer = None


    def show_texts(self):
        """Отображает все тексты с их порядковыми номерами и
         позволяет выбрать один для отображения
         подробной информации."""
        with SaveToDatabase(self.db) as db_fetch:
            db_fetch.display_texts()

    def analyze_and_save(self, show_analysis=False):
        # Подсчитываем общее количество всех без исключения токенов,
        # количество алфавитных токенов, количество пунктуационных знаков,
        # количество других символов в тексте

        cls = type(self)

        if cls._shared_segmenter is None:
            cls._shared_segmenter = Segmenter()

        if cls._shared_morph_vocab is None:
            cls._shared_morph_vocab = MorphVocab()

        if cls._shared_embedding is None:
            cls._shared_embedding = NewsEmbedding()

        if cls._shared_morph_tagger is None:
            cls._shared_morph_tagger = NewsMorphTagger(cls._shared_embedding)

        if cls._shared_syntax_parser is None:
            cls._shared_syntax_parser = NewsSyntaxParser(cls._shared_embedding)

        if cls._shared_cohesion_model is None:
            cls._shared_cohesion_model = load_cohesion_model()

        if cls._shared_cohesion_nlp is None:
            cls._shared_cohesion_nlp = init_cohesion_pipeline("ru")

        if cls._shared_cohesion_vec_cache is None:
            cls._shared_cohesion_vec_cache = OptimizedVectorCache(cls._shared_cohesion_model)

        if cls._shared_verb_analyzer is None:
            cls._shared_verb_analyzer = RuVerbAnalyzer(use_syntax=True)

        if cls._shared_noun_analyzer is None:
            cls._shared_noun_analyzer = RuNounAnalyzer()

        if cls._shared_adjective_analyzer is None:
            cls._shared_adjective_analyzer = RuAdjectiveAnalyzer()

        self.segmenter = cls._shared_segmenter
        self.morph_vocab = cls._shared_morph_vocab
        self.embedding = cls._shared_embedding
        self.morph_tagger = cls._shared_morph_tagger
        self.syntax_parser = cls._shared_syntax_parser

        self._cohesion_model = cls._shared_cohesion_model
        self._cohesion_nlp = cls._shared_cohesion_nlp
        self._cohesion_vec_cache = cls._shared_cohesion_vec_cache

        self.verb_analyzer = cls._shared_verb_analyzer
        self.noun_analyzer = cls._shared_noun_analyzer
        self.adjective_analyzer = cls._shared_adjective_analyzer

        self.fix_spacing()

        (all_tokens_count, alpha_tokens_count,
         all_punct_tokens_count, other_symbols) = count_tokens(self.text)

        # Замена аббревиатур.
        text_with_full_abbr = (
            TextPreProcessor().point_abbr_to_full(self.text)
        )

        # Парсинг всех словарных токенов в виде списка объектов Parse
        parsed_words_info = lemmatize_words(text_with_full_abbr)

        # Список всех лемм текста (Cyrillic и Latin)
        all_lemmatized_words = [
            token.normal_form for token in parsed_words_info
        ]

        # Парсинг словарных токенов в виде списка объектов Parse
        # (без стоп-слов) и кол-во удаленных стоп-слов
        parsed_text_without_stopwords, removed_stop_w_count = (
            lemmatize_words_without_stopwords(self.text)
        )

        # Список лемм знаменательных слов (Cyrillic и Latin)
        lemmatized_content_words = [
            token.normal_form for token in
            parsed_text_without_stopwords
        ]

        #  Список предложений текста.
        list_of_sentences = sent_tokenize_with_abbr(self.text)

        # Словарь с типами знаменательных слов и
        # их абсолютными частотами
        content_type_counts = (
            count_types_in_text(self.text.lower())
        )

        # Анализ Simplification_features
        if show_analysis:
            print_characteristic_header("simplification")

        lexical_density = calculate_lexical_density(
            alpha_tokens_count,
            lemmatized_content_words,
            show_analysis,
        )  # OK!

        (ttr_lex_variety, log_ttr_lex_variety,
         modified_lex_variety, standardised_lex_variety) = (
            lexical_variety(all_lemmatized_words, show_analysis)
        )  # OK!

        mean_word_length = (
            mean_word_length_char(self.text, show_analysis)
        )  # OK!
        syllable_ratio, total_syllables_count = (
            mean_word_length_syllab(self.text, show_analysis)
        )  # OK!

        tokens_mean_sent_length = (
            mean_sentence_length_in_tokens(
                list_of_sentences, show_analysis
            )
        )  # OK!
        chars_mean_sent_length = mean_sentence_length_in_chars(
            list_of_sentences, show_analysis
        )  # OK!
        mean_word_rank_1, mean_word_rank_2 = (
            calculate_mean_word_rank(all_lemmatized_words,
                                     show_analysis)
        )  # OK!
        content_words_freqs = find_n_most_frequent_words(
            alpha_tokens_count,
            content_type_counts,
            show_analysis=show_analysis
        )  # OK!

        clause_metrics = analyze_clause_structure(
            list_of_sentences,
            show_analysis=show_analysis
        )
        (punct_marks_normalized_frequency,
         punct_marks_to_all_punct_frequency,
         punctuation_counts) = analyze_punctuation(
            self.text,
            show_analysis)

        readability_index = flesh_readability_index_for_rus(
            self.text, syllable_ratio, show_analysis
        )  # OK!

        if show_analysis:
            # Анализ Normalisation_features
            print_characteristic_header("normalization")

        (repetition, repeated_content_words_count,
         repeated_content_words, total_word_tokens) = (
            calculate_repetition(
                alpha_tokens_count, content_type_counts, show_analysis)
        )  # OK!

        if show_analysis:
            # Анализ Explicitation_features
            print_characteristic_header("explicitation")

        (explicit_naming_ratio,
         found_entities_info,
         found_entities_count) = (
            calculate_explicit_naming_ratio(
                self.text,
                parsed_words_info,
                show_analysis
            ))  # OK!

        single_naming = single_naming_frequency(
            found_entities_info, show_analysis
        )  # OK!
        (mean_multiple_naming, single_entities,
         single_entities_count, multiple_entities,
         multiple_entities_count) = calculate_mean_multiple_naming(
            found_entities_info, show_analysis)  # OK!

        (total_tokens_with_dms,
         sci_markers_total_count,
         found_sci_dms,
         markers_counts,
         topic_intro_dm_count,
         topic_intro_dm_in_ord,
         topic_intro_dm_freq,
         info_sequence_count,
         info_sequence_in_ord,
         info_sequence_freq,
         illustration_dm_count,
         illustration_dm_in_ord,
         illustration_dm_freq,
         material_sequence_count,
         material_sequence_in_ord,
         material_sequence_freq,
         conclusion_dm_count,
         conclusion_dm_in_ord,
         conclusion_dm_freq,
         intro_new_addit_info_count,
         intro_new_addit_info_in_ord,
         intro_new_addit_info_freq,
         info_explanation_or_repetition_count,
         info_explanation_or_repetition_in_ord,
         info_explanation_or_repetition_freq,
         contrast_dm_count, contrast_dm_in_ord,
         contrast_dm_freq,
         examples_introduction_dm_count,
         examples_introduction_dm_in_ord,
         examples_introduction_dm_freq,
         author_opinion_count,
         author_opinion_in_ord,
         author_opinion_freq,
         author_attitude_count,
         author_attitude_in_ord,
         author_attitude_freq,
         high_certainty_modal_words_count,
         high_certainty_modal_words_in_ord,
         high_certainty_modal_words_freq,
         moderate_certainty_modal_words_count,
         moderate_certainty_modal_words_in_ord,
         moderate_certainty_modal_words_freq,
         uncertainty_modal_words_count,
         uncertainty_modal_words_in_ord,
         uncertainty_modal_words_freq,
         call_to_action_dm_count,
         call_to_action_dm_in_ord,
         call_to_action_dm_freq,
         joint_action_count,
         joint_action_in_ord,
         joint_action_freq,
         putting_emphasis_dm_count,
         putting_emphasis_dm_in_ord,
         putting_emphasis_dm_freq,
         refer_to_background_knowledge_count,
         refer_to_background_knowledge_in_ord,
         refer_to_background_knowledge_freq,
         cause_effect_dm_count,
         cause_effect_dm_str,
         cause_effect_dm_freq,
         purpose_statement_dm_count,
         purpose_statement_dm_str,
         purpose_statement_dm_freq
         ) = sci_dm_search(self.text, show_analysis)  # OK!

        cohesion_metrics = analyze_text_cohesion(
            self.text,
            self._cohesion_model,
            self._cohesion_nlp,
            self._cohesion_vec_cache,
            show_analysis=show_analysis
        )

        if show_analysis:
            # Анализ Interference_features
            print_characteristic_header("interference")
        n_grams, tokens = pos_tagger(self.text)  # OK!

        (pos_unigrams_counts, pos_unigrams_freq, pos_bigrams_counts,
         pos_bigrams_freq, pos_trigrams_counts, pos_trigrams_freq) = (
            pos_ngrams(n_grams, tokens, show_analysis=show_analysis)
        )  # OK!

        (char_unigram_counts, char_unigram_freq, char_bigram_counts,
         char_bigram_freq, char_trigram_counts, char_trigram_freq,
         char_fourgram_counts, char_fourgram_freq, char_fivegram_counts,
         char_fivegram_freq) = character_ngrams(
            self.text, show_analysis=show_analysis)  # OK!

        (token_positions_normalized_frequencies,
         token_positions_counts) = calculate_position_frequencies(
            n_grams, tokens, show_analysis)  # OK!

        token_positions_in_sent = extract_positions(
            n_grams, tokens, show_analysis
        )  # OK!

        (func_w_trigrams_freqs, func_w_trigram_with_pos_counts,
         func_w_full_contexts) = contextual_function_words_in_trigrams(
            list_of_sentences, show_analysis)  # OK!

        if show_analysis:
            # Анализ Miscellaneous_features
            print("\n" + Fore.LIGHTWHITE_EX + "*" * 80)

            print(
                Fore.GREEN + Style.BRIGHT +

                "                        "
                "             ОСТАЛЬНЫЕ" +
                Fore.LIGHTGREEN_EX + Style.BRIGHT +
                " ИНДИКАТОРЫ")
            print(Fore.LIGHTWHITE_EX + "*" * 80)
            wait_for_enter_to_analyze()

        func_words_freq, func_words_counts = (
            compute_function_word_frequencies(self.text, show_analysis)
        )  # OK

        (pers_possessive_pronouns_frequencies,
         pers_possessive_pronouns_counts) = (
            compute_pronoun_frequencies(
                self.text, pers_possessive_pronouns_analysis_list,
                show_analysis)
        )
        reflexive_pronoun_frequencies, reflexive_pronoun_counts = (
            compute_pronoun_frequencies(
                self.text, reflexive_pronoun_list, show_analysis)
        )
        (demonstrative_pronouns_frequencies,
         demonstrative_pronouns_counts) = compute_pronoun_frequencies(
            self.text, demonstrative_pronouns_list, show_analysis
        )
        (defining_pronouns_frequencies,
         defining_pronouns_counts) = compute_pronoun_frequencies(
            self.text, defining_pronouns_list, show_analysis
        )
        (relative_pronouns_frequencies,
         relative_pronouns_counts) = (
            compute_pronoun_frequencies(
                self.text, relative_pronouns_list, show_analysis)
        )
        (indefinite_pronouns_frequencies,
         indefinite_pronouns_counts) = (
            compute_pronoun_frequencies(
                self.text, indefinite_pronouns_list, show_analysis)
        )
        negative_pronouns_frequencies, negative_pronouns_counts = (
            compute_pronoun_frequencies(
                self.text, negative_pronouns_list, show_analysis)
        )
        self.verb_analyzer.analyze(self.text)
        tense = self.verb_analyzer.tense_distribution_among_finite()
        if show_analysis:
            self.verb_analyzer.print_analysis()

        (passive_to_all_v_ratio, passive_verbs,
         passive_verbs_count, all_verbs, all_verbs_count) = (
            calculate_passive_verbs_ratio(
                self.text, show_analysis,
                segmenter=self.segmenter,
                morph_tagger=self.morph_tagger,
            )
        )

        self.noun_analyzer.analyze(self.text)
        if show_analysis:
            self.noun_analyzer.print_analysis()

        self.adjective_analyzer.analyze(self.text)
        if show_analysis:
            self.adjective_analyzer.print_analysis()


        # Использование контекстного менеджера для
        # работы с базой данных
        with SaveToDatabase(self.db) as db_saver:
            db_saver.insert_simplification_features(
                lexical_density,
                ttr_lex_variety,
                log_ttr_lex_variety,
                modified_lex_variety,
                standardised_lex_variety,
                mean_word_length,
                syllable_ratio,
                total_syllables_count,
                tokens_mean_sent_length,
                chars_mean_sent_length,
                mean_word_rank_1,
                mean_word_rank_2,
                content_words_freqs,
                content_type_counts,
                all_tokens_count,
                alpha_tokens_count,
                all_punct_tokens_count,
                mean_clause_length=clause_metrics['avg_tokens_per_clause'],
                avg_clauses_per_sent=clause_metrics['avg_clauses_per_sent'],
                simple_sentences_ratio=clause_metrics['simple_ratio'],
                readability_index=readability_index,
                punct_marks_normalized_frequency=punct_marks_normalized_frequency,
                punct_marks_to_all_punct_frequency=punct_marks_to_all_punct_frequency,
                punctuation_counts=punctuation_counts,
            )

            db_saver.insert_normalisation_features(
                repetition, repeated_content_words_count,
                repeated_content_words, total_word_tokens
            )

            db_saver.insert_explicitation_features(
                explicit_naming_ratio, single_naming,
                mean_multiple_naming,
                single_entities, single_entities_count,
                multiple_entities, multiple_entities_count,
                found_entities_info,
                found_entities_count,
                total_tokens_with_dms,
                sci_markers_total_count,
                found_sci_dms, markers_counts,
                topic_intro_dm_count,
                topic_intro_dm_in_ord,
                topic_intro_dm_freq,
                info_sequence_count,
                info_sequence_in_ord,
                info_sequence_freq,
                illustration_dm_count,
                illustration_dm_in_ord,
                illustration_dm_freq,
                material_sequence_count,
                material_sequence_in_ord,
                material_sequence_freq,
                conclusion_dm_count,
                conclusion_dm_in_ord,
                conclusion_dm_freq,
                intro_new_addit_info_count,
                intro_new_addit_info_in_ord,
                intro_new_addit_info_freq,
                info_explanation_or_repetition_count,
                info_explanation_or_repetition_in_ord,
                info_explanation_or_repetition_freq,
                contrast_dm_count, contrast_dm_in_ord,
                contrast_dm_freq,
                examples_introduction_dm_count,
                examples_introduction_dm_in_ord,
                examples_introduction_dm_freq,
                author_opinion_count, author_opinion_in_ord,
                author_opinion_freq,
                author_attitude_count, author_attitude_in_ord,
                author_attitude_freq,
                high_certainty_modal_words_count,
                high_certainty_modal_words_in_ord,
                high_certainty_modal_words_freq,
                moderate_certainty_modal_words_count,
                moderate_certainty_modal_words_in_ord,
                moderate_certainty_modal_words_freq,
                uncertainty_modal_words_count,
                uncertainty_modal_words_in_ord,
                uncertainty_modal_words_freq,
                call_to_action_dm_count,
                call_to_action_dm_in_ord,
                call_to_action_dm_freq,
                joint_action_count,
                joint_action_in_ord,
                joint_action_freq,
                putting_emphasis_dm_count,
                putting_emphasis_dm_in_ord,
                putting_emphasis_dm_freq,
                refer_to_background_knowledge_count,
                refer_to_background_knowledge_in_ord,
                refer_to_background_knowledge_freq,
                cause_effect_dm_count,
                cause_effect_dm_str,
                cause_effect_dm_freq,
                purpose_statement_dm_count,
                purpose_statement_dm_str,
                purpose_statement_dm_freq,
                avg_token_pair_similarity=cohesion_metrics["avg_token_pair_similarity"],
                mean_adjacent_sentence_cosine=cohesion_metrics["mean_adjacent_sentence_cosine"],
            )

            db_saver.insert_interference_features(
                pos_unigrams_counts, pos_unigrams_freq, pos_bigrams_counts,
                pos_bigrams_freq, pos_trigrams_counts, pos_trigrams_freq,
                char_unigram_counts, char_unigram_freq, char_bigram_counts,
                char_bigram_freq, char_trigram_counts, char_trigram_freq,
                char_fourgram_counts, char_fourgram_freq,
                char_fivegram_counts, char_fivegram_freq,
                token_positions_normalized_frequencies, token_positions_counts,
                token_positions_in_sent, func_w_trigrams_freqs,
                func_w_trigram_with_pos_counts, func_w_full_contexts
            )

            db_saver.insert_miscellaneous_features(
                func_words_freq, func_words_counts,
                pers_possessive_pronouns_frequencies,
                pers_possessive_pronouns_counts,
                reflexive_pronoun_frequencies,
                reflexive_pronoun_counts,
                demonstrative_pronouns_frequencies,
                demonstrative_pronouns_counts,
                defining_pronouns_frequencies,
                defining_pronouns_counts,
                relative_pronouns_frequencies,
                relative_pronouns_counts,
                indefinite_pronouns_frequencies,
                indefinite_pronouns_counts,
                negative_pronouns_frequencies,
                negative_pronouns_counts,
                passive_to_all_v_ratio,
                passive_verbs,
                passive_verbs_count,
                all_verbs,
                all_verbs_count,

                self.pct(self.verb_analyzer.ratio_verbs_to_tokens()),
                self.pct(self.verb_analyzer.ratio_participles_to_verbs()),
                self.pct(self.verb_analyzer.ratio_converbs_to_verbs()),
                self.pct(self.verb_analyzer.ratio_short_part_to_part()),
                self.pct(self.verb_analyzer.ratio_imp_among_all_verbs()),
                self.pct(self.verb_analyzer.ratio_imp_among_finite()),
                self.verb_analyzer.pct(tense['Pres']),
                self.verb_analyzer.pct(tense['Fut']),
                self.verb_analyzer.pct(tense['Past']),
                self.pct(self.verb_analyzer.ratio_preposed_participles()),
                self.pct(self.verb_analyzer.ratio_nonpreposed_participles()),

                self.pct(self.noun_analyzer.ratio_nouns_to_tokens()),
                self.pct(self.noun_analyzer.ratio_neuter_to_nouns()),
                self.pct(self.noun_analyzer.ratio_singular_to_nouns()),
                self.pct(self.noun_analyzer.ratio_abstract_to_nouns()),

                self.pct(self.adjective_analyzer.ratio_adjectives_to_tokens()),
                self.pct(self.adjective_analyzer.ratio_short_to_adjectives()),
                self.pct(self.adjective_analyzer.ratio_comparative_to_adjectives()),
                self.pct(self.adjective_analyzer.ratio_superlative_to_adjectives())
            )

    @staticmethod
    def pct(x: float) -> str:
        return f"{x * 100:.2f}"


    def save_text_passport(self, text, title, subject_area, keywords,
                           publication_year, published_in,
                           authors, author_gender, author_birth_year):
        # Использование контекстного менеджера для работы с базой данных
        with SaveToDatabase(self.db) as db_saver:
            db_saver.insert_text_passport(
                text, title, subject_area, keywords,
                publication_year, published_in,
                authors, author_gender, author_birth_year
            )

    def create_text_passport(self):
        """Метод для ввода информации о паспорте текста
         и установки соответствующих атрибутов."""
        print("\n" + Fore.LIGHTWHITE_EX + "*" * 80)
        print(
            Fore.GREEN + Style.BRIGHT +
            "                            "
            "СОЗДАНИЕ ПАСПОРТА ТЕКСТА")
        print(Fore.LIGHTWHITE_EX + "*" * 80)

        self.text = self.text
        # --- Обязательные поля ---
        self.title = input(
            "Введите название статьи (обязательно): "
        ).strip()

        while not self.title:
            self.title = input(
                Fore.LIGHTRED_EX +
                "Название статьи не может быть пустым. "
                "\nВведите название статьи: "
            ).strip()

        self.subject_area = input(
            "Введите предметную область (обязательно): "
        ).strip()

        while not self.subject_area:
            self.subject_area = input(
                Fore.LIGHTRED_EX +
                "Предметная область не может быть пустой. "
                "\nВведите предметную область: "
            ).strip()

        # --- Спрашиваем, заполнять ли необязательные
        # поля паспорта ---

        while True:
            fill_optional = input(
                Fore.GREEN + Style.BRIGHT +
                "Заполнить остальные поля паспорта (y/n)? "
            ).strip().lower()
            if fill_optional in ("y", "n"):
                break
            print(Fore.LIGHTRED_EX +
                  "Неверный ввод. "
                  "Введите 'y' или 'n'.")

        if fill_optional == "y":
            # --- Необязательные поля ---
            self.keywords = input(
                "Введите указанные "
                "в статье ключевые слова "
                "(через запятую): "
            ).strip()

            while True:
                try:
                    self.publication_year = input(
                        "Введите год публикации: "
                    ).strip()
                    if self.publication_year:
                        self.publication_year = validate_years(
                            self.publication_year,
                            "года публикации",
                            single_year=True
                        )
                    break
                except ValueError as e:
                    print(Fore.LIGHTRED_EX + str(e))

            self.published_in = input(
                "Введите место публикации (журнал): "
            )

            self.authors = input(
                "Введите имена автора(ов) статьи: "
            )
            self.author_gender = validate_gender(
                input("Введите пол автора(ов) "
                      "(через запятую): "
                      ).strip())

            while True:
                try:
                    self.author_birth_year = input(
                        "Введите год(ы) "
                        "рождения автора(ов)"
                        " (через запятую): "
                    ).strip()
                    if self.author_birth_year:
                        self.author_birth_year = validate_years(
                            self.author_birth_year,
                            "год(ы) рождения"
                        )
                    break
                except ValueError as e:
                    print(Fore.LIGHTRED_EX + str(e))

        print(Fore.LIGHTGREEN_EX + Style.BRIGHT +
              "\n     ПАСПОРТ ТЕКСТА УСПЕШНО СОЗДАН!"
              )

    def display_text_passport(self):
        """Метод для отображения паспорта текста в
        виде таблицы."""
        print(Fore.BLUE + Style.BRIGHT +
              "\n            "
              "ПАСПОРТ ТЕКСТА")

        table = Table()
        table.add_column(
            "Поле",
            style="bold",
            min_width=20
        )
        table.add_column("Значение")

        table.add_row(
            "Название статьи",
            self.title
        )
        table.add_row(
            "Предметная область",
            self.subject_area
        )

        if self.keywords:
            table.add_row(
                "Ключевые слова",
                self.keywords
            )
        if self.publication_year:
            table.add_row(
                "Год публикации",
                str(self.publication_year)
            )
        if self.published_in:
            table.add_row(
                "Место публикации (журнал)",
                self.published_in
            )
        if self.authors:
            table.add_row(
                "Автор(ы)",
                self.authors
            )
        if self.author_gender:
            table.add_row(
                "Пол автора(ов)",
                self.author_gender
            )
        if self.author_birth_year:
            table.add_row(
                "Год рождения автора(ов)",
                self.author_birth_year
            )
        console.print(table)

    def fix_spacing(self):
        """Удаляет лишние пробелы и исправляет пунктуацию."""
        text = self.text
        # Удаление символов новой строки и замена их на один пробел
        text = re.sub(r'\n+', ' ', text)
        # Регулярное выражение для поиска шаблона
        # "Заглавная/строчная буква. Заглавная/строчная буква"
        pattern = r'([А-Яа-яЁёA-Za-z])\.([А-Яа-яЁёA-Za-z])'
        # Замена на "Заглавная буква . Пробел Заглавная буква"
        text = re.sub(pattern, r'\1. \2', text)
        # Замена любых последовательностей пробелов
        # (больше одного) на один пробел
        text = re.sub(r'\s+', ' ', text)
        # Обработка знаков препинания, сливающихся со словом
        text = re.sub(
            r'([,.!?])([А-Яа-яЁёA-Za-z])',
            r'\1 \2',
            text
        )
        # Удаление лишних пробелов перед знаками препинания
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        self.text = text
        return text

    def get_morphological_annotation(self):
        """Метод для выполнения морфологической разметки текста."""
        sentences = sent_tokenize_with_abbr(self.text)
        all_sentences_info = []
        for sentence in sentences:
            tokens_morph_info = {}
            text = re.sub(
                r'([\.\,\!\"\„\”\«\»\‘\’\(\)\?\;\%\)\)\[\]\:'
                r'\—])',
                r' \1 ',
                sentence
            )
            punctuation = set(".,!?;%)([]—")
            token_index = 1

            for token in text.split():
                token = token.strip()
                if re.match(r'\d+', token):
                    pos = 'NUMBER'
                    lemma = token
                    morph_features = 'N/A'
                elif token in punctuation:
                    pos = 'PUNCT'
                    lemma = token
                    morph_features = 'N/A'
                elif re.match(r'[a-zA-Z]', token):
                    pos = 'LATN'
                    lemma = token.lower()
                    morph_features = 'N/A'
                else:
                    parsed_token = parse_cached(token)
                    pos = (
                        parsed_token.tag.POS) if (
                        parsed_token.tag.POS) else 'UNKNOWN'
                    lemma = parsed_token.normal_form
                    morph_features = format_morphological_features(
                        parsed_token.tag
                    )

                tokens_morph_info[f'Token {token_index}'] = {
                    "token": token,
                    "lemma": lemma,
                    "POS": pos,
                    "morph_features": morph_features
                }

                token_index += 1

            all_sentences_info.append(tokens_morph_info)

        self.morphological_analysis_result = all_sentences_info
        morph_analysis_str = json.dumps(
            all_sentences_info,
            ensure_ascii=False,
            indent=4
        )

        with SaveToDatabase(self.db) as db_saver:
            db_saver.insert_morphological_annotation(
                morph_analysis_str
            )
        return all_sentences_info

    def get_syntactic_annotation(self, save=True):
        """Метод для выполнения синтаксической разметки
        текста с использованием Natasha."""
        cls = type(self)

        if cls._shared_segmenter is None:
            cls._shared_segmenter = Segmenter()
        if cls._shared_embedding is None:
            cls._shared_embedding = NewsEmbedding()
        if cls._shared_morph_tagger is None:
            cls._shared_morph_tagger = NewsMorphTagger(cls._shared_embedding)
        if cls._shared_syntax_parser is None:
            cls._shared_syntax_parser = NewsSyntaxParser(cls._shared_embedding)

        self.segmenter = cls._shared_segmenter
        self.morph_tagger = cls._shared_morph_tagger
        self.syntax_parser = cls._shared_syntax_parser

        self.syntactic_analysis_result = []

        doc = Doc(self.text)

        # Сегментация на предложения для
        # дальнейшего анализа
        doc.segment(self.segmenter)
        # Морфологический анализ для
        # дальнейшего синтаксического анализа
        doc.tag_morph(self.morph_tagger)
        # Анализ синтаксиса
        doc.parse_syntax(self.syntax_parser)
        # Сохраняем результат анализа
        self.syntactic_analysis_result = doc

        # Собираем синтаксическую разметку в строку
        tokens_info = []
        i = 0

        for token in doc.tokens:
            # Получаем информацию о каждом токене
            token_info = {
                f"TOKEN {i + 1}": token.text,
                "id": token.id,
                "head_id": token.head_id,
                "pos": token.pos,
                "dependency": token.rel,
                "features": token.feats
            }
            tokens_info.append(token_info)
            i += 1

        # Преобразуем список словарей
        # в строку в формате JSON
        synt_analysis_string = json.dumps(
            tokens_info,
            ensure_ascii=False,
            indent=4
        )
        if save:
            with SaveToDatabase(self.db) as db_saver:
                db_saver.insert_syntactic_annotation(
                    synt_analysis_string
                )
        return synt_analysis_string

    def display_syntactic_annotation(self):
        """Метод для отображения синтаксической разметки
        в виде древовидной структуры."""
        if self.syntactic_analysis_result:
            print_synt_annot_display()

            total_sentences = len(
                self.syntactic_analysis_result.sents
            )
            start = 0
            batch_size = 5

            while start < total_sentences:
                end = min(start + batch_size, total_sentences)

                for i in range(start, end):
                    print(Fore.GREEN + Style.BRIGHT +
                          f'\nПРЕДЛОЖЕНИЕ {i + 1}:\n'
                          )
                    (self.syntactic_analysis_result.sents[i].
                     syntax.print()
                     )
                    print("\n" + Fore.LIGHTBLUE_EX +
                          Style.BRIGHT + "*" * 150)

                # Обновляем значение start до фактического
                # конца отображенных предложений
                start = end
                print(
                    Fore.LIGHTWHITE_EX + Style.BRIGHT +
                    f"Отображено {start} предложений. "
                    f"Всего предложений: {total_sentences}"
                )

                if start < total_sentences:
                    user_input = input(
                        Fore.GREEN + Style.BRIGHT +
                        "Отобразить следующие 5 "
                        "предложений (y/n)?\n"
                    ).strip().lower()

                    while user_input not in ["y", "n", "т", "н"]:
                        user_input = input(
                            Fore.LIGHTRED_EX +
                            "Неверный ввод. "
                            "Пожалуйста, "
                            "выберите один из "
                            "возможных вариантов "
                            "(y/n):\n"
                        ).strip().lower()

                    if user_input in ["n", "т"]:
                        break
        else:
            print(Fore.RED +
                  "Синтаксический анализ"
                  " еще не был выполнен."
                  )

    def run_analysis_pipeline(self):
        """Запуск анализа текста."""
        self.create_text_passport()
        self.display_text_passport()

        while True:
            print(Fore.LIGHTGREEN_EX + Style.BRIGHT +
                  "\nОтобразить анализ текста "
                  "в консоль (y/n)?"
                  )

            user_display_choice = input()
            if user_display_choice.lower() == 'y':
                show_analysis = True
                break
            elif user_display_choice.lower() == 'n':
                print(Fore.RED + Style.BRIGHT +
                      "\nПОЖАЛУЙСТА, ДОЖДИТЕСЬ "
                      "ОКОНЧАНИЯ АНАЛИЗА."
                      )
                show_analysis = False
                break
            else:
                print(Fore.RED + Style.BRIGHT +
                      "Неверный ввод. Пожалуйста,"
                      " выберите один из возможных "
                      "вариантов (y/n)."
                      )
                continue

        self.analyze_and_save(show_analysis)
        print_analysis_ready()

        self.save_text_passport(
            self.text, self.title,
            self.subject_area,
            self.keywords,
            self.publication_year,
            self.published_in,
            self.authors,
            self.author_gender,
            self.author_birth_year
        )

        morph_ann = self.get_morphological_annotation()
        print_annotation_ready('морфологическая')

        while True:
            display_morph_ann = input(
                Fore.LIGHTGREEN_EX + Style.BRIGHT +
                "\nОтобразить морфологическую "
                "разметку текста (y/n)?\n"
            )
            if display_morph_ann.lower() == 'y':
                display_morphological_annotation(morph_ann)
                break
            elif display_morph_ann.lower() == 'n':
                break
            else:
                print(Fore.LIGHTRED_EX +
                      "Неверный ввод. Пожалуйста, "
                      "выберите один из возможных "
                      "вариантов (y/n)."
                      )
                continue

        self.get_syntactic_annotation()
        print_annotation_ready('синтаксическая')

        while True:
            display_synt_ann = input(
                Fore.LIGHTGREEN_EX + Style.BRIGHT +
                "\nОтобразить синтаксическую "
                "разметку текста (y/n)?\n")

            if display_synt_ann.lower() == 'y':
                self.display_syntactic_annotation()
                break
            elif display_synt_ann.lower() == 'n':
                break
            else:
                print(Fore.LIGHTRED_EX +
                      "Неверный ввод. Пожалуйста, "
                      "выберите один из возможных "
                      "вариантов (y/n)."
                      )
                continue

    def display_corpus_info(self, choice=None):
        """
        Отображает общую информацию о корпусе текстов
        """
        if not choice:
            with SaveToDatabase(self.db) as db_corpus_info:
                db_corpus_info.display_corpus_info()
        else:
            with SaveToDatabase() as db_corpora_comparison:
                db_corpora_comparison.display_corpus_info(choice)


def start_analysis():
    print_greeting()

    while True:
        choice = print_main_menu()

        if choice == "1":
            db = choose_db()
            if not db:
                print(Fore.RED +
                      Style.BRIGHT +
                      RETURN_TO_MENU)
                continue

            text_to_analyse = text_input_for_analysis()
            print(Fore.LIGHTGREEN_EX + Style.BRIGHT +
                  "\nТекст будет сохранен в базу данных" +
                  Fore.BLUE + Style.BRIGHT + f" {db}.")
            corpus_text = CorpusText(text=text_to_analyse, db=db)
            corpus_text.run_analysis_pipeline()
            wait_for_enter_to_choose_opt()

        elif choice == "2":
            db = choose_db()
            if db == "auth_texts_corpus.db":
                analyze_texts_from_directory("auth_ready", db)
            elif db == "mt_texts_corpus.db":
                analyze_texts_from_directory("mt_ready", db)
            elif db == "ht_texts_corpus.db":
                analyze_texts_from_directory("ht_ready", db)

        elif choice == "3":
            db = choose_db()
            if not db:
                print(Fore.RED + Style.BRIGHT + RETURN_TO_MENU)
                continue
            corpus = CorpusText(db=db)
            corpus.show_texts()

        elif choice == "4":
            db = choose_db()
            if not db:
                print(Fore.RED + Style.BRIGHT + RETURN_TO_MENU)
                continue
            corpus = CorpusText(db=db)
            corpus.display_corpus_info()
        elif choice == "5":
            corpus = CorpusText()
            if (
                    check_db_exists(NON_TRANSLATED_DB_NAME)
                    and check_db_exists(MACHINE_TRANSLATED_DB_NAME)
                    and check_db_exists(HUMAN_TRANSLATED_DB_NAME)
            ):
                corpus.display_corpus_info(choice='all')

            elif (
                    check_db_exists(NON_TRANSLATED_DB_NAME)
                    and check_db_exists(MACHINE_TRANSLATED_DB_NAME)
            ):
                corpus.display_corpus_info(choice='auth_mt')
            elif (
                    check_db_exists(NON_TRANSLATED_DB_NAME)
                    and check_db_exists(HUMAN_TRANSLATED_DB_NAME)
            ):
                corpus.display_corpus_info(choice='auth_ht')
            elif (
                    check_db_exists(MACHINE_TRANSLATED_DB_NAME)
                    and check_db_exists(HUMAN_TRANSLATED_DB_NAME)
            ):
                corpus.display_corpus_info(choice='mt_ht')

            else:
                print(Fore.LIGHTRED_EX + Style.BRIGHT +
                      "Для сравнения необходимо наличие"
                      " хотя бы двух корпусов."
                      )

        elif choice == "6":
            preprocess_text.main()
        elif choice == "7":
            print(Fore.LIGHTRED_EX + Style.BRIGHT +
                  "\nВыход из программы."
                  )
            exit()
        else:
            print(INCORRECT_CHOICE)


def text_input_for_analysis():
    print("\n" + Fore.LIGHTWHITE_EX + "*" * 80)
    print(
        Fore.GREEN + Style.BRIGHT +
        "                            "
        "ВВОД ТЕКСТА ДЛЯ АНАЛИЗА")
    print(Fore.LIGHTWHITE_EX + "*" * 80)
    print(
        Fore.RED + Style.BRIGHT +
        "ВНИМАНИЕ! Текст должен быть заранее "
        "предобработан и готов для "
        "последующего анализа.")

    while True:
        mode = input(
            Fore.BLUE + Style.BRIGHT +
            "Введите 'f' для обработки файла "
            "или 't' для ввода текста вручную:\n"
        ).strip().lower()
        if mode.lower().strip() == 'f':
            while True:
                print(
                    Fore.GREEN + Style.BRIGHT +
                    "Выберите директорию с"
                    " текстовым файлом (.txt).")
                print(Fore.GREEN + Style.BRIGHT + "1."
                      + Style.NORMAL + Fore.BLACK +
                      "Директория с непереводными"
                      " текстами (auth_ready)")
                print(Fore.GREEN + Style.BRIGHT + "2."
                      + Style.NORMAL + Fore.BLACK +
                      "Директория с машинными"
                      " переводами (mt_ready)")
                print(Fore.GREEN + Style.BRIGHT + "3."
                      + Style.NORMAL + Fore.BLACK +
                      "Директория с ручными переводами "
                      "(ht_ready)")
                dir_choice = input(
                    Fore.GREEN + Style.BRIGHT
                    + "Введите номер директории:\n"
                ).strip()

                if dir_choice == '1':
                    directory = "auth_ready/"
                elif dir_choice == '2':
                    directory = "mt_ready/"
                elif dir_choice == '3':
                    directory = "ht_ready/"
                else:
                    print(INCORRECT_CHOICE)
                    continue

                file_name = input(Fore.GREEN + Style.BRIGHT +
                                  "Введите название файла в"
                                  " выбранной директории "
                                  "(только файлы .txt):\n"
                                  ).strip()
                file_path = directory + file_name + '.txt'
                if os.path.isfile(file_path):

                    return process_text_from_file(file_path)
                else:
                    print(
                        Fore.LIGHTRED_EX + Style.BRIGHT +
                        "Файл не найден. "
                        "Проверьте путь и "
                        "попробуйте снова.\n"
                    )
                continue
        elif mode.lower().strip() == 't':
            print(Fore.GREEN + Style.BRIGHT +
                  "\nВведите текст для анализа "
                  "(по окончанию ввода напечатайте 'r'"
                  " с красной строки и\nнажмите 'Enter'):"
                  )
            # print(
            #     Fore.RED + Style.BRIGHT +
            #     "Для возврата в главное меню "
            #     "напечатайте 'x' с красной строки"
            #     " и нажмите 'Enter'."
            # )

            text_lines = []
            while True:
                line = input()
                if line.lower().strip() == "r":
                    break
                # elif line.lower().strip() == "x":
                #     print(
                #         Fore.RED +
                #         "Возвращение в "
                #         "главное меню...\n"
                #     )
                #     break
                text_lines.append(line)

            # if line.lower() == "x":
            #     continue

            input_text = '\n'.join(text_lines).strip()
            if not input_text:
                print(Fore.LIGHTRED_EX + Style.BRIGHT +
                      "Текст не был введен! Попробуйте снова."
                      )
                continue
            return input_text


def analyze_texts_from_directory(base_dir, db):
    """
    Анализирует все текстовые файлы
    из указанной директории.

    :param base_dir: Основная директория
    (mt_ready, ht_ready, auth_ready).
    """
    # Автоматическое создание основной директории,
    # если она не существует
    os.makedirs(base_dir, exist_ok=True)

    while True:
        dir_name = input(
            Fore.GREEN + Style.BRIGHT +
            f"Введите название директории с "
            f"txt-файлами для анализа.\n"
            f"Директория с файлами должна находиться "
            f"внутри директории {base_dir}: \n"
        ).strip()
        target_dir = os.path.join(base_dir, dir_name)

        if not os.path.exists(target_dir):
            print(Fore.LIGHTRED_EX +
                  f"Ошибка: Директория "
                  f"{target_dir} "
                  f"не существует."
                  f" Проверьте путь.")
            continue
        # Запрос предметной области
        subject_area = input(
            Fore.GREEN + Style.BRIGHT +
            "Введите предметную область, общую для"
            " всех текстов в директории: \n" +
            Fore.RESET).strip()
        while not subject_area:
            subject_area = input(
                Fore.LIGHTRED_EX +
                "Предметная область не "
                "может быть пустой. "
                "Пожалуйста, введите её: "
            ).strip()

        text_files = [os.path.join(target_dir, f)
                      for f in os.listdir(target_dir)
                      if f.endswith('.txt')]

        if not text_files:
            print(Fore.LIGHTRED_EX +
                  f"Ошибка: В директории {target_dir}"
                  f" нет файлов с расширением .txt."
                  )
            continue

        print(Fore.GREEN + Style.BRIGHT +
              f"Найдено {len(text_files)}"
              f" текстов. Анализ запущен. "
              f"Время анализа зависит "
              f"от количества "
              f"\nи объема текстов."
              f" Пожалуйста, "
              f"будьте готовы подождать "
              f"несколько минут."
              )
        for idx, file_path in enumerate(text_files, start=1):
            try:
                text_name = f"{dir_name.capitalize()}_{idx}"
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

                # Вызываем анализ текста
                corpus = CorpusText(db=db, text=text)
                corpus.analyze_and_save(show_analysis=False)

                # Сохраняем паспорт текста напрямую
                corpus.save_text_passport(
                    text=text,
                    title=text_name,
                    subject_area=subject_area,
                    keywords="",
                    publication_year="",
                    published_in="",
                    authors="",
                    author_gender="",
                    author_birth_year=""
                )
                print(Fore.GREEN +
                      f"Текст \"{text_name}\" успешно "
                      f"проанализирован и сохранён."
                      )

            except Exception as e:
                print(Fore.LIGHTRED_EX +
                      f"Ошибка при обработке "
                      f"файла {file_path}: {e}")
        # Выходим из цикла после успешного
        # завершения анализа всех файлов
        print(Fore.GREEN + Style.BRIGHT +
              "Анализ всех текстов завершён.")
        wait_for_enter_to_choose_opt()
        break


start_analysis()
