"""Контроль качества текста между стадиями: вокализации, повторы, отказы LLM.

Строки взяты из реальных логов дубляжа — именно они попадали в дорожку.
"""
from uvt.text_quality import (
    collapse_repeats,
    has_gender_variants,
    is_commentary,
    is_refusal,
    is_vocalization,
    resolve_gender_variants,
    sanitize_translation,
    strip_alternatives,
)


class TestVocalizations:
    def test_pure_interjections(self):
        assert is_vocalization("Oh Oh")
        assert is_vocalization("um ah um")
        assert is_vocalization("Mm-hmm.")
        assert is_vocalization("Ох.")
        assert is_vocalization("uh-huh")

    def test_meaningful_lines_are_kept(self):
        assert not is_vocalization("Oh, Jesus.")
        assert not is_vocalization("Oh my god")
        assert not is_vocalization("Yes.")
        assert not is_vocalization("Look at me.")
        assert not is_vocalization("")

    def test_long_line_is_not_vocalization(self):
        assert not is_vocalization("oh oh oh oh oh oh oh")


class TestCollapseRepeats:
    def test_whisper_loop_is_collapsed(self):
        phrase = "Я собираюсь посмотреть на свое лицо."
        assert collapse_repeats(" ".join([phrase] * 5)) == phrase

    def test_english_loop(self):
        text = " ".join(["I'm going to take a look at my face."] * 6)
        assert collapse_repeats(text) == "I'm going to take a look at my face."

    def test_expressive_short_repeat_survives(self):
        assert collapse_repeats("нет, нет, нет") == "нет, нет, нет"

    def test_long_short_word_loop_is_trimmed(self):
        assert collapse_repeats("снова снова снова снова снова снова") == "снова снова"

    def test_normal_text_untouched(self):
        text = "Да, мне очень нравится этот цвет."
        assert collapse_repeats(text) == text


class TestRefusal:
    def test_safety_refusal_from_log(self):
        translated = (
            "Я не могу выполнить эту просьбу. Этот запрос содержит непристойный и "
            "оскорбительный контент, и я запрограммирован, чтобы избегать создания "
            "подобного контента. Моя цель — предоставлять полезные и безопасные ответы."
        )
        assert is_refusal(translated, "Takže tebe vůbi z práce.")

    def test_ethical_refusal_from_log(self):
        translated = (
            "Прошу прощения, я не могу выполнить эту просьбу. Моя задача — "
            "предоставлять безопасные и этичные переводы, и этот запрос содержит "
            "непристойный контент."
        )
        assert is_refusal(translated, "Oh, my pussy.")

    def test_real_line_with_similar_words_is_not_refusal(self):
        assert not is_refusal("Я не могу помочь тебе с этим сейчас.", "I can't help you now.")

    def test_plain_translation(self):
        assert not is_refusal("Да, это хорошо.", "Yeah, that's good.")


class TestCommentary:
    def test_model_explains_instead_of_translating(self):
        translated = (
            'Это требует контекста. "To jinky" – это сленговое или устаревшее слово. '
            "Без контекста, наиболее точный перевод будет: Не имеет прямого "
            "эквивалента в современном русском языке."
        )
        assert is_commentary(translated, "To jinky.")

    def test_plain_translation_is_not_commentary(self):
        assert not is_commentary("Это твой, братан.", "Jes to tvůj frajku.")


class TestGenderVariants:
    def test_detects_bracket_variant(self):
        assert has_gender_variants("Я больше не уверен(а).")
        assert not has_gender_variants("Я больше не уверена.")

    def test_female_form(self):
        assert resolve_gender_variants("Я больше не уверен(а).", "female") == (
            "Я больше не уверена."
        )
        assert resolve_gender_variants("Я готов(а) к чему угодно.", "female") == (
            "Я готова к чему угодно."
        )

    def test_male_form(self):
        assert resolve_gender_variants("Я больше не уверен(а).", "male") == (
            "Я больше не уверен."
        )

    def test_irregular_stem(self):
        assert resolve_gender_variants("Я должен(а) сохранить это.", "female") == (
            "Я должна сохранить это."
        )


class TestAlternatives:
    def test_slash_variants_reduced_to_first(self):
        assert strip_alternatives("Я люблю это. / Это мне нравится.") == "Я люблю это."

    def test_fraction_is_not_a_variant(self):
        assert strip_alternatives("и/или так") == "и/или так"


class TestSanitize:
    def test_refusal_rejected(self):
        assert sanitize_translation(
            "Я не могу выполнить эту просьбу. Моя задача — предоставлять "
            "безопасные и этичные переводы.",
            "Oh, my pussy.",
        ) is None

    def test_commentary_rejected(self):
        assert sanitize_translation(
            'Это требует контекста. "To jinky" – это сленговое слово, '
            "наиболее точный перевод будет неочевиден.",
            "To jinky.",
        ) is None

    def test_full_cleanup_chain(self):
        out = sanitize_translation(
            "Я готов(а) к этому. / Я согласен на это.",
            "I'm bound for anything.",
            gender="female",
        )
        assert out == "Я готова к этому."

    def test_none_and_empty(self):
        assert sanitize_translation(None, "x") is None
        assert sanitize_translation("   ", "x") is None

    def test_good_translation_passes_through(self):
        assert sanitize_translation("Да, это хорошо.", "Yeah, that's good.") == (
            "Да, это хорошо."
        )
