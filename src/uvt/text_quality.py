"""Контроль качества текста между стадиями дубляжа.

Стадии STT → перевод → TTS передают друг другу строки, и каждая умеет
испортить дорожку по-своему:

- Whisper на неразборчивой речи зацикливается и повторяет одну фразу
  десятки раз (``collapse_repeats``);
- чистые вокализации («oh», «um ah um», «mm-hmm») не несут перевода: их
  надо оставить оригиналом, а не синтезировать заново (``is_vocalization``);
- LLM с safety-фильтром вместо перевода отдаёт отказ, и он уходит в TTS как
  реплика (``is_refusal``);
- слабые модели пишут «хотел(а)» и «Я люблю это. / Это мне нравится.» —
  скобки и варианты через слэш нельзя произносить (``sanitize_translation``).

Модуль намеренно без зависимостей и без сети: это дешёвые проверки строк,
которые вызываются на каждую реплику.
"""
from __future__ import annotations

import re

# Интеръекции и неречевые вокализации: у них нет смысла для перевода, а
# исходное звучание универсально. Только те, что не могут быть словом.
_VOCALIZATIONS = {
    # английские
    "oh", "ooh", "ohh", "ohhh", "o",
    "ah", "aah", "ahh", "ahhh",
    "uh", "uhh", "umm", "um", "ummm",
    "mm", "mmm", "mmmm", "hm", "hmm", "hmmm",
    "mhm", "mmhmm", "mm-hmm", "mmm-hmm", "uh-huh", "uh-hum",
    "huh", "hah", "ha", "haha", "hahaha", "heh", "hehe",
    "ugh", "argh", "aargh", "urgh", "err", "erm", "eh",
    "aw", "aww", "ow", "oww", "oof", "phew", "whew",
    "shh", "sh", "tsk", "gah", "nngh", "ngh", "mmf",
    # русские/украинские — на случай обратного направления перевода
    "ох", "ах", "ух", "эх", "хм", "ммм", "мм", "эм", "мхм", "угу",
    "ой", "ай", "уф", "ммф",
}

_MAX_VOCALIZATION_TOKENS = 6

# Слова с беглой гласной: female-форма не получается простым добавлением
# содержимого скобок к основе.
_GENDER_IRREGULAR = {
    "должен": "должна",
    "долженъ": "должна",
}

_ALTERNATIVE_SPLIT = re.compile(r"\s+[/|]\s+")
_GENDER_VARIANT = re.compile(r"\b([А-Яа-яЁёA-Za-z]+)\((а|ла|ась|лась|на)\)")
_PUNCT = re.compile(r"[^\w\s\-']+", re.UNICODE)

# Мета-маркеры отказа модели: обычно приходят целым абзацем вместо перевода.
_REFUSAL_MARKERS = (
    "запрограммирован, чтобы избегать",
    "оскорбительный контент",
    "не могу выполнить",
    "не могу помочь",
    "не могу предоставить",
    "не могу перевести",
    "я не буду",
    "моя задача — предоставлять",
    "моя задача - предоставлять",
    "безопасные и этичные",
    "непристойный контент",
    "непристойного контента",
    "сексуально откровен",
    "нарушает правила",
    "нарушают правила",
    "как языковая модель",
    "как ии-модель",
    "i cannot fulfill",
    "i can't fulfill",
    "i cannot assist",
    "i can't assist",
    "i cannot help with",
    "i can't help with",
    "i'm sorry, but i can",
    "i am sorry, but i can",
    "as an ai language model",
    "sexually explicit",
    "violates my guidelines",
    "against my guidelines",
)

# Отказ бывает короче исходной реплики (например «Не могу помочь.»), поэтому
# длина сама по себе не критерий, но резкий рост объёма — сильный признак.
_REFUSAL_LENGTH_RATIO = 2.5

# Модель «рассуждает» вместо перевода: разбирает слово, предлагает варианты,
# просит контекст. Такой абзац нельзя озвучивать как реплику.
_COMMENTARY_MARKERS = (
    "требует контекста",
    "без контекста",
    "наиболее точный перевод",
    "точный перевод будет",
    "варианты перевода",
    "вариант перевода",
    "не имеет прямого эквивалента",
    "нет прямого эквивалента",
    "дословно переводится",
    "буквально означает",
    "это сленговое",
    "сленговое или устаревшее",
    "в зависимости от контекста",
    "возможно, потребуется",
    "примечание:",
    "пояснение:",
    "note:",
    "translation note",
    "depending on context",
    "there is no direct equivalent",
    "could be translated as",
)
_COMMENTARY_LENGTH_RATIO = 2.0


def _tokens(text: str) -> list[str]:
    return _PUNCT.sub(" ", text.lower()).split()


def is_vocalization(text: str) -> bool:
    """Реплика состоит только из неречевых вокализаций.

    Такие реплики не переводятся и не озвучиваются: в миксе на их месте
    остаётся оригинальный звук, что и звучит естественно.
    """
    tokens = _tokens(text)
    if not tokens or len(tokens) > _MAX_VOCALIZATION_TOKENS:
        return False
    return all(token in _VOCALIZATIONS for token in tokens)


def collapse_repeats(text: str, *, max_block: int = 12) -> str:
    """Сворачивает подряд идущие повторы — зацикливание Whisper.

    Длинный блок (≥3 слов), повторённый дважды, — всегда галлюцинация:
    оставляем одну копию. Короткий блок (1–2 слова) может быть экспрессией
    («нет, нет, нет»), поэтому его сворачиваем только от четырёх повторов и
    оставляем две копии.
    """
    words = text.split()
    if len(words) < 2:
        return text

    for block in range(min(max_block, len(words) // 2), 0, -1):
        index = 0
        out: list[str] = []
        while index < len(words):
            chunk = words[index : index + block]
            if len(chunk) < block:
                out.extend(words[index:])
                break
            repeats = 1
            probe = index + block
            while (
                probe + block <= len(words)
                and [w.lower().strip(".,!?…") for w in words[probe : probe + block]]
                == [w.lower().strip(".,!?…") for w in chunk]
            ):
                repeats += 1
                probe += block
            # Блок из одного и того же слова — это экспрессия, а не зависшая
            # фраза: его решает правило для коротких блоков ниже.
            varied = len({w.lower().strip(".,!?…") for w in chunk}) >= 2
            if block >= 3 and repeats >= 2 and varied:
                out.extend(chunk)
                index = probe
                continue
            if block <= 2 and repeats >= 4:
                out.extend(chunk * 2)
                index = probe
                continue
            out.extend(chunk)
            index += block
        words = out

    return " ".join(words)


def is_refusal(translated: str, original: str = "") -> bool:
    """Перевод оказался safety-отказом модели, а не переводом."""
    lowered = " ".join(translated.split()).lower()
    if not lowered:
        return False
    if not any(marker in lowered for marker in _REFUSAL_MARKERS):
        return False
    # Реальная реплика тоже может содержать «не могу помочь» — тогда её объём
    # сопоставим с оригиналом. Отказ почти всегда многословнее исходника.
    source = " ".join(original.split())
    if source and len(lowered) < _REFUSAL_LENGTH_RATIO * len(source):
        # Короткий ответ считаем отказом только по явно мета-формулировкам.
        meta = (
            "как языковая модель",
            "как ии-модель",
            "as an ai language model",
            "безопасные и этичные",
            "моя задача — предоставлять",
            "моя задача - предоставлять",
            "against my guidelines",
            "violates my guidelines",
        )
        return any(marker in lowered for marker in meta)
    return True


def is_commentary(translated: str, original: str = "") -> bool:
    """Модель разобрала реплику вслух вместо того, чтобы её перевести.

    Мета-ответ («это требует контекста», «наиболее точный перевод будет…»)
    почти всегда заметно длиннее исходной реплики; на коротких строках
    полагаемся только на явные маркеры.
    """
    lowered = " ".join(translated.split()).lower()
    if not lowered:
        return False
    if not any(marker in lowered for marker in _COMMENTARY_MARKERS):
        return False
    source = " ".join(original.split())
    if source and len(lowered) < _COMMENTARY_LENGTH_RATIO * len(source):
        return False
    return True


def strip_alternatives(text: str) -> str:
    """«Я люблю это. / Это мне нравится.» → первый вариант.

    Слэш между двумя самостоятельными фразами — это выбор модели, который
    нельзя произнести. Дроби и «и/или» внутри слова не задеваются: разделитель
    требует пробелов с двух сторон.
    """
    parts = _ALTERNATIVE_SPLIT.split(text)
    if len(parts) < 2:
        return text
    first = parts[0].strip()
    return first or text.strip()


def has_gender_variants(text: str) -> bool:
    """В тексте есть «хотел(а)» — модель не знала пола говорящего."""
    return bool(_GENDER_VARIANT.search(text))


def resolve_gender_variants(text: str, gender: str) -> str:
    """Раскрывает «хотел(а)» в одну форму по полу говорящего."""
    female = str(gender or "").lower().startswith(("f", "ж"))

    def replace(match: re.Match[str]) -> str:
        base, suffix = match.group(1), match.group(2)
        if not female:
            return base
        irregular = _GENDER_IRREGULAR.get(base.lower())
        if irregular:
            return irregular if base.islower() else irregular.capitalize()
        return f"{base}{suffix}"

    return _GENDER_VARIANT.sub(replace, text)


def sanitize_translation(
    translated: str | None, original: str = "", gender: str = "male"
) -> str | None:
    """Единый постпроцессор перевода перед озвучкой.

    Возвращает None, если строку нельзя произносить (отказ модели или пусто) —
    вызывающая сторона повторит перевод другой моделью или оставит оригинал.
    """
    if translated is None:
        return None
    text = " ".join(str(translated).split())
    if not text:
        return None
    if is_refusal(text, original) or is_commentary(text, original):
        return None
    text = strip_alternatives(text)
    text = resolve_gender_variants(text, gender)
    text = collapse_repeats(text)
    return text.strip() or None
