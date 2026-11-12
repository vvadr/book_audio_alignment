"""Uzbek apostrophes and punctuation, written the way the NeMo model writes them.

Uzbek Latin has two apostrophe-like letters:
    ʻ  (U+02BB) in oʻ and gʻ       oʻzbek, gʻamgin, togʻ
    ’  (U+2019) the tutuq sign     in’om, va’da
People and the transcription API type the plain ' for both.

The model can only read the characters in its vocabulary (MODEL_CHARS).
Any other character becomes an "unknown" token, which the aligner cannot
place well: that word gets a very low score and its chunk is rejected.
"""

import re

ALL_APOSTROPHES = "'‘’ʻʼ`"
OKINA = "\u02bb"
TUTUQ = "\u2019"

MODEL_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ",.?-" + OKINA + TUTUQ
)

PUNCTUATION_MODEL_KNOWS = {
    "!": ".",
    "…": ".",
    ";": ",",
    ":": ",",
}


def fix_apostrophes(text, tutuq=TUTUQ):
    """Give every apostrophe inside a word the right Uzbek letter.

    g'                      -> gʻ   (also at the end of a word: togʻ)
    o' before a letter      -> oʻ
    ' between two letters   -> the tutuq sign: in’om
    anything else           -> left alone, it is a quote mark
    """
    result = []

    for i, char in enumerate(text):
        if char not in ALL_APOSTROPHES:
            result.append(char)
            continue

        before = text[i - 1] if i > 0 else ""
        after = text[i + 1] if i + 1 < len(text) else ""

        if result and result[-1] in (OKINA, tutuq):
            continue
        elif before in {"g", "G"}:
            result.append(OKINA)
        elif before in {"o", "O"} and after.isalpha():
            result.append(OKINA)
        elif before.isalpha() and after.isalpha():
            result.append(tutuq)
        else:
            result.append(char)

    return "".join(result)


def to_model_text(text):
    """Keep only characters the model knows, so every word can be aligned.

    "Podshohim,     -> Podshohim,     quote marks are not spoken
    undan:          -> undan,         : ; become a comma
    etiladi!"       -> etiladi.       ! … become a full stop
    –               -> (removed)      a dash on its own is not a word

    Run fix_apostrophes first, so the apostrophes are already letters.
    Only punctuation and symbols are removed. Digits and unusual letters
    (1871, Cyrillic) are spoken, so they stay, and align() warns about them.
    """
    words = []

    for word in text.split():
        for old, new in PUNCTUATION_MODEL_KNOWS.items():
            word = word.replace(old, new)

        word = "".join(char for char in word if char in MODEL_CHARS or char.isalnum())
        word = re.sub(r"([,.?])[,.?]+", r"\1", word)
        word = word.strip("-")

        if any(char.isalnum() for char in word):
            words.append(word)

    return " ".join(words)