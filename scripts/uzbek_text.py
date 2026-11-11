"""Uzbek apostrophes, written the way the NeMo model writes them.

Uzbek Latin has two apostrophe-like letters:
    ʻ  (U+02BB) in oʻ and gʻ       oʻzbek, gʻamgin, togʻ
    ʼ  (U+02BC) the tutuq sign     inʼom, vaʼda
People and the transcription API type the plain ' for both.
"""

ALL_APOSTROPHES = "'‘’ʻʼ`"
OKINA = "\u02bb"
TUTUQ = "\u02bc"


def fix_apostrophes(text, tutuq=TUTUQ):
    """Give every apostrophe inside a word the right Uzbek letter.

    g'                      -> gʻ   (also at the end of a word: togʻ)
    o' before a letter      -> oʻ
    ' between two letters   -> the tutuq sign: inʼom
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