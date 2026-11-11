"""Uzbek apostrophes, written the way the NeMo model writes them.

Uzbek Latin has two apostrophe-like letters:
    ʻ  (U+02BB) in oʻ and gʻ       oʻzbek, gʻamgin, togʻ
    ʼ  (U+02BC) the tutuq sign     inʼom, vaʼda
People and the transcription API type the plain ' for both.
"""

ALL_APOSTROPHES = "'‘’ʻʼ`"
OKINA = "\u02bb"   # ʻ  after o and g
TUTUQ = "\u02bc"   # ʼ  the tutuq sign


def fix_apostrophes(text, tutuq=TUTUQ):
    """Give every apostrophe inside a word the right Uzbek letter.

    g'                      -> gʻ   (also at the end of a word: togʻ)
    o' before a letter      -> oʻ
    ' between two letters   -> the tutuq sign: inʼom
    anything else           -> left alone, it is a quote mark
    """
    result = []

    for i, char in enumerate(text):
        # Most characters are not apostrophes: copy them unchanged
        if char not in ALL_APOSTROPHES:
            result.append(char)
            continue

        # The characters on each side of this apostrophe
        before = text[i - 1] if i > 0 else ""
        after = text[i + 1] if i + 1 < len(text) else ""

        if result and result[-1] in (OKINA, tutuq):
            continue                          # bog''labdi: keep only one
        elif before in {"g", "G"}:
            result.append(OKINA)              # gʻamgin, togʻ
        elif before in {"o", "O"} and after.isalpha():
            result.append(OKINA)              # oʻzbek
        elif before.isalpha() and after.isalpha():
            result.append(tutuq)              # inʼom
        else:
            result.append(char)               # a quote mark

    return "".join(result)