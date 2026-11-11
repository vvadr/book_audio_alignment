"""Stage 10: compare every spoken word with the book.

The transcript says what the narrator said, and the book (after OCR) says what
was printed. They make different kinds of mistakes, so a word that both agree
on is very likely right.
"""

from difflib import SequenceMatcher

from segments import normalize_word


def comparable(text):
    """Split text into words that can be compared with the book.

    The book was saved without hyphens, punctuation or a reliable apostrophe,
    so "Mol-u," becomes ["mol", "u"], "togʻga" becomes ["togga"] and "–"
    becomes [] (nothing to compare).
    """
    parts = []
    for part in text.replace("-", " ").split():
        part = normalize_word(part).replace("'", "")
        if part:
            parts.append(part)
    return parts


def check_against_book(words, book_text):
    """Add "confirmed": True to every word the book has in the same place, False otherwise.

    words:     the aligned words (dicts with a "word" key), in spoken order
    book_text: the whole book as one string
    """
    book = comparable(book_text)

    spoken = []
    owner = []
    for index, word in enumerate(words):
        for part in comparable(word["word"]):
            spoken.append(part)
            owner.append(index)

    agrees = [False] * len(spoken)
    matcher = SequenceMatcher(None, spoken, book, autojunk=False)
    for tag, s_start, s_end, b_start, b_end in matcher.get_opcodes():
        if "".join(spoken[s_start:s_end]) == "".join(book[b_start:b_end]):
            for i in range(s_start, s_end):
                agrees[i] = True

    for word in words:
        word["confirmed"] = True

    for agree, index in zip(agrees, owner):
        if not agree:
            words[index]["confirmed"] = False

    return words