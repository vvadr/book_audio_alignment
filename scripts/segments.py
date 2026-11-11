"""Join the transcription API's 20-second segments into one clean text.

The API cuts the audio every 20 seconds, even in the middle of a word, so the
segment texts cannot simply be joined with spaces.
"""

import re

APOSTROPHES = "'‘’ʻʼ`"


def normalize_word(word):
    """Make words comparable: 'Baxtiyor,' -> 'baxtiyor', "Qo‘li" -> "qo'li"."""
    word = word.lower()
    word = re.sub(f"[{APOSTROPHES}]+", "'", word)   # every apostrophe style -> '
    word = re.sub(r"[^\w']", "", word)               # drop punctuation
    return word.strip("'")


def is_split_word(last, first, vocabulary):
    """True if `last` + `first` is one book word that the API cut in two ("kel" + "ibdi")."""
    # A cut word has a letter on both sides of the cut.
    # "kel," ends with punctuation and "– debdi" starts with a dash: nothing was cut.
    if not (last[-1].isalnum() and first[0].isalnum()):
        return False
    joined = normalize_word(last + first)
    both_are_words = normalize_word(last) in vocabulary and normalize_word(first) in vocabulary
    return joined in vocabulary and not both_are_words


def join_segments(segments, book_words=None):
    """Join segment texts, repairing words cut or repeated at segment boundaries.

    segments:   the "segments" list from an API response (dicts with a "text" key)
    book_words: optional list of book words. With it, the function can also
                glue splits that have no apostrophe, and keeps a repeated word
                when the book repeats it too ("Baxtiyor, Baxtiyor!").
    """
    book_words = [normalize_word(word) for word in (book_words or [])]
    vocabulary = set(book_words)
    doubled = {a for a, b in zip(book_words, book_words[1:]) if a == b}

    words = []
    for segment in segments:
        piece = segment["text"].split()
        if not piece:
            continue

        if words:
            last, first = words[-1], piece[0]

            # Problem 1: a word cut in two, "qo" + "'ydiribdi".
            # The word before must end with a letter; after "dedi:" the ' opens a quote.
            if first[0] in APOSTROPHES and last[-1].isalpha():
                words[-1] = last + first
                piece = piece[1:]

            # Problem 1b: a word cut in two without an apostrophe; the book decides.
            elif vocabulary and is_split_word(last, first, vocabulary):
                words[-1] = last + first
                piece = piece[1:]

            # Problem 2: the boundary word written twice, "baxtiyor" + "Baxtiyor".
            # Keep the second copy: it starts its segment, so its capital letter is right.
            elif (normalize_word(first)
                  and normalize_word(last) == normalize_word(first)
                  and normalize_word(first) not in doubled):
                words.pop()

        words.extend(piece)

    return " ".join(words)