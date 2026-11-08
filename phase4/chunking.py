import re

def split_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_by_words(text, chunk_size=60, overlap=15):
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_by_paragraphs(text, max_words=80):
    chunks = []
    current = []
    current_words = 0

    for paragraph in split_paragraphs(text):
        n_words = len(paragraph.split())

        if n_words > max_words:
            if current:
                chunks.append("\n\n".join(current))
                current, current_words = [], 0
            chunks.extend(chunk_by_words(paragraph, max_words, overlap=15))
            continue

        if current and current_words + n_words > max_words:
            chunks.append("\n\n".join(current))
            current, current_words = [], 0

        current.append(paragraph)
        current_words += n_words

    if current:
        chunks.append("\n\n".join(current))
    return chunks

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def pack(pieces, count_tokens, max_tokens, separator):
    chunks = []
    current = []
    for piece in pieces:
        candidate = separator.join(current + [piece])
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(separator.join(current))
            current = [piece]
        else:
            current.append(piece)
    if current:
        chunks.append(separator.join(current))
    return chunks


def split_to_fit(text, count_tokens, max_tokens):
    if count_tokens(text) <= max_tokens:
        return [text]

    levels = [
        (split_paragraphs, "\n\n"),
        (split_sentences, " "),
        (str.split, " "),
    ]
    for splitter, separator in levels:
        pieces = splitter(text)
        if len(pieces) > 1:
            fitted = []
            for piece in pieces:
                fitted.extend(split_to_fit(piece, count_tokens, max_tokens))
            return pack(fitted, count_tokens, max_tokens, separator)

    return [text]