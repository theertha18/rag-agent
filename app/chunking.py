"""
Simple, dependency-free text chunker, easy to explain in an interview.

1. Split the text into paragraphs (blank-line separated).
2. Any paragraph longer than target_size is split further, at sentence
   boundaries where possible and at word boundaries as a last resort. This
   matters: the embedding model only reads the first part of each chunk
   (roughly 256 tokens), so an oversized chunk has a "blind" tail whose
   content cannot be found by search.
3. Small pieces are merged back together up to target_size.
4. Each new chunk starts with a short overlap from the end of the previous
   one (cut on a word boundary, never mid-word) so context isn't lost at
   chunk edges.

Every chunk is at most about target_size + overlap characters long.
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


# A sentence ends at . ! ? followed by whitespace; single newlines (used for
# list-like text) are also treated as boundaries.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_words(text: str, max_size: int) -> list[str]:
    """Last resort for a single huge sentence: split at word boundaries."""
    parts: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > max_size:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        parts.append(current)
    return parts


def _split_long(paragraph: str, max_size: int) -> list[str]:
    """Break a paragraph into pieces of at most max_size characters."""
    units: list[str] = []
    for sentence in _BOUNDARY.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        units.extend(_split_words(sentence, max_size) if len(sentence) > max_size else [sentence])

    pieces: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > max_size:
            pieces.append(current)
            current = unit
        else:
            current = f"{current} {unit}" if current else unit
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(buffer: str, overlap: int) -> str:
    """Last `overlap` characters of buffer, starting on a word boundary."""
    if not buffer or overlap <= 0:
        return ""
    if len(buffer) <= overlap:
        return buffer.strip()
    tail = buffer[-overlap:]
    if not buffer[-overlap - 1].isspace():
        # the cut landed inside a word: drop the partial first word
        _, _, tail = tail.partition(" ")
    return tail.strip()


def chunk_text(
    text: str,
    source: str,
    target_size: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) > target_size:
            units.extend(_split_long(paragraph, target_size))
        else:
            units.append(paragraph)

    chunks: list[Chunk] = []
    buffer = ""

    def flush(buf: str):
        if buf.strip():
            chunks.append(Chunk(text=buf.strip(), source=source, chunk_index=len(chunks)))

    for unit in units:
        if len(buffer) + len(unit) + 2 <= target_size:
            buffer = f"{buffer}\n\n{unit}" if buffer else unit
        else:
            flush(buffer)
            tail = _overlap_tail(buffer, overlap)
            buffer = f"{tail}\n\n{unit}" if tail else unit

    flush(buffer)
    return chunks
