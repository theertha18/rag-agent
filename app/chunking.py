"""
Dead-simple text chunker. No dependencies, easy to reason about and explain
in an interview: split on paragraphs first, then merge/split to hit a target
chunk size in characters, with a bit of overlap so context isn't cut at
an awkward boundary.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def chunk_text(
    text: str,
    source: str,
    target_size: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[Chunk] = []
    buffer = ""

    def flush(buf: str):
        if buf.strip():
            chunks.append(Chunk(text=buf.strip(), source=source, chunk_index=len(chunks)))

    for para in paragraphs:
        if len(buffer) + len(para) <= target_size:
            buffer = f"{buffer}\n\n{para}" if buffer else para
        else:
            flush(buffer)
            # start next buffer with a small overlap from the end of the last chunk
            tail = buffer[-overlap:] if buffer else ""
            buffer = f"{tail}\n\n{para}" if tail else para

    flush(buffer)
    return chunks
