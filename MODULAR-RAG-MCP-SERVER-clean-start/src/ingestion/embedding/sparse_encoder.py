"""Build stable local term-frequency representations for BM25 indexing."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from copy import deepcopy

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord


Tokenizer = Callable[[str], Sequence[str]]
TOKEN_SEGMENT = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
CHINESE_SEGMENT = re.compile(r"^[\u4e00-\u9fff]+$")


def default_tokenize(text: str) -> list[str]:
    """Tokenize Latin words and overlapping Chinese character bigrams."""
    tokens: list[str] = []
    for segment in TOKEN_SEGMENT.findall(text):
        if CHINESE_SEGMENT.fullmatch(segment):
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(
                    segment[index : index + 2]
                    for index in range(len(segment) - 1)
                )
        else:
            tokens.append(segment.casefold())
    return tokens


class SparseEncoder:
    def __init__(
        self, settings: Settings, tokenizer: Tokenizer | None = None
    ) -> None:
        self.settings = settings
        self.tokenizer = tokenizer or default_tokenize

    def encode(
        self, chunks: Sequence[Chunk], trace: TraceContext | None = None
    ) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        token_count = 0
        vocabulary: set[str] = set()
        for chunk in chunks:
            tokens = self._normalized_tokens(chunk.text)
            frequencies = Counter(tokens)
            sparse_vector = {
                term: float(frequency) for term, frequency in frequencies.items()
            }
            records.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=deepcopy(chunk.metadata),
                    sparse_vector=sparse_vector,
                )
            )
            token_count += len(tokens)
            vocabulary.update(frequencies)

        if trace is not None:
            trace.record_stage(
                "sparse_encoder",
                {
                    "chunk_count": len(chunks),
                    "token_count": token_count,
                    "vocabulary_size": len(vocabulary),
                },
            )
        return records

    def _normalized_tokens(self, text: str) -> list[str]:
        tokens = self.tokenizer(text)
        if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
            raise ValueError("tokenizer must return a sequence of strings")
        normalized: list[str] = []
        for token in tokens:
            if not isinstance(token, str):
                raise ValueError("tokenizer must return only strings")
            token = token.strip().casefold()
            if token:
                normalized.append(token)
        return normalized
