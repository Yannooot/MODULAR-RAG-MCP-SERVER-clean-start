from __future__ import annotations

import math
import os
import pickle
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from core.types import ChunkRecord


class BM25Indexer:
    """Build and persist the corpus statistics required for BM25 retrieval."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        index_path: str | Path = "data/db/bm25/index.pkl",
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not isinstance(k1, (int, float)) or isinstance(k1, bool) or k1 <= 0:
            raise ValueError("k1 must be positive")
        if (
            not isinstance(b, (int, float))
            or isinstance(b, bool)
            or not 0 <= b <= 1
        ):
            raise ValueError("b must be between 0 and 1")
        self.index_path = Path(index_path)
        self.k1 = float(k1)
        self.b = float(b)
        self.index = self._empty_index()
        if self.index_path.is_file():
            self.load()

    def build(self, records: Iterable[ChunkRecord]) -> None:
        self.index["documents"] = self._prepare_documents(records)
        self._rebuild_derived_index()
        self.save()

    def update(self, records: Iterable[ChunkRecord]) -> None:
        self.index["documents"].update(self._prepare_documents(records))
        self._rebuild_derived_index()
        self.save()

    def remove_document(self, source_path: str) -> int:
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("source_path must be a non-empty string")
        documents = self.index["documents"]
        identifiers = [
            identifier
            for identifier, document in documents.items()
            if document["metadata"].get("source_path") == source_path
        ]
        for identifier in identifiers:
            del documents[identifier]
        if identifiers:
            self._rebuild_derived_index()
            self.save()
        return len(identifiers)

    def query(
        self,
        query_terms: Sequence[str] | Mapping[str, float],
        *,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        terms = self._prepare_query_terms(query_terms)
        document_count = self.index["document_count"]
        if not terms or document_count == 0:
            return []

        average_length = self.index["total_document_length"] / document_count
        scores: defaultdict[str, float] = defaultdict(float)
        for term, query_weight in terms.items():
            entry = self.index["inverted_index"].get(term)
            if entry is None:
                continue
            for posting in entry["postings"]:
                tf = posting["tf"]
                length_ratio = (
                    posting["doc_length"] / average_length
                    if average_length
                    else 0.0
                )
                denominator = tf + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[posting["chunk_id"]] += (
                    query_weight
                    * entry["idf"]
                    * tf
                    * (self.k1 + 1)
                    / denominator
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        documents = self.index["documents"]
        return [
            {
                "id": identifier,
                "text": documents[identifier]["text"],
                "metadata": deepcopy(documents[identifier]["metadata"]),
                "score": score,
            }
            for identifier, score in ranked
        ]

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.index_path.with_suffix(".tmp")
        try:
            with temporary_path.open("wb") as file:
                pickle.dump(self.index, file, protocol=pickle.HIGHEST_PROTOCOL)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self.index_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def load(self) -> BM25Indexer:
        with self.index_path.open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, dict) or loaded.get("format_version") != 1:
            raise ValueError("unsupported BM25 index format")
        required = {
            "document_count",
            "total_document_length",
            "documents",
            "inverted_index",
        }
        if not required.issubset(loaded):
            raise ValueError("invalid BM25 index structure")
        self.index = loaded
        return self

    @classmethod
    def calculate_idf(cls, document_count: int, document_frequency: int) -> float:
        return math.log(
            (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

    def _prepare_documents(
        self, records: Iterable[ChunkRecord]
    ) -> dict[str, dict[str, Any]]:
        documents: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, ChunkRecord):
                raise ValueError("records must contain ChunkRecord values")
            if record.id in documents:
                raise ValueError(f"duplicate chunk id: {record.id}")
            sparse = record.sparse_vector
            if not isinstance(sparse, dict):
                raise ValueError("sparse_vector must be present")
            frequencies: dict[str, float] = {}
            for term, frequency in sparse.items():
                if (
                    not isinstance(term, str)
                    or not term.strip()
                    or not isinstance(frequency, (int, float))
                    or isinstance(frequency, bool)
                    or not math.isfinite(frequency)
                    or frequency <= 0
                ):
                    raise ValueError("sparse_vector contains invalid statistics")
                frequencies[term] = float(frequency)
            documents[record.id] = {
                "text": record.text,
                "metadata": deepcopy(record.metadata),
                "term_frequencies": frequencies,
                "document_length": float(sum(frequencies.values())),
            }
        return documents

    def _rebuild_derived_index(self) -> None:
        documents = self.index["documents"]
        document_count = len(documents)
        postings: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for identifier in sorted(documents):
            document = documents[identifier]
            for term, frequency in sorted(document["term_frequencies"].items()):
                postings[term].append(
                    {
                        "chunk_id": identifier,
                        "tf": frequency,
                        "doc_length": document["document_length"],
                    }
                )
        self.index["document_count"] = document_count
        self.index["total_document_length"] = sum(
            document["document_length"] for document in documents.values()
        )
        self.index["inverted_index"] = {
            term: {
                "idf": self.calculate_idf(document_count, len(term_postings)),
                "postings": term_postings,
            }
            for term, term_postings in sorted(postings.items())
        }

    @staticmethod
    def _prepare_query_terms(
        query_terms: Sequence[str] | Mapping[str, float],
    ) -> dict[str, float]:
        if isinstance(query_terms, Mapping):
            terms = dict(query_terms)
        elif isinstance(query_terms, Sequence) and not isinstance(
            query_terms, (str, bytes)
        ):
            terms = {term: 1.0 for term in query_terms}
        else:
            raise ValueError("query_terms must be a sequence or mapping")
        for term, weight in terms.items():
            if (
                not isinstance(term, str)
                or not term.strip()
                or not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or not math.isfinite(weight)
                or weight <= 0
            ):
                raise ValueError("query_terms contains invalid values")
        return {term: float(weight) for term, weight in terms.items()}

    @classmethod
    def _empty_index(cls) -> dict[str, Any]:
        return {
            "format_version": cls.FORMAT_VERSION,
            "document_count": 0,
            "total_document_length": 0.0,
            "documents": {},
            "inverted_index": {},
        }
