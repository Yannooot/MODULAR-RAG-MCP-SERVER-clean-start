"""Rule-based chunk cleanup with optional LLM enhancement."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


DEFAULT_PROMPT = (
    "Clean the following text while preserving its meaning and Markdown structure. "
    "Return only the cleaned text:\n\n{text}"
)


class ChunkRefiner(BaseTransform):
    def __init__(
        self,
        settings: Settings,
        llm: BaseLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        super().__init__(settings)
        self.use_llm = settings.ingestion.chunk_refiner.use_llm
        self.prompt_template = self._load_prompt(prompt_path)
        self._llm_setup_error: str | None = None
        self._last_llm_error: str | None = None
        self.llm = llm
        if self.use_llm and self.llm is None:
            try:
                self.llm = LLMFactory.create(settings)
            except Exception as exc:
                self._llm_setup_error = str(exc)

    def transform(
        self, chunks: Sequence[Chunk], trace: TraceContext | None = None
    ) -> list[Chunk]:
        refined: list[Chunk] = []
        for chunk in chunks:
            try:
                rule_text = self._rule_based_refine(chunk.text)
                metadata = dict(chunk.metadata)
                text = rule_text
                if self.use_llm:
                    llm_text = self._llm_refine(rule_text, trace)
                    if llm_text is not None:
                        text = llm_text
                        metadata["refined_by"] = "llm"
                    else:
                        metadata["refined_by"] = "rule"
                        metadata["refinement_fallback_reason"] = (
                            self._last_llm_error or "LLM unavailable"
                        )
                else:
                    metadata["refined_by"] = "rule"
                refined.append(replace(chunk, text=text, metadata=metadata))
            except Exception as exc:
                metadata = dict(chunk.metadata)
                metadata["refined_by"] = "original"
                metadata["refinement_fallback_reason"] = str(exc)
                refined.append(replace(chunk, metadata=metadata))

        if trace is not None:
            trace.record_stage(
                "chunk_refiner",
                {"chunk_count": len(chunks), "llm_enabled": self.use_llm},
            )
        return refined

    def _rule_based_refine(self, text: str) -> str:
        code_blocks: list[str] = []

        def protect_code(match: re.Match[str]) -> str:
            token = f"@@CHUNK_REFINER_CODE_{len(code_blocks)}@@"
            code_blocks.append(match.group(0))
            return token

        cleaned = re.sub(r"```[\s\S]*?```", protect_code, text)
        cleaned = re.sub(r"<!--[\s\S]*?-->", "", cleaned)

        kept_lines: list[str] = []
        for line in cleaned.splitlines():
            stripped = line.strip()
            if self._is_noise_line(stripped):
                continue
            if not stripped:
                if kept_lines and kept_lines[-1] != "":
                    kept_lines.append("")
                continue
            kept_lines.append(self._normalize_line(line))

        cleaned = "\n".join(kept_lines).strip()
        for index, code_block in enumerate(code_blocks):
            cleaned = cleaned.replace(
                f"@@CHUNK_REFINER_CODE_{index}@@", code_block
            )
        return cleaned

    def _llm_refine(self, text: str, trace: TraceContext | None = None) -> str | None:
        del trace
        self._last_llm_error = self._llm_setup_error
        if self.llm is None:
            return None
        try:
            response = self.llm.chat(
                [{"role": "user", "content": self.prompt_template.format(text=text)}]
            )
        except Exception as exc:
            self._last_llm_error = str(exc)
            return None
        if not response.strip():
            self._last_llm_error = "empty LLM response"
            return None
        self._last_llm_error = None
        return response.strip()

    def _load_prompt(self, prompt_path: str | Path | None = None) -> str:
        path = (
            Path(prompt_path)
            if prompt_path is not None
            else Path(__file__).parents[3] / "config" / "prompts" / "chunk_refinement.txt"
        )
        try:
            prompt = path.read_text(encoding="utf-8").strip()
        except OSError:
            return DEFAULT_PROMPT
        return prompt if prompt and "{text}" in prompt else DEFAULT_PROMPT

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        patterns = (
            r"(?:页眉|页脚)\s*[:：].*",
            r"(?:header|footer)\s*[:：].*",
            r"第\s*\d+\s*页(?:\s*[，,]\s*共\s*\d+\s*页)?",
            r"page\s+\d+(?:\s+of\s+\d+)?",
            r"[-=_*]{4,}",
        )
        return any(re.fullmatch(pattern, line, re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _normalize_line(line: str) -> str:
        list_item = re.match(r"^(\s*)([-+*]|\d+\.)(\s+)(.*)$", line)
        if list_item:
            indent, marker, _, content = list_item.groups()
            return f"{indent}{marker} {re.sub(r'[ \t]+', ' ', content).rstrip()}"
        return re.sub(r"[ \t]+", " ", line).strip()
