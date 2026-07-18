"""Azure OpenAI embedding provider."""

from typing import Any
from urllib.parse import quote

from libs.embedding.openai_embedding import OpenAIEmbedding


class AzureEmbedding(OpenAIEmbedding):
    provider_name = "azure"
    api_version = "2024-02-15-preview"

    def _endpoint(self, base_url: str, model: str) -> str:
        deployment = quote(model, safe="")
        return (
            f"{base_url}/openai/deployments/{deployment}/embeddings"
            f"?api-version={self.api_version}"
        )

    def _headers(self, api_key: str) -> dict[str, str]:
        return {"api-key": api_key, "Content-Type": "application/json"}

    def _payload(self, model: str, texts: list[str]) -> dict[str, Any]:
        return {"input": texts}
