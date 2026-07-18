"""Azure OpenAI chat provider."""

from urllib.parse import quote

from libs.llm.openai_llm import OpenAILLM


class AzureLLM(OpenAILLM):
    provider_name = "azure"
    api_version = "2024-02-15-preview"

    def _endpoint(self, base_url: str, model: str) -> str:
        deployment = quote(model, safe="")
        return (
            f"{base_url}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )

    def _headers(self, api_key: str) -> dict[str, str]:
        return {"api-key": api_key, "Content-Type": "application/json"}
