"""DeepSeek chat provider using the OpenAI-compatible API."""

from libs.llm.openai_llm import OpenAILLM


class DeepSeekLLM(OpenAILLM):
    provider_name = "deepseek"
