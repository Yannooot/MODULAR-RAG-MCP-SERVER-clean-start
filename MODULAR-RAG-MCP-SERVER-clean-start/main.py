"""Project entry point."""

from core.settings import load_settings
from observability.logger import get_logger


logger = get_logger(__name__)


def main() -> None:
    settings = load_settings()
    logger.info(
        "Configuration loaded: LLM=%s/%s, Embedding=%s/%s",
        settings.llm.provider,
        settings.llm.model,
        settings.embedding.provider,
        settings.embedding.model,
    )


if __name__ == "__main__":
    main()
