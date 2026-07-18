import importlib

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "package_name",
    ["mcp_server", "core", "ingestion", "libs", "observability"],
)
def test_top_level_package_is_importable(package_name: str) -> None:
    assert importlib.import_module(package_name) is not None
