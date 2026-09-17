"""프로젝트 패키지가 정상적으로 import 되는지 확인하는 기본 테스트."""

import importlib

import pytest

PACKAGES = [
    "src",
    "src.preprocessing",
    "src.embedding",
    "src.retrieval",
    "src.ranking",
    "src.recommendation",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_importable(package: str) -> None:
    assert importlib.import_module(package) is not None
