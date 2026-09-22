"""프로젝트 패키지가 정상적으로 import 되는지 확인하는 기본 테스트."""

import importlib

import pytest

PACKAGES = [
    "backend.app.main",
    "experiments",
    "recommender",
    "recommender.preprocessing",
    "experiments.labeling",
    "recommender.embedding",
    "recommender.retrieval",
    "recommender.ranking",
    "recommender.recommendation",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_importable(package: str) -> None:
    assert importlib.import_module(package) is not None
