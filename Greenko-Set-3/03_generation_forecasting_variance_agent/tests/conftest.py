"""
conftest.py – shared pytest configuration for Greenko Set 3 test suite.

Suppresses noisy sklearn / joblib UserWarnings that do not affect correctness
(e.g. deprecated `delayed` usage in older joblib / sklearn parallel workers)
so the test output stays readable for evaluators.
"""

import warnings

import pytest


def pytest_configure(config):
    """Register custom warning filters before collection begins."""
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
