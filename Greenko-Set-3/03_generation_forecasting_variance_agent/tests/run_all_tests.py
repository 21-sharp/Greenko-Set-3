"""
Unified Test Runner for Greenko Generation Forecasting & Variance Agent
========================================================================
Runs all automated unit, integration, and API tests and generates a formatted
testing report with pass/fail counts and execution times.
"""

import sys
import time
import warnings
from pathlib import Path

# Suppress sklearn / joblib UserWarnings to keep test output clean
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

def main():
    print("=" * 70)
    print("  GREENKO SET 3 - AUTOMATED TEST SUITE EXECUTION")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print()

    tests_dir = PROJECT_ROOT / "tests"
    start_time = time.time()
    
    # Run pytest programmatically
    args = [
        str(tests_dir),
        "-v",
        "--tb=short",
        "-s"
    ]
    
    exit_code = pytest.main(args)
    elapsed = time.time() - start_time
    
    print()
    print("=" * 70)
    if exit_code == 0:
        print(f"  ALL TESTS PASSED SUCCESSFULLY! (Time: {elapsed:.2f}s)")
    else:
        print(f"  TESTS FINISHED WITH EXIT CODE: {exit_code} (Time: {elapsed:.2f}s)")
    print("=" * 70)
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())

