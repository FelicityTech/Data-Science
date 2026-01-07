#!/usr/bin/env python3
"""
Test runner for Movie Recommendation System
Run all tests with: python run_tests.py
"""

import subprocess
import sys
import os

def run_tests():
    """Run the test suite"""
    print("🧪 Running Movie Recommendation System Tests")
    print("=" * 50)

    # Change to tests directory
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')

    # Run pytest
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            test_dir,
            '-v',
            '--tb=short',
            '--color=yes'
        ], capture_output=False, text=True)

        return result.returncode == 0

    except FileNotFoundError:
        print("❌ pytest not found. Install with: pip install pytest")
        return False

def run_quick_checks():
    """Run basic functionality checks"""
    print("\n🔍 Running Quick Checks")
    print("-" * 30)

    checks = [
        ("Import check", "python -c \"import pandas, numpy, sklearn; print('✅ Core ML libraries imported')\""),
        ("Data loading", "python -c \"import pandas as pd; df = pd.read_csv('tmdb_5000_movies.csv'); print(f'✅ Data loaded: {len(df)} movies')\""),
        ("Model loading", "python -c \"import joblib; models = joblib.load('models.pkl'); print(f'✅ Models loaded: {list(models.keys())}')\""),
    ]

    all_passed = True
    for name, command in checks:
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"{name}: {result.stdout.strip()}")
            else:
                print(f"❌ {name}: {result.stderr.strip()}")
                all_passed = False
        except subprocess.TimeoutExpired:
            print(f"⏰ {name}: Timeout")
            all_passed = False
        except Exception as e:
            print(f"❌ {name}: {e}")
            all_passed = False

    return all_passed

if __name__ == "__main__":
    print("🎬 Movie Recommendation System - Test Suite")
    print("=" * 50)

    # Run quick checks first
    quick_ok = run_quick_checks()

    if not quick_ok:
        print("\n❌ Quick checks failed. Please fix issues before running full tests.")
        sys.exit(1)

    # Run full test suite
    print("\n🧪 Running Full Test Suite")
    print("=" * 30)

    tests_ok = run_tests()

    if tests_ok:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)