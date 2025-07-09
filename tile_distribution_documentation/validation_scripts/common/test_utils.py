"""
Common testing utilities for validation scripts.

These functions help us write consistent, helpful validation scripts
that make it easy to understand what's happening.
"""

import sys
import traceback
from typing import Any, Callable, Optional


def print_section(title: str, width: int = 60):
    """
    Print a nice section header to organize script output.
    
    You'll see this used throughout our validation scripts to make
    the output easy to follow.
    """
    print(f"\n{'='*width}")
    print(f" {title}")
    print(f"{'='*width}")


def print_step(step_number: int, description: str):
    """
    Print a numbered step to show progress through an example.
    
    This helps you follow along with what we're doing at each stage.
    """
    print(f"\n{step_number}. {description}")
    print("-" * (len(description) + 4))


def show_result(name: str, value: Any, explanation: str = ""):
    """
    Display a result with clear formatting.
    
    We use this to show you the output of operations in a consistent way.
    """
    print(f"\n{name}:")
    print(f"  Value: {value}")
    print(f"  Type: {type(value).__name__}")
    if explanation:
        print(f"  Note: {explanation}")


def validate_example(name: str, test_func: Callable, expected_result: Any = None):
    """
    Run a test function and show whether it worked.
    
    This is our safety net - if an example doesn't work here,
    it won't work in the documentation either.
    """
    print(f"\n🧪 Testing: {name}")
    try:
        result = test_func()
        if expected_result is not None:
            if result == expected_result:
                print(f"  ✅ PASS - Got expected result: {result}")
            else:
                print(f"  ❌ FAIL - Expected {expected_result}, got {result}")
                return False
        else:
            print(f"  ✅ PASS - Completed without error")
            if result is not None:
                print(f"      Result: {result}")
        return True
    except Exception as e:
        print(f"  ❌ FAIL - Error: {e}")
        print(f"      {traceback.format_exc()}")
        return False


def explain_concept(concept: str, explanation: str):
    """
    Explain a concept in a friendly, conversational way.
    
    We use this to add context and help you understand not just
    what the code does, but why it matters.
    """
    print(f"\n💡 {concept}")
    print(f"   {explanation}")


def show_comparison(name1: str, value1: Any, name2: str, value2: Any):
    """
    Show a side-by-side comparison of two values.
    
    This is great for showing before/after or different approaches.
    """
    print(f"\n📊 Comparison:")
    print(f"  {name1}: {value1}")
    print(f"  {name2}: {value2}")
    print(f"  Same? {value1 == value2}")


def run_script_safely(script_name: str, main_func: Callable):
    """
    Run a validation script with nice error handling.
    
    Use this as your main entry point for validation scripts.
    """
    print_section(f"Running {script_name}")
    print("Let's walk through this step by step...")
    
    try:
        main_func()
        print(f"\n🎉 {script_name} completed successfully!")
        print("All examples worked - this code is ready for the docs!")
    except Exception as e:
        print(f"\n💥 {script_name} failed with error: {e}")
        print("This needs to be fixed before it goes in the documentation.")
        traceback.print_exc()
        sys.exit(1)


def check_imports():
    """
    Check that we can import our main library.
    
    This is usually the first thing we do in any validation script.
    """
    try:
        import pytensor
        print("✅ pytensor imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import pytensor: {e}")
        print("Make sure you're running from the project root directory")
        return False 