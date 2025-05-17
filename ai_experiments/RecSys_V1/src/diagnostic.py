"""
Simple test to diagnose module import issues.
"""
import os
import sys
import traceback

print("===== DIAGNOSTIC TEST =====")
print(f"Current directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    # Try importing from current directory
    print("\nAttempting to import directly...")
    import sentiment_analysis_module
    print("Import directly successful")
except Exception as e:
    print(f"Direct import failed: {e}")
    traceback.print_exc()

try:
    # Try importing with sys.path adjustment
    print("\nAttempting to import with sys.path adjustment...")
    sys.path.append(os.path.dirname(os.getcwd()))
    from src import sentiment_analysis_module
    print("Import with sys.path adjustment successful")
except Exception as e:
    print(f"Import with sys.path adjustment failed: {e}")
    traceback.print_exc()

try:
    # Try relative import
    print("\nAttempting relative import...")
    from . import sentiment_analysis_module
    print("Relative import successful")
except Exception as e:
    print(f"Relative import failed: {e}")
    traceback.print_exc()

print("\n===== END DIAGNOSTIC TEST =====")
