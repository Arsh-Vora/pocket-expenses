#!/usr/bin/env python3
import importlib
import sys

def verify_environment():
    """Performs a runtime sanity check on core project components and imports."""
    print("--- Initiating Environment Smoke Test ---")
    
    # Target modules to check sequentially
    modules_to_test = [
        ("Dataset Generator", "src.generator"),
        ("SLM Training Engine", "src.trainer"),
        ("SLM Evaluation Suite", "src.evaluator")
    ]
    
    failures = 0

    for label, module_path in modules_to_test:
        print(f"Checking component: {label:<22}...", end=" ", flush=True)
        try:
            # Dynamically import the module using its string path
            importlib.import_module(module_path)
            print("[ PASS ]")
        except Exception as e:
            print(f"[ FAIL ]\n  └─ Reason: {type(e).__name__}: {e}")
            failures += 1

    print("-" * 41)
    if failures:
        print(f"Verification completed with {failures} critical import failure(s).")
        sys.exit(1)
        
    print("All core pipelines verified successfully. Ready for execution.")

if __name__ == "__main__":
    verify_environment()
