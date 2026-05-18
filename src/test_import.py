import sys
print("starting")
try:
    print("importing generator")
    from src.generator import DatasetGenerator
    print("generator ok")
except Exception as e:
    print("generator error:", e)

try:
    print("importing trainer")
    from src.trainer import SLMTrainer
    print("trainer ok")
except Exception as e:
    print("trainer error:", e)

try:
    print("importing evaluator")
    from src.evaluator import SLMEvaluator
    print("evaluator ok")
except Exception as e:
    print("evaluator error:", e)

print("test_import done")
