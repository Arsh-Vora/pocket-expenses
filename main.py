import subprocess
import sys

def run_step(module_name):
    print(f"Running step: {module_name}")
    subprocess.run([sys.executable, "-m", module_name], check=True)

def main():
    try:
        # Step 1: Dataset collection
        run_step("src.generator")

        # Step 2 & 3: Fine-Tuning Execution
        run_step("src.trainer")

        # Step 4: Verification 
        run_step("src.evaluator")
        
        print("Pipeline completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed at step {e.cmd[2]}")
        sys.exit(1)

if __name__ == "__main__":
    main()