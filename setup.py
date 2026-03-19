import subprocess
import sys

def install():
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("\nDone! You can now run the pipeline with:")
    print("  python run_analysis.py")

if __name__ == "__main__":
    install()
