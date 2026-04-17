import argparse
from src.trainer import train
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, required=True, help="Path to dataset")
    parser.add_argument("--target", type=str, required=True, help="Target column")

    args = parser.parse_args()

    os.makedirs("artifacts", exist_ok=True)

    print(" Running AutoML Pipeline...\n")

    score, problem_type = train(args.file, args.target)

    print("\n Done!")
    print(f"Problem Type: {problem_type}")
    print(f"Best Score: {score}")