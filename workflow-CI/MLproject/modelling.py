import pandas as pd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="heart_preprocessed.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
