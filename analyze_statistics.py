import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str)

    args = parser.parse_args()

    with open(args.file, "r") as f:
        data = json.load(f)

        sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[1], reverse=True)}

        print(list(sorted_data.items())[0:10])
        print(list(sorted_data.items())[-10:])



