import argparse
from datasets import load_dataset
from pathlib import Path

def load_dataset_test():
    # not important but maybe make a prettier testing suite
    #? learn to use unittest module: https://www.dataquest.io/blog/unit-tests-python/
    path = (Path(__file__).parent / "../data/GSM8K").resolve().__str__()
    GSM8K = load_dataset(path)
    print("Load Data Test Passed")


TESTMAP = {
    "loaddata": load_dataset_test,
}

def _parse_args() -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", default=None, choices=TESTMAP.keys(), help="Test the functionality")

    return vars(parser.parse_args())



if __name__ == "__main__":
    args = _parse_args()

    if args["test"] is not None:
        TESTMAP[args["test"]]()
