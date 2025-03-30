from argparse import ArgumentParser
from datasets import load_dataset
from enum import Enum

class Datasets(Enum):
    GSM8K = "gsm8k"
    MATH = "math"

def _parse_args() -> dict:
    parser = ArgumentParser(
        # prog="Downloader",
        # usage="run python Downloader.py --<dataset> to download the corresponding dataset. Supported datasets are currently GSM8k and MATH."
    )

    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument("--gsm8k", action="store_true", help="Downloads the Grade School Math 8k Dataset")
    dataset.add_argument("--math", action="store_true", help="Downloads the MATH dataset/benchmark")

    return vars(parser.parse_args())

def _get_dataset(dataset: str):
    match dataset: # requires python 3.10+
        case Datasets.GSM8K.value:
            ds = load_dataset("openai/gsm8k", "main")
            ds.save_to_disk("GSM8K")
        case Datasets.MATH.value:
            ds = load_dataset("nlile/hendrycks-MATH-benchmark")
            ds.save_to_disk("MATH")
        case _:
            raise ValueError(f"Unknown dataset {dataset}") # note this cannot happen

if __name__ == "__main__":
    args = _parse_args()
    key = [k for k, v in args.items() if v][0]
    _get_dataset(Datasets[key.upper()].value)
