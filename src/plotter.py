import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# https://ai.meta.com/blog/meta-llama-3-1/


gsm8k_unsorted_models = [
    Path(__file__).parent / "../models/gsm8k_0/training.npz",
    Path(__file__).parent / "../models/gsm8k_1/training.npz",
    Path(__file__).parent / "../models/gsm8k_2/training.npz",
]

math_unsorted_models = [
    Path(__file__).parent / "../models/math_0/training.npz",
    Path(__file__).parent / "../models/math_1/training.npz",
    Path(__file__).parent / "../models/math_2/training.npz",
]

# models trained on gsm8k then math
gsm8k_math_unsorted_models = [
    Path(__file__).parent / "../models/gsm8k_math_0/training.npz",
    Path(__file__).parent / "../models/gsm8k_math_1/training.npz",
    Path(__file__).parent / "../models/gsm8k_math_2/training.npz",
]

gsm8k_sorted_models = {
    1: [
        Path(__file__).parent / "../models/model_1/training1.npz",
        Path(__file__).parent / "../models/model_2/training1.npz",
        Path(__file__).parent / "../models/model_3/training1.npz",
    ],
    2: [
        Path(__file__).parent / "../models/model_1/training2.npz",
        Path(__file__).parent / "../models/model_2/training2.npz",
        Path(__file__).parent / "../models/model_3/training2.npz",
    ],
    3: [
        Path(__file__).parent / "../models/model_1/training3.npz",
        Path(__file__).parent / "../models/model_2/training3.npz",
        Path(__file__).parent / "../models/model_3/training3.npz",
    ],
    4: [
        Path(__file__).parent / "../models/model_1/training4.npz",
        Path(__file__).parent / "../models/model_2/training4.npz",
        Path(__file__).parent / "../models/model_3/training4.npz",
    ],
    5: [
        Path(__file__).parent / "../models/model_1/training5.npz",
        Path(__file__).parent / "../models/model_2/training5.npz",
        Path(__file__).parent / "../models/model_3/training5.npz",
    ],
}

math_sorted_models = {
    1: [
        Path(__file__).parent / "../models/math_sorted1_0/training.npz",
        Path(__file__).parent / "../models/math_sorted1_1/training.npz",
        Path(__file__).parent / "../models/math_sorted1_2/training.npz",
    ],
    2: [
        Path(__file__).parent / "../models/math_sorted2_0/training.npz",
        Path(__file__).parent / "../models/math_sorted2_1/training.npz",
        Path(__file__).parent / "../models/math_sorted2_2/training.npz",
    ],
    3: [
        Path(__file__).parent / "../models/math_sorted3_0/training.npz",
        Path(__file__).parent / "../models/math_sorted3_1/training.npz",
        Path(__file__).parent / "../models/math_sorted3_2/training.npz",
    ],
    4: [
        Path(__file__).parent / "../models/math_sorted4_0/training.npz",
        Path(__file__).parent / "../models/math_sorted4_1/training.npz",
        Path(__file__).parent / "../models/math_sorted4_2/training.npz",
    ],
    5: [
        Path(__file__).parent / "../models/math_sorted5_0/training.npz",
        Path(__file__).parent / "../models/math_sorted5_1/training.npz",
        Path(__file__).parent / "../models/math_sorted5_2/training.npz",
    ],
}

gsm8k_math_sorted_models = {
    1: [
        Path(__file__).parent / "../models/model1/training1.npz",
        Path(__file__).parent / "../models/model2/training1.npz",
        Path(__file__).parent / "../models/model3/training1.npz",
    ],
    2: [
        Path(__file__).parent / "../models/model1/training2.npz",
        Path(__file__).parent / "../models/model2/training2.npz",
        Path(__file__).parent / "../models/model3/training2.npz",
    ],
    3: [
        Path(__file__).parent / "../models/model1/training3.npz",
        Path(__file__).parent / "../models/model2/training3.npz",
        Path(__file__).parent / "../models/model3/training3.npz",
    ],
    4: [
        Path(__file__).parent / "../models/model1/training4.npz",
        Path(__file__).parent / "../models/model2/training4.npz",
        Path(__file__).parent / "../models/model3/training4.npz",
    ],
    5: [
        Path(__file__).parent / "../models/model1/training5.npz",
        Path(__file__).parent / "../models/model2/training5.npz",
        Path(__file__).parent / "../models/model3/training5.npz",
    ],
}

combined_models = {
    Path(__file__).parent / "../models/combined3/training.npz",
    Path(__file__).parent / "../models/combined4/training.npz",
    Path(__file__).parent / "../models/combined5/training.npz",
}


def every_k(x: np.ndarray, k: int=100) -> np.ndarray:
    return x.copy()[::k]


def plot_gsm8k(k: int = 25) -> None:
    losses = []
    steps = None
    for model_path in gsm8k_unsorted_models:
        loadedarr = np.load(model_path)
        losses.append(loadedarr['loss'])
        steps = loadedarr['steps'] #it should be the same for all loaded instances
        # plt.plot(loadedarr['steps'], loadedarr['loss'])

    means = np.mean(np.array(losses), axis=0)

    steps, means = every_k(steps, k), every_k(means, k)

    plt.plot(steps, means)
    plt.title("Avg GSM8K Unsorted Train Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")

    save_path = Path(__file__).parent / "../images/gsm8k_unsorted.png"
    plt.savefig(save_path)
    plt.show()


def plot_math(k: int = 25) -> None:
    losses = []
    steps = None
    for model_path in math_unsorted_models:
        loadedarr = np.load(model_path)
        losses.append(loadedarr['loss'])
        steps = loadedarr['steps'] #it should be the same for all loaded instances
        # plt.plot(loadedarr['steps'], loadedarr['loss'])

    means = np.mean(np.array(losses), axis=0)

    steps, means = every_k(steps, k), every_k(means, k)

    plt.plot(steps, means)
    plt.title("Avg MATH Unsorted Train Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")

    save_path = Path(__file__).parent / "../images/math_unsorted.png"
    plt.savefig(save_path)
    plt.show()



def plot_gsm8k_math(k: int = 25) -> None:
    math_losses = []
    steps = None
    for model_path in math_unsorted_models:
        loadedarr = np.load(model_path)
        math_losses.append(loadedarr['loss'])
        steps = loadedarr['steps'] #it should be the same for all loaded instances
        # plt.plot(loadedarr['steps'], loadedarr['loss'])

    math_means = np.mean(np.array(math_losses), axis=0)

    losses = []
    steps = None
    for model_path in gsm8k_math_unsorted_models:
        loadedarr = np.load(model_path)
        losses.append(loadedarr['loss'])
        steps = loadedarr['steps'] #it should be the same for all loaded instances
        # plt.plot(loadedarr['steps'], loadedarr['loss'])

    means = np.mean(np.array(losses), axis=0)

    steps, means, math_means = every_k(steps, k), every_k(means, k), every_k(math_means, k)

    plt.plot(steps, means, label="GSM8K+MATH")
    plt.plot(steps, math_means, label="MATH only")
    plt.title("Avg GSM8K-MATH Unsorted Train Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()

    save_path = Path(__file__).parent / "../images/gsm8k_math_unsorted.png"
    plt.savefig(save_path)
    plt.show()


def plot_gsm8k_sorted(k: int = 25) -> None:
    for level, paths in gsm8k_sorted_models.items():
        losses = []
        steps = None
        for p in paths:
            loadedarr = np.load(p)
            losses.append(loadedarr['loss'])
            steps = loadedarr['steps'] #it should be the same for all loaded instances
            # plt.plot(loadedarr['steps'], loadedarr['loss'])

        means = np.mean(np.array(losses), axis=0)

        steps, means = every_k(steps, k), every_k(means, k)

        plt.plot(steps, means)
        plt.title(f"Avg GSM8K Level {level} Train Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")

        save_path = Path(__file__).parent / f"../images/gsm8k_level{level}.png"
        plt.savefig(save_path)
        plt.show()



def plot_math_sorted(k: int = 25) -> None:
    for level, paths in math_sorted_models.items():
        losses = []
        steps = None
        for p in paths:
            loadedarr = np.load(p)
            losses.append(loadedarr['loss'])
            steps = loadedarr['steps'] #it should be the same for all loaded instances
            # plt.plot(loadedarr['steps'], loadedarr['loss'])

        means = np.mean(np.array(losses), axis=0)

        steps, means = every_k(steps, k), every_k(means, k)

        plt.plot(steps, means)
        plt.title(f"Avg MATH Level {level} Train Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")

        save_path = Path(__file__).parent / f"../images/math_level{level}.png"
        plt.savefig(save_path)
        plt.show()


def plot_gsm8k_math_sorted(k: int = 25) -> None:
    for level, paths in math_sorted_models.items():
        math_losses = []
        steps = None
        for p in paths:
            loadedarr = np.load(p)
            math_losses.append(loadedarr['loss'])
            steps = loadedarr['steps'] #it should be the same for all loaded instances
            # plt.plot(loadedarr['steps'], loadedarr['loss'])

        losses = []
        for p in gsm8k_math_sorted_models[level]:
            loadedarr = np.load(p)
            losses.append(loadedarr['loss'])
            steps = loadedarr['steps'] #it should be the same for all loaded instances
            # plt.plot(loadedarr['steps'], loadedarr['loss'])

        math_means = np.mean(np.array(math_losses), axis=0)
        means = np.mean(np.array(losses), axis=0)
        steps, math_means, means = every_k(steps, k), every_k(math_means, k), every_k(means, k)

        plt.plot(steps, means, label="GSM8K+MATH")
        plt.plot(steps, math_means, label="MATH Only")
        plt.title(f"Avg MATH Level {level} Train Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()

        save_path = Path(__file__).parent / f"../images/gsm8k_math_level{level}.png"
        plt.savefig(save_path)
        plt.show()



def plot_combined(k: int = 25) -> None:
    losses = []
    steps = None
    for model_path in combined_models:
        loadedarr = np.load(model_path)
        losses.append(loadedarr['loss'])
        steps = loadedarr['steps'] #it should be the same for all loaded instances
        # plt.plot(loadedarr['steps'], loadedarr['loss'])

    means = np.mean(np.array(losses), axis=0)

    steps, means = every_k(steps, k), every_k(means, k)

    plt.plot(steps, means)
    plt.title("Avg Combined Train Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")

    save_path = Path(__file__).parent / "../images/combined.png"
    plt.savefig(save_path)
    plt.show()



class Config:
    GSM8K = "GSM8K"
    MATH = "MATH"
    GSM8K_MATH = "GSM8K_MATH"
    GSM8K_SORTED = "GSM8K_SORTED"
    MATH_SORTED = "MATH_SORTED"
    GSM8K_MATH_SORTED = "GSM8K_MATH_SORTED"
    COMBINED = "COMBINED"

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gsm8k", dest="func", const=Config.GSM8K, action="store_const", help="Plots GSM8K training losses.")
    group.add_argument("--math", dest="func", const=Config.MATH, action="store_const", help="Plots MATH training losses.")
    group.add_argument("--gsm8k-math", dest="func", const=Config.GSM8K_MATH, action="store_const", help="Plots MATH training losses on models pretrained on GSM8K.")

    group.add_argument("--gsm8k-sorted", dest="func", const=Config.GSM8K_SORTED, action="store_const", help="Plots GSM8K sorted training losses.")
    group.add_argument("--math-sorted", dest="func", const=Config.MATH_SORTED, action="store_const", help="Plots MATH sorted training losses.")
    group.add_argument("--gsm8k-math-sorted", dest="func", const=Config.GSM8K_MATH_SORTED, action="store_const", help="Plots MATH sorted training losses on models pretrained on GSM8K sorted.")

    group.add_argument("--combined", dest="func", const=Config.COMBINED, action="store_const", help="Plots the combined dataset training losses.")

    parser.add_argument("--k", type=int, default=25, help="Plots made at every k training step")

    return parser.parse_args()

FUNC_MAP = {
    Config.GSM8K: plot_gsm8k,
    Config.MATH: plot_math,
    Config.GSM8K_MATH: plot_gsm8k_math,
    Config.GSM8K_SORTED: plot_gsm8k_sorted,
    Config.MATH_SORTED: plot_math_sorted,
    Config.GSM8K_MATH_SORTED: plot_gsm8k_math_sorted,
    Config.COMBINED: plot_combined,
}

if __name__ == "__main__":
    args = _parse_args()
    FUNC_MAP[args.func](k=args.k)
