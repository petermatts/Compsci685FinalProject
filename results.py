import json
import numpy as np
from pathlib import Path

# files = ["op_base_final.json"] # base
# files = ["op_gsm8k_0_final.json", "op_gsm8k_1_final.json", "op_gsm8k_2_final.json"] # gsm8k unsorted
files = ["op_gsm8k_sorted_1_final.json", "op_gsm8k_sorted_2_final.json", "op_gsm8k_sorted_3_final.json"] # gsm8k sorted
# files = ["op_math_0_final.json", "op_math_1_final.json", "op_math_2_final.json"] # math unsorted
# files = ["op_math_sorted_0_final.json", "op_math_sorted_1_final.json", "op_math_sorted_2_final.json"] # math sorted
# files = ["op_gsm8k_math_0_final.json", "op_gsm8k_math_1_final.json", "op_gsm8k_math_2_final.json"] # gsm8k + math unsorted
# files = ["op_gsm8k_math_sorted_1_final.json", "op_gsm8k_math_sorted_2_final.json", "op_gsm8k_math_sorted_3_final.json"] # gsm8k + math sorted
# files = ["op_combined_3.json", "op_combined_4.json", "op_combined_5.json"]

level1 = []
level2 = []
level3 = []
level4 = []
level5 = []

overall = []

for file in files:
    with open(Path(__file__).parent / f"results/{file}", "r") as f:
        data = json.load(f)

    level1.append(data["accuracy_by_level"]["1"])
    level2.append(data["accuracy_by_level"]["2"])
    level3.append(data["accuracy_by_level"]["3"])
    level4.append(data["accuracy_by_level"]["4"])
    level5.append(data["accuracy_by_level"]["5"])
    overall.append(data["accuracy"])



print(f"level1:  {np.mean(level1)*100:2.1f}%")
print(f"level2:  {np.mean(level2)*100:2.1f}%")
print(f"level3:  {np.mean(level3)*100:2.1f}%")
print(f"level4:  {np.mean(level4)*100:2.1f}%")
print(f"level5:  {np.mean(level5)*100:2.1f}%")
print(f"overall: {np.mean(overall)*100:2.1f}%")
