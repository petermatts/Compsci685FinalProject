"""
This script fetches and downloads (as a PDF) any of the reference papers
listed in the Papers dictionary below. Papers are saved to the references
directory
"""

from pathlib import Path
import requests
import os

# references are a key value pair: key=name of paper, value=url to paper pdf
Papers = {
    "All You Need is Attention": "https://arxiv.org/pdf/1706.03762",
    "Adaptive Mixtures of Local Experts": "https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf",
    "Curriculum Learning: A Survey": "https://arxiv.org/pdf/2101.10382",
    "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models": "https://arxiv.org/pdf/2201.11903",
    "Self-Consistency Improves Chain of Thought Reasoning in Language Models": "https://arxiv.org/pdf/2203.11171",
    "Metacognitive Capabilities of LLMs: An Exploration in Mathematical Problem Solving": "https://arxiv.org/pdf/2405.12205",
    "Curriculum Learning": "https://ronan.collobert.com/pub/2009_curriculum_icml.pdf",
    "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models": "https://arxiv.org/pdf/2205.10625",
    "Formal Mathematics Statement Curriculum Learning": "https://arxiv.org/pdf/2202.01344v1",
    "Large Language Models for Mathematical Reasoning": "https://arxiv.org/pdf/2402.00157v3",

    "Automatic Curriculum Expert Iteration For Reliable LLM Reasoning": "https://openreview.net/pdf?id=3ogIALgghF",
    "Lets Be Self-Generated Via Step By Step A Curriculum Learning Approach to Automated Reasoning with Large Language Models": "https://arxiv.org/pdf/2410.21728v1",
    "Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning": "https://arxiv.org/pdf/2402.05808",
}

BASE_PATH = Path(__file__).parent / "references"
os.makedirs(BASE_PATH, exist_ok=True)

downloaded = False
for k, v in Papers.items():
    response = requests.get(v)
    file_path = BASE_PATH / f'{k.replace(" ", "_").replace(":", "")}.pdf'

    if os.path.isfile(file_path):
        continue
    else:
        downloaded = True
        name = (k + '.pdf' if len(k) < 64 else k[:64] + '... .pdf').ljust(72, ' ')
        print(f"Getting: {name} ... ", end='')
        with open(file_path, "wb") as f:
            f.write(response.content)
            print(" Done")

if not downloaded:
    print("Refs upto date :)")
    