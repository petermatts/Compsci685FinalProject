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
    "Adaptive Mixtures of Local Experts": "https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf",
    "Curriculum Learning: A Survey": "https://arxiv.org/pdf/2101.10382",
    "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models": "https://arxiv.org/pdf/2201.11903",
    "Self-Consistency Improves Chain of Thought Reasoning in Language Models": "https://arxiv.org/pdf/2203.11171",
    "Metacognitive Capabilities of LLMs: An Exploration in Mathematical Problem Solving": "https://arxiv.org/pdf/2405.12205",
    "Curriculum Learning": "https://ronan.collobert.com/pub/2009_curriculum_icml.pdf",
    "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models": "https://arxiv.org/pdf/2205.10625",
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
    