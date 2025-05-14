# Compsci685FinalProject

## A quick, code overview to our project.

- `Training.ipynb` - Our first fine-tuning notebook that focuses on using the unsorted GSM8K and MATH datasets
- `SortedTraining.ipynb` - As the name suggests, this is our fine-tuning notebook that utilizes sorting on the GSM8K and MATH datasets\*
- `CombinedTraining.ipynb` - Our combined dataset run over both the GSM8K and Math datasets in one fine-tuning run.
- `Inference.ipynb` - This notebook was used to run our fine-tuned models from above on the MATH test set for evaluation saving the results into CSV files.
- `inf_stats.ipynb` - This notebook was used to help us verify and perform analysis on our models' inference outputs

ote that all our notebooks were run in [Google Colab ](https://colab.research.google.com/) using Nvidia A100 GPUs.
\* To run `SortedTraining.ipynb` you will need to upload `src/gsm8k_ordered.csv` in order to sort the GSM8K dataset per our batch processing results to OpenAI GPT-4o mini.


## Models and Inference Results

Our trained models are stored in a `models` directory but it is gitignored due to their collective size. They can be found in our [Google Drive](https://drive.google.com/drive/folders/1c9M_CVFZ_8zSa0XiTof1kKm8ZtdOZ1B7?usp=drive_link).

Our inference results, produced by `Inference.ipynb` can also be found in our drive. 