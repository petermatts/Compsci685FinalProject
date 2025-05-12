from datasets import load_dataset
import json
import re

# Step 1: construct jsonl file of requests
dataset_train = load_dataset('openai/gsm8k', 'main', split='train')

i = 0
with open('batch-v2.jsonl', 'w') as f:
  for data in dataset_train:
    question: str = data['question']
    question = question.replace('\n', '')

    answer: str = data['answer']
    answer = re.sub(r'<<.*?>>', '', answer)
    answer = re.sub(r'###.*$', '', answer)
    answer = answer.replace('\n', ' ')

    f.write(json.dumps({
      'custom_id': f'gsm8k-{i:04}',
      'method': 'POST',
      'url': '/v1/chat/completions',
      'body': {
        'model': 'gpt-4.1-nano-2025-04-14',
        'messages': [
          { 'role': 'system', 'content': 'You are a math problem classifier bot. You will be given a math question-answer pair, and your task is to classify its difficulty with an integer between 1-5, where 1 is lower elementary school level and 5 is upper elementary school level. Level 1 should contain up to counting and basic addition/subtraction (up to 20). Level 2 should contain up to addition/subtraction (any number), basic multiplication/division (up to 20), and basic fractions and decimals. Level 3 should contain up to multiplication/division (any number), complex fractions and decimals, and basic geometry (area, perimeter). Level 4 should contain up to advanced geometry (complex shapes), basic algebra (variables), and percentages/ratios. Level 5 should contain up to standard algebra, negative numbers, data analysis (mean, median, mode), and exponentiation. ONLY output the difficulty level, NOTHING ELSE.' },
          { 'role': 'user', 'content': f'Question: {question}\nAnswer: {answer}' }
        ],
        'max_completion_tokens': 100
      }
    }) + '\n')

    i += 1
