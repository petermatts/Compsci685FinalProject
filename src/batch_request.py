from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

with open('batch-v2.id', 'w') as f:
  client = OpenAI()

  # Upload LLM requests
  batch_input_file = client.files.create(
    file=open(f'batch-v2.jsonl', 'rb'), purpose='batch'
  )

  # Start adaptive inference
  batch_request = client.batches.create(
    input_file_id=batch_input_file.id,
    endpoint='/v1/chat/completions',
    completion_window='24h'
  )

  print(batch_request.model_dump())
  f.write(batch_request.id)
