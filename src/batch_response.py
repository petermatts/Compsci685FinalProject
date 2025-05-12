import datetime
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAI compatible API
client = OpenAI()

with open('batch-v2.id', 'r') as f:
  batch_request_id = f.read().strip()

response = client.batches.retrieve(batch_request_id)
data = response.model_dump()
print(data)

if data['status'] != "completed":
  print(f'Batch not complete yet ({datetime.datetime.now()})')
  sys.exit()

print('Batch is complete!')

# Download results
result_file_id = response.output_file_id
assert result_file_id is not None
llm_inference_results = client.files.content(result_file_id).content

with open('result-v2.txt', 'wb') as f:
  f.write(llm_inference_results)
