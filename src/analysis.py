import json

freq = {}

with open('result.txt', 'r') as f:
  for line in f:
    data = json.loads(line)
    content = data['response']['body']['choices'][0]['message']['content']
    freq[content] = freq.get(content, 0) + 1

print('Response Frequency')
for k, v in freq.items():
  print(f'{k}: {v}')
