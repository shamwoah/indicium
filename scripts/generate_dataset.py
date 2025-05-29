import json
import re
import pandas as pd

results = []
with open('hallucinated_answers.jsonl', 'r') as file:
    for line in file:
        results.append(json.loads(line.strip()))

with open('counter.json', 'r') as file:
    for line in file:
        counter = json.loads(line.strip())

prompts = []
with open('tasks.jsonl', 'r') as file:
    for line in file:
        prompts.append(json.loads(line.strip())['body']['input'])

rows = []
for r in results:
    id = int(re.findall(r'\d+', r['custom_id'])[0])

    if id in counter['indices']['none-conflicting']:
        hallu_type = 'none-conflicting'
        hallu_type_int = 0
    elif id in counter['indices']['fact-conflicting']:
        hallu_type = 'fact-conflicting'
        hallu_type_int = 1
    elif id in counter['indices']['input-conflicting']:
        hallu_type = 'input-conflicting'
        hallu_type_int = 2
    elif id in counter['indices']['context-conflicting']:
        hallu_type = 'context-conflicting'
        hallu_type_int = 3

    row = {
        'question': prompts[id],
        'answer': r['response']['body']['output'][0]['content'][0]['text'],
        'hallu_type': hallu_type,
        'hallu_type_int': hallu_type_int
    }
    rows.append(row)

df = pd.DataFrame.from_dict(rows)
df.to_csv('HalluQA.csv', index=False)

with open('HalluQA.jsonl', 'w') as f:
    for obj in rows:
        f.write(json.dumps(obj) + '\n')
