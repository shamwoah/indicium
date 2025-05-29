import csv
import json
from pathlib import Path

p = Path(__file__).parent.parent
path = str(p) + '/dataset/HalluQA.csv'
with open(path, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

total = 0
for row in rows:
    if "Hallucinated answer" in row['answer'] or "Hallucinated Answer" in row['answer']:
        total += 1
        row['answer'] = row['answer'].replace("Hallucinated answer", "")
        row['answer'] = row['answer'].replace("Hallucinated Answer", "")

print(total)

names = [
    'question',
    'answer',
    'hallu_type',
    'hallu_type_int'
]
path = str(p) + '/dataset/HalluQA2.csv'
with open(path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=names)
    writer.writeheader()
    writer.writerows(rows)

path = str(p) + '/dataset/HalluQA2.jsonl'
with open(path, mode='w') as f:
    for row in rows:
        f.write(json.dumps(row) + '\n')
