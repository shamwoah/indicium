import pandas as pd
import csv
import json

df = pd.read_csv('medquad.csv')
df = df[df['focus_area'].str.contains('disease|cancer|diabetes|glaucoma', regex=True, case=False, na=False)].reset_index()
questions = df['question']

with open('HalluQA.csv', mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

for i in range(len(rows)):
    rows[i]['question'] = questions[i]

names = [
    'question',
    'answer',
    'hallu_type',
    'hallu_type_int'
]
with open('HalluQA2.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=names)
    writer.writeheader()
    writer.writerows(rows)

with open('HalluQA2.jsonl', 'w') as f:
    for row in rows:
        f.write(json.dumps(row) + '\n')
