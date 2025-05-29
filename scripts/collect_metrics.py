import json
import csv

results = []
with open('detection_tests.jsonl', mode='r') as file:
    for line in file:
        results.append(json.loads(line.strip()))

with open('HalluQA.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

outcomes = {
    'true-pos': 0,
    'false-pos': 0,
    'true-neg': 0,
    'false-neg': 0
}
correct = 0
total = 0
for i in range(len(results)):
    answer = results[i]['response']['body']['output'][0]['content'][0]['text']

    if 'yes' in answer[:7].lower() and int(rows[i]['hallu_type_int']) > 0:
        outcomes['true-pos'] += 1
        correct += 1
        total += 1
    elif 'yes' in answer[:7].lower():
        outcomes['false-pos'] += 1
        total += 1
    elif 'no' in answer[:7].lower() and int(rows[i]['hallu_type_int']) == 0:
        outcomes['true-neg'] += 1
        correct += 1
        total += 1
    elif 'no' in answer[:7].lower():
        outcomes['false-neg'] += 1
        total += 1

accuracy = correct / total
precision = outcomes['true-pos'] / (outcomes['true-pos'] + outcomes['false-pos'])
recall = outcomes['true-pos'] / (outcomes['true-pos'] + outcomes['false-neg'])
f_score = 2 * ((precision * recall) / (precision + recall))

metrics = {
    'total': total,
    'correct': correct,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1-score': f_score
}

with open('metrics.jsonl', mode='w') as file:
    file.write(json.dumps(metrics))
