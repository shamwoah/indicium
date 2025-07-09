import csv
import json
from pathlib import Path

p = str(Path(__file__).parent.parent)

path = p + "/dataset/HalluQA.csv"
with open(path, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

tasks = []

i = 0
for row in rows:
    prompt = f"""I want you to act as an answer judge. Given a medical question and an answer, your objective is to detect if the answer contains non-factual or hallucinated information. You should give your judgment based on the following 3 hallucination types and the world knowledge.
            
    1. Input-conflicting hallucination: When the generated answer deviates from the correct answer.
    2. Context-conflicting hallucination: When the generated answer conflicts with itself.
    3. Fact-conflicting hallucination: When the generated answer contradicts the established world knowledge.
            
    You should try your best to determine if the answer contains non-factual or hallucinated information according to the above hallucination types. The answer must be either “Yes” or “No”. If “Yes”, provide the non-factual or hallucinated text spans from the answer in a bullet format without any other information.

    Question: {row['question']}

    Answer: {row['answer']}"""

    task = {
        "custom_id": str(i),
        "body": {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        },
    }

    tasks.append(task)
    i += 1

path = p + "/batches_and_tasks/mistral_tasks.jsonl"
with open(path, mode="w", encoding="utf-8") as file:
    for task in tasks:
        file.write(json.dumps(task) + "\n")
