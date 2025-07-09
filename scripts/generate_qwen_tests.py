from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path
import csv
import json

p = str(Path(__file__).parent.parent)

path = p + "/dataset/HalluQA.csv"
with open(path, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

model_name = "Qwen/Qwen3-8B"

generator = pipeline(
    "text-generation",
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
generator.tokenizer.padding_side = "left"

batch = []
for row in rows:
    prompt = f"""I want you to act as an answer judge. Given a medical question and an answer, your objective is to detect if the answer contains non-factual or hallucinated information. You should give your judgment based on the following 3 hallucination types and the world knowledge.
            
    1. Input-conflicting hallucination: When the generated answer deviates from the correct answer.
    2. Context-conflicting hallucination: When the generated answer conflicts with itself.
    3. Fact-conflicting hallucination: When the generated answer contradicts the established world knowledge.
            
    You should try your best to determine if the answer contains non-factual or hallucinated information according to the above hallucination types. The answer must be either “Yes” or “No”. If “Yes”, provide the non-factual or hallucinated text spans from the answer in a bullet format without any other information.

    Question: {row['question']}

    Answer: {row['answer']}"""

    message = [{"role": "user", "content": prompt}]
    batch.append(message)

results = generator(batch, max_new_tokens=32768, batch_size=len(batch))
batch = [result[0] for result in results]

path = p + "/batches_and_tasks/qwen_tasks.jsonl"
with open(path, mode="w", encoding="utf-8") as file:
    for task in batch:
        file.write(json.dumps(task) + "\n")
