from mistralai import File, Mistral
from pathlib import Path
import time
from io import BytesIO
import csv
import json
from dotenv import load_dotenv
import os


def print_stats(batch_job):
    """
    Print the statistics of the batch job.

    Args:
        batch_job: The batch job object containing job statistics.
    """
    print(f"Total requests: {batch_job.total_requests}")
    print(f"Failed requests: {batch_job.failed_requests}")
    print(f"Successful requests: {batch_job.succeeded_requests}")
    print(
        f"Percent done: {round((batch_job.succeeded_requests + batch_job.failed_requests) / batch_job.total_requests, 4) * 100}"
    )


def download_file(client, file_id, output_path):
    """
    Download a file from the Mistral server.

    Args:
        client (Mistral): The Mistral client instance.
        file_id (str): The ID of the file to download.
        output_path (str): The path where the file will be saved.
    """
    if file_id is not None:
        print(f"Downloading file to {output_path}")
        output_file = client.files.download(file_id=file_id)
        with open(output_path, mode="w", encoding="utf-8") as f:
            for chunk in output_file.stream:
                f.write(chunk.decode("utf-8"))
        print(f"Downloaded file to {output_path}")


def create_tasks(client, path):
    with open(path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    buffer = BytesIO()
    i = 0
    for row in rows:
        prompt = f"""I want you to act as an answer judge. Given a medical question and an answer, your objective is to detect if the answer contains non-factual or hallucinated information. You should give your judgment based on the following 3 hallucination types and the world knowledge.
            
        1. Input-conflicting hallucination: When the generated answer deviates from the correct answer.
        2. Context-conflicting hallucination: When the generated answer conflicts with itself.
        3. Fact-conflicting hallucination: When the generated answer contradicts the established world knowledge.
            
        You should try your best to determine if the answer contains non-factual or hallucinated information according to the above hallucination types. The answer must be either “Yes” or “No”. If “Yes”, provide the non-factual or hallucinated text spans from the answer in a bullet format without any other information.

        Question: {row['question']}

        Answer: {row['answer']}"""

        request = {
            "custom_id": str(i),
            "body": {
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        buffer.write(json.dumps(request).encode("utf-8"))
        buffer.write("\n".encode("utf-8"))
        i += 1

    return client.files.upload(
        file=File(file_name="mistral_tasks.jsonl", content=buffer.getvalue()),
        purpose="batch",
    )


p = str(Path(__file__).parent.parent)
path = p + "/batches_and_tasks/mistral_tasks.jsonl"
output_path = p + "/batches_and_tasks/mistral_tests.jsonl"
error_path = p + "/batches_and_tasks/mistral_errors.jsonl"

model = "mistral-medium-2505"

load_dotenv()

client = Mistral(api_key=os.getenv("mistral_api_key"))

file = create_tasks(client, p + "/dataset/HalluQA.csv")

batch_job = client.batch.jobs.create(
    input_files=[file.id],
    model=model,
    endpoint="/v1/chat/completions",
    metadata={"job_type": "testing"},
)

while batch_job.status in ["QUEUED", "RUNNING"]:
    batch_job = client.batch.jobs.get(job_id=batch_job.id)
    print_stats(batch_job)
    time.sleep(1)

print(f"Batch job {batch_job.id} completed with status: {batch_job.status}")
print(f"Job duration: {batch_job.completed_at - batch_job.created_at} seconds")
download_file(client, batch_job.error_file, error_path)
download_file(client, batch_job.output_file, output_path)
