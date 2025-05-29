import kagglehub

# Download latest version
path = kagglehub.dataset_download("pythonafroz/medquad-medical-question-answer-for-ai-research")

print("Path to dataset files:", path)