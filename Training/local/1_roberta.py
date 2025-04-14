# compute_emotion_vectors.py

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from tqdm import tqdm

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# HF token
hf_token = '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'

# Load dataset
csv_path = hf_hub_download(repo_id="aoxo/temp", filename="chat_conversation.csv", repo_type='dataset', token=hf_token)
dataset = pd.read_csv(csv_path).fillna('')
speaker1, speaker2 = dataset.columns[:2]

# Load emotion model
emotion_tokenizer = RobertaTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
emotion_model = RobertaForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
emotion_model.config.pad_token_id = 50256
emotion_model.to(device)
emotion_model.eval()

# Function to compute emotion vector
def get_emotion_vector(text):
    if not text.strip():
        return np.zeros(28)
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    logits = outputs.logits.squeeze()
    probabilities = torch.sigmoid(logits)
    return probabilities.cpu().numpy()

# Compute emotion vectors
emotion_vectors = []
for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Computing emotion vectors"):
    combined_text = str(row[speaker1]) + " " + str(row[speaker2])
    emotion_vec = get_emotion_vector(combined_text)
    emotion_vectors.append(emotion_vec)

# Save to .npy file
emotion_vectors = np.stack(emotion_vectors)
np.save("emotion_vectors.npy", emotion_vectors)
print("Saved emotion vectors to emotion_vectors.npy")