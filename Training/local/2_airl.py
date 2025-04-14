# train_airl_model.py

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download, HfApi
from tqdm import tqdm

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

batch_size = 2
hf_token = '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'

# Load dataset
csv_path = hf_hub_download(repo_id="aoxo/temp", filename="chat_conversation.csv", repo_type='dataset', token=hf_token)
dataset = pd.read_csv(csv_path).fillna('')
speaker1, speaker2 = dataset.columns[:2]
print(f"Loaded dataset with columns: {speaker1}, {speaker2}")

# Load emotion vectors
emotion_vectors = np.load("emotion_vectors.npy")
assert len(emotion_vectors) == len(dataset), "Mismatch between dataset and emotion vectors"

# Define Dataset class
class DialogueDataset(Dataset):
    def __init__(self, df, tokenizer, emotion_vectors, max_length=512):
        self.data = df
        self.speaker1, self.speaker2 = df.columns[:2]
        self.tokenizer = tokenizer
        self.emotion_vectors = emotion_vectors
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = str(self.data.loc[idx, self.speaker1])
        response = str(self.data.loc[idx, self.speaker2])
        emotion_vec = torch.tensor(self.emotion_vectors[idx], dtype=torch.float32)

        text = prompt + " " + response
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors="pt")

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "reward": torch.tensor(1.0),
            "emotion": emotion_vec
        }

# Define AIRL model
class AIRLRewardModel(nn.Module):
    def __init__(self, base_model_ckpt, emotion_dim=28, hf_token=None):
        super().__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            base_model_ckpt,
            num_labels=1,
            trust_remote_code=True,
            token=hf_token,
            ignore_mismatched_sizes=True
        )
        self.emotion_proj = nn.Linear(emotion_dim, self.transformer.config.hidden_size)
        self.transformer.config.pad_token_id = 50256
        self.transformer.config.use_cache = True

    def forward(self, input_ids, attention_mask, emotion_vec=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if emotion_vec is not None:
            emotion_features = self.emotion_proj(emotion_vec)
            logits = logits + emotion_features.mean(dim=1, keepdim=True)
        return logits

# Training setup
reward_model_ckpt = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_ckpt, token=hf_token)
if reward_tokenizer.pad_token is None:
    reward_tokenizer.pad_token = reward_tokenizer.eos_token

reward_model = AIRLRewardModel(reward_model_ckpt, hf_token=hf_token)
reward_model.to(device)

# Dataloader
reward_dataset = DialogueDataset(dataset, reward_tokenizer, emotion_vectors)
reward_dataloader = DataLoader(
    reward_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'reward': torch.stack([x['reward'] for x in batch]),
        'emotion': torch.stack([x['emotion'] for x in batch])
    }
)

# AIRL training
irl_criterion = nn.MSELoss()
irl_optimizer = optim.Adam(reward_model.parameters(), lr=1e-5)
num_irl_epochs = 3
accumulation_steps = 8

print("Starting AIRL Training...")
for epoch in range(num_irl_epochs):
    reward_model.train()
    total_loss = 0
    count = 0
    accumulation_count = 0

    for batch_idx, batch in enumerate(tqdm(reward_dataloader, desc=f"AIRL Epoch {epoch+1}/{num_irl_epochs}")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        rewards = batch["reward"].to(device)
        emotions = batch["emotion"].to(device)

        logits = reward_model(input_ids=input_ids, attention_mask=attention_mask, emotion_vec=emotions)
        loss = irl_criterion(logits.squeeze(-1), rewards)
        loss = loss / accumulation_steps
        loss.backward()

        total_loss += loss.item() * accumulation_steps
        count += 1
        accumulation_count += 1

        if accumulation_count % accumulation_steps == 0:
            irl_optimizer.step()
            irl_optimizer.zero_grad()
            if (batch_idx + 1) % (accumulation_steps * 5) == 0:
                print(f"[Epoch {epoch+1}] Batch {batch_idx+1} | Avg Loss: {total_loss / count:.4f}")

    if accumulation_count % accumulation_steps != 0:
        irl_optimizer.step()
        irl_optimizer.zero_grad()

    print(f"[AIRL] Epoch {epoch+1} complete | Avg Loss: {total_loss / count:.4f}")

# Save model
torch.save(reward_model.state_dict(), "airl_reward_model.pt")

# Upload to Hugging Face
api = HfApi(token=hf_token)
output_dir = f"./posterity_airl_{reward_model_ckpt.split('/')[-1]}"
reward_model.transformer.save_pretrained(output_dir)
torch.save(reward_model.emotion_proj.state_dict(), f"{output_dir}/emotion_proj.pt")

api.upload_large_folder(
    folder_path=output_dir,
    repo_id=f"aoxo/posterity_airl_{reward_model_ckpt.split('/')[-1]}",
    repo_type="model",
)

print(f"Model uploaded to https://huggingface.co/aoxo/posterity_airl_{reward_model_ckpt.split('/')[-1]}")