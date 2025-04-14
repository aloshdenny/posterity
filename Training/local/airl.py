import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download, HfApi
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification
)
from tqdm import tqdm
import numpy as np

# Set device to GPU if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

batch_size = 7
hf_token = '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'

# Load and clean dataset
csv_path = hf_hub_download(repo_id="aoxo/temp", filename="chat_conversation.csv", repo_type='dataset', token=hf_token)
dataset = pd.read_csv(csv_path)

# Fill NaN values with empty strings
dataset = dataset.fillna('')

# Verify dataset structure
if len(dataset.columns) < 2:
    raise ValueError("Dataset must have at least 2 columns (speaker1 and speaker2)")
speaker1, speaker2 = dataset.columns[:2]
print(f"Dataset loaded with columns: {speaker1} and {speaker2}")

############################################
# Emotion Classifier Setup
############################################
emotion_tokenizer = RobertaTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
emotion_model = RobertaForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
emotion_model.config.pad_token_id=50256
emotion_model.to(device)
emotion_model.eval()

def get_emotion_vector(text):
    if not text.strip():  # Handle empty strings
        return np.zeros(28)  # 28 emotions in go_emotions
    
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    logits = outputs.logits.squeeze()
    if len(logits.shape) == 0:  # Handle single sample case
        logits = logits.unsqueeze(0)
    probabilities = torch.sigmoid(logits)
    return probabilities.cpu().numpy()

############################################
# Dataset Class with Emotion Support
############################################
class DialogueDataset(Dataset):
    """Dataset class for AIRL reward modeling"""
    def __init__(self, df, tokenizer, max_length=1024):
        self.data = df
        self.speaker1, self.speaker2 = df.columns[:2]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Convert all text columns to strings and handle NaN values
        self.data[self.speaker1] = self.data[self.speaker1].astype(str)
        self.data[self.speaker2] = self.data[self.speaker2].astype(str)
        
        # Precompute emotion vectors
        self.emotion_vectors = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing emotion vectors"):
            speaker1_text = str(row[self.speaker1])
            speaker2_text = str(row[self.speaker2])
            emotion_vec = get_emotion_vector(speaker1_text + " " + speaker2_text)
            self.emotion_vectors.append(emotion_vec)
        
        # Set padding token if not set
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
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "reward": torch.tensor(1.0),  # Using 1.0 as positive reward for all examples
            "emotion": emotion_vec
        }

############################################
# AIRL: Adversarial IRL with Emotion Vectors
############################################
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
        self.transformer.config.use_cache = False
        self.transformer.config.pretraining_tp = 1
        self.transformer.config._attn_implementation = "eager"

    def forward(self, input_ids, attention_mask, emotion_vec=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        if emotion_vec is not None:
            emotion_features = self.emotion_proj(emotion_vec)
            logits = logits + emotion_features.mean(dim=1, keepdim=True)

        return logits

reward_model_ckpt = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_ckpt, token=hf_token)

# Fix tokenizer
if reward_tokenizer.pad_token is None:
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
if not hasattr(reward_tokenizer, 'pad_token') or reward_tokenizer.pad_token is None:
    reward_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Initialize reward model
# reward_model = AutoModelForSequenceClassification.from_pretrained(
#     reward_model_ckpt, 
#     num_labels=1, 
#     trust_remote_code=True, 
#     token=hf_token,
#     ignore_mismatched_sizes=True
# )

# Add emotion projection layer
# reward_model.emotion_proj = nn.Linear(28, reward_model.transformer.config.hidden_size)  # 28 emotions in go_emotions
reward_model = AIRLRewardModel(reward_model_ckpt, hf_token=hf_token)
reward_model.transformer.config.pad_token_id=50256
reward_model.to(device)

# Initialize weights
for param in reward_model.parameters():
    if param.requires_grad:
        param.data.normal_(mean=0.0, std=0.02)

reward_model.transformer.config.use_cache = False
reward_model.transformer.config.pretraining_tp = 1
reward_model.transformer.config._attn_implementation = "eager"

# Prepare dataset and dataloader
reward_dataset = DialogueDataset(dataset, reward_tokenizer)
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

# AIRL training setup
irl_criterion = nn.MSELoss()
irl_optimizer = optim.Adam(reward_model.parameters(), lr=1e-5)
num_irl_epochs = 3  # You can adjust this
accumulation_steps = 4  # Accumulate gradients over 16 steps

print("Starting AIRL Training with Gradient Accumulation...")
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

        # Forward pass
        # outputs = reward_model(input_ids=input_ids, attention_mask=attention_mask)
        # emotion_features = reward_model.emotion_proj(emotions)
        # modified_logits = outputs.logits + emotion_features.mean(dim=1, keepdim=True)
        modified_logits = reward_model(input_ids=input_ids, attention_mask=attention_mask, emotion_vec=emotions)
        loss = irl_criterion(modified_logits.squeeze(-1), rewards)
        
        # Normalize the loss to account for accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        total_loss += loss.item() * accumulation_steps  # Scale back up for reporting
        count += 1
        accumulation_count += 1

        # Only step and zero gradients after accumulation steps
        if accumulation_count % accumulation_steps == 0:
            irl_optimizer.step()
            irl_optimizer.zero_grad()
            
            # Print progress
            if (batch_idx + 1) % (accumulation_steps * 5) == 0:  # Print every 5 accumulation steps
                avg_loss_so_far = total_loss / count
                print(f"[Epoch {epoch+1}] Batch {batch_idx+1} | Avg Loss: {avg_loss_so_far:.4f}")

    # Handle remaining gradients if not divisible by accumulation_steps
    if accumulation_count % accumulation_steps != 0:
        irl_optimizer.step()
        irl_optimizer.zero_grad()

    avg_loss = total_loss / count
    print(f"[AIRL] Epoch {epoch+1}/{num_irl_epochs} Completed | Avg Loss: {avg_loss:.4f}")

# Save the trained reward model
torch.save(reward_model.state_dict(), "airl_reward_model_accum.pt")

############################################
# Test the Reward Function
############################################
def get_reward(text, emotion_vec=None):
    """Compute reward for a given text and optional emotion vector"""
    # Tokenize input
    inputs = reward_tokenizer(
        text,
        truncation=True,
        max_length=1024,
        padding='max_length',
        return_tensors="pt"
    ).to(device)
    
    # Convert emotion vector if provided
    if emotion_vec is None:
        emotion_vec = get_emotion_vector(text)
    emotions = torch.tensor(emotion_vec, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get reward
    with torch.no_grad():
        # outputs = reward_model(input_ids=inputs["input_ids"], 
        #                     attention_mask=inputs["attention_mask"])
        # emotion_features = reward_model.emotion_proj(emotions)
        # modified_logits = outputs.logits + emotion_features.mean(dim=1, keepdim=True)
        modified_logits = reward_model(input_ids=input_ids, attention_mask=attention_mask, emotion_vec=emotions)
        reward = modified_logits.squeeze().item()
    
    return reward

# Test the reward function with some examples
test_texts = [
    "Hello, how are you today?",
    "I'm feeling great! The weather is wonderful.",
    "I hate everything about this situation.",
    "This is the best day of my life!"
]

print("\nTesting Reward Function:")
for text in test_texts:
    reward = get_reward(text)
    print(f"Text: {text[:50]}... | Reward: {reward:.4f}")

api = HfApi(token=hf_token)

model_id = reward_model_ckpt.split('/')[-1]
output_dir = f"./posterity_airl_{model_id}"

# Upload to Hugging Face Hub
print(f"Uploading model {model_id} to Hugging Face...")
hf_model_name = f"posterity_airl_{model_id}"

print(f"Saving model {reward_model_ckpt}...")
reward_model.transformer.save_pretrained(output_dir)
torch.save(reward_model.emotion_proj.state_dict(), f"{output_dir}/emotion_proj.pt")

api.upload_large_folder(
    folder_path=output_dir,
    repo_id=f"aoxo/{hf_model_name}",
    repo_type="model",
)

print(f"Model uploaded to https://huggingface.co/aoxo/{hf_model_name}")