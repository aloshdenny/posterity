import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from tqdm import tqdm
import pandas as pd
from huggingface_hub import hf_hub_download
from peft import PeftModel, PeftConfig

# Config
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

hf_token = "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"
batch_size = 1
accumulation_steps = 16
beta = 0.1  # DPO temperature parameter

# Load models
base_sft_model_path = "google/gemma-3-1b-it"  # Base SFT model
sft_model_path = "aoxo/posterity_sft_gemma-3-1b-it"  # Your SFT-tuned model
airl_model_path = "Qwen/Qwen2.5-0.5B"  # Your AIRL reward model

class AIRLRewardModel(nn.Module):
    def __init__(self, base_model_ckpt, hf_token=None):
        super().__init__()
        self.llm = AutoModelForSequenceClassification.from_pretrained(
            base_model_ckpt,
            num_labels=1,
            trust_remote_code=True,
            token=hf_token
        )
        
        # Initialize the classification head properly
        if hasattr(self.llm, 'score'):
            nn.init.normal_(self.llm.score.weight, mean=0.0, std=0.02)
            if self.llm.score.bias is not None:
                nn.init.zeros_(self.llm.score.bias)
        elif hasattr(self.llm, 'classifier'):
            if hasattr(self.llm.classifier, 'weight'):
                nn.init.normal_(self.llm.classifier.weight, mean=0.0, std=0.02)
            if hasattr(self.llm.classifier, 'bias') and self.llm.classifier.bias is not None:
                nn.init.zeros_(self.llm.classifier.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

    @property
    def config(self):
        return self.llm.config

def load_dataset():
    csv_path = hf_hub_download(repo_id="aoxo/temp", filename="chat_conversation.csv", 
                            repo_type='dataset', token=hf_token)
    df = pd.read_csv(csv_path).fillna('')
    if len(df.columns) < 2:
        raise ValueError("Dataset needs at least 2 columns")
    return df

class DPODataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024):
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = str(self.data.iloc[idx, 0])
        response = str(self.data.iloc[idx, 1])
        
        # Tokenize separately for DPO
        prompt_enc = self.tokenizer(
            prompt, 
            truncation=True, 
            max_length=self.max_length//2, 
            return_tensors="pt"
        )
        response_enc = self.tokenizer(
            response,
            truncation=True,
            max_length=self.max_length//2,
            return_tensors="pt"
        )
        
        return {
            "prompt_input_ids": prompt_enc["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_enc["attention_mask"].squeeze(0),
            "response_input_ids": response_enc["input_ids"].squeeze(0),
            "response_attention_mask": response_enc["attention_mask"].squeeze(0)
        }

# Load models
print("Loading models...")

tokenizer = AutoTokenizer.from_pretrained(base_sft_model_path, token=hf_token)
base_model = AutoModelForCausalLM.from_pretrained(base_sft_model_path, token=hf_token).to(device)
peft_config = PeftConfig.from_pretrained(sft_model_path)
policy_model = PeftModel.from_pretrained(base_model, sft_model_path)
print(f"PEFT model loaded with {policy_model.num_parameters()} parameters")

reward_model = AIRLRewardModel(airl_model_path, hf_token=hf_token).to(device)

# Load dataset
df = load_dataset()
dpo_dataset = DPODataset(df, tokenizer)
dpo_loader = DataLoader(dpo_dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = optim.Adam(policy_model.parameters(), lr=5e-6)

# Training loop
print("Starting DPO Training...")
for epoch in range(3):  # 3 epochs
    policy_model.train()
    total_loss = 0
    accumulation_count = 0
    
    for batch_idx, batch in enumerate(tqdm(dpo_loader)):
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 1. Get log probabilities for policy responses
        policy_outputs = policy_model(
            input_ids=batch["response_input_ids"],
            attention_mask=batch["response_attention_mask"],
            labels=batch["response_input_ids"]
        )
        policy_log_probs = -policy_outputs.loss
        
        # 2. Get rewards from AIRL model
        with torch.no_grad():
            rewards = reward_model(
                input_ids=batch["response_input_ids"],
                attention_mask=batch["response_attention_mask"]
            ).squeeze(-1)

        # 3. Advantage Calculation
        advantages = rewards - rewards.mean()  # Simple advantage
        
        # 4. DPO Loss with Advantage Weighting
        # Get log probs for shuffled "negative" responses
        shuffled_ids = batch["response_input_ids"][torch.randperm(batch["response_input_ids"].size(0))]
        neg_outputs = policy_model(
            input_ids=shuffled_ids,
            attention_mask=batch["response_attention_mask"],
            labels=shuffled_ids
        )
        neg_log_probs = -neg_outputs.loss
        
        # Advantage-weighted DPO loss
        log_ratio = policy_log_probs - neg_log_probs
        loss = -torch.mean(advantages * torch.log(torch.sigmoid(beta * log_ratio))) / accumulation_steps
        loss.backward()
        
        total_loss += loss.item() * accumulation_steps
        accumulation_count += 1
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"Batch {batch_idx+1} | Loss: {total_loss/(batch_idx+1):.4f}")
    
    print(f"Epoch {epoch+1} Complete | Avg Loss: {total_loss/len(dpo_loader):.4f}")

# Save final model
policy_model.save_pretrained("dpo_trained_model")
print("DPO training complete!")