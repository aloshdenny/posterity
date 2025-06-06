import modal

# Create a Modal image with the required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "huggingface_hub[hf_xet]",
        "pandas",
        "wandb",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "sentencepiece"
    )
    .apt_install("gcc", "python3-dev")
)

app = modal.App("posterity-experimentals-dpo")

@app.function(gpu="H100", image=image, timeout=86400)
def dpo():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        RobertaTokenizer,
        RobertaForSequenceClassification
    )
    from tqdm import tqdm
    import os
    import pandas as pd
    from huggingface_hub import hf_hub_download

    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_token = "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"
    batch_size = 1
    accumulation_steps = 16
    beta = 0.1  # DPO temperature parameter

    # Load models
    sft_model_path = "aoxo/posterity_sft_gemma-3-1b-it"  # Your SFT-tuned model
    airl_model_path = "aoxo/posterity_airl_DeepSeek-R1-Distill-Qwen-1.5B"  # Your AIRL reward model

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

        @property
        def config(self):
            return self.transformer.config

    ############################################
    # Load Dataset (Same as Before)
    ############################################
    def load_dataset():
        csv_path = hf_hub_download(repo_id="aoxo/temp", filename="chat_conversation.csv", 
                                repo_type='dataset', token=hf_token)
        df = pd.read_csv(csv_path).fillna('')
        if len(df.columns) < 2:
            raise ValueError("Dataset needs at least 2 columns")
        return df

    ############################################
    # Emotion Utilities (Same as Before)
    ############################################
    def init_emotion_model():
        model = RobertaForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions").to(device)
        tokenizer = RobertaTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        model.eval()
        return model, tokenizer

    def get_emotion_vector(text, emotion_model, emotion_tokenizer):
        inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = emotion_model(**inputs)
        return torch.sigmoid(outputs.logits).cpu().numpy()

    ############################################
    # DPO Dataset Class
    ############################################
    class DPODataset(Dataset):
        def __init__(self, df, tokenizer, emotion_model, emotion_tokenizer, max_length=1024):
            self.data = df
            self.tokenizer = tokenizer
            self.emotion_model = emotion_model
            self.emotion_tokenizer = emotion_tokenizer
            self.max_length = max_length
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            prompt = str(self.data.iloc[idx, 0])
            response = str(self.data.iloc[idx, 1])
            
            # Get emotion vector
            full_text = prompt + " " + response
            emotion_vec = get_emotion_vector(full_text, self.emotion_model, self.emotion_tokenizer)
            
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
                "response_attention_mask": response_enc["attention_mask"].squeeze(0),
                "emotion": torch.tensor(emotion_vec, dtype=torch.float32)
            }

    ############################################
    # A2C-Enhanced DPO Training
    ############################################
    # Load models
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, token=hf_token)
    policy_model = AutoModelForCausalLM.from_pretrained(sft_model_path, token=hf_token).to(device)
    # reward_model = AutoModelForSequenceClassification.from_pretrained(airl_model_path, token=hf_token).to(device)
    reward_model = AIRLRewardModel(airl_model_path, hf_token=hf_token).to(device)
    emotion_proj_path = hf_hub_download(repo_id=airl_model_path, filename="emotion_proj.pt", token=hf_token)
    reward_model.emotion_proj.load_state_dict(torch.load(emotion_proj_path, map_location=device))

    # Emotion model
    emotion_model, emotion_tokenizer = init_emotion_model()

    # Load dataset
    df = load_dataset()
    dpo_dataset = DPODataset(df, tokenizer, emotion_model, emotion_tokenizer)
    dpo_loader = DataLoader(dpo_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = optim.Adam(policy_model.parameters(), lr=5e-6)

    # Training loop
    print("Starting DPO-A2C Training...")
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
            
            # 2. Get rewards from AIRL model (with emotion)
            with torch.no_grad():
                reward_outputs = reward_model(
                    input_ids=batch["response_input_ids"],
                    attention_mask=batch["response_attention_mask"]
                )
                # rewards = reward_outputs.logits.squeeze(-1)
                
                # # Emotion modulation
                # rewards += torch.matmul(batch["emotion"], reward_model.emotion_proj.weight.T).mean(dim=1)
                rewards = reward_model(
                    input_ids=batch["response_input_ids"],
                    attention_mask=batch["response_attention_mask"],
                    emotion_vec=batch["emotion"]
                ).squeeze(-1)

            # 3. A2C Advantage Calculation
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
    policy_model.save_pretrained("dpo_a2c_trained_model")
    print("DPO-A2C training complete!")

@app.local_entrypoint()
def main():
    dpo.remote()