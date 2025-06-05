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

app = modal.App("posterity-experimentals-airl")

@app.function(gpu="H100", image=image, timeout=86400)
def adversarial_inverse_rl():
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from huggingface_hub import hf_hub_download, HfApi
    from tqdm import tqdm
    from torch.amp import GradScaler, autocast

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        num_workers = 0
        pin_memory = False
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        num_workers = 2
        pin_memory = True
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Configuration
    batch_size = 2
    hf_token = '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'
    accumulation_steps = 4
    num_irl_epochs = 5
    learning_rate = 1e-4

    # Load dataset
    csv_path = hf_hub_download(repo_id="aoxo/temp", filename="chat_conversation.csv", repo_type='dataset', token=hf_token)
    dataset = pd.read_csv(csv_path).fillna('')
    speaker1, speaker2 = dataset.columns[:2]
    print(f"Loaded dataset with {len(dataset)} examples")

    # Dataset class
    class DialogueDataset(Dataset):
        def __init__(self, df, tokenizer, max_length=1024):
            self.data = df
            self.tokenizer = tokenizer
            self.max_length = max_length
            
            # Ensure tokenizer has padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            prompt = str(self.data.iloc[idx, 0])
            response = str(self.data.iloc[idx, 1])
            
            # Combine prompt and response with separator
            text = f"{prompt} {self.tokenizer.sep_token} {response}"
            
            # Tokenize with truncation and padding
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
                "reward": torch.tensor(1.0)  # Default reward (will be learned)
            }

    # Simplified AIRL model
    class AIRLRewardModel(nn.Module):
        def __init__(self, base_model_ckpt, hf_token=None):
            super().__init__()
            self.llm = AutoModelForSequenceClassification.from_pretrained(
                base_model_ckpt,
                num_labels=1,  # Single scalar reward output
                trust_remote_code=True,
                token=hf_token
            )
            self.llm.config.pad_token_id = 50256
            self.llm.config.use_cache = True
            
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
            # Return raw logits (unbounded rewards)
            return outputs.logits

    # Initialize model and tokenizer
    reward_model_ckpt = "Qwen/Qwen2.5-0.5B"
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_ckpt, token=hf_token)
    reward_model = AIRLRewardModel(reward_model_ckpt, hf_token=hf_token).to(device)

    # DataLoader
    reward_dataset = DialogueDataset(dataset, reward_tokenizer)
    reward_dataloader = DataLoader(
        reward_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'reward': torch.stack([x['reward'] for x in batch])
        }
    )

    # Training setup
    irl_criterion = nn.MSELoss()
    irl_optimizer = optim.AdamW(reward_model.parameters(), lr=learning_rate)

    # Training loop
    print("Starting AIRL Training...")
    for epoch in range(num_irl_epochs):
        reward_model.train()
        total_loss = 0
        total_samples = 0
        
        progress_bar = tqdm(reward_dataloader, desc=f"Epoch {epoch+1}/{num_irl_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rewards = batch["reward"].to(device)
            
            # Forward pass
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                predicted_rewards = reward_model(input_ids, attention_mask).squeeze(-1)
                loss = irl_criterion(predicted_rewards, rewards)
            
            # Backward pass with gradient accumulation
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            
            # Update metrics
            batch_size = input_ids.size(0)
            total_loss += loss.item() * accumulation_steps * batch_size
            total_samples += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / total_samples,
                'avg_reward': predicted_rewards.mean().item()
            })
            
            # Step optimizer
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(irl_optimizer)
                scaler.update()
                irl_optimizer.zero_grad()
        
        print(f"Epoch {epoch+1} complete | Avg Loss: {total_loss / total_samples:.4f}")

    # Save and upload model
    output_dir = f"./posterity_airl_{reward_model_ckpt.split('/')[-1]}"
    reward_model.llm.save_pretrained(output_dir)

    api = HfApi(token=hf_token)
    api.upload_large_folder(
        folder_path=output_dir,
        repo_id=f"aoxo/posterity_airl_{reward_model_ckpt.split('/')[-1]}",
        repo_type="model",
    )

    print(f"Model uploaded to: https://huggingface.co/aoxo/posterity_airl_{reward_model_ckpt.split('/')[-1]}")

@app.local_entrypoint()
def main():
    adversarial_inverse_rl.remote()