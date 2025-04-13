import modal

# Create a Modal image with the required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "huggingface_hub",
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

app = modal.App("posterity-experimentals")

@app.function(gpu="H100", image=image, timeout=86400)
def train_and_upload():
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from huggingface_hub import hf_hub_download
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        AutoModelForCausalLM, 
        DataCollatorForLanguageModeling
    )
    from tqdm import tqdm
    import random

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 16

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
    # Utility: Compute Perplexity
    ############################################
    def compute_perplexity(model, dataloader):
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              labels=input_ids)
                loss = outputs.loss
                batch_tokens = attention_mask.sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss))
        model.train()
        return perplexity.item()

    ############################################
    # 1. IRL: Reward Model Training
    ############################################
    class DialogueDataset(Dataset):
        """Unified dataset class for all training stages"""
        def __init__(self, df, tokenizer, max_length=1024, mode="irl"):
            self.data = df
            self.speaker1, self.speaker2 = df.columns[:2]
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.mode = mode

            # Convert all text columns to strings and handle NaN values
            self.data[self.speaker1] = self.data[self.speaker1].astype(str)
            self.data[self.speaker2] = self.data[self.speaker2].astype(str)

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

             # For Qwen models specifically
            if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            prompt = str(self.data.loc[idx, self.speaker1])
            response = str(self.data.loc[idx, self.speaker2])
            
            if self.mode == "irl":
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
                    "reward": torch.tensor(1.0)
                }
            elif self.mode == "sft":
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
                    "attention_mask": encoded["attention_mask"].squeeze(0)
                }
            elif self.mode == "dpo":
                encoded_prompt = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors="pt"
                )
                encoded_response = self.tokenizer(
                    response,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors="pt"
                )
                return {
                    "prompt_input_ids": encoded_prompt["input_ids"].squeeze(0),
                    "prompt_attention_mask": encoded_prompt["attention_mask"].squeeze(0),
                    "response_input_ids": encoded_response["input_ids"].squeeze(0),
                    "response_attention_mask": encoded_response["attention_mask"].squeeze(0)
                }

    # Initialize reward model and tokenizer
    reward_model_ckpt = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_ckpt, token=hf_token)

    # Fix for padding token error
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
        if not hasattr(reward_tokenizer, 'pad_token') or reward_tokenizer.pad_token is None:
            reward_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    reward_dataset = DialogueDataset(dataset, reward_tokenizer, mode="irl")
    reward_dataloader = DataLoader(
        reward_dataset,
        batch_size=batch_size,  # Reduced batch size for stability
        shuffle=True,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'reward': torch.stack([x['reward'] for x in batch])
        }
    )

    # Initialize the reward model with proper configuration
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_ckpt, 
        num_labels=1, 
        trust_remote_code=True, 
        token=hf_token,
        ignore_mismatched_sizes=True  # This will suppress the warning
    )

    # Then properly initialize the classification head
    if hasattr(reward_model, 'score'):
        nn.init.normal_(reward_model.score.weight, std=0.02)
        if reward_model.score.bias is not None:
            nn.init.zeros_(reward_model.score.bias)
    reward_model.config.pad_token_id=50256
    reward_model.to(device)
    reward_model.train()

    # Fix for newly initialized weights warning
    for param in reward_model.parameters():
        if param.requires_grad:
            param.data.normal_(mean=0.0, std=0.02)

    # Configure for better SDPA handling
    reward_model.config.use_cache = False
    reward_model.config.pretraining_tp = 1
    reward_model.config._attn_implementation = "eager"

    irl_criterion = nn.MSELoss()
    irl_optimizer = optim.Adam(reward_model.parameters(), lr=1e-5)

    num_irl_epochs = 1
    print("Starting IRL Training...")
    for epoch in range(num_irl_epochs):
        total_loss = 0
        count = 0
        # Wrapping the batch iteration with tqdm
        for batch in tqdm(reward_dataloader, desc=f"IRL Epoch {epoch+1}/{num_irl_epochs}", total=len(reward_dataloader), ncols=100):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rewards = batch["reward"].to(device)

            irl_optimizer.zero_grad()
            outputs = reward_model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.squeeze(-1)
            loss = irl_criterion(predictions, rewards)
            loss.backward()
            irl_optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"[IRL] Epoch {epoch+1}/{num_irl_epochs} Loss: {avg_loss:.4f}")

    torch.save(reward_model.state_dict(), "reward_model.pt")

    ############################################
    # 2. SFT: Supervised Fine-Tuning of Policy Model
    ############################################
    policy_ckpt = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_ckpt, token=hf_token)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    if not hasattr(reward_tokenizer, 'pad_token') or reward_tokenizer.pad_token is None:
        reward_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    policy_model = AutoModelForCausalLM.from_pretrained(policy_ckpt)
    policy_model.config.pad_token_id=50256
    policy_model.to(device)
    policy_model.train()

    sft_dataset = DialogueDataset(dataset, policy_tokenizer, mode="sft")
    sft_dataloader = DataLoader(sft_dataset, batch_size=batch_size, shuffle=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=policy_tokenizer, mlm=False)
    sft_optimizer = optim.Adam(policy_model.parameters(), lr=5e-6)
    num_sft_epochs = 1

    print("Starting SFT Training...")
    # Inside the SFT Training Loop:
    for epoch in range(num_sft_epochs):
        total_loss = 0
        count = 0

        # Create batches using the data collator
        batch_indices = list(range(len(sft_dataset)))
        random.shuffle(batch_indices)

        # Wrapping the batch iteration with tqdm
        for i in tqdm(range(0, len(batch_indices), 4), desc=f"SFT Epoch {epoch+1}/{num_sft_epochs}", total=len(batch_indices)//4, ncols=100):
            batch_indices_chunk = batch_indices[i:i+4]
            batch_samples = [sft_dataset[idx] for idx in batch_indices_chunk]

            # Use the data collator to properly prepare the batch
            batch = data_collator(batch_samples)
            batch = {k: v.to(device) for k, v in batch.items()}

            sft_optimizer.zero_grad()
            outputs = policy_model(**batch)
            loss = outputs.loss
            loss.backward()
            sft_optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        ppl = compute_perplexity(policy_model, sft_dataloader, policy_tokenizer)
        print(f"[SFT] Epoch {epoch+1}/{num_sft_epochs} Loss: {avg_loss:.4f} | Perplexity: {ppl:.2f}")

    policy_model.save_pretrained("policy_model_sft")

    ############################################
    # 3. DPO: Direct Preference Optimization Training
    ############################################
    dpo_dataset = DialogueDataset(dataset, policy_tokenizer, mode="dpo")
    dpo_dataloader = DataLoader(dpo_dataset, batch_size=batch_size, shuffle=True)
    dpo_optimizer = optim.Adam(policy_model.parameters(), lr=5e-6)
    beta = 1.0
    num_dpo_epochs = 1

    def compute_log_probs(model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device),
                        labels=input_ids.to(device))
        return -outputs.loss

    print("Starting DPO Training...")
    # Inside the DPO Training Loop:
    for epoch in range(num_dpo_epochs):
        total_dpo_loss = 0
        count = 0
        # Wrapping the batch iteration with tqdm
        for batch in tqdm(dpo_dataloader, desc=f"DPO Epoch {epoch+1}/{num_dpo_epochs}", total=len(dpo_dataloader), ncols=100):
            dpo_optimizer.zero_grad()

            # Get prompt and preferred response
            prompts = policy_tokenizer.batch_decode(
                batch["prompt_input_ids"],
                skip_special_tokens=True
            )
            preferred = policy_tokenizer.batch_decode(
                batch["response_input_ids"],
                skip_special_tokens=True
            )

            # Preferred responses
            preferred_concat = [p + " " + r for p, r in zip(prompts, preferred)]
            encoded_pref = policy_tokenizer(
                preferred_concat,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors="pt"
            )
            logp_pref = compute_log_probs(
                policy_model,
                encoded_pref["input_ids"],
                encoded_pref["attention_mask"]
            )

            # Negative responses
            shuffled_ids = batch["response_input_ids"][torch.randperm(batch["response_input_ids"].size(0))]
            negatives = policy_tokenizer.batch_decode(shuffled_ids, skip_special_tokens=True)
            nonpref_concat = [p + " " + n for p, n in zip(prompts, negatives)]
            encoded_nonpref = policy_tokenizer(
                nonpref_concat,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors="pt"
            )
            logp_nonpref = compute_log_probs(
                policy_model,
                encoded_nonpref["input_ids"],
                encoded_nonpref["attention_mask"]
            )

            dpo_loss = torch.mean(torch.log1p(torch.exp(-beta * (logp_pref - logp_nonpref))))
            dpo_loss.backward()
            dpo_optimizer.step()

            total_dpo_loss += dpo_loss.item()
            count += 1

        avg_dpo_loss = total_dpo_loss / count if count > 0 else 0.0
        ppl = compute_perplexity(policy_model, dpo_dataloader, policy_tokenizer)
        print(f"[DPO] Epoch {epoch+1}/{num_dpo_epochs} Loss: {avg_dpo_loss:.4f} | Perplexity: {ppl:.2f}")

    policy_model.save_pretrained("policy_model_dpo")

@app.local_entrypoint()
def main():
    train_and_upload.remote()