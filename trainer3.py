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

app = modal.App("posterity-experimentals-2")

def clear_cuda_memory():
    import torch
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

@app.function(gpu="A100", image=image, timeout=86400)
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
        DataCollatorForLanguageModeling,
        RobertaTokenizer,
        RobertaForSequenceClassification
    )
    from tqdm import tqdm
    import random
    import numpy as np
    from peft import LoraConfig, get_peft_model
    import gc

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 4
    hf_token = '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'

    # Load and clean dataset
    csv_path = hf_hub_download(repo_id="aoxo/temp", filename="chat_conversation.csv", repo_type='dataset', token=hf_token)
    dataset = pd.read_csv(csv_path)
    dataset = dataset.fillna('')

    if len(dataset.columns) < 2:
        raise ValueError("Dataset must have at least 2 columns (speaker1 and speaker2)")
    speaker1, speaker2 = dataset.columns[:2]
    print(f"Dataset loaded with columns: {speaker1} and {speaker2}")

    ############################################
    # Utility: Compute Perplexity
    ############################################
    def compute_perplexity(model, dataloader, tokenizer):
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
    # Emotion Classifier Setup
    ############################################
    def load_emotion_model():
        emotion_tokenizer = RobertaTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        emotion_model = RobertaForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
        emotion_model.config.pad_token_id = 50256
        emotion_model.to(device)
        emotion_model.eval()
        return emotion_tokenizer, emotion_model

    def unload_emotion_model(emotion_model):
        del emotion_model
        clear_cuda_memory()

    def get_emotion_vector(text, emotion_model, emotion_tokenizer):
        if not text.strip():
            return np.zeros(28)
        
        inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = emotion_model(**inputs)
        logits = outputs.logits.squeeze()
        if len(logits.shape) == 0:
            logits = logits.unsqueeze(0)
        probabilities = torch.sigmoid(logits)
        return probabilities.cpu().numpy()

    ############################################
    # Dataset Class with Emotion Support
    ############################################
    class DialogueDataset(Dataset):
        def __init__(self, df, tokenizer, max_length=1024, mode="irl"):
            self.data = df
            self.speaker1, self.speaker2 = df.columns[:2]
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.mode = mode
            
            # Load emotion model temporarily for preprocessing
            emotion_tokenizer, emotion_model = load_emotion_model()
            
            self.data[self.speaker1] = self.data[self.speaker1].astype(str)
            self.data[self.speaker2] = self.data[self.speaker2].astype(str)
            
            self.emotion_vectors = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing emotion vectors"):
                speaker1_text = str(row[self.speaker1])
                speaker2_text = str(row[self.speaker2])
                emotion_vec = get_emotion_vector(speaker1_text + " " + speaker2_text, emotion_model, emotion_tokenizer)
                self.emotion_vectors.append(emotion_vec)
            
            # Clean up emotion model
            unload_emotion_model(emotion_model)
            
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
                    "reward": torch.tensor(1.0),
                    "emotion": emotion_vec
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
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "emotion": emotion_vec
                }
            elif self.mode == "dpo" or self.mode == "npo":
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
                    "response_attention_mask": encoded_response["attention_mask"].squeeze(0),
                    "emotion": emotion_vec
                }

    ############################################
    # 1. AIRL: Adversarial IRL with Emotion Vectors
    ############################################
    def train_airl():
        print("\n" + "="*50)
        print("Starting AIRL Training with Emotion Vectors...")
        print("="*50 + "\n")
        
        reward_model_ckpt = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_ckpt, token=hf_token)
        
        if reward_tokenizer.pad_token is None:
            reward_tokenizer.pad_token = reward_tokenizer.eos_token
        if not hasattr(reward_tokenizer, 'pad_token') or reward_tokenizer.pad_token is None:
            reward_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_ckpt, 
            num_labels=1, 
            trust_remote_code=True, 
            token=hf_token,
            ignore_mismatched_sizes=True
        )
        reward_model.config.pad_token_id = 50256
        
        reward_model.emotion_proj = nn.Linear(28, reward_model.config.hidden_size)
        reward_model.to(device)
        
        for param in reward_model.parameters():
            if param.requires_grad:
                param.data.normal_(mean=0.0, std=0.02)
        
        reward_model.config.use_cache = False
        reward_model.config.pretraining_tp = 1
        reward_model.config._attn_implementation = "eager"

        reward_dataset = DialogueDataset(dataset, reward_tokenizer, mode="irl")
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

        irl_criterion = nn.MSELoss()
        irl_optimizer = optim.Adam(reward_model.parameters(), lr=1e-5)
        num_irl_epochs = 1

        for epoch in range(num_irl_epochs):
            total_loss = 0
            count = 0
            for batch in tqdm(reward_dataloader, desc=f"AIRL Epoch {epoch+1}/{num_irl_epochs}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                rewards = batch["reward"].to(device)
                emotions = batch["emotion"].to(device)

                irl_optimizer.zero_grad()
                
                outputs = reward_model(input_ids=input_ids, attention_mask=attention_mask)
                emotion_features = reward_model.emotion_proj(emotions)
                modified_logits = outputs.logits + emotion_features.mean(dim=1, keepdim=True)
                
                loss = irl_criterion(modified_logits.squeeze(-1), rewards)
                loss.backward()
                irl_optimizer.step()

                total_loss += loss.item()
                count += 1

            avg_loss = total_loss / count
            print(f"[AIRL] Epoch {epoch+1}/{num_irl_epochs} Loss: {avg_loss:.4f}")

        torch.save(reward_model.state_dict(), "airl_reward_model.pt")
        
        # Clean up
        del reward_model, reward_tokenizer, reward_dataset, reward_dataloader
        clear_cuda_memory()

    ############################################
    # 2. SFT: Supervised Fine-Tuning with LoRA
    ############################################
    def train_sft():
        print("\n" + "="*50)
        print("Starting SFT Training with LoRA...")
        print("="*50 + "\n")
        
        policy_ckpt = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        policy_tokenizer = AutoTokenizer.from_pretrained(policy_ckpt, token=hf_token)
        if policy_tokenizer.pad_token is None:
            policy_tokenizer.pad_token = policy_tokenizer.eos_token

        policy_model = AutoModelForCausalLM.from_pretrained(policy_ckpt)
        policy_model.config.pad_token_id = 50256
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        policy_model = get_peft_model(policy_model, lora_config)
        policy_model.config.pad_token_id = 50256
        policy_model.to(device)
        policy_model.print_trainable_parameters()

        sft_dataset = DialogueDataset(dataset, policy_tokenizer, mode="sft")
        sft_dataloader = DataLoader(sft_dataset, batch_size=batch_size, shuffle=True)
        data_collator = DataCollatorForLanguageModeling(tokenizer=policy_tokenizer, mlm=False)
        sft_optimizer = optim.Adam(policy_model.parameters(), lr=5e-6)
        num_sft_epochs = 1

        for epoch in range(num_sft_epochs):
            total_loss = 0
            count = 0
            batch_indices = list(range(len(sft_dataset)))
            random.shuffle(batch_indices)

            for i in tqdm(range(0, len(batch_indices), batch_size), 
                          desc=f"SFT Epoch {epoch+1}/{num_sft_epochs}"):
                batch_indices_chunk = batch_indices[i:i+batch_size]
                batch_samples = [sft_dataset[idx] for idx in batch_indices_chunk]
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

        policy_model.save_pretrained("policy_model_sft_lora")
        
        # Clean up
        del policy_model, policy_tokenizer, sft_dataset, sft_dataloader
        clear_cuda_memory()

    ############################################
    # 3. DPO: Direct Preference Optimization with Emotion Vectors
    ############################################
    def train_dpo():
        print("\n" + "="*50)
        print("Starting DPO Training with Emotion Vectors...")
        print("="*50 + "\n")
        
        # Reload model for DPO
        policy_ckpt = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        policy_tokenizer = AutoTokenizer.from_pretrained(policy_ckpt, token=hf_token)
        if policy_tokenizer.pad_token is None:
            policy_tokenizer.pad_token = policy_tokenizer.eos_token

        policy_model = AutoModelForCausalLM.from_pretrained(policy_ckpt)
        policy_model.config.pad_token_id = 50256
        
        # Load LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        policy_model = get_peft_model(policy_model, lora_config)
        policy_model.config.pad_token_id = 50256
        policy_model.to(device)
        
        # Load from SFT checkpoint if available
        try:
            from peft import PeftModel
            policy_model = PeftModel.from_pretrained(policy_model, "policy_model_sft_lora")
            print("Loaded SFT LoRA weights successfully")
        except:
            print("Could not load SFT LoRA weights, training from scratch")

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

        for epoch in range(num_dpo_epochs):
            total_dpo_loss = 0
            count = 0
            for batch in tqdm(dpo_dataloader, desc=f"DPO Epoch {epoch+1}/{num_dpo_epochs}"):
                dpo_optimizer.zero_grad()

                prompts = policy_tokenizer.batch_decode(
                    batch["prompt_input_ids"],
                    skip_special_tokens=True
                )
                preferred = policy_tokenizer.batch_decode(
                    batch["response_input_ids"],
                    skip_special_tokens=True
                )
                emotions = batch["emotion"].to(device)

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

                # Negative responses (shuffled)
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

                # Emotion-guided DPO loss
                emotion_weights = emotions.mean(dim=1)
                dpo_loss = torch.mean(emotion_weights * torch.log1p(torch.exp(-beta * (logp_pref - logp_nonpref))))
                dpo_loss.backward()
                dpo_optimizer.step()

                total_dpo_loss += dpo_loss.item()
                count += 1

            avg_dpo_loss = total_dpo_loss / count
            ppl = compute_perplexity(policy_model, dpo_dataloader, policy_tokenizer)
            print(f"[DPO] Epoch {epoch+1}/{num_dpo_epochs} Loss: {avg_dpo_loss:.4f} | Perplexity: {ppl:.2f}")

        policy_model.save_pretrained("policy_model_dpo_emotion")
        
        # Clean up
        del policy_model, policy_tokenizer, dpo_dataset, dpo_dataloader
        clear_cuda_memory()

    ############################################
    # 4. NPO: Negative Policy Optimization
    ############################################
    def train_npo():
        print("\n" + "="*50)
        print("Starting NPO Training...")
        print("="*50 + "\n")
        
        # Reload model for NPO
        policy_ckpt = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        policy_tokenizer = AutoTokenizer.from_pretrained(policy_ckpt, token=hf_token)
        if policy_tokenizer.pad_token is None:
            policy_tokenizer.pad_token = policy_tokenizer.eos_token

        policy_model = AutoModelForCausalLM.from_pretrained(policy_ckpt)
        policy_model.config.pad_token_id = 50256
        
        # Load LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        policy_model = get_peft_model(policy_model, lora_config)
        policy_model.config.pad_token_id = 50256
        policy_model.to(device)
        
        # Load from DPO checkpoint if available
        try:
            from peft import PeftModel
            policy_model = PeftModel.from_pretrained(policy_model, "policy_model_dpo_emotion")
            print("Loaded DPO LoRA weights successfully")
        except:
            print("Could not load DPO LoRA weights, training from scratch")

        npo_dataset = DialogueDataset(dataset, policy_tokenizer, mode="npo")
        npo_dataloader = DataLoader(npo_dataset, batch_size=batch_size, shuffle=True)
        npo_optimizer = optim.Adam(policy_model.parameters(), lr=5e-6)
        num_npo_epochs = 1

        def compute_log_probs(model, input_ids, attention_mask):
            outputs = model(input_ids=input_ids.to(device),
                          attention_mask=attention_mask.to(device),
                          labels=input_ids.to(device))
            return -outputs.loss

        for epoch in range(num_npo_epochs):
            total_npo_loss = 0
            count = 0
            for batch in tqdm(npo_dataloader, desc=f"NPO Epoch {epoch+1}/{num_npo_epochs}"):
                npo_optimizer.zero_grad()

                prompts = policy_tokenizer.batch_decode(
                    batch["prompt_input_ids"],
                    skip_special_tokens=True
                )
                responses = policy_tokenizer.batch_decode(
                    batch["response_input_ids"],
                    skip_special_tokens=True
                )
                emotions = batch["emotion"].to(device)

                # Compute response log probs
                full_texts = [p + " " + r for p, r in zip(prompts, responses)]
                encoded = policy_tokenizer(
                    full_texts,
                    truncation=True,
                    max_length=256,
                    padding='max_length',
                    return_tensors="pt"
                )
                logp_responses = compute_log_probs(
                    policy_model,
                    encoded["input_ids"],
                    encoded["attention_mask"]
                )

                # Compute NPO loss (maximize divergence from negative examples)
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

                # Emotion-guided NPO loss
                emotion_weights = emotions.mean(dim=1)
                npo_loss = -torch.mean(emotion_weights * (logp_responses - logp_nonpref))
                npo_loss.backward()
                npo_optimizer.step()

                total_npo_loss += npo_loss.item()
                count += 1

            avg_npo_loss = total_npo_loss / count
            ppl = compute_perplexity(policy_model, npo_dataloader, policy_tokenizer)
            print(f"[NPO] Epoch {epoch+1}/{num_npo_epochs} Loss: {avg_npo_loss:.4f} | Perplexity: {ppl:.2f}")

        policy_model.save_pretrained("policy_model_npo_emotion")
        
        # Clean up
        del policy_model, policy_tokenizer, npo_dataset, npo_dataloader
        clear_cuda_memory()

    ############################################
    # Run all training stages sequentially
    ############################################
    train_airl()
    train_sft()
    train_dpo()
    train_npo()

@app.local_entrypoint()
def main():
    train_and_upload.remote()