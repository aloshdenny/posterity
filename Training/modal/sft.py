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

@app.function(gpu="A100", image=image, timeout=86400)
def sft():
    import pandas as pd
    import torch
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from huggingface_hub import hf_hub_download, HfApi
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        DataCollatorForLanguageModeling,
    )
    from tqdm import tqdm
    import numpy as np
    from peft import LoraConfig, get_peft_model

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    hf_token = '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'
    batch_size = 4  # Actual batch size
    accumulation_steps = 8  # Effective batch size = batch_size * accumulation_steps
    model_ckpt = "google/gemma-3-1b-it"

    ############################################
    # Utility: Compute Perplexity
    ############################################
    def compute_perplexity(model, dataloader, tokenizer):
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device and ensure we have attention_mask
                batch = {k: v.to(device) for k, v in batch.items()}
                if 'attention_mask' not in batch:
                    batch['attention_mask'] = (batch['input_ids'] != tokenizer.pad_token_id).float()
                
                outputs = model(**batch)
                loss = outputs.loss
                batch_tokens = batch['attention_mask'].sum().item()
                
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss))
        model.train()
        return perplexity.item()

    ############################################
    # Dataset Class
    ############################################
    class DialogueDataset(Dataset):
        def __init__(self, df, tokenizer, max_length=1024):
            self.data = df
            self.speaker1, self.speaker2 = df.columns[:2]
            self.tokenizer = tokenizer
            self.max_length = max_length
            
            # Convert all text columns to strings and handle NaN values
            self.data[self.speaker1] = self.data[self.speaker1].astype(str)
            self.data[self.speaker2] = self.data[self.speaker2].astype(str)
            
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
            
            # Combine prompt and response for SFT
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
            }

    ############################################
    # Main Training Script
    ############################################
    # Load dataset
    csv_path = hf_hub_download(repo_id="aoxo/temp", filename="chat_conversation.csv", 
                            repo_type='dataset', token=hf_token)
    dataset = pd.read_csv(csv_path).fillna('')
    
    if len(dataset.columns) < 2:
        raise ValueError("Dataset must have at least 2 columns")
    print(f"Dataset loaded with columns: {dataset.columns[:2]}")

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_ckpt, token=hf_token)
    model.config.pad_token_id = 50256

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8 if tokenizer.pad_token_id is not None else None)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    # Prepare dataset and dataloader
    train_dataset = DialogueDataset(dataset, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    
    # Validation split (10%)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    num_epochs = 3

    print(f"\nStarting SFT Training with Gradient Accumulation (batch_size={batch_size}, accum_steps={accumulation_steps})...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        accumulation_count = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps  # Normalize loss
            loss.backward()
            
            total_loss += loss.item() * accumulation_steps  # Scale back for reporting
            accumulation_count += 1

            # Optimizer step and zero gradients
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                
                # Update progress bar
                avg_loss_so_far = total_loss / (step + 1)
                progress_bar.set_postfix({
                    'loss': f"{avg_loss_so_far:.4f}",
                    'lr': optimizer.param_groups[0]['lr']
                })

        # End of epoch validation
        val_ppl = compute_perplexity(model, val_dataloader, tokenizer)
        avg_train_loss = total_loss / len(train_dataloader)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Perplexity: {val_ppl:.2f}")

    # Save final model
    api = HfApi(token=hf_token)

    model_id = model_ckpt.split('/')[-1]
    output_dir = f"./posterity_sft_{model_id}"

    # Upload to Hugging Face Hub
    print(f"Uploading model {model_id} to Hugging Face...")
    hf_model_name = f"posterity_sft_{model_id}"

    print(f"Saving model {model_ckpt}...")
    model.save_pretrained(output_dir)
    
    api.upload_large_folder(
        folder_path=output_dir,
        repo_id=f"aoxo/{hf_model_name}",
        repo_type="model",
    )
    
    print(f"Model uploaded to https://huggingface.co/aoxo/{hf_model_name}")

@app.local_entrypoint()
def main():
    sft.remote()