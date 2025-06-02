import modal

# Modal image definition
image = (
    modal.Image.debian_slim()
    .apt_install("git", "gcc", "python3-dev")
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
    )
)

app = modal.App("posterity-experimentals-sft-improved")

@app.function(gpu="A100", image=image, timeout=86400)
def sft():
    import pandas as pd
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from huggingface_hub import hf_hub_download, HfApi
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        DataCollatorForLanguageModeling,
        get_linear_schedule_with_warmup
    )
    from tqdm import tqdm
    import numpy as np
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset as HFDataset
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    hf_token = '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'
    batch_size = 4
    accumulation_steps = 8
    model_ckpt = "google/gemma-3-1b-it"
    num_epochs = 3
    max_length = 1024
    num_workers = 4

    ############################################
    # Utility: Compute Perplexity
    ############################################
    def compute_perplexity(model, dataloader):
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                tokens = batch["attention_mask"].sum().item()
                total_loss += loss.item() * tokens
                total_tokens += tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss))
        model.train()
        return perplexity.item()

    ############################################
    # Data: Load and Pre-tokenize
    ############################################
    csv_path = hf_hub_download(
        repo_id="aoxo/mine", filename="conversation.csv",
        repo_type='dataset', token=hf_token
    )
    dataset_df = pd.read_csv(csv_path).fillna('')
    if len(dataset_df.columns) < 2:
        raise ValueError("Dataset must have at least 2 columns")
    print(f"Loaded CSV with columns: {dataset_df.columns[:2]}")
    speaker1, speaker2 = dataset_df.columns[:2]

    # Convert DataFrame to HuggingFace Dataset
    hf_dataset = HFDataset.from_pandas(
        dataset_df[[speaker1, speaker2]].rename(
            columns={speaker1: "prompt", speaker2: "response"}
        )
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-tokenize the dataset (fast!)
    def tokenize_fn(example):
        text = (example["prompt"] or "") + " " + (example["response"] or "")
        return tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    tokenized_dataset = hf_dataset.map(tokenize_fn, batched=False, remove_columns=["prompt", "response"])
    # Split into train/val
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    # DataCollator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if tokenizer.pad_token_id is not None else None
    )

    # PyTorch DataLoaders (use num_workers, pin_memory)
    def collate_fn(batch):
        return data_collator(batch)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: data_collator(batch), pin_memory=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: data_collator(batch), pin_memory=True, num_workers=num_workers
    )

    ############################################
    # Model & LoRA
    ############################################
    model = AutoModelForCausalLM.from_pretrained(model_ckpt, token=hf_token, attn_implementation="eager")
    model.config.pad_token_id = tokenizer.pad_token_id
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    ############################################
    # Optimizer and LR Scheduler
    ############################################
    optimizer = optim.AdamW(model.parameters(), lr=5e-6)
    total_steps = num_epochs * (len(train_dataloader) // accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 20),
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler()  # For AMP

    ############################################
    # Training Loop (with Gradient Accumulation + AMP)
    ############################################
    print(f"\nStarting SFT Training with AMP & Pre-tokenized Data ...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()
        for step, batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps
            scaler.scale(loss).backward()
            total_loss += loss.item() * accumulation_steps
            # Step optimizer & scheduler
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            if (step + 1) % 10 == 0:
                avg_loss = total_loss / (step + 1)
                progress_bar.set_postfix({'loss': f"{avg_loss:.4f}", 'lr': optimizer.param_groups[0]['lr']})

        # Validation
        val_ppl = compute_perplexity(model, val_dataloader)
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1} Results: Train Loss={avg_train_loss:.4f} | Val Perplexity={val_ppl:.2f}")

    ############################################
    # Save and Upload
    ############################################
    api = HfApi(token=hf_token)
    model_id = model_ckpt.split('/')[-1]
    output_dir = f"./posterity_sft_{model_id}"
    print(f"Saving model {model_ckpt} to {output_dir} ...")
    model.save_pretrained(output_dir)
    print(f"Uploading model to HuggingFace ...")
    hf_model_name = f"posterity_sft_{model_id}"
    api.upload_large_folder(
        folder_path=output_dir,
        repo_id=f"aoxo/{hf_model_name}",
        repo_type="model",
    )
    print(f"Model uploaded: https://huggingface.co/aoxo/{hf_model_name}")

@app.local_entrypoint()
def main():
    sft.remote()