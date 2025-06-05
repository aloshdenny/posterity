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
        "pillow",
    )
)

app = modal.App("posterity-gemma-3-12b-it")

@app.function(gpu="H100", image=image, timeout=86400)
def sft():
    import pandas as pd
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from huggingface_hub import hf_hub_download, HfApi
    from transformers import (
        AutoProcessor,
        AutoModelForCausalLM,
        get_linear_schedule_with_warmup
    )
    from tqdm import tqdm
    import numpy as np
    from peft import LoraConfig, get_peft_model
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HUB_ENABLE_HF_XET"] = "1"  # Enable XET for large model support
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"  # Disable telemetry for cleaner output
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    hf_token = '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'
    batch_size = 2
    accumulation_steps = 16
    model_ckpt = "google/gemma-3-12b-it"
    num_epochs = 10
    max_length = 1024
    num_workers = 0

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
    # Data: Load, Windowing, Tokenization (multi-turn)
    ############################################
    csv_path = hf_hub_download(
        repo_id="aoxo/mine", filename="conversation.csv",
        repo_type='dataset', token=hf_token
    )
    df = pd.read_csv(csv_path).fillna('')
    col1, col2 = df.columns[:2]
    input_speaker = "Alosh"    # Set your prompt speaker (user)
    target_speaker = "Joanne"  # Set your response persona (assistant)
    system_prompt = "You are Joanne, Alosh's girlfriend. Talk exactly like Joanne."  # Persona!

    window_size = 6  # Number of *turns* (not rows!) per training example

    # Flatten the dialogue into (speaker, text) tuples
    dialogue = []
    for idx, row in df.iterrows():
        if row[col1].strip():
            dialogue.append((col1, row[col1].strip()))
        if row[col2].strip():
            dialogue.append((col2, row[col2].strip()))

    # Create windows for SFT
    chat_windows = []
    for i in range(window_size, len(dialogue) + 1):
        window = dialogue[i - window_size:i]
        # Window must END with the target_speaker
        if window[-1][0] != target_speaker:
            continue
        # Must have at least one input_speaker turn in context
        user_turns = [t for t in window[:-1] if t[0] == input_speaker]
        if not user_turns:
            continue
        # Build message history
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]
        for speaker, text in window[:-1]:
            role = "user" if speaker == input_speaker else "assistant"
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})
        # Add target (Joanne's) turn as assistant for the final label
        target_text = window[-1][1]
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target_text}]})
        chat_windows.append(messages)

    print(f"Total multi-turn windows created: {len(chat_windows)}")

    # Tokenize each window
    processor = AutoProcessor.from_pretrained(model_ckpt, token=hf_token)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(messages):
        out = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            return_dict=True
        )
        return {
            "input_ids": out["input_ids"].squeeze(0),
            "attention_mask": out["attention_mask"].squeeze(0),
            "labels": out["input_ids"].squeeze(0),
        }

    import datasets
    raw_dataset = datasets.Dataset.from_list([{"messages": x} for x in chat_windows])
    tokenized_dataset = raw_dataset.map(
        lambda ex: tokenize_fn(ex["messages"]), batched=False, remove_columns=["messages"]
    )
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    # PyTorch DataLoaders
    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(batch):
        max_len = max_length
        input_ids = [torch.tensor(ex["input_ids"][:max_len]) for ex in batch]
        attention_mask = [torch.tensor(ex["attention_mask"][:max_len]) for ex in batch]
        labels = [torch.tensor(ex["labels"][:max_len]) for ex in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        # Truncate in case any are longer
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]
        labels = labels[:, :max_len]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, pin_memory=False, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, pin_memory=False, num_workers=num_workers
    )

    ############################################
    # Model & LoRA
    ############################################
    model = AutoModelForCausalLM.from_pretrained(model_ckpt, token=hf_token, attn_implementation="eager")
    model.config.pad_token_id = tokenizer.pad_token_id
    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16).to(device)
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

    ############################################
    # Training Loop (with Gradient Accumulation + AMP)
    ############################################
    print(f"\nStarting SFT Training with AMP & Multi-turn Roleplay Windows ...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()
        for step, batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps
            total_loss += loss.item() * accumulation_steps
            # Step optimizer & scheduler
            loss.backward()
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
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