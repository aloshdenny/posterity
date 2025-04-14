import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch
import wandb
from huggingface_hub import HfApi

# Authentication tokens
wandb_login = '70eb8127394ad088f9024024836193c92b58d46a'
hf_token = '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'

# Log in with your WandB token
wandb.login(key=wandb_login)

# List of models to fine-tune
models_to_train = [
    "Qwen/Qwen2.5-0.5B",
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2.5-1.5B",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "google/gemma-2-9b",
    "Qwen/Qwen2.5-14B",
    "google/gemma-2-27b",
]

# ------------------------
# DATA PREPARATION - Do this once for all models
# ------------------------

# Load and prepare the conversation dataset
print("Loading datasets...")
dataset = load_dataset("aoxo/temp", data_files="joanne.csv", token=hf_token)
df = dataset['train'].to_pandas()

# Load and prepare the audio transcriptions dataset
audio_transcriptions = load_dataset("aoxo/temp", data_files="joanne.json", token=hf_token)
audio_df = audio_transcriptions["train"].to_pandas()
audio_df = audio_df.melt(var_name="key", value_name="value")

# Process conversation data
def process_row(row):
    joe_lines = row["joe"].strip().split("\n") if pd.notna(row["joe"]) else []
    Joachii_lines = row["Joachii"].strip().split("\n") if pd.notna(row["Joachii"]) else []
    
    conversation_pairs = []
    for joe, Joachii in zip(joe_lines, Joachii_lines):
        conversation_pairs.append(
            f"<|user|> {joe.strip()} <|enduser|>\n<|assistant|> {Joachii.strip()} <|endassistant|>"
        )
    return conversation_pairs

# Apply the process_row function to each row
df["conversation_pairs"] = df.apply(process_row, axis=1)

# Flatten the conversation pairs
conversation_samples = [pair for pairs in df["conversation_pairs"] for pair in pairs]
dataset = Dataset.from_dict({"conversation": conversation_samples})

# Process audio transcription data
audio_conversation_samples = []
for row in audio_df.itertuples(index=False):
    transcript_text = row.value.strip()
    sample = (
        f"<|user|> <|enduser|>\n"
        f"<|assistant|> {transcript_text} <|endassistant|>"
    )
    audio_conversation_samples.append(sample)

audio_dataset = Dataset.from_dict({"conversation": audio_conversation_samples})

print("Datasets prepared successfully.")

# Define inference function (outside the training loop)
def ask_question(model, tokenizer, question, max_new_tokens=128):
    prompt = f"<|user|> {question} <|enduser|>\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    assistant_response = response.split("<|assistant|>")[-1].strip()
    return assistant_response

# Create HF API client for model upload
api = HfApi(token=hf_token)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ------------------------
# START MODEL TRAINING LOOP
# ------------------------
for model_name in models_to_train:
    print(f"Processing model: {model_name}")
    
    # Tokenize data for current model
    print(f"Tokenizing data for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=hf_token)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["conversation"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    # Tokenize both datasets
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_audio_dataset = audio_dataset.map(tokenize_function, batched=True)

    # Load model
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation='eager',
        token=hf_token
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    # Wrap model with LoRA adapter
    model = get_peft_model(model, lora_config)
    
    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # First training pass - conversation data
    model_id = model_name.split('/')[-1]
    output_dir = f"./joanne_experimental_lora_{model_id}"
    
    print(f"Starting first training pass for {model_name}...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        optim="adamw_torch",
        push_to_hub=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    
    # Second training pass - audio transcription data
    print(f"Starting second training pass for {model_name}...")
    second_output_dir = f"./joanne_experimental_lora_second_pass_{model_id}"
    
    second_training_args = TrainingArguments(
        output_dir=second_output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        logging_steps=10,
        optim="adamw_torch",
        fp16=True,
    )

    trainer_2 = Trainer(
        model=model,
        args=second_training_args,
        train_dataset=tokenized_audio_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer_2.train()

    # Save the final model
    print(f"Saving model {model_name}...")
    model.save_pretrained(output_dir)
    
    # Test the model
    prompt = "How do you feel"
    test_response = ask_question(model, tokenizer, prompt)
    print(prompt + "\n")
    print(f"{model_name}: {test_response}")

    # Upload to Hugging Face Hub
    print(f"Uploading model {model_name} to Hugging Face...")
    hf_model_name = f"joanne_experimental_lora_{model_id}"
    
    api.upload_large_folder(
        folder_path=output_dir,
        repo_id=f"aoxo/{hf_model_name}",
        repo_type="model",
    )
    
    print(f"Model uploaded to https://huggingface.co/aoxo/{hf_model_name}")
    print(f"Completed processing for {model_name}")
    
    # Free up GPU memory
    del model
    torch.cuda.empty_cache()

print("All models processed successfully")