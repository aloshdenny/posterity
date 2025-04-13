import modal
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use the same image as in the training script with required dependencies
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
        "accelerate",
        "bitsandbytes",
        "sentencepiece"
    )
    .apt_install("gcc", "python3-dev")
)

app = modal.App("posterity-perplexity")

@app.function(gpu="H100", image=image, timeout=86400)
def calculate_perplexity():
    # List of original model names used during training
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
    
    # Dictionary to store perplexity results
    perplexity_results = {}

    hf_token = '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'

    # Define a sample text on which perplexity will be computed.
    # For more robust evaluation, consider using a dedicated evaluation dataset.
    test_text = "How are you feeling?"

    for model_name in models_to_train:
        # The model was saved with a repository name "joanne_experimental_lora_{model_id}"
        model_id = model_name.split("/")[-1]
        hf_model_name = f"joanne_experimental_lora_{model_id}"
        repo_path = f"aoxo/{hf_model_name}"
        
        print(f"Downloading model from repository: {repo_path} ...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                token = hf_token
            )
            tokenizer = AutoTokenizer.from_pretrained(repo_path, token = hf_token)
        except Exception as e:
            print(f"Error loading {repo_path}: {e}")
            continue

        # Ensure the tokenizer has a pad token (use eos_token if not present)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize the input text
        inputs = tokenizer(test_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)

        # Put model in evaluation mode and compute loss over the test text
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        perplexity = math.exp(loss.item())
        perplexity_results[hf_model_name] = perplexity
        print(f"Model: {hf_model_name} -- Perplexity: {perplexity:.2f}")

        # Clean up to free GPU memory
        del model
        torch.cuda.empty_cache()

    print("\nFinal Perplexity Results:")
    for name, ppl in perplexity_results.items():
        print(f"{name}: {ppl:.2f}")


@app.local_entrypoint()
def main():
    calculate_perplexity.remote()