import modal

# Modal image: add all necessary dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git", "gcc", "python3-dev")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers",
        "peft",
        "accelerate",
        "huggingface_hub[hf_xet]"
    )
)

app = modal.App("posterity-experimentals-inference")

@app.function(gpu="H100", image=image, timeout=1800)
def run_inference(prompt, max_new_tokens=128):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel
    import torch
    torch.set_float32_matmul_precision('high')

    # Configuration
    base_model = "google/gemma-3-1b-it"
    lora_repo = "aoxo/posterity_sft_gemma-3-12b-it"  # change if your repo name differs
    hf_token = "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"

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, token=hf_token, attn_implementation="eager"
    )
    model = model.to("cuda")

    # Load and apply LoRA/PEFT weights
    model = PeftModel.from_pretrained(model, lora_repo, token=hf_token)
    model = model.merge_and_unload()  # optional: merges adapters into base for faster inference

    # Inference
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.amp.autocast("cuda"):
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result

@app.local_entrypoint()
def main():
    prompt = "How do you feel joanne?"
    output = run_inference.remote(prompt)
    print("\n=== Generated Text ===\n", output)