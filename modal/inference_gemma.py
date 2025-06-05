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
def run_inference(
    user_message,
    system_prompt="You are Joanne, Alosh's girlfriend. Talk exactly like Joanne.",
    history=None,  # list of (role, message) pairs
    max_new_tokens=128
):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoProcessor
    from peft import PeftModel
    import torch
    torch.set_float32_matmul_precision('high')

    # Configuration
    base_model = "google/gemma-3-12b-it"  # match SFT base
    lora_repo = "aoxo/posterity_sft_gemma-3-12b-it"  # your LoRA fine-tuned repo
    hf_token = "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"

    # Load base model and processor/tokenizer
    processor = AutoProcessor.from_pretrained(base_model, token=hf_token)
    tokenizer = processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model, token=hf_token, attn_implementation="eager"
    ).to("cuda")

    # Load and apply LoRA/PEFT weights
    model = PeftModel.from_pretrained(model, lora_repo, token=hf_token)
    model = model.merge_and_unload()  # merges adapters for speed

    # Build chat messages (matching SFT)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]
    if history:
        for role, msg in history:
            messages.append({
                "role": role,  # "user" or "assistant"
                "content": [{"type": "text", "text": msg}]
            })
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": user_message}]
    })

    # Tokenize using chat template (as in SFT)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,   # leave the prompt open for gen
        tokenize=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt",
        return_dict=True
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Inference
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

    # Optionally: Extract only the new assistant response
    # (Gemma-style templates often echo the full context; you may need to split after the last user message)
    return result

@app.local_entrypoint()
def main():
    prompt = "How do you feel, Joanne?"
    output = run_inference.remote(prompt)
    print("\n=== Generated Text ===\n", output)