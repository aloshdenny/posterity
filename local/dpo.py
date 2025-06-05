import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
    # Consider adding DataCollatorWithPadding if padding issues arise
    # from transformers import DataCollatorWithPadding
)
from tqdm import tqdm
import pandas as pd
from huggingface_hub import hf_hub_download
from peft import PeftModel, PeftConfig
import warnings

# Suppress specific warnings if needed (optional)
warnings.filterwarnings("ignore", message=".*PAD token.*") # Example

# --- Config ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# --- Parameters ---
hf_token = "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" # Replace with your actual token if needed
batch_size = 1
accumulation_steps = 16
beta = 0.1  # DPO temperature parameter
learning_rate = 5e-6
num_epochs = 3
max_length = 1024 # Max sequence length for tokenizers

# Define model paths
base_sft_model_path = "google/gemma-3-1b-it"  # Base SFT model (used for policy)
sft_model_path = "aoxo/posterity_sft_gemma-3-1b-it"  # Your SFT-tuned model (policy)
airl_model_path = "Qwen/Qwen2.5-0.5B"  # Your AIRL reward model

# --- Reward Model Definition ---
class AIRLRewardModel(nn.Module):
    def __init__(self, base_model_ckpt, hf_token=None):
        super().__init__()
        self.llm = AutoModelForSequenceClassification.from_pretrained(
            base_model_ckpt,
            num_labels=1,
            trust_remote_code=True,
            token=hf_token
        )

        # Initialize the classification head properly (attempt common attribute names)
        output_layer = None
        if hasattr(self.llm, 'score'):
             output_layer = self.llm.score
        elif hasattr(self.llm, 'classifier'):
             output_layer = self.llm.classifier
        elif hasattr(self.llm, 'value_head'): # Another possible name
            output_layer = self.llm.value_head

        if output_layer is not None and hasattr(output_layer, 'weight'):
            print(f"Initializing weights for layer: {output_layer}")
            nn.init.normal_(output_layer.weight, mean=0.0, std=0.02)
            if hasattr(output_layer, 'bias') and output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias)
        else:
            print("Warning: Could not find standard classification head ('score', 'classifier', 'value_head') to initialize.")


    def forward(self, input_ids, attention_mask):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

    @property
    def config(self):
        return self.llm.config

# --- Dataset Loading Function ---
def load_dataset(hf_token=None):
    try:
        csv_path = hf_hub_download(repo_id="aoxo/temp", filename="chat_conversation.csv",
                                   repo_type='dataset', token=hf_token)
        df = pd.read_csv(csv_path).fillna('')
        if len(df.columns) < 2:
            raise ValueError("Dataset needs at least 2 columns (prompt, response)")
        print(f"Loaded dataset with {len(df)} rows.")
        # Optional: Rename columns for clarity if needed
        # df = df.rename(columns={df.columns[0]: 'prompt', df.columns[1]: 'response'})
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

# --- DPO Dataset Class (Modified) ---
class DPODataset(Dataset):
    def __init__(self, df, policy_tokenizer, reward_tokenizer, max_length=1024):
        """
        Initializes the DPODataset.

        Args:
            df (pd.DataFrame): DataFrame containing 'prompt' and 'response' columns.
                                Assumes first column is prompt, second is response.
            policy_tokenizer: Tokenizer for the policy model.
            reward_tokenizer: Tokenizer for the reward model.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.data = df
        self.policy_tokenizer = policy_tokenizer
        self.reward_tokenizer = reward_tokenizer # Store reward tokenizer
        self.max_length = max_length
        self.prompt_col_idx = 0 # Assuming first column is prompt
        self.response_col_idx = 1 # Assuming second column is response

        # Ensure padding tokens are set (handled outside before instantiation)
        if self.policy_tokenizer.pad_token is None:
             raise ValueError("Policy tokenizer must have a pad token set.")
        if self.reward_tokenizer.pad_token is None:
             raise ValueError("Reward tokenizer must have a pad token set.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = str(self.data.iloc[idx, self.prompt_col_idx])
        response = str(self.data.iloc[idx, self.response_col_idx])

        # --- Tokenization ---
        # Adjust max_length allocations if needed, e.g., allow more for response
        prompt_max_len = self.max_length // 3
        response_max_len = self.max_length - prompt_max_len # Or a fixed value

        # Tokenize prompt (only needed once, using policy tokenizer)
        prompt_enc = self.policy_tokenizer(
            prompt,
            truncation=True,
            max_length=prompt_max_len,
            return_tensors="pt",
            # padding='max_length', # Optional: Pad prompts too if needed downstream
            # return_attention_mask=True
        )

        # Tokenize response with POLICY tokenizer
        policy_response_enc = self.policy_tokenizer(
            response,
            truncation=True,
            max_length=response_max_len,
            return_tensors="pt",
            # padding='max_length', # Consider padding for consistent tensor shapes if using default dataloader
            # return_attention_mask=True
        )

        # Tokenize response with REWARD tokenizer
        # Use a potentially different max_length if reward model handles longer sequences
        reward_response_enc = self.reward_tokenizer(
            response, # Tokenize the same text response
            truncation=True,
            max_length=self.max_length, # Reward model might see full context or just response
            return_tensors="pt",
            # padding='max_length', # Consider padding
            # return_attention_mask=True
        )

        # --- Prepare Output Dictionary ---
        # .squeeze(0) removes the batch dimension added by return_tensors="pt"
        return {
            # Prompt (optional, for reference or KL term)
            "prompt_input_ids": prompt_enc["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_enc["attention_mask"].squeeze(0),

            # Response tokenized for POLICY model
            "policy_response_input_ids": policy_response_enc["input_ids"].squeeze(0),
            "policy_response_attention_mask": policy_response_enc["attention_mask"].squeeze(0),

            # Response tokenized for REWARD model
            "reward_response_input_ids": reward_response_enc["input_ids"].squeeze(0),
            "reward_response_attention_mask": reward_response_enc["attention_mask"].squeeze(0),
        }

# --- Main Script ---

# --- Load Tokenizers ---
print("Loading tokenizers...")
policy_tokenizer = AutoTokenizer.from_pretrained(base_sft_model_path, token=hf_token)
reward_tokenizer = AutoTokenizer.from_pretrained(airl_model_path, token=hf_token)

# --- Set Padding Tokens ---
# Important: Ensure pad tokens are set BEFORE creating the Dataset instance
if policy_tokenizer.pad_token is None:
    policy_tokenizer.pad_token = policy_tokenizer.eos_token
    print(f"Set policy_tokenizer pad_token to eos_token ({policy_tokenizer.eos_token})")
if reward_tokenizer.pad_token is None:
    reward_tokenizer.pad_token = reward_tokenizer.eos_token # Use EOS is common, check model card if unsure
    print(f"Set reward_tokenizer pad_token to eos_token ({reward_tokenizer.eos_token})")

# Optional: Set padding side (check reward model docs, often 'left' for reward models)
# policy_tokenizer.padding_side = 'right' # Default for causal LMs
# reward_tokenizer.padding_side = 'left'
# print(f"Policy tokenizer padding side: {policy_tokenizer.padding_side}")
# print(f"Reward tokenizer padding side: {reward_tokenizer.padding_side}")


# --- Load Models ---
print("Loading models...")
# Load base model for PEFT
base_model = AutoModelForCausalLM.from_pretrained(
    base_sft_model_path,
    token=hf_token,
    # torch_dtype=torch.bfloat16 # Optional: Use lower precision if memory is an issue
).to(device)

# Load PEFT model (Policy)
try:
    policy_model = PeftModel.from_pretrained(base_model, sft_model_path)
    print(f"PEFT policy model loaded from {sft_model_path}")
    # Ensure the merged model is on the correct device
    policy_model = policy_model.to(device)
except Exception as e:
    print(f"Could not load PEFT model from {sft_model_path}. Using base SFT model {base_sft_model_path} as policy. Error: {e}")
    policy_model = base_model # Fallback to base SFT model if PEFT fails


# Load Reward Model
reward_model = AIRLRewardModel(airl_model_path, hf_token=hf_token).to(device)
reward_model.eval() # Set reward model to evaluation mode

# --- Load Dataset and DataLoader ---
df = load_dataset(hf_token=hf_token)
dpo_dataset = DPODataset(df, policy_tokenizer, reward_tokenizer, max_length=max_length)

# Consider using a DataCollator for dynamic padding, especially if sequences vary significantly
# data_collator = DataCollatorWithPadding(tokenizer=policy_tokenizer) # Or handle padding manually
dpo_loader = DataLoader(
    dpo_dataset,
    batch_size=batch_size,
    shuffle=True,
    # collate_fn=data_collator # Add if using DataCollator
)

print("--- Checking Trainable Parameters in policy_model ---")
trainable_params = 0
all_param = 0
for name, param in policy_model.named_parameters():
    all_param += param.numel()
    param.requires_grad = True
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"Trainable: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
print(f"Trainable params: {trainable_params} || All params: {all_param} || Trainable %: {100 * trainable_params / all_param:.2f}")
print("--- End Check ---")

if trainable_params == 0:
   raise ValueError("CRITICAL: No trainable parameters found in the policy model. Check PEFT loading and model configuration.")

# --- Optimizer ---
# Ensure optimizer only gets parameters that require gradients
optimizer = optim.Adam(filter(lambda p: p.requires_grad, policy_model.parameters()), lr=learning_rate)

# --- Optimizer ---
# optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)

# --- Training Loop ---
print("Starting DPO Training...")
total_steps = len(dpo_loader) // accumulation_steps * num_epochs
print(f"Total training steps: {total_steps}")

global_step = 0
for epoch in range(num_epochs):
    policy_model.train() # Set policy model to training mode
    epoch_loss = 0.0
    optimizer.zero_grad() # Clear gradients at the start of epoch and after each optimizer step

    progress_bar = tqdm(enumerate(dpo_loader), total=len(dpo_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, batch in progress_bar:
        # Move batch to device
        try:
            batch = {k: v.to(device) for k, v in batch.items()}
        except Exception as e:
            print(f"\nError moving batch {batch_idx} to device: {e}")
            print("Skipping batch.")
            continue # Skip problematic batch

        # --- Forward Pass ---
        # 1. Get log probabilities for policy responses using POLICY tokens
        try:
            policy_outputs = policy_model(
                input_ids=batch["policy_response_input_ids"],
                attention_mask=batch["policy_response_attention_mask"],
                labels=batch["policy_response_input_ids"] # Labels are the same for Causal LM loss
            )
            # Log probability of the chosen response under the policy model
            # The loss returned by CausalLM is the negative log likelihood (NLL)
            # So, log_prob = -NLL = -loss
            policy_log_probs = -policy_outputs.loss
        except Exception as e:
            print(f"\nError during policy model forward pass (chosen): {e}")
            print(f"Input shapes: {batch['policy_response_input_ids'].shape}")
            continue # Skip batch


        # 2. Get rewards from Reward model using REWARD tokens
        with torch.no_grad(): # No gradients needed for reward model
            try:
                rewards = reward_model(
                    input_ids=batch["reward_response_input_ids"],
                    attention_mask=batch["reward_response_attention_mask"]
                ).squeeze(-1) # Remove the last dimension (num_labels=1)
            except Exception as e:
                print(f"\nError during reward model forward pass: {e}")
                print(f"Input shapes: {batch['reward_response_input_ids'].shape}")
                # Check vocab size vs max input ID
                max_id = batch["reward_response_input_ids"].max().item()
                vocab_size = reward_model.config.vocab_size
                print(f"Max ID in reward input: {max_id}, Reward model vocab size: {vocab_size}")
                if max_id >= vocab_size:
                     print("INDEX ERROR DETECTED: Max input ID exceeds reward model vocabulary size!")
                continue # Skip batch


        # 3. Advantage Calculation (Simplified: reward - mean reward)
        # For single sample batches (batch_size=1), mean is just the reward itself, advantage is 0.
        # This simple advantage works better with larger batch sizes or across accumulated steps.
        # Consider using a baseline or accumulated reward stats for better advantage estimation.
        if rewards.numel() > 1:
             advantages = rewards - rewards.mean()
        else:
             advantages = torch.zeros_like(rewards) # Advantage is 0 for batch_size=1 with this method


        # 4. DPO Loss Calculation
        # Get log probs for "negative" examples (using shuffled responses here as a simple proxy)
        # Ensure shuffling doesn't pick the same sample if batch size allows
        if batch["policy_response_input_ids"].size(0) > 1:
             perm_indices = torch.randperm(batch["policy_response_input_ids"].size(0))
             # Avoid shuffling an item to its original position if possible in small batches
             # This simple shuffle might still compare a sample to itself if batch_size=1
             # A better approach uses explicit chosen/rejected pairs if available
        else:
             perm_indices = torch.tensor([0], device=device) # No shuffling possible for batch_size=1

        # Select the shuffled policy inputs and masks
        shuffled_policy_ids = batch["policy_response_input_ids"][perm_indices]
        shuffled_policy_mask = batch["policy_response_attention_mask"][perm_indices]

        try:
            # Calculate log probs for the shuffled ("negative") responses using the policy model
            neg_outputs = policy_model(
                input_ids=shuffled_policy_ids,
                attention_mask=shuffled_policy_mask,
                labels=shuffled_policy_ids
            )
            neg_log_probs = -neg_outputs.loss
        except Exception as e:
             print(f"\nError during policy model forward pass (negative): {e}")
             print(f"Input shapes: {shuffled_policy_ids.shape}")
             continue # Skip batch


        # Advantage-weighted DPO-like loss (using advantage instead of implicit preference)
        # Standard DPO: loss = -log_sigmoid(beta * (log_prob_chosen - log_prob_rejected))
        # Here, we weight by advantage: loss = -advantage * log_sigmoid(beta * (log_prob_chosen - log_prob_negative))
        # Note: Ensure advantage sign makes sense (positive advantage -> increase log_prob_chosen relative to log_prob_negative)

        log_ratio = policy_log_probs - neg_log_probs # log(pi_policy(chosen) / pi_policy(negative))

        # We want to increase likelihood of high-advantage responses and decrease for low-advantage ones.
        # Sigmoid term ranges (0, 1). Log sigmoid ranges (-inf, 0).
        # If advantage > 0: We want log_ratio to be large. log_sigmoid becomes less negative (closer to 0).
        #                   We multiply by positive advantage. Loss becomes less negative (moves towards 0). Minimizing loss achieves this.
        # If advantage < 0: We want log_ratio to be small. log_sigmoid becomes more negative.
        #                   We multiply by negative advantage. Loss becomes less negative (moves towards 0). Minimizing loss achieves this.
        # This seems consistent. Need to average over the batch.

        # Loss per sample
        loss = -torch.log(torch.sigmoid(beta * log_ratio))

        # Weight by advantages (using .detach() as advantages shouldn't require gradients themselves)
        # This step needs careful consideration based on the exact RL algorithm interpretation (e.g., Actor-Critic, PPO-clip like)
        # Simple weighting:
        weighted_loss = advantages.detach() * loss

        # Normalize loss by accumulation steps
        normalized_loss = weighted_loss.mean() / accumulation_steps # Average over batch, then normalize

        # --- Backward Pass and Accumulation ---
        normalized_loss.backward()
        epoch_loss += normalized_loss.item() * accumulation_steps # Accumulate the un-normalized batch loss


        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping (optional but recommended)
            # torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)

            optimizer.step() # Update weights
            optimizer.zero_grad() # Reset gradients for the next accumulation cycle
            global_step += 1

            # Log accumulated loss
            avg_acc_loss = epoch_loss / (batch_idx + 1) # Average loss per batch so far in epoch
            progress_bar.set_postfix({
                "loss": f"{avg_acc_loss:.4f}",
                "rewards_mean": f"{rewards.mean().item():.2f}" # Log mean reward of last batch
            })

    # --- End of Epoch ---
    avg_epoch_loss = epoch_loss / len(dpo_loader)
    print(f"\nEpoch {epoch+1} Complete | Average Epoch Loss: {avg_epoch_loss:.4f}")

# --- Save Final Model ---
print("DPO training complete! Saving final policy model...")
policy_model.save_pretrained("dpo_trained_policy_model")
# Optionally save the tokenizer too
policy_tokenizer.save_pretrained("dpo_trained_policy_model")
print("Model saved to 'dpo_trained_policy_model'")