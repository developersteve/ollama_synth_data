import os
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from solcx import compile_source, install_solc, set_solc_version
import re

# Install Solidity compiler
print("Installing Solidity compiler...")
install_solc(version="0.8.20")

# Seed, number of contracts, and token standard as environment inputs
seed = int(os.environ.get("SEED", 42))  # Default to 42 if SEED is not provided
num_contracts = int(os.environ.get("NUM_CONTRACTS", 1))  # Default to 1 contract if NUM_CONTRACTS is not provided
token_standard = os.environ.get("TOKEN_STANDARD", "ERC-20")  # Default to ERC-20 if TOKEN_STANDARD is not provided

print(f"Environment Variables - SEED: {seed}, NUM_CONTRACTS: {num_contracts}, TOKEN_STANDARD: {token_standard}")

# Set random seed
random.seed(seed)
torch.manual_seed(seed)

# Use local paths for model and dataset
model_name = "/app/model"
contracts_path = "/app/dataset/contracts"  # Directory where ERC-20 contracts are stored

print(f"Using model path: {model_name}")
print(f"Using contract path: {contracts_path}")

# bitsandbytes parameters
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# Load model and tokenizer
print("Loading model and tokenizer...")
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Model and tokenizer loaded successfully.")

# Function to load Solidity contracts from the directory
def load_contract_examples(path_to_contracts: str):
    print("Loading contract examples from:", path_to_contracts)
    contract_texts = []
    for filename in os.listdir(path_to_contracts):
        if filename.endswith(".sol"):  # Only load Solidity files
            with open(os.path.join(path_to_contracts, filename), 'r') as file:
                contract_text = file.read()
                contract_texts.append(contract_text)
    print(f"Loaded {len(contract_texts)} contracts from {path_to_contracts}")
    return contract_texts

# Function to clean and add necessary pragma and imports
def add_pragma_and_imports(synthetic_contract):
    pragma_statement = "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\n\n"
    if not synthetic_contract.startswith("pragma solidity"):
        synthetic_contract = pragma_statement + synthetic_contract

    # Clean up unnecessary or incomplete comment blocks (e.g., /** ... */)
    synthetic_contract = re.sub(r'/\*\*.*?\*/', '', synthetic_contract, flags=re.DOTALL)

    return synthetic_contract

# Create a custom dataset for fine-tuning
class SolidityDataset(Dataset):
    def __init__(self, contracts, tokenizer, max_length=512):
        self.contracts = contracts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.contracts)

    def __getitem__(self, idx):
        contract = self.contracts[idx]
        tokenized = self.tokenizer(
            contract,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Set the labels equal to the input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["labels"].squeeze()  # Labels for loss computation
        }

# Fine-tune the model using the loaded contracts
def finetune_model(model, tokenizer, contract_examples):
    print("Preparing the dataset for fine-tuning...")
    dataset = SolidityDataset(contract_examples, tokenizer)

    print("Starting the fine-tuning process...")
    training_args = TrainingArguments(
        output_dir="/app/results",
        num_train_epochs=1,
        per_device_train_batch_size=1,  # Keep batch size low for memory efficiency
        gradient_accumulation_steps=4,  # Adjust this to maintain effective batch size
        save_strategy="no",  # Disable saving during training
        logging_dir="/app/logs",
        logging_steps=10,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    print("Fine-tuning completed.")
    return model

# Function to generate synthetic smart contract
def generate_synthetic_contract(model, tokenizer, prompt, max_length=512):
    print("Generating synthetic contract...")
    device = model.device  # Get the device the model is using (GPU or CPU)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    synthetic_contract = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return add_pragma_and_imports(synthetic_contract)

# Function to validate smart contract using py-solc-x
def validate_contract(contract_code: str):
    print("Validating contract...")
    try:
        set_solc_version('0.8.20')
        compiled_sol = compile_source(contract_code, output_values=["abi", "bin"])
        print("Contract validation successful.")
        return True, None
    except Exception as e:
        print(f"Contract validation failed: {e}")
        return False, str(e)

# Function to categorize errors for targeted fine-tuning
def categorize_error(error_message):
    if "pragma" in error_message or "import" in error_message:
        return "pragma_or_import_issue"
    elif "primary expression" in error_message or re.search(r'!+', error_message):
        return "syntax_noise"  # Handling invalid characters like '!'
    return "general_error"

# Function to perform basic syntactical cleaning
def basic_syntax_check(contract_code):
    # Allow more flexibility: only reject contracts with extreme issues like excessive repeated characters
    if re.search(r'!{5,}', contract_code):  # Check if there are more than 5 consecutive '!'
        return False
    # Ensure there is no hanging multi-line comment
    if re.search(r'/\*\*[^*]*$', contract_code):  # Incomplete comment block
        return False
    return True

# Load the ERC-20 contract examples from the specified directory
contract_examples = load_contract_examples(contracts_path)

# Fine-tune the model using the contract examples
model = finetune_model(model, tokenizer, contract_examples)

# Generate the required number of valid contracts
valid_contracts = []
generated_count = 0
error_feedback = []

while len(valid_contracts) < num_contracts:
    print(f"Generating contract {generated_count + 1}/{num_contracts}...")
    prompt = random.choice(contract_examples)[:512]
    synthetic_contract = generate_synthetic_contract(model, tokenizer, prompt)

    # Perform basic syntax validation before compiling
    if not basic_syntax_check(synthetic_contract):
        print("Contract skipped due to obvious syntax issues.")
        continue

    # Validate the synthetic contract
    is_valid, error_message = validate_contract(synthetic_contract)

    if is_valid:
        valid_contracts.append(synthetic_contract)
        generated_count += 1
        print(f"Contract {generated_count} is valid.")
    else:
        error_type = categorize_error(error_message)
        error_feedback.append((error_type, synthetic_contract))
        print(f"Contract {generated_count + 1} is invalid. Error: {error_message}")

        # Skip fine-tuning if the error is due to syntax noise like repeated '!'
        if error_type == "syntax_noise":
            print("Skipping fine-tuning due to noise (syntax).")
            continue

        # Fine-tune the model based on error feedback
        if len(error_feedback) > 0:
            feedback_contracts = [entry[1] for entry in error_feedback if entry[0] == error_type]
            model = finetune_model(model, tokenizer, feedback_contracts)

print(f"Generated {len(valid_contracts)} valid contracts.")
