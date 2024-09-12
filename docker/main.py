import os
import torch
import random
import logging
import re
from typing import List, Tuple

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments

from solcx import compile_source, install_solc
from solcx.exceptions import SolcError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_contract_examples(contracts_path: str) -> List[str]:
    """Load contract examples from a directory."""
    logger.info(f"Loading contract examples from: {contracts_path}")
    contract_examples = []

    for filename in os.listdir(contracts_path):
        if filename.endswith(".sol"):
            with open(os.path.join(contracts_path, filename), "r") as f:
                contract_examples.append(f.read())

    logger.info(f"Loaded {len(contract_examples)} contracts from {contracts_path}")
    return contract_examples


def add_pragma_and_imports(synthetic_contract: str) -> str:
    """Ensure correct pragma version, clean up imports, and ensure necessary elements are present."""
    
    # Define the correct pragma and import statements
    correct_pragma = "pragma solidity ^0.8.20;"
    spdx_license = "// SPDX-License-Identifier: UNLICENSED"
    
    # Path to where OpenZeppelin contracts are located in the container
    openzeppelin_path = 'import "../openzeppelin-contracts/openzeppelin-contracts-4.8.0/contracts/'
    
    # Remove incorrect pragma versions (e.g., `0.8.0` or any other)
    synthetic_contract = re.sub(r"pragma solidity\s+[^\;]+;", "", synthetic_contract)
    
    # Add the correct pragma at the top if it's not already there
    if correct_pragma not in synthetic_contract:
        synthetic_contract = correct_pragma + "\n" + synthetic_contract
    
    # Add the SPDX-License-Identifier at the top if it's not already there
    if spdx_license not in synthetic_contract:
        synthetic_contract = spdx_license + "\n" + synthetic_contract
    
    # Remove unnecessary multi-line comments (e.g., block comments) that appear at the start
    synthetic_contract = re.sub(r"/\*.*?\*/", "", synthetic_contract, flags=re.DOTALL)
    
    # Replace OpenZeppelin imports with the correct path
    synthetic_contract = re.sub(r'import\s+"@openzeppelin/contracts/', openzeppelin_path, synthetic_contract)
    
    # Ensure there are no duplicated imports (e.g., OpenZeppelin libraries)
    # List any required imports that should always be present
    required_imports = [
        f'{openzeppelin_path}token/ERC20/ERC20.sol";',
        f'{openzeppelin_path}access/Ownable.sol";'
    ]
    
    # Add each required import if it's not already in the contract
    for imp in required_imports:
        if imp not in synthetic_contract:
            synthetic_contract = imp + "\n" + synthetic_contract

    # Return the cleaned-up contract
    return synthetic_contract.strip()


def validate_contract(contract_code: str) -> Tuple[bool, str]:
    """Validate the Solidity contract code."""
    logger.info("Validating the contract...")
    try:
        compile_source(contract_code, output_values=["abi", "bin"])
        logger.info("Contract is valid.")
        return True, ""
    except SolcError as e:
        logger.error(f"Contract validation failed: {e}")
        return False, str(e)


def generate_synthetic_contract(model, tokenizer, prompt: str, max_length: int = 512) -> str:
    """Generate a synthetic contract using the given model and tokenizer."""
    logger.info("Generating synthetic contract...")
    device = next(model.parameters()).device

    # Providing a better prompt structure
    prompt = (
        "Generate an ERC20 token contract that includes the following functionality:\n"
        "- Basic token properties like name, symbol, and supply\n"
        "- Minting and burning functionality\n"
        "- Ownership and access control\n"
        "Here is the contract:\n"
        + prompt
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    
    # Log the generated output
    logger.info(f"Raw model output: {outputs}")

    synthetic_contract = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Ensure basic Solidity elements are present
    if "contract" not in synthetic_contract and "function" not in synthetic_contract:
        synthetic_contract += "\n\ncontract GeneratedContract {\n    function example() public {}\n}"

    return add_pragma_and_imports(synthetic_contract)


def prepare_dataset(contracts: List[str], tokenizer) -> Dataset:
    """Prepare dataset using the Hugging Face datasets library."""
    logger.info("Preparing the dataset...")
    
    # Define a preprocessing function
    def preprocess_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    # Create a dataset from a list of dictionaries
    raw_dataset = Dataset.from_dict({"text": contracts})
    
    # Tokenize the dataset
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)
    return tokenized_dataset


def setup_model_and_tokenizer(model_name: str):
    """Set up the model and tokenizer."""
    logger.info("Loading model and tokenizer...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically map model to devices
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"

    return model, tokenizer


def setup_training_config(output_dir: str):
    """Set up the training configuration."""
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Increased epochs to allow more fine-tuning
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
    )

    return peft_config, training_arguments


def finetune_model(model, dataset, peft_config, training_arguments, tokenizer):
    """Fine-tune the model."""
    logger.info("Starting the fine-tuning process...")
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=False,
        )
        trainer.train()
        logger.info("Fine-tuning completed successfully")
        return trainer.model
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise


def generate_contracts(model, tokenizer, contract_examples, num_contracts):
    """Generate valid contracts."""
    valid_contracts = []

    while len(valid_contracts) < num_contracts:
        logger.info(f"Generating contract {len(valid_contracts) + 1}/{num_contracts}...")
        prompt = random.choice(contract_examples)[:512]
        synthetic_contract = generate_synthetic_contract(model, tokenizer, prompt)

        logger.info("Generated Contract:")
        logger.info(synthetic_contract)
        logger.info("-" * 50)

        is_valid, error_message = validate_contract(synthetic_contract)

        if is_valid:
            valid_contracts.append(synthetic_contract)
            logger.info(f"Contract {len(valid_contracts)} is valid.")
        else:
            logger.info(f"Contract is invalid. Error: {error_message}")

    return valid_contracts


def main():
    """Main function to run the Smart Contract Generator."""
    logger.info("Smart Contract Generator started")

    # Environment setup
    install_solc(version="0.8.20")
    seed = int(os.environ.get("SEED", 42))
    num_contracts = int(os.environ.get("NUM_CONTRACTS", 1))
    token_standard = os.environ.get("TOKEN_STANDARD", "ERC20")

    logger.info(
        f"Environment Variables - SEED: {seed}, "
        f"NUM_CONTRACTS: {num_contracts}, TOKEN_STANDARD: {token_standard}"
    )

    random.seed(seed)
    torch.manual_seed(seed)

    # Paths and configurations
    model_name = "/app/model"
    contracts_path = "/app/dataset/contracts"
    output_dir = "/app/results"

    # Setup model, tokenizer, and configurations
    model, tokenizer = setup_model_and_tokenizer(model_name)
    peft_config, training_arguments = setup_training_config(output_dir)

    # Load dataset
    contract_examples = load_contract_examples(contracts_path)

    # Prepare dataset using Hugging Face dataset format
    dataset = prepare_dataset(contract_examples, tokenizer)

    # Fine-tune the model
    finetuned_model = finetune_model(
        model, dataset, peft_config, training_arguments, tokenizer
    )

    # Generate contracts using the fine-tuned model
    valid_contracts = generate_contracts(
        finetuned_model, tokenizer, contract_examples, num_contracts
    )

    logger.info(f"Generated {len(valid_contracts)} valid contracts.")

    # Optional: Save the fine-tuned model
    finetuned_model.save_pretrained(os.path.join(output_dir, "finetuned_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "finetuned_model"))


if __name__ == "__main__":
    main()
