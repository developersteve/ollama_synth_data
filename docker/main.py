import os
import random
import logging
import re
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
)
from solcx import compile_source, set_solc_version
from peft import LoraConfig
from trl import SFTTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_contract_examples(path_to_contracts: str) -> List[str]:
    """Load Solidity contract examples from a directory."""
    logger.info(f"Loading contract examples from: {path_to_contracts}")
    contract_texts = []
    for filename in os.listdir(path_to_contracts):
        if filename.endswith(".sol"):
            with open(os.path.join(path_to_contracts, filename), "r") as file:
                contract_texts.append(file.read())
    logger.info(f"Loaded {len(contract_texts)} contracts from {path_to_contracts}")
    return contract_texts

def add_pragma_and_imports(synthetic_contract: str) -> str:
    """Ensure correct pragma version, clean up imports, and ensure necessary elements are present."""
    
    correct_pragma = "pragma solidity ^0.8.20;"
    spdx_license = "// SPDX-License-Identifier: UNLICENSED"
    
    openzeppelin_path = 'import "../openzeppelin-contracts/openzeppelin-contracts-4.8.0/contracts/'
    
    synthetic_contract = re.sub(r"pragma solidity\s+[^\;]+;", "", synthetic_contract)
    
    if correct_pragma not in synthetic_contract:
        synthetic_contract = correct_pragma + "\n" + synthetic_contract
    
    if spdx_license not in synthetic_contract:
        synthetic_contract = spdx_license + "\n" + synthetic_contract
    
    synthetic_contract = re.sub(r"/\*.*?\*/", "", synthetic_contract, flags=re.DOTALL)
    
    synthetic_contract = re.sub(r'import\s+"@openzeppelin/contracts/', openzeppelin_path, synthetic_contract)
    
    required_imports = [
        f'{openzeppelin_path}token/ERC20/ERC20.sol";',
        f'{openzeppelin_path}access/Ownable.sol";'
    ]
    
    for imp in required_imports:
        if imp not in synthetic_contract:
            synthetic_contract = imp + "\n" + synthetic_contract

    return synthetic_contract.strip()

class SolidityDataset(Dataset):
    """Dataset for Solidity contracts."""

    def __init__(self, contracts: List[str], tokenizer, max_length: int = 512):
        self.contracts = contracts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.contracts)

    def __getitem__(self, idx: int) -> dict:
        contract = self.contracts[idx]
        tokenized = self.tokenizer(
            contract,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

def generate_synthetic_contract(
    model, tokenizer, prompt: str, max_length: int = 512
) -> str:
    """Generate a synthetic contract using the given model and tokenizer."""
    logger.info("Generating synthetic contract...")
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)

    synthetic_contract = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return add_pragma_and_imports(synthetic_contract)

def validate_contract(contract_code: str) -> Tuple[bool, str]:
    """Validate a Solidity contract."""
    logger.info("Validating contract...")
    try:
        set_solc_version("0.8.20")
        compile_source(contract_code, output_values=["abi", "bin"])
        logger.info("Contract validation successful.")
        return True, None
    except Exception as e:
        logger.error(f"Contract validation failed: {e}")
        return False, str(e)

def setup_model_and_tokenizer(model_name: str):
    """Set up the model and tokenizer."""
    logger.info("Loading model and tokenizer...")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map={"": 0},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def setup_training_config(output_dir: str):
    """Set up the training configuration."""
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="SEQ2SEQ_LM",
    )

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
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
            tokenizer=tokenizer,
            args=training_arguments,
            dataset_text_field="input_ids"  # Ensure dataset uses correct text field for training
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

    seed = int(os.environ.get("SEED", 42))
    num_contracts = int(os.environ.get("NUM_CONTRACTS", 1))
    token_standard = os.environ.get("TOKEN_STANDARD", "ERC-20")

    logger.info(
        f"Environment Variables - SEED: {seed}, "
        f"NUM_CONTRACTS: {num_contracts}, TOKEN_STANDARD: {token_standard}"
    )

    random.seed(seed)
    torch.manual_seed(seed)

    model_name = "/app/model"
    contracts_path = "/app/dataset/contracts"
    output_dir = "/app/results"

    model, tokenizer = setup_model_and_tokenizer(model_name)
    peft_config, training_arguments = setup_training_config(output_dir)

    contract_examples = load_contract_examples(contracts_path)
    dataset = SolidityDataset(contract_examples, tokenizer)

    finetuned_model = finetune_model(
        model, dataset, peft_config, training_arguments, tokenizer
    )

    valid_contracts = generate_contracts(
        finetuned_model, tokenizer, contract_examples, num_contracts
    )

    logger.info(f"Generated {len(valid_contracts)} valid contracts.")

    finetuned_model.save_pretrained(os.path.join(output_dir, "finetuned_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "finetuned_model"))

if __name__ == "__main__":
    main()
