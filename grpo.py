#!/usr/bin/env python3
"""
WORKING SOLUTION: SFT Training for Llama 3.2 3B Reasoning Model
WITH CHECKPOINTING SUPPORT - Can resume from interruptions

This approach teaches the model to reason by training on examples with explicit reasoning steps
"""

# Critical: Import order matters for Unsloth
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import json
import numpy as np
import re
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
import gc
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Memory cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print("🚀 Starting WORKING SFT Training for Llama 3.2 3B Reasoning Model")
if torch.cuda.is_available():
    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
else:
    raise RuntimeError("GPU not available. Training requires CUDA.")


# ==============================================================================
# CONFIGURATION - RTX 3060 OPTIMIZED
# ==============================================================================
@dataclass
class TrainingConfig:
    model_name: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    max_seq_length: int = 2048  # Increased for reasoning examples
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    per_device_train_batch_size: int = 2  # SFT can handle larger batches
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 1000
    save_steps: int = 100  # Save checkpoint every 100 steps
    logging_steps: int = 10
    max_samples: int = 2000
    output_dir: str = "./llama-3.2-3b-reasoning-sft"
    checkpoint_dir: str = "./checkpoints-sft"


config = TrainingConfig()


# ==============================================================================
# MODEL LOADING
# ==============================================================================
def load_model_and_tokenizer():
    logger.info("📦 Loading Llama 3.2 3B with Unsloth optimizations...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # Configure tokenizer properly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    logger.info(f"✅ Model loaded with {model.get_memory_footprint() / 1e9:.2f}GB memory")
    return model, tokenizer


# ==============================================================================
# DATA PREPARATION - CREATE REASONING EXAMPLES
# ==============================================================================
def extract_hash_answer(text: str) -> str:
    """Extract answer from GSM8K format (#### answer)"""
    if not isinstance(text, str):
        return ""
    if "####" not in text:
        return ""
    return text.split("####")[1].strip()


def format_question_with_choices(example: Dict) -> str:
    """Format multiple choice questions"""
    question = example.get("question", "")
    choices = example.get("choices", {})

    if not choices:
        return question

    choices_text = []
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    for label, text in zip(labels, texts):
        choices_text.append(f"{label}: {text}")

    return f"{question}\n\nChoices:\n" + "\n".join(choices_text)


def create_reasoning_example(question: str, answer: str, domain: str) -> str:
    """Create a reasoning example with explicit thinking steps"""

    if domain == "math":
        # Math reasoning template
        reasoning_template = f"""Let me solve this step by step:

<thinking>
First, I need to understand what the question is asking. The question is: {question}

I'll break this down:
1. Identify the key information and what we're solving for
2. Set up the equation or approach
3. Solve step by step
4. Verify my answer

Let me work through this systematically:
</thinking>

<answer>
{answer}
</answer>"""
        return reasoning_template

    elif domain == "science":
        # Science reasoning template
        reasoning_template = f"""Let me think through this science problem:

<thinking>
This is a science question: {question}

To answer this, I need to:
1. Recall relevant scientific principles
2. Apply them to the specific scenario
3. Eliminate incorrect options if it's multiple choice
4. Arrive at the correct conclusion

Let me reason through this:
</thinking>

<answer>
{answer}
</answer>"""
        return reasoning_template

    else:  # logic
        # Logic reasoning template
        reasoning_template = f"""Let me analyze this logical reasoning question:

<thinking>
The question presents a logical scenario: {question}

To solve this, I need to:
1. Understand the given information
2. Identify the logical relationships
3. Apply deductive or inductive reasoning
4. Draw the correct conclusion

Let me work through this logically:
</thinking>

<answer>
{answer}
</answer>"""
        return reasoning_template


def load_and_prepare_datasets():
    """Load datasets and prepare reasoning examples"""
    logger.info("📊 Loading and preparing reasoning datasets...")

    datasets_info = []

    # GSM8K - Math reasoning
    try:
        gsm8k = load_dataset("openai/gsm8k", "main", split="train")
        if len(gsm8k) > config.max_samples // 3:
            gsm8k = gsm8k.select(range(config.max_samples // 3))
        datasets_info.append(("gsm8k", gsm8k, "math"))
        logger.info(f"✅ GSM8K loaded: {len(gsm8k)} examples")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load GSM8K: {e}")

    # ARC Challenge - Science reasoning
    try:
        arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        if len(arc) > config.max_samples // 3:
            arc = arc.select(range(config.max_samples // 3))
        datasets_info.append(("arc", arc, "science"))
        logger.info(f"✅ ARC loaded: {len(arc)} examples")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load ARC: {e}")

    # CommonsenseQA - Logic reasoning
    try:
        csqa = load_dataset("tau/commonsense_qa", split="train")
        if len(csqa) > config.max_samples // 3:
            csqa = csqa.select(range(config.max_samples // 3))
        datasets_info.append(("csqa", csqa, "logic"))
        logger.info(f"✅ CommonsenseQA loaded: {len(csqa)} examples")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load CommonsenseQA: {e}")

    if not datasets_info:
        raise RuntimeError("No datasets could be loaded!")

    return datasets_info


def create_sft_dataset(tokenizer, datasets_info):
    """Create SFT dataset with reasoning examples"""
    logger.info("🛠️ Creating SFT dataset with reasoning examples...")

    formatted_data = []

    for dataset_name, dataset, domain in datasets_info:
        logger.info(f"Processing {dataset_name}...")

        for example in dataset:
            try:
                # Extract question and answer based on dataset type
                if domain == "math":  # GSM8K
                    question = example.get("question", "")
                    answer = extract_hash_answer(example.get("answer", ""))
                elif domain in ["science", "logic"]:  # ARC, CommonsenseQA
                    question = format_question_with_choices(example)
                    answer = example.get("answerKey", "")
                else:
                    continue

                if not question or not answer:
                    continue

                # Create reasoning example
                reasoning_response = create_reasoning_example(question, answer, domain)

                # Create conversation for SFT
                messages = [
                    {"role": "system",
                     "content": "You are a helpful assistant that thinks step by step. When solving problems, show your reasoning process clearly using <thinking> and <answer> tags."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": reasoning_response}
                ]

                # Check token length
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                if len(tokenizer.encode(prompt_text)) > config.max_seq_length:
                    continue

                # Add to dataset
                formatted_data.append({
                    "messages": messages,
                    "text": prompt_text,  # For SFT trainer
                    "domain": domain,
                })
            except Exception as e:
                logger.warning(f"Error processing example: {e}")
                continue

    if not formatted_data:
        raise RuntimeError("No valid examples created!")

    # Create dataset and shuffle
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.shuffle(seed=42)

    logger.info(f"✅ SFT dataset created with {len(dataset)} examples")
    return dataset


# ==============================================================================
# TRAINING SETUP - SFT TRAINER WITH CHECKPOINTING
# ==============================================================================
def setup_training(model, tokenizer, dataset):
    """Setup SFT trainer with checkpointing"""
    logger.info("⚙️ Setting up SFTTrainer with checkpointing...")

    # Create directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # SFT Configuration with checkpointing - KEY ADDITIONS
    training_args = SFTConfig(
        output_dir=config.checkpoint_dir,  # Checkpoints go here
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="adamw_8bit",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,  # KEY: Save checkpoints every N steps
        dataset_text_field="text",
        packing=False,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="none",
        gradient_checkpointing=True,
        save_strategy="steps",  # KEY: Save based on steps
        save_total_limit=3,  # KEY: Keep only last 3 checkpoints
    )

    # Initialize trainer with max_seq_length parameter
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        max_seq_length=config.max_seq_length,
    )

    logger.info("✅ SFTTrainer setup complete with checkpointing")
    return trainer


# ==============================================================================
# MAIN TRAINING PIPELINE WITH RESUME CAPABILITY
# ==============================================================================
def main():
    """Main training pipeline with resume capability"""
    logger.info("🚀 Starting SFT training pipeline with checkpointing...")

    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()

        # Load and prepare datasets
        datasets_info = load_and_prepare_datasets()
        sft_dataset = create_sft_dataset(tokenizer, datasets_info)

        # Setup training
        trainer = setup_training(model, tokenizer, sft_dataset)

        # Check for existing checkpoints and resume if found
        checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)

        if checkpoint_path:
            logger.info(f"🔄 Found checkpoint at {checkpoint_path}. Resuming training...")
            trainer_stats = trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            logger.info("🔥 Starting fresh training...")
            trainer_stats = trainer.train()

        # Save final model
        logger.info("💾 Saving final model...")
        final_path = os.path.join(config.output_dir, "final_model")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)

        # Also save LoRA adapters separately
        logger.info("💾 Saving LoRA adapters...")
        model.save_lora("./llama-3.2-3b-reasoning-lora")

        logger.info(f"🎉 Training completed! Model saved to: {final_path}")

        # Test the model
        logger.info("🧪 Testing the model...")
        test_reasoning(model, tokenizer)

    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("🧹 Cleanup completed")


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the checkpoint directory"""
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        return None

    # Look for checkpoint directories (checkpoint-XXXX)
    checkpoints = list(checkpoint_path.glob("checkpoint-*"))

    if not checkpoints:
        return None

    # Sort by step number and return the latest
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
    return str(latest_checkpoint)


def test_reasoning(model, tokenizer):
    """Test the model's reasoning capabilities"""
    FastLanguageModel.for_inference(model)

    test_questions = [
        "What is 15 + 27?",
        "If a train travels 60 km/h for 2 hours, how far does it go?",
        "What comes next in the pattern: 2, 4, 8, 16, ?"
    ]

    for question in test_questions:
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that thinks step by step. When solving problems, show your reasoning process clearly using <thinking> and <answer> tags."},
            {"role": "user", "content": question}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            use_cache=True
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nQuestion: {question}")
        print(f"Response: {response}")
        print("-" * 50)


if __name__ == "__main__":
    main()