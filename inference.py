#!/usr/bin/env python3
"""
INFERENCE SCRIPT for the Fine-Tuned Llama 3.2 3B Reasoning Model
"""

import torch
from unsloth import FastLanguageModel
import os

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# IMPORTANT: This must be the path where the final model was saved.
# This was set by `output_dir` in your training script.
MODEL_PATH = "./llama-3.2-3b-reasoning-sft/final_model"


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_inference_model():
    """Load the fine-tuned model for inference."""
    logger.info(f"🚀 Loading fine-tuned model from: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model path not found: {MODEL_PATH}. Make sure you have the correct path to the 'final_model' directory.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )

    # Put the model in inference mode
    FastLanguageModel.for_inference(model)

    logger.info("✅ Model loaded and ready for inference.")
    return model, tokenizer


# ==============================================================================
# INFERENCE LOGIC
# ==============================================================================

def run_inference(model, tokenizer, question: str):
    """Run a single inference query."""

    SYSTEM_PROMPT = "You are a helpful assistant that thinks step by step. When solving problems, show your reasoning process clearly using <thinking> and <answer> tags."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1,  # Use a low temperature for more deterministic reasoning
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up the output to only show the assistant's response
    assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]

    print("\n" + "=" * 20 + " INFERENCE " + "=" * 20)
    print(f"❓ Question: {question}")
    print(f"🧠 Response:\n{assistant_response}")
    print("=" * 51 + "\n")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        model, tokenizer = load_inference_model()

        # --- ASK YOUR QUESTIONS HERE ---


        # Interactive loop
        while True:
            user_question = input("Ask a question (or type 'quit' to exit): ")
            if user_question.lower() == 'quit':
                break
            run_inference(model, tokenizer, user_question)

    except Exception as e:
        logger.error(f"❌ An error occurred: {e}", exc_info=True)
    finally:
        logger.info("✅ Inference script finished.")