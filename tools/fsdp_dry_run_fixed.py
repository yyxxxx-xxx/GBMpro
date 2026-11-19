#!/usr/bin/env python3
"""DeepSpeed ZeRO-3 dry-run loader: use init_empty_weights to build model, then load with ZeRO-3."""
import time
import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights

MODEL = os.environ.get("MODEL_PATH", "models/Qwen2-7B-Instruct")

def main():
    print("=== DeepSpeed ZeRO-3 dry-run loader ===")
    print(f"Model: {MODEL}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    print("Tokenizer loaded.")

    print("Building model with accelerate init_empty_weights...")
    try:
        t0 = time.time()
        # Use accelerate init_empty_weights to build model on meta device
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        # Now initialize DeepSpeed ZeRO-3 with the empty model
        model = deepspeed.initialize(
            model=model,
            config="deepspeed_zero3_config.json"
        )[0]

        t1 = time.time()
        print(f"Model built and initialized with ZeRO-3 in {t1-t0:.1f}s")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
        print(f"Model device: {next(model.parameters()).device}")

    except Exception as e:
        print("Failed to build model with ZeRO-3:", e)
        raise

    print("Dry-run load successful with ZeRO-3. Sleeping 10s to allow monitoring capture...")
    time.sleep(10)
    print("Exiting dry-run.")

if __name__ == '__main__':
    main()


