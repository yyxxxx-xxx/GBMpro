#!/usr/bin/env python3
"""Dry-run loader: load base LLM with 4-bit + device_map=auto and check memory usage."""
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = os.environ.get("MODEL_PATH", "models/Qwen2-7B-Instruct")

def main():
    print("=== FSDP dry-run loader ===")
    print(f"Model: {MODEL}")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    print("BitsAndBytesConfig:", bnb_config)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    print("Tokenizer loaded.")

    print("Loading model with device_map=auto, low_cpu_mem_usage=True ...") 
    try:
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            device_map="auto",
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        t1 = time.time()
        print(f"Model loaded in {t1-t0:.1f}s")
        # print some model stats
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
    except Exception as e:
        print("Failed to load model:", e)
        raise

    print("Dry-run load successful. Sleeping 10s to allow monitoring capture...")
    time.sleep(10)
    print("Exiting dry-run.")

if __name__ == '__main__':
    main()








