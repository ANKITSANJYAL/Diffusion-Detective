#!/usr/bin/env python3
"""Simple LLM reasoning generator smoke test.

Loads a 4-bit-ready Mistral instruct model and prints a Chain-of-Thought style output.
"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    parser.add_argument("--prompt", default="Decompose the prompt: 'A cat wearing a crown' into 4 short reasoning steps.")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    inputs = tokenizer(args.prompt, return_tensors="pt").to(next(model.parameters()).device)
    out = model.generate(**inputs, max_new_tokens=args.max_tokens)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
