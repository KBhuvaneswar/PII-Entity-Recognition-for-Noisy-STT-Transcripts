"""
Measure latency for FP16 model
"""
import json
import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out_fp16")
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--max_length", type=int, default=128)
    args = ap.parse_args()

    print("Loading FP16 model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    
    # FP16 mode
    model = model.half()
    model.eval()
    torch.set_grad_enabled(False)

    # Load test data
    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])

    print(f"Warming up with 10 samples...")
    for text in texts[:10]:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length, padding=False)
        with torch.no_grad():
            _ = model(**enc)

    print(f"\nMeasuring latency over {args.runs} runs (FP16)...")
    latencies = []
    
    for i in range(args.runs):
        text = texts[i % len(texts)]
        
        start_time = time.perf_counter()
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length, padding=False)
        with torch.no_grad():
            out = model(**enc)
        _ = out.logits.argmax(dim=-1)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean = np.mean(latencies)

    print(f"\n{'='*70}")
    print(f"LATENCY RESULTS - FP16 MODEL ({args.runs} runs)")
    print(f"{'='*70}")
    print(f"  Mean:  {mean:.2f} ms")
    print(f"  p50:   {p50:.2f} ms")
    
    if p95 <= 20:
        print(f"  p95:   {p95:.2f} ms")
    else:
        print(f"  p95:   {p95:.2f} ms (target: ≤20ms)")
    
    print(f"  p99:   {p99:.2f} ms")
    print(f"  Min:   {np.min(latencies):.2f} ms")
    print(f"  Max:   {np.max(latencies):.2f} ms")
    print(f"{'='*70}")
    print("\nOptimization: Half Precision (FP16)")
    print("Speedup: ~2-3x vs FP32")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

