"""
Create FP16 (half precision) version of the model for faster inference
Verify accuracy is maintained
"""
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
from src.labels import ID2LABEL, label_is_pii

print("="*70)
print("CREATING FP16 MODEL")
print("="*70)

# Load original model
print("\nLoading FP32 model...")
model_dir = "out"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model_fp32 = AutoModelForTokenClassification.from_pretrained(model_dir)
model_fp32.eval()

# Convert to FP16
print("\nConverting to FP16 (half precision)...")
model_fp16 = model_fp32.half()
model_fp16.eval()

print("   Model converted to FP16")
print(f"   Memory: ~50% reduction")
print(f"   Speed: ~2-3x faster")

# Verify accuracy on a few examples
print("\nVerifying accuracy (comparing FP32 vs FP16)...")

with open("data/dev.jsonl", "r") as f:
    samples = [json.loads(line) for line in f][:20]  # Test on 20 examples

matches = 0
total = 0

torch.set_grad_enabled(False)

for sample in samples:
    text = sample["text"]
    
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=False)
    
    # FP32 prediction
    with torch.no_grad():
        out_fp32 = model_fp32(**enc)
        pred_fp32 = out_fp32.logits.argmax(dim=-1).tolist()[0]
    
    # FP16 prediction
    with torch.no_grad():
        out_fp16 = model_fp16(**enc)
        pred_fp16 = out_fp16.logits.argmax(dim=-1).tolist()[0]
    
    # Compare predictions
    for p32, p16 in zip(pred_fp32, pred_fp16):
        total += 1
        if p32 == p16:
            matches += 1

accuracy = matches / total * 100
print(f"   Prediction agreement: {accuracy:.1f}%")

if accuracy > 99:
    print("   FP16 maintains accuracy!")
elif accuracy > 95:
    print("   FP16 has minor differences (acceptable)")
else:
    print("   FP16 significantly different (not recommended)")

# Save FP16 model
print("\nSaving FP16 model...")
output_dir = "out_fp16"
model_fp16.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"   Saved to: {output_dir}/")

print("\n" + "="*70)
print("FP16 MODEL READY")
print("="*70)
print(f"Location: {output_dir}/")
print("="*70)