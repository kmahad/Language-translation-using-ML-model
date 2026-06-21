"""Quick checkpoint inspection script."""
import json
import sys
import os

checkpoint_path = 'checkpoints/best.json'
if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint not found at {checkpoint_path}")
    sys.exit(1)

with open(checkpoint_path, 'r', encoding='utf-8') as f:
    cp = json.load(f)

print("Checkpoint submodels:")
for key in cp.keys():
    if key == "weights":
        print(f"  weights: {cp[key]}")
    elif key == "phrase_table":
        rules_count = sum(len(cand) for cand in cp[key]["phrase_probs"].values())
        print(f"  phrase_table: {len(cp[key]['phrase_probs'])} source phrases, {rules_count} total translation rules")
    elif key == "language_model":
        print(f"  language_model: vocab size = {len(cp[key]['vocab'])}, order = {cp[key]['order']}")
    else:
        print(f"  {key}: present")
