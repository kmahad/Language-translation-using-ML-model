"""Quick checkpoint inspection script."""
import torch
import sys

cp = torch.load('checkpoints/best.pt', map_location='cpu')
print("Keys:", list(cp.keys()))
print("Epoch:", cp.get('epoch', '?'))
print("Best val loss:", cp.get('best_val_loss', '?'))

hist = cp.get('training_history', [])
print(f"History entries: {len(hist)}")
if hist:
    print("\nTraining history (last 10):")
    for h in hist[-10:]:
        e = h.get('epoch', '?')
        tl = h.get('train_loss', 0)
        vl = h.get('val_loss', 0)
        print(f"  Epoch {e}: train={tl:.4f}, val={vl:.4f}")

# Check if model trained meaningfully
if hist:
    first_vl = hist[0].get('val_loss', 0)
    last_vl = hist[-1].get('val_loss', 0)
    print(f"\nVal loss: {first_vl:.4f} -> {last_vl:.4f} (improvement: {first_vl - last_vl:.4f})")
