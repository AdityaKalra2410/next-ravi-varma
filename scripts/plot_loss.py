"""
Plot the LoRA training loss: raw per-step loss (faint) + a moving average (bold).

Diffusion loss is noisy because every step uses a random noise level, so the moving
average is what shows the trend.

Run from the repo root:  python scripts/plot_loss.py runs/lora_v2 loss_curve_v2.png
"""
import glob, sys
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/lora_v2"
out = sys.argv[2] if len(sys.argv) > 2 else "loss_curve_v2.png"
WINDOW = 50

files = glob.glob(f"{run_dir}/logs/**/events.out.tfevents.*", recursive=True)
if not files:
    sys.exit(f"No TensorBoard logs found under {run_dir}/logs")
ea = EventAccumulator(max(files, key=len), size_guidance={"scalars": 0}); ea.Reload()
ev = ea.Scalars("train_loss")
steps, loss = [e.step for e in ev], [e.value for e in ev]

avg = []
for i in range(len(loss)):
    w = loss[max(0, i - WINDOW + 1): i + 1]
    avg.append(sum(w) / len(w))

plt.figure(figsize=(8, 4))
plt.plot(steps, loss, color="#9FB7D6", linewidth=0.6, label="per-step loss")
plt.plot(steps, avg, color="#6B1F1F", linewidth=2, label=f"moving average ({WINDOW} steps)")
plt.xlabel("Training step"); plt.ylabel("Loss (noise-prediction MSE)")
plt.title(f"LoRA training loss ({steps[-1]} steps)")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(out, dpi=200)
print(f"saved {out}; moving average {avg[WINDOW]:.4f} at step {steps[WINDOW]} -> {avg[-1]:.4f} at the end")
