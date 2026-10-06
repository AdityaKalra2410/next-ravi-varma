"""
Midsem evaluation: base SD 1.5 vs. "in the style of Raja Ravi Varma" vs. our LoRA.

For each epic-scene prompt and seed, generates 3 images (same seed, so only the
condition changes) and scores them with CLIP:
  - clip_score : how well the image matches the scene text (prompt WITHOUT style words), 0-100
  - style_sim  : similarity to the 9 HELD-OUT test paintings (never seen in training)
  - max_train  : similarity to the closest training painting (memorization check; >0.95 = suspicious)

Run on Colab T4 from the repo root:
    !pip install -q diffusers transformers accelerate peft
    !python scripts/evaluate.py --dry-run          # just print prompts/conditions
    !python scripts/evaluate.py                     # ~5-10 min
Outputs in eval/: grid.png (for the slide), scores.csv (per image), summary.csv (per condition)
"""
import argparse, csv, os
from pathlib import Path

PROMPTS = [
    "Sita being abducted by Ravana in the sky while the eagle Jatayu attacks",
    "Arjuna drawing his bow on a chariot on the battlefield of Kurukshetra",
    "Karna giving away his golden armour to a sage",
    "Draupadi pleading in the royal court of Hastinapura",
    "Rama, Sita and Lakshmana walking through a forest in exile",
    "Hanuman kneeling before Rama and offering a ring",
    "Damayanti speaking to a golden swan beside a lotus pond",
    "Bhishma lying on a bed of arrows surrounded by warriors",
    "a queen seated on a throne in a palace hall",
    "Krishna playing the flute to Radha by the Yamuna river at dusk",
]
CONDITIONS = {
    "base":       "{p}",
    "base+style": "{p}, in the style of Raja Ravi Varma, oil painting",
    "lora":       "rrvarma oil painting, {p}",
}
NEG = "blurry, deformed, extra limbs, bad anatomy, text, watermark"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                    help="must be the same base model the LoRA was trained on")
    ap.add_argument("--lora", default="pytorch_lora_weights.safetensors")
    ap.add_argument("--report", default="data/captions_report.csv")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="eval")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        for c, t in CONDITIONS.items():
            print(f"[{c}] {t.format(p=PROMPTS[0])}")
        print(f"{len(PROMPTS)} prompts x {len(CONDITIONS)} conditions x {args.seeds} seeds "
              f"= {len(PROMPTS) * len(CONDITIONS) * args.seeds} images")
        return

    import torch
    from PIL import Image, ImageDraw
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from transformers import CLIPModel, CLIPProcessor

    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)
    split = {r["filename"]: r["split"] for r in csv.DictReader(open(args.report, encoding="utf-8"))}

    # ---------- 1. generate ----------
    pipe = StableDiffusionPipeline.from_pretrained(args.base, torch_dtype=torch.float16,
                                                   safety_checker=None).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    def gen(text, seed, **kw):
        g = torch.Generator("cuda").manual_seed(seed)
        return pipe(text, negative_prompt=NEG, num_inference_steps=args.steps,
                    guidance_scale=7.5, generator=g, **kw).images[0]

    records = []
    for cond in CONDITIONS:  # base conditions first, then load the LoRA
        kw = {}
        if cond == "lora":
            pipe.load_lora_weights(".", weight_name=args.lora)
            kw = {"cross_attention_kwargs": {"scale": args.lora_scale}}
        for pi, p in enumerate(PROMPTS):
            for s in range(args.seeds):
                path = out / "images" / f"{cond}_p{pi:02d}_s{s}.png"
                if not path.exists():
                    gen(CONDITIONS[cond].format(p=p), s, **kw).save(path)
                records.append({"condition": cond, "prompt_id": pi, "seed": s,
                                "prompt": p, "path": str(path)})
                print(f"{cond:10s} p{pi:02d} s{s} done")

    del pipe; torch.cuda.empty_cache()

    # ---------- 2. score with CLIP ----------
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    @torch.no_grad()
    def img_emb(paths):
        embs = []
        for i in range(0, len(paths), 16):
            ims = [Image.open(x).convert("RGB") for x in paths[i:i + 16]]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to("cuda")
            e = clip.visual_projection(clip.vision_model(pixel_values=px).pooler_output)
            embs.append(e / e.norm(dim=-1, keepdim=True))
        return torch.cat(embs)

    @torch.no_grad()
    def txt_emb(texts):
        t = proc.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        e = clip.text_projection(clip.text_model(**t).pooler_output)
        return e / e.norm(dim=-1, keepdim=True)

    train = [os.path.join(args.images, f) for f, s in split.items() if s == "train"]
    test = [os.path.join(args.images, f) for f, s in split.items() if s == "test"]
    train_e, test_e = img_emb(train), img_emb(test)
    test_centroid = test_e.mean(0); test_centroid /= test_centroid.norm()

    gen_e = img_emb([r["path"] for r in records])
    txt_e = txt_emb([r["prompt"] for r in records])  # plain scene text, no style words
    for i, r in enumerate(records):
        r["clip_score"] = round(100 * max((gen_e[i] @ txt_e[i]).item(), 0), 2)
        r["style_sim"] = round((gen_e[i] @ test_centroid).item(), 4)
        r["max_train"] = round((gen_e[i] @ train_e.T).max().item(), 4)

    # reference: how "Varma-like" are real held-out paintings by this measure
    ref = (test_e @ test_centroid).mean().item()

    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)

    summary = []
    for cond in CONDITIONS:
        rs = [r for r in records if r["condition"] == cond]
        summary.append({"condition": cond,
                        "clip_score": round(sum(r["clip_score"] for r in rs) / len(rs), 2),
                        "style_sim": round(sum(r["style_sim"] for r in rs) / len(rs), 4),
                        "max_train": round(max(r["max_train"] for r in rs), 4),
                        "n_over_0.95": sum(r["max_train"] > 0.95 for r in rs)})
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    # ---------- 3. grid for the slide (seed 0) ----------
    S, H = 256, 28
    grid = Image.new("RGB", (S * len(CONDITIONS), H + S * len(PROMPTS)), "white")
    d = ImageDraw.Draw(grid)
    for ci, cond in enumerate(CONDITIONS):
        d.text((ci * S + 8, 8), cond, fill="black")
        for pi in range(len(PROMPTS)):
            im = Image.open(out / "images" / f"{cond}_p{pi:02d}_s0.png").resize((S, S))
            grid.paste(im, (ci * S, H + pi * S))
    grid.save(out / "grid.png")

    print("\n=== summary ===")
    for s in summary:
        print(s)
    print(f"reference style_sim of real held-out paintings: {ref:.4f}")
    print(f"saved: {out}/grid.png, {out}/scores.csv, {out}/summary.csv")


if __name__ == "__main__":
    main()
