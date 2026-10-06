"""
Reconstruction test: can each setup re-create the 9 HELD-OUT paintings from their own captions?

For each of the 9 test paintings, we take its training-style caption (from captions_report.csv),
generate it under the same 3 conditions as evaluate.py, and compare each image to the REAL painting.

Conditions (content text = the caption without the trigger word):
  base       : content text
  base+style : "Raja Ravi Varma oil painting, " + content text (prefix, so it isn't truncated)
  lora       : full caption with trigger word, LoRA loaded

Scores (CLIP ViT-L/14):
  img_sim    : similarity of the generated image to ITS OWN real painting (main number, higher = closer)
  top1       : is its own painting the closest of the 9 test paintings? (1/0; chance = 1/9 = 11%)
  clip_score : match to the content text, 0-100

Run on Colab T4 from the repo root (same setup as evaluate.py):
    !pip uninstall -y torchao
    !python scripts/reconstruct.py --dry-run
    !python scripts/reconstruct.py            # 54 images, ~5 min
Outputs in eval_recon/: grid.png, scores.csv, summary.csv
"""
import argparse, csv
from pathlib import Path

TRIGGER = "rrvarma oil painting"
STYLE = "Raja Ravi Varma oil painting"  # put FIRST: captions are near the 77-token limit, so a suffix would be cut off
NEG = "blurry, deformed, extra limbs, bad anatomy, text, watermark"
CONDS = ["base", "base+style", "lora"]


def load_test(report):
    rows = [r for r in csv.DictReader(open(report, encoding="utf-8")) if r["split"] == "test"]
    for r in rows:
        cap = r["caption"].strip()
        content = cap[len(TRIGGER):].lstrip(", ").strip() if cap.lower().startswith(TRIGGER) else cap
        r["prompts"] = {"base": content, "base+style": f"{STYLE}, {content}", "lora": f"{TRIGGER}, {content}"}
        r["content"] = content
    return rows


def size_for(path):
    from PIL import Image
    w, h = Image.open(path).size
    r = w / h
    if r < 0.8:
        return 512, 768  # portrait
    if r > 1.25:
        return 768, 512  # landscape
    return 512, 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--lora", default="pytorch_lora_weights.safetensors")
    ap.add_argument("--report", default="data/captions_report.csv")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="eval_recon")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tests = load_test(args.report)
    if args.dry_run:
        for t in tests:
            print(f"{t['filename']}  ({len(t['content'].split())} words)")
            for c in CONDS:
                print(f"   [{c}] {t['prompts'][c][:110]}")
        print(f"{len(tests)} paintings x {len(CONDS)} conditions x {args.seeds} seeds = "
              f"{len(tests) * len(CONDS) * args.seeds} images")
        return

    import torch
    from PIL import Image, ImageDraw
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from transformers import CLIPModel, CLIPProcessor

    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)

    # ---------- 1. generate ----------
    pipe = StableDiffusionPipeline.from_pretrained(args.base, torch_dtype=torch.float16,
                                                   safety_checker=None).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    records = []
    for cond in CONDS:  # base conditions first, then load the LoRA
        kw = {}
        if cond == "lora":
            pipe.load_lora_weights(".", weight_name=args.lora)
            kw = {"cross_attention_kwargs": {"scale": args.lora_scale}}
        for t in tests:
            w, h = size_for(Path(args.images) / t["filename"])
            for s in range(args.seeds):
                path = out / "images" / f"{cond}_{Path(t['filename']).stem}_s{s}.png"
                if not path.exists():
                    g = torch.Generator("cuda").manual_seed(s)
                    pipe(t["prompts"][cond], negative_prompt=NEG, num_inference_steps=args.steps,
                         guidance_scale=7.5, width=w, height=h, generator=g, **kw).images[0].save(path)
                records.append({"condition": cond, "target": t["filename"], "seed": s,
                                "content": t["content"], "path": str(path)})
                print(f"{cond:10s} {t['filename']} s{s} done")
    del pipe; torch.cuda.empty_cache()

    # ---------- 2. score ----------
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

    names = [t["filename"] for t in tests]
    real_e = img_emb([str(Path(args.images) / n) for n in names])
    gen_e = img_emb([r["path"] for r in records])
    txt_e = txt_emb([r["content"] for r in records])
    for i, r in enumerate(records):
        sims = gen_e[i] @ real_e.T  # similarity to each of the 9 real paintings
        own = names.index(r["target"])
        r["img_sim"] = round(sims[own].item(), 4)
        r["top1"] = int(sims.argmax().item() == own)
        r["clip_score"] = round(100 * max((gen_e[i] @ txt_e[i]).item(), 0), 2)

    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)

    summary = []
    for cond in CONDS:
        rs = [r for r in records if r["condition"] == cond]
        summary.append({"condition": cond,
                        "img_sim": round(sum(r["img_sim"] for r in rs) / len(rs), 4),
                        "top1_acc": f"{sum(r['top1'] for r in rs)}/{len(rs)}",
                        "clip_score": round(sum(r["clip_score"] for r in rs) / len(rs), 2)})
    # per-painting wins: on how many of the 9 paintings is LoRA the closest condition (seed-averaged)?
    wins = 0
    for n in names:
        avg = {c: sum(r["img_sim"] for r in records if r["condition"] == c and r["target"] == n) / args.seeds
               for c in CONDS}
        wins += max(avg, key=avg.get) == "lora"
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    # ---------- 3. grid: real | base | base+style | lora (seed 0) ----------
    S, H = 200, 24
    grid = Image.new("RGB", (S * 4, H + S * len(tests)), "white")
    d = ImageDraw.Draw(grid)
    for ci, lab in enumerate(["real (held-out)", *CONDS]):
        d.text((ci * S + 6, 6), lab, fill="black")
    for ri, t in enumerate(tests):
        stem = Path(t["filename"]).stem
        ims = [Image.open(Path(args.images) / t["filename"])] + \
              [Image.open(out / "images" / f"{c}_{stem}_s0.png") for c in CONDS]
        for ci, im in enumerate(ims):
            im = im.convert("RGB"); im.thumbnail((S, S))
            grid.paste(im, (ci * S + (S - im.width) // 2, H + ri * S + (S - im.height) // 2))
    grid.save(out / "grid.png")

    print("\n=== reconstruction summary ===")
    for s in summary:
        print(s)
    print(f"LoRA is the closest condition on {wins}/{len(names)} paintings")
    print(f"(top1 chance level = 1/{len(names)})")
    print(f"saved: {out}/grid.png, {out}/scores.csv, {out}/summary.csv")


if __name__ == "__main__":
    main()
