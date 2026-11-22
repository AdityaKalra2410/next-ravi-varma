"""
Milestone 3 evaluation: does LLM prompt expansion help?

For 10 short epic scenes (the kind a user would type), three conditions with SD 1.5 + our LoRA:
  raw      : "rrvarma oil painting, <scene>"
  expanded : LLM fields in full caption format (Qwen2.5-7B)
  hybrid   : "rrvarma oil painting, <scene>" + LLM figures, pose, setting, composition
  hybrid_neg     : hybrid + style-aware negative prompt (no calendar art / poster / flat colours)
  raw_neg_best4  : raw    + negative prompt + best-of-4 selection   (fairness control)
  hybrid_neg_best4: hybrid + negative prompt + best-of-4 selection  (final system)
Best-of-4 is picked by a DIFFERENT CLIP (ViT-B/32) against the TRAINING paintings, never the test set.
Same seeds for both. Scores (CLIP ViT-L/14, same method as evaluate.py):
  scene_match : similarity to the user's ORIGINAL scene text (did we draw what was asked?), 0-100
  style_sim   : similarity to the 9 held-out Varma paintings

Loads the LLM once, expands everything, frees it, then loads SD once.
Colab T4, from the repo root (~10-15 min):
    !python scripts/eval_m3.py
Outputs in eval_m3/: expansions.csv, scores.csv, summary.csv, grid.png
"""
import argparse, csv, gc, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import expand_prompt as ep

SCENES = [
    "Yudhishthira weeping over the body of Karna on the battlefield of Kurukshetra",
    "Sita sitting alone in the Ashoka grove, guarded by demonesses",
    "Arjuna drawing his bow on his chariot while Krishna holds the reins",
    "Hanuman kneeling before Rama and offering a ring",
    "Bhishma lying on a bed of arrows",
    "Draupadi pleading in the royal court",
    "Damayanti talking to a golden swan by a lotus pond",
    "Shakuntala looking back at King Dushyanta in the forest",
    "Ravana carrying Sita away in his flying chariot",
    "Ganga descending from the heavens onto Shiva's hair",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="models/lora_v1.safetensors")
    ap.add_argument("--report", default="data/captions_report.csv")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="eval_m3")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    import torch
    from PIL import Image, ImageDraw
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)

    # ---------- 1. expand all scenes (LLM loaded once) ----------
    exp_file = out / "expansions.csv"
    rows = list(csv.DictReader(open(exp_file, encoding="utf-8"))) if exp_file.exists() else []
    if rows and "hybrid" in rows[0]:
        prompts = {r["scene"]: r["expanded"] for r in rows}
        hybrid = {r["scene"]: r["hybrid"] for r in rows}
    else:
        prompts, hybrid, raws = {}, {}, {}
        for i, sc in enumerate(SCENES):
            _, raw = ep.expand(sc)
            fields = ep.fix_fields(raw, sc)
            prompts[sc] = ep.assemble(fields, sc, "full")
            hybrid[sc] = ep.assemble(fields, sc, "hybrid")
            raws[sc] = raw.strip()
            print(f"[{i}] {sc}\n    full   -> {prompts[sc]}\n    hybrid -> {hybrid[sc]}")
        with open(exp_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["scene", "expanded", "hybrid", "tokens_full", "tokens_hybrid", "llm_json"])
            for sc in SCENES:
                w.writerow([sc, prompts[sc], hybrid[sc], ep.clip_tokens(prompts[sc]),
                            ep.clip_tokens(hybrid[sc]), raws[sc]])
        ep._llm = ep._tok = None; gc.collect(); torch.cuda.empty_cache()   # free GPU for SD

    # ---------- 2. generate ----------
    raw_p = lambda sc: f"{ep.TRIGGER}, {sc}"
    conds = {"raw": raw_p, "expanded": lambda sc: prompts[sc], "hybrid": lambda sc: hybrid[sc],
             "hybrid_neg": lambda sc: hybrid[sc], "raw_neg_best4": raw_p, "hybrid_neg_best4": lambda sc: hybrid[sc]}
    records = []
    for cond, make in conds.items():
        for si, sc in enumerate(SCENES):
            for s in range(args.seeds):
                path = out / "images" / f"{cond}_p{si:02d}_s{s}.png"
                if not path.exists():
                    if cond.endswith("best4"):
                        img, _ = ep.generate_best(make(sc), sc, n=4, seed=s, lora=args.lora, negative=ep.NEG_PLUS)
                    elif cond.endswith("neg"):
                        img = ep.generate(make(sc), seed=s, lora=args.lora, negative=ep.NEG_PLUS)
                    else:
                        img = ep.generate(make(sc), seed=s, lora=args.lora)
                    img.save(path)
                records.append({"condition": cond, "scene_id": si, "seed": s, "scene": sc,
                                "prompt": make(sc), "path": str(path)})
                print(f"{cond:8s} p{si:02d} s{s} done")
    ep._pipe = ep._judge = None; gc.collect(); torch.cuda.empty_cache()

    # ---------- 3. score with CLIP ----------
    from transformers import CLIPModel, CLIPProcessor
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

    test = [str(Path(args.images) / r["filename"]) for r in csv.DictReader(open(args.report, encoding="utf-8"))
            if r["split"] == "test"]
    centroid = img_emb(test).mean(0); centroid /= centroid.norm()
    g = img_emb([r["path"] for r in records]); t = txt_emb([r["scene"] for r in records])
    for i, r in enumerate(records):
        r["scene_match"] = round(100 * max((g[i] @ t[i]).item(), 0), 2)
        r["style_sim"] = round((g[i] @ centroid).item(), 4)

    with open(out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    summary = []
    for cond in conds:
        rs = [r for r in records if r["condition"] == cond]
        summary.append({"condition": cond,
                        "scene_match": round(sum(r["scene_match"] for r in rs) / len(rs), 2),
                        "style_sim": round(sum(r["style_sim"] for r in rs) / len(rs), 4)})
    wins = {c: 0 for c in conds}
    for si in range(len(SCENES)):
        m = {c: sum(r["scene_match"] for r in records if r["condition"] == c and r["scene_id"] == si) for c in conds}
        wins[max(m, key=m.get)] += 1
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    # ---------- 4. grid: raw | expanded (seed 0) ----------
    S, H = 256, 28
    show = ["raw", "hybrid", "hybrid_neg_best4"]
    grid = Image.new("RGB", (S * len(show), H + S * len(SCENES)), "white"); d = ImageDraw.Draw(grid)
    for ci, cond in enumerate(show):
        d.text((ci * S + 8, 8), cond, fill="black")
        for si in range(len(SCENES)):
            grid.paste(Image.open(out / "images" / f"{cond}_p{si:02d}_s0.png").resize((S, S)), (ci * S, H + si * S))
    grid.save(out / "grid.png")

    print("\n=== M3 summary ===")
    for s in summary:
        print(s)
    print("best scene match per scene:", wins)
    print(f"saved: {out}/expansions.csv, scores.csv, summary.csv, grid.png")


if __name__ == "__main__":
    main()
