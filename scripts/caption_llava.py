"""
Step 4: generate dense template tags for every oil painting with LLaVA.

Input : image + cleaned title (no other context)
Output: data/metadata_tags.csv  -> one row per painting, one column per template field

Run from the repo root on a GPU (Colab T4 is enough):
    pip install -q transformers accelerate bitsandbytes
    python scripts/caption_llava.py --limit 3 --overwrite   # test on 3 images first
    python scripts/caption_llava.py                # full run (resumes if interrupted)
    python scripts/caption_llava.py --dry-run      # just print the prompts, no model
"""
import argparse, csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from caption import clean_title  # reuse the regex title cleaner

# ---- the rigid template (Step 3) ----
FIELDS = {
    "figures":     "each main person or animal visible, each with a short description, not just a number",
    "characters":  "only names written in the title, else none",
    "attire":      "clothing, colours, jewellery, crowns",
    "pose":        "body position, gesture, gaze",
    "action":      "what is happening, or none for a still portrait",
    "setting":     "location and background",
    "composition": "layout and framing, e.g. full scene, single-figure portrait, oval frame, close-up detail",
    "mood":        "emotional tone",
    "lighting":    "light direction and contrast",
    "palette":     "dominant colours",
}

PROMPT = """This is the oil painting "{title}" by Raja Ravi Varma.
Describe it by filling in the template below.

Rules:
- Use short phrases separated by commas, not full sentences.
- Describe only what you can actually see in the image. Do not add people or events from the story that are not visible.
- Mention each thing once. Do not repeat phrases.
- Do not mention painting style or technique, except in the lighting and palette fields.
- If a field does not apply, write none.
- Keep every label exactly as written, one field per line. Stop after the palette line and write nothing else.

{template}"""


def build_prompt(title):
    template = "\n".join(f"{k}: <{v}>" for k, v in FIELDS.items())
    return PROMPT.format(title=title or "untitled", template=template)


def parse(text):
    """Turn 'field: value' lines into a dict. Missing fields stay empty."""
    out = {k: "" for k in FIELDS}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().strip("*-# ").lower()
        if key in out and not out[key]:
            out[key] = val.strip().strip("<>").rstrip(".").strip()
    return out


def load_titles(meta_path, review_path):
    """Oil paintings only. Prefer the hand-fixed title from captions_review.csv if present."""
    review = {}
    if Path(review_path).exists():
        review = {r["filename"]: r["title"] for r in csv.DictReader(open(review_path, encoding="utf-8"))}
    rows = []
    for r in csv.DictReader(open(meta_path, encoding="utf-8")):
        if r["keep"] == "y" and r["type"] == "oil":
            rows.append({"filename": r["filename"], "split": r["split"],
                         "title": review.get(r["filename"]) or clean_title(r["title"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/metadata.csv")
    ap.add_argument("--review", default="data/captions_review.csv")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="data/metadata_tags.csv")
    ap.add_argument("--model", default="llava-hf/llava-v1.6-mistral-7b-hf",
                    help="LLaVA-NeXT (v1.6) by default; llava-hf/llava-1.5-7b-hf also works")
    ap.add_argument("--overwrite", action="store_true", help="delete old output and start fresh")
    ap.add_argument("--no-4bit", action="store_true", help="load in fp16 (needs ~15 GB VRAM)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N images (testing)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load_titles(args.meta, args.review)
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} oil paintings")

    if args.dry_run:
        print(build_prompt(rows[0]["title"]))
        for r in rows:
            print(r["filename"], "->", r["title"])
        return

    # resume: skip images already in the output file
    cols = ["filename", "title", "split", *FIELDS, "parse_ok", "checked", "raw"]
    done = set()
    if args.overwrite and Path(args.out).exists():
        Path(args.out).unlink()
    if Path(args.out).exists():
        done = {r["filename"] for r in csv.DictReader(open(args.out, encoding="utf-8"))}
    todo = [r for r in rows if r["filename"] not in done]
    print(f"{len(done)} already done, {len(todo)} to go")
    if not todo:
        return

    import torch
    from PIL import Image
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              LlavaForConditionalGeneration, LlavaNextForConditionalGeneration)

    if not torch.cuda.is_available():
        sys.exit("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")
    quant = None if args.no_4bit else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    ModelCls = LlavaNextForConditionalGeneration if "v1.6" in args.model else LlavaForConditionalGeneration
    model = ModelCls.from_pretrained(
        args.model, torch_dtype=torch.float16, quantization_config=quant, device_map="auto")
    proc = AutoProcessor.from_pretrained(args.model)

    new_file = not Path(args.out).exists()
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        for i, r in enumerate(todo, 1):
            img = Image.open(Path(args.images) / r["filename"]).convert("RGB")
            conv = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": build_prompt(r["title"])}]}]
            prompt = proc.apply_chat_template(conv, add_generation_prompt=True)
            inputs = proc(images=img, text=prompt, return_tensors="pt").to(model.device, torch.float16)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=250, do_sample=False,
                                     repetition_penalty=1.15)
            text = proc.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            tags = parse(text)
            ok = all(tags.values())
            w.writerow({**r, **tags, "parse_ok": "y" if ok else "n", "checked": "", "raw": text})
            f.flush()  # so a Colab disconnect doesn't lose finished rows
            print(f"[{i}/{len(todo)}] {r['filename']} {'' if ok else '(PARSE PROBLEM, check raw)'}")
            print("   ", tags["figures"], "|", tags["action"])

    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
