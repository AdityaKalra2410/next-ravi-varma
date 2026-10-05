"""
Tag every oil painting with Qwen3-VL-8B (open-weight) using the fixed 10-field template.

Output: data/metadata_tags.csv  (one row per painting, one column per field)
Same columns as caption_gemini.py, so the rest of the pipeline doesn't change.

Run from the repo root on a GPU (Colab T4):
    python scripts/caption_qwen.py --dry-run           # prompt + image list, no model
    python scripts/caption_qwen.py --limit 3           # test on 3 images
    python scripts/caption_qwen.py                     # all remaining (resumes)
    python scripts/caption_qwen.py --no-title          # don't show the title (if it hallucinates story characters)

Needs: pip install -U transformers accelerate bitsandbytes pillow
"""
import argparse, csv, json, re, functools
from pathlib import Path

print = functools.partial(print, flush=True)  # show progress live in Colab

FIELDS = {
    # content fields (go into training captions)
    "figures":     "every visible person or animal, each as a short phrase, e.g. 'bearded old man', 'young woman', 'white horse'",
    "characters":  "names of characters who are clearly visible, judged from the title; 'none' if unsure",
    "attire":      "clothing and how it drapes or folds, colours, jewellery, armour, headwear of the main figures",
    "pose":        "body posture, gesture and gaze direction of the main figures",
    "action":      "what is happening; 'none' for a still portrait",
    "setting":     "location and background",
    "composition": "layout and framing, e.g. single-figure portrait, full scene, oval frame, close-up detail",
    # style fields (table only, never in training captions)
    "mood":        "emotional tone",
    "lighting":    "light direction and contrast",
    "palette":     "dominant colours",
}

PROMPT = """You are tagging a painting for a dataset.{title_line}

Look carefully at the image and fill in this JSON object. Rules:
- Describe ONLY what you can actually see. Never add people, objects or events that are not visible,
  even if the story is famous. If you cannot see someone, do not list them.
- Short descriptive phrases separated by commas. No full sentences. Do not repeat phrases.
- Except in mood, lighting and palette, do not mention painting style or technique.
- Use "none" if a field does not apply.

Fields:
{fields}

Answer with only the JSON object, keys in this order: {keys}"""

TITLE_LINE = '\nIt is an oil painting by Raja Ravi Varma titled "{title}". Use the title only to name figures you can clearly see.'


def clean_title(t: str) -> str:
    t = str(t)
    t = re.split(r'(?:label|title) QS:', t)[0]
    t = re.sub(r'^Malayalam:\s*', '', t)
    t = re.sub(r'[^\x00-\x7F]+', ' ', t)
    t = re.sub(r'[,-]?\s*(by\s+)?Raja Ravi Varma\s*$', '', t, flags=re.I)
    t = re.sub(r'^Ravi Varma\s*-\s*', '', t, flags=re.I)
    t = re.sub(r'\[\d+\]', '', t)
    t = t.replace('"', '').strip(' ,-')
    return re.sub(r'\s+', ' ', t)


def load_titles(review_csv):
    if not Path(review_csv).exists():
        return {}
    return {r['filename']: r['title'] for r in csv.DictReader(open(review_csv, encoding='utf-8'))}


def build_prompt(title, use_title=True):
    fields = "\n".join(f'- "{k}": {v}' for k, v in FIELDS.items())
    tl = TITLE_LINE.format(title=title) if use_title else ''
    return PROMPT.format(title_line=tl, fields=fields, keys=", ".join(FIELDS))


def dedupe(value: str, max_items=12) -> str:
    """Remove repeated phrases (guards against generation loops)."""
    seen, out = set(), []
    for p in (x.strip() for x in str(value).split(',')):
        if p and p.lower() not in seen:
            seen.add(p.lower()); out.append(p)
    return ', '.join(out[:max_items])


def parse(text: str):
    """Return (dict, ok). Tries JSON first, then 'key: value' lines."""
    m = re.search(r'\{.*\}', text, flags=re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return {k: dedupe(d.get(k, '')) for k in FIELDS}, all(k in d for k in FIELDS)
        except json.JSONDecodeError:
            pass
    d = {}
    for line in text.splitlines():
        mm = re.match(r'\s*"?(\w+)"?\s*:\s*"?(.*?)"?,?\s*$', line)
        if mm and mm.group(1).lower() in FIELDS:
            d[mm.group(1).lower()] = dedupe(mm.group(2))
    return {k: d.get(k, '') for k in FIELDS}, len(d) == len(FIELDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', default='data/metadata.csv')
    ap.add_argument('--review', default='data/captions_review.csv')
    ap.add_argument('--images', default='data/images')
    ap.add_argument('--out', default='data/metadata_tags.csv')
    ap.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    ap.add_argument('--max-side', type=int, default=1024, help='resize images so the longer side is at most this')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no-title', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    use_title = not args.no_title

    fixed = load_titles(args.review)
    rows = [r for r in csv.DictReader(open(args.meta, encoding='utf-8'))
            if r['keep'] == 'y' and r['type'] == 'oil'
            and (Path(args.images) / r['filename']).exists()]
    for r in rows:
        r['clean_title'] = fixed.get(r['filename']) or clean_title(r['title'])
    print(f"{len(rows)} oil paintings")

    if args.dry_run:
        print(build_prompt(rows[0]['clean_title'], use_title))
        for r in rows:
            print(r['filename'], '->', r['clean_title'])
        return

    import torch
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SystemExit('No GPU found. In Colab: Runtime > Change runtime type > T4 GPU.')

    print(f"Loading {args.model} in 4-bit (first time downloads ~17 GB) ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16),   # T4 has no bfloat16
        dtype=torch.float16, device_map='auto')
    processor = AutoProcessor.from_pretrained(args.model)

    out = Path(args.out)
    cols = ['filename', 'title', 'split', *FIELDS, 'parse_ok', 'checked', 'raw']
    if args.overwrite and out.exists():
        out.unlink()
    done = set()
    if out.exists():
        done = {r['filename'] for r in csv.DictReader(open(out, encoding='utf-8'))}
    else:
        with open(out, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()

    todo = [r for r in rows if r['filename'] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} already done, tagging {len(todo)} now (title {'on' if use_title else 'off'})")

    for i, r in enumerate(todo, 1):
        print(f"  working on {r['filename']} ...")
        img = Image.open(Path(args.images) / r['filename']).convert('RGB')
        img.thumbnail((args.max_side, args.max_side))
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img},
            {'type': 'text', 'text': build_prompt(r['clean_title'], use_title)}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=500, do_sample=False,
                                 repetition_penalty=1.05)
        text = processor.batch_decode(ids[:, inputs['input_ids'].shape[1]:],
                                      skip_special_tokens=True)[0].strip()
        tags, ok = parse(text)

        row = {'filename': r['filename'], 'title': r['clean_title'], 'split': r.get('split', ''),
               'parse_ok': 'y' if ok else 'n', 'checked': '', 'raw': text, **tags}
        with open(out, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=cols).writerow(row)
        print(f"[{i}/{len(todo)}] {r['filename']} {r['clean_title']} | parse_ok={row['parse_ok']} | {tags['figures'][:70]}")
        torch.cuda.empty_cache()

    print(f"Done. Results in {out}")


if __name__ == '__main__':
    main()
