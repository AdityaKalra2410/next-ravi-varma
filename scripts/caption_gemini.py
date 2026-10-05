"""
Tag every oil painting with Gemini using a fixed 10-field template.

Output: data/metadata_tags.csv  (one row per painting, one column per field)
This table is the source of truth. Training captions are built from it later
(build_captions.py), after a human has checked every row.

Run from the repo root:
    python scripts/caption_gemini.py --dry-run          # show prompt + image list, no API calls
    python scripts/caption_gemini.py --limit 3          # test on 3 images
    python scripts/caption_gemini.py                    # all remaining images (resumes)
    python scripts/caption_gemini.py --overwrite        # start fresh

Needs: pip install google-genai pillow   and env var GEMINI_API_KEY
"""
import argparse, csv, json, os, re, time
from pathlib import Path

FIELDS = {
    # content fields (go into training captions)
    "figures":     "every visible person or animal, each as a short phrase with gender/type, e.g. 'bearded old man', 'young woman', 'white horse'",
    "characters":  "names of characters who are clearly visible, judged from the title; 'none' if the title names no one or they are not visible",
    "attire":      "clothing, colours, jewellery, headwear, for each main figure",
    "pose":        "body position, gesture and gaze of the main figures",
    "action":      "what is happening; 'none' for a still portrait",
    "setting":     "location and background",
    "composition": "layout and framing, e.g. single-figure portrait, full scene, oval frame, close-up detail",
    # style fields (table only, never in training captions)
    "mood":        "emotional tone",
    "lighting":    "light direction and contrast",
    "palette":     "dominant colours",
}

PROMPT = """You are tagging an oil painting by Raja Ravi Varma for a dataset.
Its title is: "{title}".

Fill every field of the JSON schema by LOOKING AT THE IMAGE.
Rules:
- Describe only what is actually visible. The title may help you NAME a figure you can see,
  but never add characters, objects or events from the story that are not in the image.
- Use short descriptive phrases separated by commas. No full sentences.
- In all fields except mood, lighting and palette: do not mention painting style or technique
  (no 'oil painting', 'realistic', 'brushwork', 'chiaroscuro' etc.).
- Write 'none' if a field does not apply.

Fields:
{fields}"""

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
    """Hand-fixed titles from captions_review.csv, if it exists."""
    if not Path(review_csv).exists():
        return {}
    return {r['filename']: r['title'] for r in csv.DictReader(open(review_csv, encoding='utf-8'))}

def build_prompt(title):
    fields = "\n".join(f"- {k}: {v}" for k, v in FIELDS.items())
    return PROMPT.format(title=title, fields=fields)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', default='data/metadata.csv')
    ap.add_argument('--review', default='data/captions_review.csv')
    ap.add_argument('--images', default='data/images')
    ap.add_argument('--out', default='data/metadata_tags.csv')
    ap.add_argument('--model', default='gemini-3.6-flash')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    fixed = load_titles(args.review)
    rows = [r for r in csv.DictReader(open(args.meta, encoding='utf-8'))
            if r['keep'] == 'y' and r['type'] == 'oil'
            and (Path(args.images) / r['filename']).exists()]
    for r in rows:
        r['clean_title'] = fixed.get(r['filename']) or clean_title(r['title'])
    print(f"{len(rows)} oil paintings")

    if args.dry_run:
        print(build_prompt(rows[0]['clean_title']))
        for r in rows:
            print(r['filename'], '->', r['clean_title'])
        return

    from google import genai
    from google.genai import types
    from PIL import Image

    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise SystemExit('GEMINI_API_KEY is not set. See the Colab Secrets step.')
    client = genai.Client(api_key=key)

    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={k: types.Schema(type=types.Type.STRING) for k in FIELDS},
        required=list(FIELDS),
        property_ordering=list(FIELDS),
    )
    config = types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=schema,
    )

    out = Path(args.out)
    cols = ['filename', 'title', 'split', *FIELDS, 'checked', 'raw']
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
    print(f"{len(done)} already done, tagging {len(todo)} now with {args.model}")

    for i, r in enumerate(todo, 1):
        img = Image.open(Path(args.images) / r['filename']).convert('RGB')
        img.thumbnail((2048, 2048))
        prompt = build_prompt(r['clean_title'])

        for attempt in range(5):
            try:
                resp = client.models.generate_content(
                    model=args.model, contents=[img, prompt], config=config)
                tags = json.loads(resp.text)
                break
            except Exception as e:
                msg = str(e)
                if 'API_KEY_INVALID' in msg or 'NOT_FOUND' in msg:
                    raise SystemExit(f"Stopping: {msg[:200]}")
                wait = 10 * (attempt + 1)
                print(f"  error on {r['filename']}: {e}. Retrying in {wait}s")
                time.sleep(wait)
        else:
            print(f"  giving up on {r['filename']}, rerun later to retry it")
            continue

        row = {'filename': r['filename'], 'title': r['clean_title'],
               'split': r.get('split', ''), 'checked': '', 'raw': resp.text}
        row.update({k: str(tags.get(k, '')).strip() for k in FIELDS})
        with open(out, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=cols).writerow(row)
        print(f"[{i}/{len(todo)}] {r['filename']} {r['clean_title']} | {row['figures'][:70]}")

    print(f"Done. Results in {out}")

if __name__ == '__main__':
    main()
