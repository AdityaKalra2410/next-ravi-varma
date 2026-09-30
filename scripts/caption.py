"""
Caption the oil paintings with BLIP-2 + trigger word.

Caption format:
    rrvarma oil painting, <clean title>, <BLIP-2 description>
    e.g. rrvarma oil painting, Shakuntala, a woman in a sari standing in a forest

Run from the repo root (on a GPU, e.g. Colab T4):
    python scripts/caption.py

Outputs:
    data/captions/<image name>.txt   one caption per image (same name as the image)
    data/captions_review.csv         all captions in one table, for checking by hand
"""
import csv, re, argparse
from pathlib import Path

TRIGGER = "rrvarma oil painting"

def clean_title(t: str) -> str:
    """Strip the Wikimedia/Wikidata junk out of Commons titles."""
    t = str(t)
    # 'Foolabel QS:Len,"Foo"...' / 'Footitle QS:P1476,...'  -> keep text before the template
    t = re.split(r'(?:label|title) QS:', t)[0]
    # 'Malayalam:  <malayalam text> English title' -> keep the English part
    t = re.sub(r'^Malayalam:\s*', '', t)
    t = re.sub(r'[^\x00-\x7F]+', ' ', t)             # drop non-English scripts
    t = re.sub(r'[,-]?\s*(by\s+)?Raja Ravi Varma\s*$', '', t, flags=re.I)
    t = re.sub(r'^Ravi Varma\s*-\s*', '', t, flags=re.I)
    t = re.sub(r'\[\d+\]', '', t)                    # footnote marks like [1]
    t = t.replace('"', '').strip(' ,-')
    return re.sub(r'\s+', ' ', t)

def clean_desc(d: str) -> str:
    d = d.strip().rstrip('.')
    # the trigger already says it's a painting, so drop "a painting of ..."
    d = re.sub(r'^(an? )?(oil )?(painting|picture|image) of\s+', '', d, flags=re.I)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', default='data/metadata.csv')
    ap.add_argument('--images', default='data/images')
    ap.add_argument('--out', default='data/captions')
    ap.add_argument('--model', default='Salesforce/blip2-opt-2.7b')
    ap.add_argument('--dry-run', action='store_true', help='only print cleaned titles, no model')
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.meta, encoding='utf-8'))
            if r['keep'] == 'y' and r['type'] == 'oil']
    print(f'{len(rows)} oil paintings to caption')

    if args.dry_run:
        for r in rows:
            print(r['filename'], '->', clean_title(r['title']))
        return

    import torch
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('WARNING: no GPU found. Switch Colab to a T4 GPU (Runtime > Change runtime type).')
    dtype = torch.float16 if device == 'cuda' else torch.float32
    proc = Blip2Processor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    review = []
    for i, r in enumerate(rows, 1):
        img = Image.open(Path(args.images) / r['filename']).convert('RGB')
        inputs = proc(images=img, return_tensors='pt').to(device, dtype)
        ids = model.generate(**inputs, max_new_tokens=40, num_beams=3, repetition_penalty=1.3)
        desc = clean_desc(proc.batch_decode(ids, skip_special_tokens=True)[0])
        title = clean_title(r['title'])
        caption = ', '.join(p for p in [TRIGGER, title, desc] if p)
        (out / (Path(r['filename']).stem + '.txt')).write_text(caption, encoding='utf-8')
        review.append({'filename': r['filename'], 'title': title, 'blip2': desc, 'caption': caption})
        print(f'[{i}/{len(rows)}] {caption}')

    with open('data/captions_review.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['filename', 'title', 'blip2', 'caption'])
        w.writeheader(); w.writerows(review)
    print(f'Done. {len(review)} captions in {out}/ and data/captions_review.csv')

if __name__ == '__main__':
    main()
