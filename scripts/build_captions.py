"""
Build LoRA training captions from data/metadata_tags.csv.

Caption = trigger, then the 7 content fields in priority order:
    rrvarma oil painting, <characters>, <figures>, <action>, <pose>, <attire>, <setting>, <composition>
- mood, lighting and palette are NOT used (style must be absorbed by the trigger token)
- fields that are 'none' or empty are skipped, repeated phrases are dropped
- each field contributes at most a few phrases (CAPS), so every field gets a share
- every caption is kept within CLIP's 77-token limit (SD 1.5 text encoder);
  if too long, the lowest-priority phrases at the end are dropped

Run from the repo root:
    pip install transformers
    python scripts/build_captions.py

Outputs:
    data/captions/<image>.txt     one caption per painting
    data/captions_report.csv      caption, token count, and whether it was trimmed
"""
import argparse, csv
from pathlib import Path

ORDER = ['characters', 'figures', 'action', 'pose', 'attire', 'setting', 'composition']
# max phrases per field, so long fields can't crowd out setting/composition
CAPS = {'characters': 3, 'figures': 4, 'action': 2, 'pose': 2, 'attire': 3, 'setting': 3, 'composition': 1}
MAX_TOKENS = 77  # CLIP limit, including start and end tokens


def get_counter():
    try:
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')  # SD 1.5's tokenizer
        return lambda s: len(tok(s).input_ids)
    except Exception as e:
        print(f'WARNING: CLIP tokenizer unavailable ({e}); using a rough word-based estimate')
        return lambda s: int(len(s.replace(',', ' , ').split()) * 1.3) + 2


def phrases(value):
    v = str(value or '').strip()
    if not v or v.lower() in ('none', 'nan', 'n/a'):
        return []
    return [p.strip() for p in v.split(',') if p.strip() and p.strip().lower() != 'none']


def build(row, trigger, count):
    parts, seen, trimmed = [trigger], set(), False
    for field in ORDER:
        for p in phrases(row.get(field))[:CAPS[field]]:
            if p.lower() in seen:
                continue
            candidate = ', '.join(parts + [p])
            if count(candidate) > MAX_TOKENS:
                trimmed = True
                break
            parts.append(p); seen.add(p.lower())
        if trimmed:
            break
    caption = ', '.join(parts)
    return caption, count(caption), trimmed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='data/metadata_tags.csv')
    ap.add_argument('--out', default='data/captions')
    ap.add_argument('--report', default='data/captions_report.csv')
    ap.add_argument('--trigger', default='rrvarma oil painting')
    args = ap.parse_args()

    count = get_counter()
    rows = list(csv.DictReader(open(args.tags, encoding='utf-8')))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    report = []
    for r in rows:
        caption, n, trimmed = build(r, args.trigger, count)
        (out / (Path(r['filename']).stem + '.txt')).write_text(caption, encoding='utf-8')
        report.append({'filename': r['filename'], 'split': r.get('split', ''),
                       'tokens': n, 'trimmed': 'y' if trimmed else 'n', 'caption': caption})

    with open(args.report, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['filename', 'split', 'tokens', 'trimmed', 'caption'])
        w.writeheader(); w.writerows(report)

    expected = {Path(r['filename']).stem for r in rows}
    stale = sorted(p.name for p in out.glob('*.txt') if p.stem not in expected)
    if stale:
        print(f'WARNING: caption files with no row in the tag table: {stale}')

    n_trim = sum(r['trimmed'] == 'y' for r in report)
    print(f'Wrote {len(report)} captions to {out}/ ({n_trim} trimmed to fit {MAX_TOKENS} tokens)')
    print(f'Report: {args.report}')


if __name__ == '__main__':
    main()
