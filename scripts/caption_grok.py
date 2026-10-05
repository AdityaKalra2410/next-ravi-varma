"""
Step 4 (Grok version): same template, input and output as caption_gemini.py.

Setup in Colab (any runtime, no GPU needed):
    1. Secrets (key icon) -> add XAI_API_KEY -> enable notebook access
    2. In a cell:
         import os; from google.colab import userdata
         os.environ["XAI_API_KEY"] = userdata.get("XAI_API_KEY")
    3. !pip install -q openai
    4. !python scripts/caption_grok.py --limit 3 --overwrite   # test
       !python scripts/caption_grok.py                         # full run (resumes)
"""
import argparse, base64, csv, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from caption_llava import FIELDS, load_titles
from caption_gemini import build_prompt, load_image_bytes

JSON_RULE = ("\n\nReturn ONLY a JSON object with exactly these keys, all string values: "
             + ", ".join(FIELDS) + ". No markdown, no extra text.")


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/metadata.csv")
    ap.add_argument("--review", default="data/captions_review.csv")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="data/metadata_tags.csv")
    ap.add_argument("--model", default="grok-4.7")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("XAI_API_KEY")
    if not key:
        sys.exit("XAI_API_KEY not set. See the setup steps at the top of this file.")

    rows = load_titles(args.meta, args.review)
    if args.limit:
        rows = rows[:args.limit]
    cols = ["filename", "title", "split", *FIELDS, "parse_ok", "checked", "raw"]
    if args.overwrite and Path(args.out).exists():
        Path(args.out).unlink()
    done = set()
    if Path(args.out).exists():
        done = {r["filename"] for r in csv.DictReader(open(args.out, encoding="utf-8"))}
    todo = [r for r in rows if r["filename"] not in done]
    print(f"{len(rows)} oil paintings, model = {args.model}; {len(done)} done, {len(todo)} to go")
    if not todo:
        return

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")

    new_file = not Path(args.out).exists()
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        for i, r in enumerate(todo, 1):
            b64 = base64.b64encode(load_image_bytes(Path(args.images) / r["filename"])).decode()
            messages = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                {"type": "text", "text": build_prompt(r["title"]) + JSON_RULE},
            ]}]
            text = None
            for attempt in range(6):  # waits 15s, 30s, 60s, 2m, 4m, 8m
                try:
                    resp = client.chat.completions.create(model=args.model, messages=messages, temperature=0)
                    text = resp.choices[0].message.content
                    break
                except Exception as e:
                    msg = str(e)
                    if any(s in msg.lower() for s in ("credit", "billing", "permission", "api key", "401", "403")):
                        sys.exit(f"Account/key problem, retrying won't help:\n{msg}")
                    wait = 15 * 2 ** attempt
                    print(f"   error: {msg[:200]} -> retrying in {wait}s")
                    time.sleep(wait)
            if text is None:
                sys.exit(f"Giving up on {r['filename']}. Rerun later; finished rows are saved.")

            data = parse_json(text)
            tags = {k: str(data.get(k, "")).strip() for k in FIELDS}
            ok = all(tags.values())
            w.writerow({**r, **tags, "parse_ok": "y" if ok else "n", "checked": "", "raw": text})
            f.flush()
            print(f"[{i}/{len(todo)}] {r['filename']} {'' if ok else '(MISSING FIELDS, check raw)'}")
            print("   ", tags["figures"], "|", tags["action"])

    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
