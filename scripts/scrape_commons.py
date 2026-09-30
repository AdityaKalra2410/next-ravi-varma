"""Download Raja Ravi Varma paintings from Wikimedia Commons + write metadata.csv.

Usage: python scripts/scrape_commons.py --depth 3 --min-side 768
Everything it downloads still needs a manual review pass (fill the `keep` and `type` columns).
"""
import argparse, csv, html, os, re, time
import requests

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks for a descriptive User-Agent with contact info
HEADERS = {"User-Agent": "NextRaviVarma-course-project/0.1 (BITS Goa; contact: your-email@example.com)"}
ROOT_CAT = "Category:Paintings by Raja Ravi Varma"


def api(params):
    params = {**params, "format": "json"}
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def list_files(cat, depth, seen):
    files, cont = [], {}
    while True:
        d = api({"action": "query", "list": "categorymembers", "cmtitle": cat,
                 "cmtype": "file|subcat", "cmlimit": "500", **cont})
        for m in d["query"]["categorymembers"]:
            if m["ns"] == 6:
                files.append(m["title"])
            elif m["ns"] == 14 and depth > 0 and m["title"] not in seen:
                seen.add(m["title"])
                files += list_files(m["title"], depth - 1, seen)
        if "continue" not in d:
            return files
        cont = d["continue"]


def clean(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=3, help="subcategory recursion depth")
    ap.add_argument("--min-side", type=int, default=768)
    ap.add_argument("--width", type=int, default=1536, help="download width (thumbnail API)")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    titles = sorted(set(list_files(ROOT_CAT, args.depth, {ROOT_CAT})))
    print(f"found {len(titles)} files")

    rows = []
    for i in range(0, len(titles), 50):
        d = api({"action": "query", "titles": "|".join(titles[i:i + 50]), "prop": "imageinfo",
                 "iiprop": "url|size|mime|extmetadata", "iiurlwidth": str(args.width)})
        for page in d["query"]["pages"].values():
            info = (page.get("imageinfo") or [None])[0]
            if not info or info["mime"] not in ("image/jpeg", "image/png"):
                continue
            if min(info["width"], info["height"]) < args.min_side:
                continue
            meta = info.get("extmetadata", {})
            ext = ".png" if info["mime"] == "image/png" else ".jpg"
            fname = f"rrv_{len(rows):04d}{ext}"
            url = info.get("thumburl") or info["url"]
            img = requests.get(url, headers=HEADERS, timeout=60)
            if img.status_code != 200:
                continue
            with open(os.path.join(args.out, fname), "wb") as f:
                f.write(img.content)
            rows.append({
                "filename": fname,
                "commons_title": page["title"],
                "title": clean(meta.get("ObjectName", {}).get("value")),
                "date": clean(meta.get("DateTimeOriginal", {}).get("value")),
                "license": clean(meta.get("LicenseShortName", {}).get("value")),
                "orig_width": info["width"], "orig_height": info["height"],
                "source_url": info["descriptionurl"],
                "type": "",   # fill manually: oil / oleograph / sketch / photo_of_painting
                "keep": "",   # fill manually: y / n
                "split": "",  # train / test (set after review)
            })
            time.sleep(0.5)  # be polite to Wikimedia
        print(f"processed {min(i + 50, len(titles))}/{len(titles)}, kept {len(rows)}")

    if not rows:
        print("nothing downloaded; check the category name / network")
        return
    with open("data/metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote data/metadata.csv with {len(rows)} rows")


if __name__ == "__main__":
    main()
