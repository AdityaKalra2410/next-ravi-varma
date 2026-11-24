"""
Fix the v2 training data before retraining.

1. Re-downloads every kept image that isn't a real image (Wikimedia error pages saved as .jpg),
   slowly and with retries, and checks each one opens as a picture.
2. Removes rrv_0016 (print version of Tilottama, Wellcome collection) from training.
3. Checks the final training set: all real images, no test paintings, no rrv_0097.

Run from the repo root:  python scripts/fix_v2.py
Then:                    python scripts/build_captions.py
                         cd data && python ../scripts/Filtering_images.py && cd ..
"""
import csv, io, sys, time
import requests
from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
H = {"User-Agent": "next-ravi-varma/1.0 (BITS Pilani student project; course CS F407)"}
META, TAGS, IMG = "data/metadata.csv", "data/metadata_tags.csv", "data/images"
TEST = {"rrv_0022.jpg", "rrv_0027.jpg", "rrv_0043.jpg", "rrv_0047.jpg", "rrv_0051.jpg",
        "rrv_0056.jpg", "rrv_0086.jpg", "rrv_0093.jpg", "rrv_0103.jpg"}
DROP = {"rrv_0016.jpg": "print version of rrv_0064 (Wellcome chromolithograph), excluded: oil-only"}


def is_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def download(title, path):
    for attempt in range(5):
        try:
            j = requests.get(API, headers=H, timeout=30, params={
                "action": "query", "titles": title, "prop": "imageinfo",
                "iiprop": "url", "iiurlwidth": 1536, "format": "json"}).json()
            info = next(iter(j["query"]["pages"].values()))["imageinfo"][0]
            r = requests.get(info.get("thumburl") or info["url"], headers=H, timeout=60)
            if r.ok and r.content[:2] == b"\xff\xd8":       # real JPEG bytes
                Image.open(io.BytesIO(r.content)).verify()
                open(path, "wb").write(r.content)
                return True
            print(f"   got HTTP {r.status_code} / not a JPEG, retrying")
        except Exception as e:
            print(f"   error: {e}")
        time.sleep(10 * (attempt + 1))                       # back off: 10s, 20s, 30s...
    return False


meta = list(csv.DictReader(open(META, encoding="utf-8")))
cols = list(meta[0].keys())

# 1. re-download broken images
broken = [m for m in meta if m["keep"] == "y" and not is_image(f"{IMG}/{m['filename']}")
          and m["filename"] not in DROP]
print(f"{len(broken)} broken image files:", [m["filename"] for m in broken])
for m in broken:
    ok = download(m["commons_title"], f"{IMG}/{m['filename']}")
    print(("fixed " if ok else "FAILED ") + m["filename"])
    time.sleep(3)

# 2. drop rrv_0016 from training
for m in meta:
    if m["filename"] in DROP:
        m.update(keep="n", split="", drop_reason=DROP[m["filename"]])
with open(META, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(meta)
tags = [t for t in csv.DictReader(open(TAGS, encoding="utf-8")) if t["filename"] not in DROP]
with open(TAGS, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=tags[0].keys()); w.writeheader(); w.writerows(tags)

# 3. final checks
train = [m["filename"] for m in meta if m["keep"] == "y" and m["split"] == "train"]
bad = [f for f in train if not is_image(f"{IMG}/{f}")]
leak = [f for f in train if f in TEST or f == "rrv_0097.jpg"]
print(f"\ntrain images: {len(train)} (expected 63)")
print("not real images:", bad or "none")
print("test leakage:", leak or "none")
if bad or leak:
    sys.exit("Fix the problems above before training.")
print("OK - now run build_captions.py and Filtering_images.py")
