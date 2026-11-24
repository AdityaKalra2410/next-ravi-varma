"""
Milestone 4: download the pose reference photos (free licences, Wikimedia Commons) into poses/.

Each photo shows ONE person in a clear full-body pose. OpenPose turns it into a skeleton and ControlNet
makes the painted figure follow it. Writes:
    poses/<name>.jpg       the photos (verified: they must open as images, not HTML error pages)
    poses/m4_scenes.csv    pose file + the epic scene we paint with it (used by eval_m4.py)
    poses/CREDITS.md       author + licence of each photo (needed for the report / video)

Colab, from the repo root:
    !python scripts/get_poses.py
"""
import csv, io, time
from pathlib import Path

# name, url (original file; resized to 1280 px after download), source page, author, licence, scene painted with this pose
POSES = [
    ("kneeling_side",
     "https://upload.wikimedia.org/wikipedia/commons/e/e6/Raja_vajrasana.jpg",
     "https://commons.wikimedia.org/wiki/File:Raja_vajrasana.jpg", "Marcocarvalho", "CC BY 3.0",
     "Hanuman kneeling in devotion, hands resting on his knees"),
    ("kneeling_front",
     "https://upload.wikimedia.org/wikipedia/commons/e/ec/A_style_of_Vajrasana_1.JPG",
     "https://commons.wikimedia.org/wiki/File:A_style_of_Vajrasana_1.JPG", "Thamizhpparithi Maari", "CC BY-SA 3.0",
     "Bharata kneeling on the ground in the forest, mourning his father"),
    ("archer",
     "https://upload.wikimedia.org/wikipedia/commons/7/7d/Archer_Doreen_Roberts_%2814845626809%29.jpg",
     "https://commons.wikimedia.org/wiki/File:Archer_Doreen_Roberts_(14845626809).jpg",
     "Archives of Ontario", "Public domain",
     "Arjuna drawing his bow at Draupadi's swayamvara"),
    ("warrior",
     "https://upload.wikimedia.org/wikipedia/commons/1/1d/Virabhadrasana_II_-_Warrior_II_Pose.jpg",
     "https://commons.wikimedia.org/wiki/File:Virabhadrasana_II_-_Warrior_II_Pose.jpg", "lululemon athletica",
     "CC BY 2.0",
     "Bhima standing in a wide battle stance with arms outstretched on the battlefield of Kurukshetra"),
    ("sage_meditating",
     "https://upload.wikimedia.org/wikipedia/commons/b/b5/Babaji.jpg",
     "https://commons.wikimedia.org/wiki/File:Babaji.jpg", "Sananda Lal Ghosh", "Public domain",
     "Sage Vishwamitra meditating cross-legged in the forest"),
    ("seated_cross_legged",
     "https://upload.wikimedia.org/wikipedia/commons/9/9a/A_style_of_Padmasana.JPG",
     "https://commons.wikimedia.org/wiki/File:A_style_of_Padmasana.JPG", "Thamizhpparithi Maari", "CC BY-SA 3.0",
     "Valmiki seated cross-legged at his hermitage, composing the Ramayana"),
    ("folded_hands",
     "https://upload.wikimedia.org/wikipedia/commons/1/1c/A_view_of_Surya_namaskar_position.JPG",
     "https://commons.wikimedia.org/wiki/File:A_view_of_Surya_namaskar_position.JPG", "Thamizhpparithi Maari",
     "CC BY-SA 3.0",
     "Vibhishana standing with folded hands, asking Rama for refuge"),
    ("arms_raised",
     "https://upload.wikimedia.org/wikipedia/commons/0/00/Tadasana_yoga_pose.jpg",
     "https://commons.wikimedia.org/wiki/File:Tadasana_yoga_pose.jpg", "Hindisign", "CC BY-SA 4.0",
     "Bhagiratha standing with both arms raised, praying for the Ganga to descend"),
]

UA = {"User-Agent": "rrv-course-project/1.0 (student project; python-requests)"}


def fetch(url, tries=6):
    import requests
    from PIL import Image
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200:
                im = Image.open(io.BytesIO(r.content)); im.load()      # fails on HTML error pages
                return im.convert("RGB")
            wait = int(r.headers.get("Retry-After", 0) or 0) or 15 * (k + 1)
            print(f"  HTTP {r.status_code}, waiting {wait} s")
        except Exception as e:
            wait = 15 * (k + 1)
            print(f"  {type(e).__name__}: {e}, waiting {wait} s")
        time.sleep(min(wait, 120))                                     # Wikimedia rate limits
    raise RuntimeError(f"could not download {url}")


def main():
    from PIL import Image
    out = Path("poses"); out.mkdir(exist_ok=True)
    for name, url, *_ in POSES:
        path = out / f"{name}.jpg"
        try:
            Image.open(path).verify(); print(f"ok (exists)  {path}"); continue
        except Exception:
            pass
        im = fetch(url)
        im.thumbnail((1280, 1280))                                     # originals can be 5000 px
        im.save(path, quality=95)
        print(f"downloaded   {path}")
        time.sleep(3)
    with open(out / "m4_scenes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["pose", "scene"])
        for name, *_, scene in POSES:
            w.writerow([f"poses/{name}.jpg", scene])
    with open(out / "CREDITS.md", "w", encoding="utf-8") as f:
        f.write("# Pose reference photos (Wikimedia Commons)\n\n"
                "Used only as pose references: OpenPose extracts the skeleton, nothing else is copied.\n\n")
        for name, _, page, author, lic, _ in POSES:
            f.write(f"- `{name}.jpg`: {author}, {lic}, {page}\n")
    print(f"saved {len(POSES)} poses, poses/m4_scenes.csv, poses/CREDITS.md")


if __name__ == "__main__":
    main()
