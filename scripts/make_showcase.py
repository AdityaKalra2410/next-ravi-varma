"""
Showcase paintings for the demo video (heroines of the epics, Varma's strongest subject).

For every scene and seed, saves the final painting from the same pipeline as the app:
  text only : paint(scene)                          -> showcase/<name>_s<seed>.jpg
  with pose : paint_with_pose(scene, photo)         -> showcase/<name>_s<seed>.jpg  (+ _skeleton.png)
              and the same prompt/seed WITHOUT ControlNet -> showcase/<name>_nocn_s<seed>.jpg (before/after)
Pick the best seed of each by eye for the video; the evaluation numbers stay those of eval_m4/.

Colab T4, from the repo root (~15 min, skips images that already exist):
    !python scripts/make_showcase.py
    !zip -qr showcase.zip showcase && echo done      # then download showcase.zip from the Files panel
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import expand_prompt as ep
import controlnet_pose as cp

TEXT = [
    ("damayanti", "Damayanti talking to a golden swan by a lotus pond"),
    ("sita", "Sita sitting alone in the Ashoka grove, guarded by demonesses"),
    ("radha", "Radha waiting by the Yamuna river at dusk, holding a garland"),
    ("saraswati", "Saraswati seated on a rock by a lotus pond, playing the veena"),
    ("shakuntala", "Shakuntala looking back at King Dushyanta in the forest"),
]
POSE = [
    ("chitrangada", "poses/archer.jpg", "Chitrangada, the warrior princess, drawing her bow"),
    ("satyabhama", "poses/warrior.jpg", "Satyabhama in a wide battle stance with arms outstretched on the battlefield"),
    ("draupadi", "poses/woman_warrior1.jpg", "Draupadi with arms outstretched, pleading in the royal court of Hastinapura"),
    ("shakuntala_seated", "poses/woman_seated.jpg", "Shakuntala seated cross-legged in Kanva's hermitage, lost in thought"),
]
SEEDS = [0, 1]


def main():
    import torch
    from PIL import Image
    out = Path("showcase"); out.mkdir(exist_ok=True)
    pipe = ep.load_pipe()
    for name, scene in TEXT:
        for s in SEEDS:
            f = out / f"{name}_s{s}.jpg"
            if f.exists():
                continue
            img, prompt = ep.paint(scene, best_of=4, seed=s)
            img.save(f, quality=95); print("saved", f, "|", prompt)
    for name, photo, scene in POSE:
        ref = Image.open(photo)
        for s in SEEDS:
            f = out / f"{name}_s{s}.jpg"
            if not f.exists():
                img, prompt, skel, info = ep.paint_with_pose(scene, ref, best_of=4, seed=s, return_info=True)
                img.save(f, quality=95); skel.save(out / f"{name}_skeleton.png")
                print("saved", f, "| pose", [round(x, 2) if x is not None else None for x in info["pose_sims"]], "|", prompt)
                # same prompt, same first seed and size, no ControlNet: for a before/after wipe
                g = torch.Generator("cuda").manual_seed(info["seeds"][info["best"]])
                nocn = pipe(prompt, negative_prompt=ep.NEG_POSE, width=skel.width, height=skel.height,
                            num_inference_steps=30, guidance_scale=7.5, generator=g).images[0]
                nocn.save(out / f"{name}_nocn_s{s}.jpg", quality=95)
    print("done:", len(list(out.glob("*.jpg"))), "images in", out)


if __name__ == "__main__":
    main()
