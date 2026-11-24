"""
Milestone 4 evaluation: does ControlNet (OpenPose) make the painted figure follow the user's pose,
and what does it cost in scene match and style?

8 pose photos (poses/m4_scenes.csv, from Wikimedia Commons) x 2 seeds, three conditions. Everything is
identical (prompt with character looks, negative prompt, image size, best-of-4, high-res refine) EXCEPT:
  no_cn      : no ControlNet (plain SD 1.5 + LoRA): the pose comes from the text only
  cn         : + ControlNet OpenPose, best-of-4 picked by the CLIP judge (as in Milestone 3)
  cn_posesel : + ControlNet, best-of-4 also checks the pose (final M4 pipeline = the app)
cn and cn_posesel use the SAME 4 candidates; only the selection differs.

Scores on the final images:
  pose_pck    : OpenPose is run again on the painting; share of the photo's joints that land within
                0.1 x body size of the right place (PCK@0.1). 1.0 = every joint in place.
  person_found: OpenPose detects a person in the painting at all
  scene_match : CLIP ViT-L/14 similarity to the scene text, 0-100 (same as eval_m3)
  style_sim   : CLIP ViT-L/14 similarity to the 9 held-out Varma paintings (same as eval_m3)
Honest caveats: cn_posesel's selector uses OpenPose too, so part of its pose gain is "picking what the
metric likes" (like CLIP best-of-N in M3); 16 images per condition, so small gaps are only indicative.
The best-of-N CLIP judge (ViT-B/32, training paintings) is different from the evaluator (ViT-L/14, test paintings).

Colab T4, from the repo root (~25 min, resumable: finished images are skipped):
    !python scripts/eval_m4.py
Outputs in eval_m4/: scores.csv, summary.csv, grid.png (photo | skeleton | no_cn | cn_posesel)
"""
import argparse, csv, gc, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import expand_prompt as ep
import controlnet_pose as cp

CONDS = ["no_cn", "cn", "cn_posesel"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="poses/m4_scenes.csv")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--best-of", type=int, default=4)
    ap.add_argument("--lora", default="models/lora_v1.safetensors")
    ap.add_argument("--report", default="data/captions_report.csv")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="eval_m4")
    args = ap.parse_args()

    import torch
    from PIL import Image, ImageDraw
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.scenes, encoding="utf-8")))

    # ---------- 1. generate ----------
    meta_file = out / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    for pi, r in enumerate(rows):
        ref_img = Image.open(r["pose"])
        for s in range(args.seeds):
            key = f"p{pi:02d}_s{s}"
            paths = {c: out / "images" / f"{c}_{key}.png" for c in CONDS}
            if all(p.exists() for p in paths.values()) and key in meta:
                continue
            ep.load_pipe(args.lora)                                  # SD + LoRA once; ControlNet shares it
            # ControlNet candidates (pose_select decides the final pick; CLIP-only pick computed here)
            _, prompt, pose, info = ep.paint_with_pose(r["scene"], ref_img, best_of=args.best_of, seed=s,
                                                       lora=args.lora, hires=False, return_info=True)
            cands, clip_scores, sims = info["candidates"], info["clip_scores"], info["pose_sims"]
            pick = {"cn": max(range(len(cands)), key=lambda i: clip_scores[i]), "cn_posesel": info["best"]}
            # no-ControlNet candidates: same prompt, negatives, size and seeds, plain SD + LoRA
            pipe = ep.load_pipe(args.lora)
            plain = []
            for sd in info["seeds"]:
                g = torch.Generator("cuda").manual_seed(sd)
                plain.append(pipe(prompt, negative_prompt=ep.NEG_POSE, width=pose.width, height=pose.height,
                                  num_inference_steps=30, guidance_scale=7.5, generator=g).images[0])
            plain_scores = ep.judge_scores(plain, r["scene"])
            pick_plain = max(range(len(plain)), key=lambda i: plain_scores[i])
            # high-res refine of each chosen image (same for every condition)
            base = cp.load_cn_pipe(args.lora)
            refined = {}
            for c in ("cn", "cn_posesel"):
                i = pick[c]
                if i not in refined:
                    refined[i] = ep.refine(cands[i], prompt, info["seeds"][i], ep.NEG_POSE, base=base)
                refined[i].save(paths[c])
            ep.refine(plain[pick_plain], prompt, info["seeds"][pick_plain], ep.NEG_POSE, base=base).save(paths["no_cn"])
            pose.save(out / "images" / f"skeleton_{key}.png")
            meta[key] = {"pose": r["pose"], "scene": r["scene"], "prompt": prompt, "aspect": pose.width / pose.height,
                         "ref": info["ref"], "cand_pose_pck": sims, "picks": {**pick, "no_cn": pick_plain}}
            meta_file.write_text(json.dumps(meta, indent=1))
            print(f"{key} done | pose of the 4 ControlNet candidates: {[round(x, 2) if x is not None else None for x in sims]}")

    # ---------- 2. pose accuracy (OpenPose again on the final paintings) ----------
    records = []
    for pi, r in enumerate(rows):
        for s in range(args.seeds):
            key = f"p{pi:02d}_s{s}"; m = meta[key]
            ref = [tuple(k) if k else None for k in m["ref"]] if m["ref"] else None
            for c in CONDS:
                path = out / "images" / f"{c}_{key}.png"
                people = cp.keypoints(Image.open(path))
                person = cp.main_person(people) if people else None
                pck = cp.pose_similarity(ref, person, m["aspect"])
                records.append({"condition": c, "pose_id": pi, "seed": s, "pose": m["pose"], "scene": m["scene"],
                                "prompt": m["prompt"], "path": str(path),
                                "person_found": int(person is not None and sum(k is not None for k in person) >= 4),
                                "pose_pck": round(pck, 3) if pck is not None else ""})
    ep._pipe = ep._judge = ep._i2i = None; cp._cnpipe = None; gc.collect(); torch.cuda.empty_cache()

    # ---------- 3. scene match and style (CLIP ViT-L/14, same as eval_m3) ----------
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    @torch.no_grad()
    def img_emb(paths):
        embs = []
        for i in range(0, len(paths), 16):
            ims = [Image.open(x).convert("RGB") for x in paths[i:i + 16]]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to("cuda")
            e = clip.visual_projection(clip.vision_model(pixel_values=px).pooler_output)
            embs.append(e / e.norm(dim=-1, keepdim=True))
        return torch.cat(embs)

    @torch.no_grad()
    def txt_emb(texts):
        t = proc.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        e = clip.text_projection(clip.text_model(**t).pooler_output)
        return e / e.norm(dim=-1, keepdim=True)

    test = [str(Path(args.images) / r["filename"]) for r in csv.DictReader(open(args.report, encoding="utf-8"))
            if r["split"] == "test"]
    centroid = img_emb(test).mean(0); centroid /= centroid.norm()
    g = img_emb([r["path"] for r in records]); t = txt_emb([r["scene"] for r in records])
    for i, r in enumerate(records):
        r["scene_match"] = round(100 * max((g[i] @ t[i]).item(), 0), 2)
        r["style_sim"] = round((g[i] @ centroid).item(), 4)

    with open(out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)

    def mean(xs):
        xs = [x for x in xs if x != ""]
        return sum(xs) / len(xs) if xs else float("nan")

    summary = []
    for c in CONDS:
        rs = [r for r in records if r["condition"] == c]
        summary.append({"condition": c, "pose_pck": round(mean([r["pose_pck"] for r in rs]), 3),
                        "person_found": round(mean([r["person_found"] for r in rs]), 3),
                        "scene_match": round(mean([r["scene_match"] for r in rs]), 2),
                        "style_sim": round(mean([r["style_sim"] for r in rs]), 4), "n": len(rs)})
    wins = {c: 0 for c in CONDS}                                     # best pose per (pose photo), avg over seeds
    for pi in range(len(rows)):
        m = {c: mean([r["pose_pck"] for r in records if r["condition"] == c and r["pose_id"] == pi]) for c in CONDS}
        wins[max(m, key=m.get)] += 1
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    # ---------- 4. grid: photo | skeleton | no_cn | cn_posesel (seed 0) ----------
    S, H = 220, 26
    heads = ["pose photo", "skeleton", "no ControlNet", "ControlNet (final)"]
    grid = Image.new("RGB", (S * 4, H + S * len(rows)), "white"); d = ImageDraw.Draw(grid)
    for ci, h in enumerate(heads):
        d.text((ci * S + 8, 8), h, fill="black")
    for pi, r in enumerate(rows):
        ims = [Image.open(r["pose"]), Image.open(out / "images" / f"skeleton_p{pi:02d}_s0.png"),
               Image.open(out / "images" / f"no_cn_p{pi:02d}_s0.png"),
               Image.open(out / "images" / f"cn_posesel_p{pi:02d}_s0.png")]
        for ci, im in enumerate(ims):
            im = im.convert("RGB"); im.thumbnail((S, S))
            grid.paste(im, (ci * S + (S - im.width) // 2, H + pi * S + (S - im.height) // 2))
    grid.save(out / "grid.png")

    print("\n=== M4 summary ===")
    for s in summary:
        print(s)
    print("best pose accuracy per pose photo:", wins)
    print(f"saved: {out}/scores.csv, summary.csv, grid.png")


if __name__ == "__main__":
    main()
