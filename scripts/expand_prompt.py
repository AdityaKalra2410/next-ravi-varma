"""
Milestone 3: text parsing, prompt expansion and generation.

FRONT END: call  paint(scene, expand=False, best_of=4, seed=0) -> (image, prompt)

Final pipeline (best scores in eval_m3):
    "rrvarma oil painting, <scene>"  +  style-aware negative prompt  +  best-of-4 selection
    (4 candidates, the best picked by CLIP ViT-B/32 against the TRAINING paintings)
Optional LLM expansion (expand=True): Qwen2.5-7B-Instruct (4-bit) fills a JSON template with the same
fields as our training tags; "hybrid" keeps the user's sentence and adds figures, pose, setting,
composition. Useful for very short inputs.

Colab (T4) from the repo root:
    !pip install -q diffusers transformers accelerate peft bitsandbytes
    !pip uninstall -y torchao
    !python scripts/expand_prompt.py --generate "Bhishma lying on a bed of arrows"            # final pipeline
    !python scripts/expand_prompt.py --generate --expand "Bhishma dying"                      # with LLM
    !python scripts/expand_prompt.py "Arjuna aiming his bow"                                  # show expansion only
"""
import argparse, re
from pathlib import Path

TRIGGER = "rrvarma oil painting"
LLM_ID = "Qwen/Qwen2.5-7B-Instruct"   # 4-bit: ~5.5 GB, fits on a T4 next to SD 1.5
SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
NEG = "blurry, deformed, extra limbs, bad anatomy, text, watermark"
# Also steer away from the calendar-print / poster look that "Ravi Varma" usually produces online
NEG_PLUS = NEG + ", calendar art, poster, flat colors, oversaturated, cartoon, anime, photograph, 3d render"
# With a pose and no clothes in the prompt, SD sometimes paints the figure nude
NEG_POSE = NEG_PLUS + ", nude, naked, extra figures, extra people, deformed face, deformed hands"

FIELDS = ["characters", "figures", "action", "pose", "attire", "setting", "composition"]

SYSTEM = """You plan images for a Stable Diffusion model fine-tuned on Raja Ravi Varma's oil paintings.
Turn the user's scene from the Indian epics into a JSON object with exactly these keys:
- "characters": the named characters, spelled exactly as in the scene (add only those the scene clearly implies)
- "figures": every visible person/animal with gender and age, e.g. "bearded king in his fifties, young warrior lying dead"
- "action": what is happening
- "pose": body positions and gestures of each figure (kneeling, lying, arms raised...)
- "attire": 19th-century Indian clothing and jewellery as Ravi Varma painted it (saris, dhotis, angavastrams, armour, gold crowns, necklaces, armlets), with colours
- "setting": place and background
- "composition": one of "single-figure portrait", "two-figure composition", "group scene", "full scene"
Rules:
- Each value is a few short comma-separated phrases, not sentences.
- Describe only what should be visible. Count the figures correctly.
- Saris are for women only; men wear dhotis, angavastrams, armour, crowns and jewellery.
- No style or technique words (no "oil painting", "chiaroscuro", "realistic", "dramatic lighting", "masterpiece").
- Output only the JSON object."""

EXAMPLES = [
    ("Draupadi pleading in the royal court of Hastinapura",
     {"characters": "Draupadi, Dushasana, Bhishma",
      "figures": "young woman, young prince, white-bearded elder, seated nobles",
      "action": "woman pleading for help, prince pulling her sari",
      "pose": "woman with hands joined in prayer, elders looking down in shame",
      "attire": "red silk sari with gold border, gold crowns, pearl necklaces, armlets",
      "setting": "royal court with carved pillars and a throne",
      "composition": "full scene"}),
    ("Krishna playing the flute for Radha by the Yamuna at dusk",
     {"characters": "Krishna, Radha",
      "figures": "young blue-skinned man, young woman",
      "action": "man playing the flute, woman listening shyly",
      "pose": "standing close together, her head tilted down",
      "attire": "yellow dhoti, peacock-feather crown, green sari with gold border, gold jewellery",
      "setting": "riverbank at dusk, trees",
      "composition": "two-figure composition"}),
]

STYLE_WORDS = re.compile(r"\b(chiaroscuro|masterpiece|realistic|photorealistic|dramatic lighting|oil painting texture|"
                         r"oil canvas|oil on canvas|in the style of[^,]*|ravi varma style|highly detailed|8k|4k)\b", re.I)

# ---------- character looks ----------
# SD 1.5 does not know what most epic characters look like (it painted Bharata as an old bald sage and
# Arjuna as a woman) and, with no clothes in the prompt, it picks anything (even nude). This short look-up
# adds each named character's gender, age and traditional attire/attributes, as Ravi Varma painted them.
# Content only, no style words. Spelling variants map to one entry.
LOOKS = {
    "rama":         "Rama a young man with blue skin, gold crown, yellow silk dhoti, holding a bow",
    "lakshmana":    "Lakshmana a young man, gold crown, dhoti, holding a bow",
    "sita":         "Sita a young woman in a silk sari with gold jewellery",
    "hanuman":      "Hanuman the monkey god with a monkey face, long tail, gold crown, holding a mace",
    "ravana":       "Ravana a dark demon king with a moustache, tall gold crown, rich robes",
    "vibhishana":   "Vibhishana a middle-aged man with a moustache, gold crown, silk dhoti",
    "bharata":      "Bharata a young man with a moustache, simple white dhoti, no crown",
    "dasharatha":   "Dasharatha an old king with a white beard, gold crown",
    "krishna":      "Krishna a young man with blue skin, peacock-feather crown, yellow silk dhoti",
    "radha":        "Radha a young woman in a silk sari with gold jewellery",
    "arjuna":       "Arjuna a young warrior man with a moustache, gold crown, armour, holding a tall wooden bow with a bowstring",
    "bhima":        "Bhima a muscular warrior man with a moustache, gold crown, holding a mace",
    "yudhishthira": "Yudhishthira a bearded king in white silk, gold crown",
    "karna":        "Karna a warrior man with gold armour and earrings",
    "duryodhana":   "Duryodhana a warrior king with a moustache, gold crown, holding a mace",
    "dushasana":    "Dushasana a young prince with a moustache, gold crown",
    "draupadi":     "Draupadi a young woman in a red silk sari with gold jewellery",
    "bhishma":      "Bhishma an old warrior with a long white beard, armour",
    "drona":        "Drona an old man with a white beard, holding a bow",
    "shakuntala":   "Shakuntala a young woman in a simple sari, flowers in her hair",
    "dushyanta":    "Dushyanta a young king with a moustache, gold crown",
    "damayanti":    "Damayanti a young woman in a silk sari with gold jewellery",
    "nala":         "Nala a young king with a moustache, gold crown",
    "vishwamitra":  "Vishwamitra an old sage with a long white beard and matted hair, saffron cloth",
    "valmiki":      "Valmiki an old sage with a long white beard, saffron cloth",
    "vasishtha":    "Vasishtha an old sage with a long white beard, saffron cloth",
    "bhagiratha":   "Bhagiratha a young man with a moustache, bare chest, white dhoti, empty hands",
    "shiva":        "Shiva a man with ash-blue skin, matted hair, crescent moon, tiger skin, snake around his neck",
    "saraswati":    "Saraswati a young woman in a white sari, holding a veena",
    "lakshmi":      "Lakshmi a young woman in a red silk sari, standing on a lotus",
    "jatayu":       "Jatayu a giant eagle",
}
ALIASES = {"droupathi": "draupadi", "sakunthala": "shakuntala", "sairandhri": "draupadi", "sairandri": "draupadi",
           "vishvamitra": "vishwamitra", "bheema": "bhima", "bheeshma": "bhishma", "sri rama": "rama", "ram": "rama",
           "vibheeshana": "vibhishana", "yudhisthira": "yudhishthira", "dushyant": "dushyanta"}


def character_looks(scene):
    """Looks of the characters named in the scene, in the order they appear."""
    found = []
    for key in list(LOOKS) + list(ALIASES):
        m = re.search(rf"\b{key}\b", scene, re.I)
        if m:
            name = ALIASES.get(key, key)
            if name not in [n for _, n in found]:
                found.append((m.start(), name))
    return [LOOKS[n] for _, n in sorted(found)]


def add_looks(prompt, scene, max_chars=None):
    """prompt + the named characters' looks, within CLIP's 77 tokens (the scene itself always stays).
    max_chars: with a pose, only the first N characters (N = people in the skeleton) get a look, otherwise
    e.g. "Vibhishana asking Rama for refuge" would give the ONE posed figure Rama's blue skin and bow."""
    looks = character_looks(scene)[:max_chars]
    out = prompt
    for look in looks:
        cand = f"{out}, {look}"
        if clip_tokens(cand) > 77:
            break
        out = cand
    return out


_llm = _tok = _clip_tok = _pipe = _judge = _i2i = None


def clip_tokens(text):
    global _clip_tok
    if _clip_tok is None:
        from transformers import CLIPTokenizer
        _clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    return len(_clip_tok(text).input_ids)


def scene_names(scene):
    """Capitalised words in the scene that look like names (Yudhishthira, Karna, Kurukshetra...)."""
    stop = {"The", "A", "An", "On", "In", "Of", "At", "By", "And", "With", "Over", "His", "Her",
            "King", "Queen", "Lord", "Prince", "Princess", "Sage", "Goddess", "God"}
    return [w for w in re.findall(r"\b[A-Z][a-z]+\b", scene) if w not in stop]


def parse_json(raw):
    import json
    m = re.search(r"\{.*\}", raw, flags=re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return {k: str(d.get(k, "")).strip() for k in FIELDS}


HYBRID_FIELDS = ["figures", "pose", "setting", "composition"]


def fix_fields(raw, scene):
    """Parse the LLM JSON and repair it: scene names kept, places moved to setting,
    composition consistent with the number of figures. Returns None if unusable."""
    fields = parse_json(raw)
    if sum(bool(v) for v in fields.values()) < 3:
        return None
    # keep every name the user typed; if the LLM garbled one, also drop names it invented
    names = scene_names(scene)
    llm_chars = [c.strip() for c in fields["characters"].split(",") if c.strip()]
    missing = [n for n in names if n.lower() not in raw.lower()]
    # a missing name right after "of/at/in/on/near" is a place (e.g. "battlefield of Kurukshetra")
    places = [n for n in missing if re.search(rf"\b(of|at|in|on|near|by)\s+(the\s+)?{n}\b", scene)]
    people = [n for n in missing if n not in places]
    if people:
        llm_chars = [c for c in llm_chars if any(w.lower() in scene.lower() for w in c.split())]
    fields["characters"] = ", ".join(people + [c for c in llm_chars if c not in people])
    if places:
        fields["setting"] = ", ".join(places + ([fields["setting"]] if fields["setting"] else []))
    # composition must agree with the number of characters
    n_chars = len([c for c in fields["characters"].split(",") if c.strip()])
    def count(item):
        if re.search(r"\b(three|several|many|crowd|group|soldiers|guards|army)\b", item): return 3
        if re.search(r"\b(two|pair|couple)\b|\b(women|men|demonesses|sages|warriors|maids|attendants)\b", item): return 2
        return 1
    n_figs = sum(count(f.strip()) for f in fields["figures"].lower().split(",") if f.strip())
    n_chars = max(n_chars, n_figs)
    if n_chars >= 2 and "single" in fields["composition"].lower():
        fields["composition"] = "two-figure composition" if n_chars == 2 else "group scene"
    return fields


def assemble(fields, scene, mode="hybrid"):
    """mode="full":   trigger + all fields in caption order (pure caption format)
       mode="hybrid": trigger + the user's scene sentence + figures, pose, setting, composition"""
    scene = scene.strip().rstrip(".")
    if fields is None:
        return f"{TRIGGER}, {scene}"
    keys = FIELDS if mode == "full" else HYBRID_FIELDS
    phrases = [scene] if mode == "hybrid" else []
    for k in keys:
        phrases += [p.strip() for p in fields.get(k, "").split(",") if p.strip()]
    phrases = [STYLE_WORDS.sub("", p).strip(" .") for p in phrases]
    phrases = [p for p in phrases if p and p.lower() not in (TRIGGER, "none")]
    if not phrases:
        phrases = [scene]
    seen, out = set(), []
    for p in phrases:                                 # drop repeated phrases
        if p.lower() not in seen:
            seen.add(p.lower()); out.append(p)
    prompt = ", ".join([TRIGGER] + out)
    while clip_tokens(prompt) > 77 and len(out) > 1:  # trim least important (last) phrases first
        out.pop(); prompt = ", ".join([TRIGGER] + out)
    return prompt


def clean(raw, scene, mode="full"):
    return assemble(fix_fields(raw, scene), scene, mode)


def load_llm(four_bit=True):
    global _llm, _tok
    if _llm is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if four_bit else None
        _tok = AutoTokenizer.from_pretrained(LLM_ID)
        _llm = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.float16,
                                                    quantization_config=q, device_map="auto")
    return _llm, _tok


def expand(scene, mode="hybrid"):
    """Short scene description -> (expanded prompt, raw LLM output). mode: "hybrid" or "full"."""
    import torch
    llm, tok = load_llm()
    msgs = [{"role": "system", "content": SYSTEM}]
    import json
    for user, answer in EXAMPLES:
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": json.dumps(answer)}]
    msgs.append({"role": "user", "content": scene})
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = tok(text, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=250, do_sample=False, repetition_penalty=1.1)
    raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return clean(raw, scene, mode), raw


def load_pipe(lora="models/lora_v1.safetensors", lora_scale=1.0):
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        _pipe = StableDiffusionPipeline.from_pretrained(SD_ID, torch_dtype=torch.float16,
                                                        safety_checker=None).to("cuda")
        _pipe.scheduler = DPMSolverMultistepScheduler.from_config(_pipe.scheduler.config)
        lp = Path(lora)
        _pipe.load_lora_weights(str(lp.parent), weight_name=lp.name)
        _pipe.fuse_lora(lora_scale=lora_scale)
    return _pipe


def generate(prompt, seed=0, steps=30, guidance=7.5, lora="models/lora_v1.safetensors", negative=NEG):
    import torch
    pipe = load_pipe(lora)
    g = torch.Generator("cuda").manual_seed(seed)
    return pipe(prompt, negative_prompt=negative, num_inference_steps=steps,
                guidance_scale=guidance, generator=g).images[0]


# ---------- best-of-N selection ----------
# The judge is CLIP ViT-B/32: a DIFFERENT model from the ViT-L/14 used for evaluation, and its
# style reference is the TRAINING paintings only, so the 9 held-out test paintings are never used.
JUDGE_ID = "openai/clip-vit-base-patch32"


def load_judge(report="data/captions_report.csv", images="data/images"):
    global _judge
    if _judge is None:
        import csv, torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
        m = CLIPModel.from_pretrained(JUDGE_ID).to("cuda").eval()
        pr = CLIPProcessor.from_pretrained(JUDGE_ID)
        train = [Path(images) / r["filename"] for r in csv.DictReader(open(report, encoding="utf-8"))
                 if r["split"] == "train"]
        ims = []
        for f in train:                       # skip files that aren't real images (e.g. failed downloads)
            try:
                ims.append(Image.open(f).convert("RGB"))
            except Exception:
                print(f"judge: skipping unreadable {f}")
        with torch.no_grad():
            px = pr(images=ims, return_tensors="pt")["pixel_values"].to("cuda")
            e = m.visual_projection(m.vision_model(pixel_values=px).pooler_output)
        e = e / e.norm(dim=-1, keepdim=True)
        c = e.mean(0); c = c / c.norm()
        _judge = (m, pr, c)
    return _judge


def judge_scores(images, text):
    """score = similarity to the scene text + similarity to the training-set style centroid"""
    import torch
    m, pr, centroid = load_judge()
    with torch.no_grad():
        px = pr(images=[im.convert("RGB") for im in images], return_tensors="pt")["pixel_values"].to("cuda")
        ie = m.visual_projection(m.vision_model(pixel_values=px).pooler_output)
        t = pr.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to("cuda")
        te = m.text_projection(m.text_model(**t).pooler_output)
    ie = ie / ie.norm(dim=-1, keepdim=True); te = te / te.norm(dim=-1, keepdim=True)
    return ((ie @ te.T).squeeze(1) + ie @ centroid).tolist()


def generate_best(prompt, scene=None, n=4, seed=0, lora="models/lora_v1.safetensors", negative=NEG_PLUS, gen=None):
    """Generate n candidates (seeds seed*100 .. seed*100+n-1) and return (best image, all scores).
    `gen` lets ControlNet reuse this: gen(prompt, seed, negative) -> image."""
    gen = gen or (lambda p, s, neg: generate(p, s, lora=lora, negative=neg))
    cands = [gen(prompt, seed * 100 + k, negative) for k in range(n)]
    scores = judge_scores(cands, scene or prompt.replace(TRIGGER + ", ", ""))
    best = max(range(n), key=lambda i: scores[i])
    return cands[best], scores


def paint(scene, expand=False, best_of=4, seed=0, lora="models/lora_v1.safetensors", keep_llm=False):
    """ONE CALL FOR THE FRONT END.
    scene   : what the user typed
    expand  : False (default, best scores) -> "rrvarma oil painting, <scene>"
              True  -> LLM hybrid expansion (useful for very short inputs like "Bhishma dying")
    best_of : candidates generated; the best is kept (1 = fastest, 4 = best quality, ~25 s on a T4)
    keep_llm: True keeps the LLM in GPU memory between calls (use in the app: avoids a ~1 min reload)
    returns : (PIL image, prompt used)"""
    if expand:
        prompt, _ = expand_fn(scene, "hybrid")
        if not keep_llm:
            _llm_free()                              # free GPU memory for Stable Diffusion
    else:
        prompt = f"{TRIGGER}, {scene.strip().rstrip('.')}"
    if best_of > 1:
        img, _ = generate_best(prompt, scene, best_of, seed, lora, negative=NEG_PLUS)
    else:
        img = generate(prompt, seed, lora=lora, negative=NEG_PLUS)
    return img, prompt


def refine(image, prompt, seed=0, negative=NEG_PLUS, scale=1.5, strength=0.35, steps=30, base=None):
    """High-res fix: upscale the chosen painting 1.5x and repaint it lightly (img2img, strength 0.35) with the
    same SD + LoRA. At 512 px a full-body figure's face is ~40 px, which SD 1.5 renders badly; at 768 px with
    the composition fixed, faces, hands and jewellery come out much cleaner. Pose and layout are kept
    (only 35% noise is added). base: the pipeline whose parts (UNet with the LoRA fused) are reused."""
    global _i2i
    import torch
    from PIL import Image
    if _i2i is None:
        from diffusers import StableDiffusionImg2ImgPipeline
        base = base or _pipe or load_pipe()
        parts = {k: v for k, v in base.components.items() if k not in ("controlnet", "safety_checker")}
        _i2i = StableDiffusionImg2ImgPipeline(**parts, safety_checker=None, requires_safety_checker=False)
        _i2i.enable_vae_slicing()
    W, H = int(image.width * scale) // 8 * 8, int(image.height * scale) // 8 * 8
    big = image.resize((W, H), Image.LANCZOS)
    g = torch.Generator("cuda").manual_seed(seed)
    return _i2i(prompt, image=big, negative_prompt=negative, strength=strength, num_inference_steps=steps,
                guidance_scale=7.5, generator=g).images[0]


def paint_with_pose(scene, pose_image, expand=False, best_of=4, seed=0, cn_scale=1.0, is_skeleton=False,
                    lora="models/lora_v1.safetensors", keep_llm=False, looks=True, pose_select=True,
                    hires=True, return_info=False):
    """Milestone 4: like paint(), but the figures follow the pose in pose_image (ControlNet OpenPose).
    pose_image : PIL image, a photo of someone posing (or an OpenPose skeleton if is_skeleton=True)
    cn_scale   : how strictly to follow the pose (0.5 loose .. 1.2 strict)
    looks      : add the named characters' gender, age and attire (fixes "Arjuna painted as a woman")
    pose_select: best-of-N also checks the pose: OpenPose is run again on every candidate, candidates
                 clearly worse at following the pose are dropped, then the CLIP judge picks as in paint()
    hires      : repaint the chosen image at 1.5x resolution (cleaner faces and hands, +~10 s)
    returns    : (PIL image, prompt used, skeleton image)  [+ info dict if return_info]"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import controlnet_pose as cp
    pose = cp.extract_pose(pose_image, is_skeleton)
    if not cp.has_person(pose):
        raise ValueError("No person found in the pose image. Try a clearer full-body photo.")
    ref_people = None if is_skeleton else cp.keypoints(pose_image)
    ref = cp.main_person(ref_people) if ref_people else None
    n_people = len(ref_people) if ref_people else 1
    if expand:
        prompt, _ = expand_fn(scene, "hybrid")
        if not keep_llm:
            _llm_free()
    else:
        prompt = f"{TRIGGER}, {scene.strip().rstrip('.')}"
    if looks:
        prompt = add_looks(prompt, scene, max_chars=n_people)
    gen = lambda s: cp.generate_with_pose(prompt, pose, s, cn_scale=cn_scale, lora=lora, negative=NEG_POSE)
    aspect = pose.width / pose.height
    seeds = [seed * 100 + k for k in range(best_of)] if best_of > 1 else [seed]
    cands = [gen(s) for s in seeds]
    pose_sims = [None] * len(cands)
    if ref is not None and (pose_select or return_info):
        for i, c in enumerate(cands):
            people = cp.keypoints(c)
            pose_sims[i] = cp.pose_similarity(ref, cp.main_person(people) if people else None, aspect)
    if len(cands) == 1:
        best, clip_scores = 0, [None]
    else:
        clip_scores = judge_scores(cands, scene)
        ok = list(range(len(cands)))
        if pose_select and None not in pose_sims:
            top = max(pose_sims)
            ok = [i for i in ok if pose_sims[i] >= top - 0.15]   # drop clearly worse poses
        best = max(ok, key=lambda i: clip_scores[i])
    final = refine(cands[best], prompt, seeds[best], NEG_POSE, base=cp.load_cn_pipe(lora)) if hires else cands[best]
    if return_info:
        return final, prompt, pose, {"seeds": seeds, "pose_sims": pose_sims, "clip_scores": clip_scores,
                                           "best": best, "candidates": cands, "ref": ref}
    return final, prompt, pose


def _llm_free():
    global _llm, _tok
    import gc, torch
    _llm = _tok = None; gc.collect(); torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", help="short scene description in plain English")
    ap.add_argument("--generate", action="store_true", help="also render the image with SD 1.5 + LoRA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora", default="models/lora_v1.safetensors")
    ap.add_argument("--out", default="m3_outputs")
    ap.add_argument("--best-of", type=int, default=4, help="candidates to generate and pick from (1 = no selection)")
    ap.add_argument("--mode", choices=["hybrid", "full"], default="hybrid")
    ap.add_argument("--expand", action="store_true", help="use the LLM to expand the scene (default: off)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.dry_run:
        print(SYSTEM); print("\nfallback if the LLM fails:", f"{TRIGGER}, {a.scene}")
        return
    if not a.generate:                               # just show the expansion
        prompt, raw = expand(a.scene, a.mode)
        print("scene   :", a.scene); print("LLM raw :", raw.strip())
        print(f"prompt  : {prompt}   ({clip_tokens(prompt)} tokens)")
        return
    img, prompt = paint(a.scene, expand=a.expand, best_of=a.best_of, seed=a.seed, lora=a.lora)
    print("scene   :", a.scene)
    print(f"prompt  : {prompt}   ({clip_tokens(prompt)} tokens)")
    Path(a.out).mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", a.scene.lower())[:50].strip("_")
    path = Path(a.out) / f"{slug}_s{a.seed}.png"
    img.save(path)
    print("saved   :", path)

if __name__ == "__main__":
    main()


expand_fn = expand   # alias so paint(expand=...) can still reach the expand() function
