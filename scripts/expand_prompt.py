"""
Milestone 3: text parsing and prompt expansion.

A short epic-scene description is expanded by an open-weight LLM (Qwen2.5-7B-Instruct, 4-bit) into
a prompt in the SAME format as our LoRA training captions. The LLM fills a JSON template
(same fields as our Qwen3-VL tags) and the code assembles it in caption order:
    rrvarma oil painting, <characters>, <figures>, <action>, <pose>, <attire>, <setting>, <composition>
then trimmed to CLIP's 77-token limit and (optionally) rendered with SD 1.5 + our LoRA.

The functions expand() and generate() are reused by the Gradio app in Milestone 4.

Colab (T4) from the repo root:
    !pip install -q diffusers transformers accelerate peft bitsandbytes
    !pip uninstall -y torchao
    !python scripts/expand_prompt.py --dry-run "Arjuna aiming his bow"           # show the LLM instructions
    !python scripts/expand_prompt.py "Yudhishthira weeping over the body of Karna at Kurukshetra"
    !python scripts/expand_prompt.py --generate --seed 0 "Sita in the Ashoka grove, guarded by demonesses"
"""
import argparse, re
from pathlib import Path

TRIGGER = "rrvarma oil painting"
LLM_ID = "Qwen/Qwen2.5-7B-Instruct"   # 4-bit: ~5.5 GB, fits on a T4 next to SD 1.5
SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
NEG = "blurry, deformed, extra limbs, bad anatomy, text, watermark"

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

_llm = _tok = _clip_tok = _pipe = None


def clip_tokens(text):
    global _clip_tok
    if _clip_tok is None:
        from transformers import CLIPTokenizer
        _clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    return len(_clip_tok(text).input_ids)


def scene_names(scene):
    """Capitalised words in the scene that look like names (Yudhishthira, Karna, Kurukshetra...)."""
    stop = {"The", "A", "An", "On", "In", "Of", "At", "By", "And", "With", "Over", "His", "Her"}
    return [w for w in re.findall(r"\b[A-Z][a-z]+\b", scene) if w not in stop]


def parse_json(raw):
    import json
    m = re.search(r"\{.*\}", raw, flags=re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return {k: str(d.get(k, "")).strip() for k in FIELDS}


def clean(raw, scene):
    """LLM output -> valid prompt: trigger first, caption-order fields, scene names kept,
    no style words, no repeats, <= 77 CLIP tokens."""
    fields = parse_json(raw)
    if sum(bool(v) for v in fields.values()) < 3:     # no usable JSON: fall back to the raw scene
        return f"{TRIGGER}, {scene.strip().rstrip('.')}"
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
    if n_chars >= 2 and "single" in fields["composition"].lower():
        fields["composition"] = "two-figure composition" if n_chars == 2 else "group scene"
    phrases = []
    for k in FIELDS:
        phrases += [p.strip() for p in fields.get(k, "").split(",") if p.strip()]
    phrases = [STYLE_WORDS.sub("", p).strip(" .") for p in phrases]
    phrases = [p for p in phrases if p and p.lower() not in (TRIGGER, "none")]
    if len(phrases) < 3:                              # LLM failed: fall back to the raw scene
        phrases = [scene.strip().rstrip(".")]
    seen, out = set(), []
    for p in phrases:
        if p.lower() not in seen:
            seen.add(p.lower()); out.append(p)
    prompt = ", ".join([TRIGGER] + out)
    while clip_tokens(prompt) > 77 and len(out) > 1:  # trim least important (last) phrases first
        out.pop(); prompt = ", ".join([TRIGGER] + out)
    return prompt


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


def expand(scene):
    """Short scene description -> (expanded prompt, raw LLM output)."""
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
    return clean(raw, scene), raw


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


def generate(prompt, seed=0, steps=30, guidance=7.5, lora="models/lora_v1.safetensors"):
    import torch
    pipe = load_pipe(lora)
    g = torch.Generator("cuda").manual_seed(seed)
    return pipe(prompt, negative_prompt=NEG, num_inference_steps=steps,
                guidance_scale=guidance, generator=g).images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", help="short scene description in plain English")
    ap.add_argument("--generate", action="store_true", help="also render the image with SD 1.5 + LoRA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora", default="models/lora_v1.safetensors")
    ap.add_argument("--out", default="m3_outputs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.dry_run:
        print(SYSTEM); print("\nfallback if the LLM fails:", f"{TRIGGER}, {a.scene}")
        return
    prompt, raw = expand(a.scene)
    print("scene   :", a.scene)
    print("LLM raw :", raw.strip())
    print(f"prompt  : {prompt}   ({clip_tokens(prompt)} tokens)")
    if a.generate:
        Path(a.out).mkdir(exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", a.scene.lower())[:50].strip("_")
        path = Path(a.out) / f"{slug}_s{a.seed}.png"
        generate(prompt, a.seed, lora=a.lora).save(path)
        print("saved   :", path)


if __name__ == "__main__":
    main()
