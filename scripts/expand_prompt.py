"""
Milestone 3: text parsing and prompt expansion.

A short epic-scene description is expanded by an open-weight LLM (Qwen2.5-3B-Instruct) into
a prompt in the SAME format as our LoRA training captions:
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
LLM_ID = "Qwen/Qwen2.5-3B-Instruct"
SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
NEG = "blurry, deformed, extra limbs, bad anatomy, text, watermark"

SYSTEM = f"""You write image prompts for a Stable Diffusion model fine-tuned on Raja Ravi Varma's oil paintings.
Rewrite the user's scene as ONE line of short comma-separated phrases, in exactly this order:
1. "{TRIGGER}"
2. character names (only names from the scene)
3. the visible figures (who: gender, age, count)
4. the main action
5. poses and gestures
6. attire: 19th-century Indian clothing and jewellery as Ravi Varma painted it (saris, dhotis, angavastrams, gold crowns, necklaces, armlets), with colours
7. setting and background
8. composition (e.g. full scene, two-figure composition, single-figure portrait)
Rules:
- Describe only what should be visible. Short phrases, not sentences. No more than about 45 words.
- Do NOT add style or technique words (no "oil painting texture", "chiaroscuro", "realistic", "dramatic lighting", "masterpiece"): the trigger phrase already carries the style.
- Keep the characters faithful to the epic; do not add characters the scene does not imply.
- Output only the prompt line, nothing else."""

EXAMPLES = [
    ("Draupadi pleading in the royal court of Hastinapura",
     f"{TRIGGER}, Draupadi, Dushasana, Bhishma, woman in red sari, crowned princes, seated elders, "
     "woman pleading with hands joined, man pulling her sari, elders looking down, red silk sari with gold border, "
     "gold crowns and jewellery, royal court with carved pillars, full scene"),
    ("Krishna playing the flute for Radha by the Yamuna at dusk",
     f"{TRIGGER}, Krishna, Radha, young blue-skinned man with flute, young woman, playing the flute, "
     "woman listening shyly, standing close together, yellow dhoti, peacock-feather crown, green sari with gold border, "
     "riverbank at dusk, trees, two-figure composition"),
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


def clean(raw, scene):
    """Turn the LLM output into a valid prompt: one line, trigger first, no style words, <= 77 tokens."""
    line = next((l for l in raw.splitlines() if l.strip()), "").strip().strip('"').strip()
    line = re.sub(r"^(prompt|output)\s*:\s*", "", line, flags=re.I)
    phrases = [p.strip() for p in line.split(",") if p.strip()]
    phrases = [STYLE_WORDS.sub("", p).strip() for p in phrases]
    phrases = [p for p in phrases if p and p.lower() != TRIGGER]
    if len(phrases) < 3:                           # LLM failed: fall back to the raw scene
        phrases = [scene.strip().rstrip(".")]
    seen, out = set(), []
    for p in phrases:                              # drop repeated phrases
        if p.lower() not in seen:
            seen.add(p.lower()); out.append(p)
    prompt = ", ".join([TRIGGER] + out)
    while clip_tokens(prompt) > 77 and len(out) > 1:   # trim least important (last) phrases first
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
    for user, answer in EXAMPLES:
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    msgs.append({"role": "user", "content": scene})
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = tok(text, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=120, do_sample=False, repetition_penalty=1.1)
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
