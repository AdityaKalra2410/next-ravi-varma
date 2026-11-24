"""
Milestone 4: structural guardrailing with ControlNet (OpenPose).

Input : a scene description + a pose reference (a photo of a person posing, or a ready OpenPose skeleton)
Steps : 1. OpenPose extracts the stick-figure skeleton from the photo
        2. the scene is expanded into a hybrid prompt (Milestone 3, Qwen2.5-7B)
        3. SD 1.5 + our LoRA + ControlNet-OpenPose generates a painting whose figures follow the skeleton
Output: m4_outputs/<name>_pose.png, <name>.png, <name>_compare.png (reference | skeleton | painting)

The functions here are reused by the Gradio app (app.py).

Colab T4, from the repo root:
    !pip install -q diffusers transformers accelerate peft bitsandbytes controlnet_aux
    !pip uninstall -y torchao
    !python scripts/controlnet_pose.py --pose poses/kneeling.jpg \
        "Yudhishthira weeping over the body of Karna on the battlefield of Kurukshetra"
Options: --is-skeleton  (pose image is already an OpenPose skeleton)
         --cn-scale 1.0 (how strictly to follow the pose, 0.5-1.2)   --no-expand  (skip the LLM)
"""
import argparse, gc, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import expand_prompt as ep

CN_ID = "lllyasviel/control_v11p_sd15_openpose"
_detector = _cnpipe = None


def fit_size(w, h, short=512, long_max=768):
    """Keep the reference's aspect ratio: short side 512, long side <= 768, multiples of 8."""
    s = short / min(w, h)
    W, H = w * s, h * s
    if max(W, H) > long_max:
        k = long_max / max(W, H); W, H = W * k, H * k
    return int(W) // 8 * 8, int(H) // 8 * 8


def extract_pose(image, is_skeleton=False):
    """Photo -> OpenPose skeleton image (body only). Returns the reference itself if it is already a skeleton."""
    global _detector
    from PIL import Image
    image = image.convert("RGB")
    W, H = fit_size(*image.size)
    if is_skeleton:
        return image.resize((W, H))
    if _detector is None:
        from controlnet_aux import OpenposeDetector
        _detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    pose = _detector(image, hand_and_face=False, detect_resolution=512, image_resolution=max(W, H))
    return pose.resize((W, H), Image.BILINEAR)


def load_cn_pipe(lora="models/lora_v1.safetensors", lora_scale=1.0):
    global _cnpipe
    if _cnpipe is None:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler
        cn = ControlNetModel.from_pretrained(CN_ID, torch_dtype=torch.float16)
        _cnpipe = StableDiffusionControlNetPipeline.from_pretrained(
            ep.SD_ID, controlnet=cn, torch_dtype=torch.float16, safety_checker=None).to("cuda")
        _cnpipe.scheduler = DPMSolverMultistepScheduler.from_config(_cnpipe.scheduler.config)
        lp = Path(lora)
        _cnpipe.load_lora_weights(str(lp.parent), weight_name=lp.name)
        _cnpipe.fuse_lora(lora_scale=lora_scale)
    return _cnpipe


def generate_with_pose(prompt, pose, seed=0, steps=30, guidance=7.5, cn_scale=1.0,
                       lora="models/lora_v1.safetensors"):
    import torch
    pipe = load_cn_pipe(lora)
    g = torch.Generator("cuda").manual_seed(seed)
    return pipe(prompt, image=pose, negative_prompt=ep.NEG, num_inference_steps=steps,
                guidance_scale=guidance, controlnet_conditioning_scale=cn_scale,
                width=pose.width, height=pose.height, generator=g).images[0]


def free_llm():
    import torch
    ep._llm = ep._tok = None; gc.collect(); torch.cuda.empty_cache()


def side_by_side(images, h=512):
    from PIL import Image
    ims = [im.convert("RGB").resize((int(im.width * h / im.height), h)) for im in images]
    out = Image.new("RGB", (sum(i.width for i in ims), h), "white"); x = 0
    for i in ims:
        out.paste(i, (x, 0)); x += i.width
    return out


def main():
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--pose", required=True, help="photo of a person posing, or an OpenPose skeleton")
    ap.add_argument("--is-skeleton", action="store_true")
    ap.add_argument("--no-expand", action="store_true", help="use 'trigger, scene' instead of the LLM prompt")
    ap.add_argument("--cn-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora", default="models/lora_v1.safetensors")
    ap.add_argument("--out", default="m4_outputs")
    a = ap.parse_args()

    ref = Image.open(a.pose)
    pose = extract_pose(ref, a.is_skeleton)
    if a.no_expand:
        prompt = f"{ep.TRIGGER}, {a.scene.strip().rstrip('.')}"
    else:
        prompt, _ = ep.expand(a.scene, "hybrid"); free_llm()
    print(f"prompt : {prompt}   ({ep.clip_tokens(prompt)} tokens)")
    print(f"size   : {pose.width}x{pose.height}")

    img = generate_with_pose(prompt, pose, a.seed, cn_scale=a.cn_scale, lora=a.lora)
    Path(a.out).mkdir(exist_ok=True)
    name = re.sub(r"[^a-z0-9]+", "_", a.scene.lower())[:40].strip("_") + f"_s{a.seed}"
    pose.save(Path(a.out) / f"{name}_pose.png")
    img.save(Path(a.out) / f"{name}.png")
    side_by_side([ref, pose, img]).save(Path(a.out) / f"{name}_compare.png")
    print("saved  :", Path(a.out) / f"{name}_compare.png")


if __name__ == "__main__":
    main()
