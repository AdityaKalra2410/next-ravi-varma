"""
The Next Ravi Varma: Gradio web app (Milestone 3 front end, with room for the M4 pose controls).

Run in Colab (T4 GPU) from the repo root:
    !pip install -q diffusers transformers accelerate peft bitsandbytes gradio
    !pip uninstall -y torchao
    !python app.py
Open the public link it prints (https://....gradio.live).

UI work without a GPU (returns training paintings instead of generating):
    python app.py --mock --no-share
"""
import argparse, base64, io, random, sys, time
from pathlib import Path

sys.path.insert(0, "scripts")
import gradio as gr
import expand_prompt as ep

ap = argparse.ArgumentParser()
ap.add_argument("--mock", action="store_true", help="no GPU: fake the model with training paintings")
ap.add_argument("--no-share", action="store_true", help="don't create a public gradio.live link")
ap.add_argument("--port", type=int, default=None)
ARGS, _ = ap.parse_known_args()

GR6 = int(gr.__version__.split(".")[0]) >= 6
IMAGES = Path("data/images")

# Full sentences work best. The first five scored highest with the default settings in eval_m3.
EXAMPLES = [
    ["Damayanti talking to a golden swan by a lotus pond", False],
    ["Shakuntala looking back at King Dushyanta in the forest", False],
    ["Sita sitting alone in the Ashoka grove, guarded by demonesses", False],
    ["Draupadi pleading in the royal court", False],
    ["Arjuna drawing his bow on his chariot while Krishna holds the reins", False],
    ["Bhishma lying on a bed of arrows", False],
    ["Hanuman kneeling before Rama and offering a ring", False],
    ["Arjuna and Krishna", True],
    ["Ravana and Jatayu", True],
]
QUALITY = ["Best (4 candidates, ~25 s)", "Fast (1 image, ~6 s)"]
HISTORY_MAX = 12

# Pose control (Milestone 4). The components are in the layout; they get wired up once
# expand_prompt.paint_with_pose(scene, pose_image, ...) exists.
POSE_READY = hasattr(ep, "paint_with_pose")


# ---------------------------------------------------------------- model
if ARGS.mock:
    _mock_files = sorted(IMAGES.glob("*.jpg"))

    def paint(scene, expand, best_of, seed):
        time.sleep(3 if best_of > 1 else 1)
        from PIL import Image
        im = Image.open(random.Random(seed).choice(_mock_files)).convert("RGB")
        s = min(im.size)  # centre-crop to 512x512 like SD 1.5 output
        im = im.crop(((im.width - s) // 2, (im.height - s) // 2,
                      (im.width + s) // 2, (im.height + s) // 2)).resize((512, 512))
        prompt = f"{ep.TRIGGER}, {scene.strip().rstrip('.')}"
        if expand:
            prompt += ", (mock) figures, pose, setting, composition"
        return im, prompt
else:
    print("Loading Stable Diffusion + LoRA and the image judge (first time ~2 min)...")
    ep.load_pipe()
    ep.load_judge()
    print("Ready.")

    def paint(scene, expand, best_of, seed):
        if hasattr(ep, "paint"):
            return ep.paint(scene, expand=expand, best_of=best_of, seed=seed, keep_llm=True)
        # Same pipeline built from the functions on main: raw scene (or LLM-expanded hybrid
        # prompt), extra negatives, best-of-N picked by the CLIP judge.
        prompt = ep.expand(scene)[0] if expand else ep.assemble(None, scene)
        if best_of > 1:
            img, _ = ep.generate_best(prompt, scene, n=best_of, seed=seed)
        else:
            img = ep.generate(prompt, seed, negative=ep.NEG_PLUS)
        return img, prompt


def run(scene, expand, quality, seed, history, progress=gr.Progress()):
    if not scene or not scene.strip():
        raise gr.Error("Please describe a scene first, e.g. "
                       "\"Sita sitting alone in the Ashoka grove, guarded by demonesses\".")
    scene = scene.strip()
    seed = random.randint(0, 9999) if seed is None or seed < 0 else int(seed)
    best_of = 4 if quality.startswith("Best") else 1
    progress(0, desc="Expanding your prompt, then painting..." if expand else "Painting...")
    t = time.time()
    try:
        img, prompt = paint(scene, expand, best_of, seed)
    except Exception as e:
        print(f"paint failed: {e!r}")
        raise gr.Error(f"Something went wrong while painting: {type(e).__name__}. "
                       "Try Fast mode or another seed. Details are in the server log.")
    secs = time.time() - t
    status = (f"Painted in **{secs:.1f} s** · "
              f"{'best of 4 candidates' if best_of > 1 else 'fast preview'} · "
              f"seed **{seed}**{' · prompt expanded by AI' if expand else ''}")
    history = ([(img, f"{scene} · seed {seed}")] + (history or []))[:HISTORY_MAX]
    return img, prompt, seed, status, history, history


# ---------------------------------------------------------------- look & feel
def header_strip(names=("rrv_0094.jpg", "rrv_0008.jpg", "rrv_0104.jpg", "rrv_0049.jpg",
                        "rrv_0107.jpg", "rrv_0062.jpg"), h=220):
    """Small strip of Ravi Varma's own paintings (public domain, from our training set)."""
    from PIL import Image
    tags = []
    for n in names:
        try:
            im = Image.open(IMAGES / n).convert("RGB")
        except Exception:
            continue
        im.thumbnail((h * 2, h))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
        tags.append(f'<img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}" alt="">')
    return f'<div class="rrv-strip">{"".join(tags)}</div>' if tags else ""


HEADER = f"""
<div class="rrv-hero">
  <div class="rrv-hero-text">
    <div class="rrv-kicker">Team The Unsupervised{' · mock mode' if ARGS.mock else ''}</div>
    <h1>The Next Ravi Varma</h1>
    <p>Describe a scene from the <b>Ramayana</b> or <b>Mahabharata</b> and get a new painting
    in the style of Raja Ravi Varma's oil paintings. Stable Diffusion 1.5 + our LoRA, trained
    on his work.</p>
  </div>
  {header_strip()}
</div>
"""

ABOUT = """
**How it works**

1. **Your scene** is turned into a prompt starting with our trigger words `rrvarma oil painting`.
   With *Expand my prompt with AI* ticked, Qwen2.5-7B first adds the figures, poses and setting.
   That helps short inputs like "Arjuna and Krishna". Full sentences score best without it.
2. **Stable Diffusion 1.5 + our LoRA** paints the scene. The LoRA was fine-tuned on Raja Ravi Varma's
   oil paintings from Wikimedia Commons (public domain).
3. **Best of 4:** four candidates are painted and a CLIP judge keeps the one that matches your
   scene and the style of the training paintings most closely.

The same seed and settings always give the same painting, so note the seed to reproduce a result.
"""

MAROON = gr.themes.Color(c50="#fbf3f2", c100="#f5e2e0", c200="#e9c0bd", c300="#d8938f",
                         c400="#bf5e59", c500="#9e3530", c600="#862525", c700="#6e1c1f",
                         c800="#57161a", c900="#431215", c950="#2a0a0c", name="maroon")
GOLD = gr.themes.Color(c50="#fdf9ec", c100="#f9efcc", c200="#f2dd96", c300="#e9c65c",
                       c400="#dfb03a", c500="#c9952a", c600="#a87622", c700="#86591f",
                       c800="#6e4820", c900="#5c3c1e", c950="#341f0e", name="gold")

THEME = gr.themes.Soft(
    primary_hue=MAROON, secondary_hue=GOLD, neutral_hue="stone",
    font=[gr.themes.GoogleFont("Lato"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#f8f1e4",
    body_background_fill_dark="#18110f",
    block_background_fill="#fffcf6",
    block_background_fill_dark="#241a17",
    block_border_color="#e6d6b8",
    block_border_color_dark="#3d2c25",
    block_label_text_color="*primary_700",
    block_label_text_color_dark="*secondary_300",
    block_title_text_color="*primary_700",
    block_title_text_color_dark="*secondary_300",
    input_background_fill="#fffdf9",
    input_background_fill_dark="#1d1513",
    button_primary_background_fill="linear-gradient(135deg, *primary_600, *primary_800)",
    button_primary_background_fill_hover="linear-gradient(135deg, *primary_500, *primary_700)",
    button_primary_background_fill_dark="linear-gradient(135deg, *primary_600, *primary_800)",
    button_primary_text_color="#fdf3dc",
    button_primary_border_color="*secondary_500",
    button_primary_border_color_dark="*secondary_600",
    checkbox_background_color_selected="*primary_600",
    slider_color="*primary_600",
)

HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700'
        '&display=swap" rel="stylesheet">')

CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
.rrv-hero { background: radial-gradient(120% 140% at 0% 0%, #7a1f24 0%, #4a1014 55%, #2b090b 100%);
  border: 1px solid #b8892e; border-radius: 14px; padding: 26px 28px 22px; color: #f6e8c8;
  box-shadow: 0 10px 30px rgba(60, 15, 15, .25), inset 0 0 0 4px rgba(201, 149, 42, .18); }
.rrv-kicker { font-size: 12px; letter-spacing: .18em; text-transform: uppercase; color: #d9b25a; }
.rrv-hero h1 { font-family: 'Cormorant Garamond', Georgia, serif; font-weight: 700;
  font-size: clamp(34px, 5vw, 52px); line-height: 1.05; margin: 6px 0 8px; color: #f9e7b8;
  letter-spacing: .01em; }
.rrv-hero p { max-width: 760px; margin: 0; font-size: 15.5px; line-height: 1.55; color: #ecdcbc; }
.rrv-hero b { color: #f3cf7a; font-weight: 600; }
.rrv-strip { display: flex; gap: 10px; margin-top: 20px; overflow: hidden; }
.rrv-strip img { flex: 1 1 0; min-width: 0; height: 120px; object-fit: cover; border-radius: 6px;
  border: 2px solid #c9952a; box-shadow: 0 4px 12px rgba(0, 0, 0, .35); }
@media (max-width: 700px) { .rrv-strip img:nth-child(n+4) { display: none; }
  .rrv-strip img { height: 110px; } .rrv-hero { padding: 20px 18px; } }
#paint-btn { font-size: 18px; font-weight: 700; letter-spacing: .04em; min-height: 52px; }
#painting { border: 2px solid #c9952a !important; }
#status { min-height: 24px; }
#history .grid-container { grid-template-columns: repeat(6, minmax(0, 1fr)) !important; }
#history .grid-wrap { height: auto !important; min-height: 120px; max-height: none !important; }
#history { height: auto !important; min-height: 0 !important; }
@media (max-width: 700px) {
  #history .grid-container { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; } }
#seed-row { align-items: center; }
#reuse-btn { align-self: center; max-height: 40px; }
.rrv-pills-title { margin-top: 6px; }
.rrv-pills { flex-wrap: wrap !important; gap: 8px !important; }
.rrv-pills button { flex: 0 1 auto !important; min-width: 0 !important; width: auto !important;
  border-radius: 999px !important; border: 1px solid #d9bf86 !important; font-weight: 500;
  text-align: left; white-space: normal; padding: 6px 14px !important; }
.rrv-pills button:hover { border-color: #9e3530 !important; color: #6e1c1f; }
.rrv-section h2, .rrv-section h3 { font-family: 'Cormorant Garamond', Georgia, serif; }
.rrv-note { font-size: 13px; opacity: .8; }
footer { opacity: .6; }
"""

STYLE = dict(theme=THEME, css=CSS, head=HEAD)
copy_btn = {"buttons": ["copy"]} if GR6 else {"show_copy_button": True}


# ---------------------------------------------------------------- layout
with gr.Blocks(title="The Next Ravi Varma", **({} if GR6 else STYLE)) as demo:
    history = gr.State([])
    gr.HTML(HEADER)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            scene = gr.Textbox(
                label="Scene", lines=3, max_lines=6, autofocus=True,
                placeholder="e.g. Yudhishthira weeping over Karna at Kurukshetra")

            with gr.Accordion("Pose reference (optional)" + ("" if POSE_READY else " · coming soon"),
                              open=False):
                pose_img = gr.Image(label="Pose reference: a photo of someone posing, or a stick figure",
                                    type="pil", height=220, interactive=POSE_READY)
                pose_strength = gr.Slider(0.5, 1.2, value=1.0, step=0.05, label="Pose strength",
                                          interactive=POSE_READY)
                if not POSE_READY:
                    gr.Markdown("The painting will follow the pose you upload. Arriving with "
                                "Milestone 4 (ControlNet).", elem_classes="rrv-note")

            expand = gr.Checkbox(label="Expand my prompt with AI (helps for very short inputs)",
                                 value=False)
            quality = gr.Radio(QUALITY, value=QUALITY[0], label="Quality")
            with gr.Row(equal_height=False, elem_id="seed-row"):
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0, scale=3)
                reuse = gr.Button("↺ Reuse last seed", size="sm", scale=2, elem_id="reuse-btn")
            btn = gr.Button("Paint", variant="primary", elem_id="paint-btn")

            gr.Markdown("**Try a scene** · ✨ = short input, uses *Expand my prompt with AI*",
                        elem_classes="rrv-pills-title")
            with gr.Row(elem_classes="rrv-pills"):
                example_btns = [(gr.Button(("✨ " if e else "") + t, size="sm", variant="secondary"),
                                 t, e) for t, e in EXAMPLES]

        with gr.Column(scale=6):
            out_img = gr.Image(label="Painting", type="pil", height=560, elem_id="painting",
                               interactive=False, format="png")
            status = gr.Markdown(elem_id="status")
            with gr.Row():
                out_prompt = gr.Textbox(label="Prompt sent to the model", interactive=False,
                                        lines=2, scale=4, **copy_btn)
                out_seed = gr.Number(label="Seed used", interactive=False, precision=0, scale=1)
            out_pose = gr.Image(label="Extracted pose skeleton", type="pil", height=260,
                                interactive=False, visible=POSE_READY)

    with gr.Column(elem_classes="rrv-section"):
        gallery = gr.Gallery(label="Your paintings this session", columns=6, height="auto",
                             object_fit="cover", show_label=True, allow_preview=True,
                             elem_id="history")
        clear = gr.Button("Clear session gallery", size="sm", variant="secondary")
        with gr.Accordion("About the project", open=False):
            gr.Markdown(ABOUT)

    # ------------------------------------------------------------ events
    go = dict(fn=run, inputs=[scene, expand, quality, seed, history],
              outputs=[out_img, out_prompt, out_seed, status, history, gallery])
    if int(gr.__version__.split(".")[0]) >= 5:
        go["show_progress_on"] = [out_img]   # one progress bar, on the painting
    btn.click(**go)
    scene.submit(**go)
    for b, t, e in example_btns:
        b.click(lambda t=t, e=e: (t, e), None, [scene, expand])
    reuse.click(lambda s: s if s is not None else -1, out_seed, seed)
    clear.click(lambda: ([], []), None, [history, gallery])
    # M4: wire pose_img / pose_strength / out_pose to ep.paint_with_pose here.

if __name__ == "__main__":
    demo.queue().launch(share=not ARGS.no_share, server_port=ARGS.port,
                        debug=not ARGS.mock, **(STYLE if GR6 else {}))
