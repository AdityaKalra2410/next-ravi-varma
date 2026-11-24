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

    def paint_pose(scene, expand, best_of, seed, pose_image, cn_scale):
        im, prompt = paint(scene, expand, best_of, seed)
        return im, prompt, pose_image.convert("L").convert("RGB")   # (mock) the "skeleton"
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

    def paint_pose(scene, expand, best_of, seed, pose_image, cn_scale):
        return ep.paint_with_pose(scene, pose_image, expand=expand, best_of=best_of, seed=seed,
                                  cn_scale=cn_scale, keep_llm=True)


def run(scene, expand, quality, seed, pose_image, cn_scale, history, progress=gr.Progress()):
    if not scene or not scene.strip():
        raise gr.Error("Please describe a scene first, e.g. "
                       "\"Sita sitting alone in the Ashoka grove, guarded by demonesses\".")
    scene = scene.strip()
    seed = random.randint(0, 9999) if seed is None or seed < 0 else int(seed)
    best_of = 4 if quality.startswith("Best") else 1
    progress(0, desc="Expanding your prompt, then painting..." if expand else "Painting...")
    use_pose = POSE_READY and pose_image is not None
    t = time.time()
    try:
        if use_pose:
            img, prompt, skeleton = paint_pose(scene, expand, best_of, seed, pose_image, float(cn_scale))
        else:
            (img, prompt), skeleton = paint(scene, expand, best_of, seed), None
    except ValueError as e:                  # e.g. no person found in the pose photo
        raise gr.Error(str(e))
    except Exception as e:
        print(f"paint failed: {e!r}")
        raise gr.Error(f"Something went wrong while painting: {type(e).__name__}. "
                       "Try Fast mode or another seed. Details are in the server log.")
    secs = time.time() - t
    status = (f"Painted in **{secs:.1f} s** · "
              f"{'best of 4 candidates' if best_of > 1 else 'fast preview'} · "
              f"seed **{seed}**{' · prompt expanded by AI' if expand else ''}"
              f"{f' · following your pose (strength {float(cn_scale):.2f})' if use_pose else ''}")
    history = ([(img, f"{scene} · seed {seed}")] + (history or []))[:HISTORY_MAX]
    return img, prompt, seed, status, history, history, skeleton


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
    <div class="rrv-feats"><span>Stable Diffusion 1.5 + LoRA</span><span>Best-of-4, picked by a
    CLIP judge</span><span>Optional prompt expansion (Qwen2.5-7B)</span><span>{'Pose control (ControlNet)' if POSE_READY
    else 'Pose control: coming soon'}</span></div>
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
    block_title_background_fill="transparent",
    block_title_background_fill_dark="transparent",
    block_title_text_weight="700",
    block_title_text_size="*text_sm",
    block_label_background_fill="rgba(255, 250, 240, .9)",
    block_label_background_fill_dark="rgba(30, 22, 19, .85)",
    block_radius="12px",
    block_shadow="0 1px 2px rgba(80, 40, 20, .06)",
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
  .rrv-strip img { height: 110px; } .rrv-hero { padding: 20px 18px; } .rrv-feats { display: none; } }
.rrv-feats { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.rrv-feats span { font-size: 12.5px; padding: 4px 11px; border-radius: 999px; color: #f3dfae;
  border: 1px solid rgba(217, 178, 90, .45); background: rgba(0, 0, 0, .18); }

/* cards */
.rrv-card { background: #fffcf6 !important; border: 1px solid #e6d6b8 !important;
  border-radius: 14px !important; padding: 14px 16px 16px !important;
  box-shadow: 0 2px 10px rgba(90, 50, 20, .06) !important; }
.rrv-card > div, .rrv-card .block { box-shadow: none !important; }
.rrv-card .block { border: none !important; background: transparent !important; padding-left: 0 !important;
  padding-right: 0 !important; }
.rrv-card .block textarea, .rrv-card .block input[type=number] { border: 1px solid #e3cfa6 !important; }
.rrv-step { font-family: 'Cormorant Garamond', Georgia, serif; font-size: 22px; font-weight: 700;
  color: #6e1c1f; margin: 0 0 2px; display: flex; align-items: center; gap: 10px; }
.rrv-step i { font-style: normal; display: inline-grid; place-items: center; width: 26px; height: 26px;
  border-radius: 50%; background: linear-gradient(135deg, #c9952a, #8a5f1c); color: #fff8e6;
  font-family: Lato, sans-serif; font-size: 13px; font-weight: 700; }
.rrv-step-row { margin-bottom: -4px; }

/* paint button */
#paint-btn { font-size: 19px; font-weight: 700; letter-spacing: .06em; min-height: 56px;
  border-radius: 12px !important; border: 1px solid #c9952a !important;
  box-shadow: 0 6px 18px rgba(110, 28, 31, .35), inset 0 1px 0 rgba(255, 230, 170, .25);
  transition: transform .15s ease, box-shadow .15s ease; }
#paint-btn:hover { transform: translateY(-1px);
  box-shadow: 0 10px 24px rgba(110, 28, 31, .42), inset 0 1px 0 rgba(255, 230, 170, .3); }

/* gilded frame around the painting */
#frame { padding: 14px !important; border-radius: 6px !important;
  background: linear-gradient(135deg, #8a5f1c 0%, #e9c65c 22%, #a87622 45%, #f2dd96 62%,
    #86591f 82%, #c9952a 100%) !important;
  box-shadow: 0 18px 40px rgba(60, 25, 10, .35), inset 0 0 0 2px rgba(60, 30, 5, .45) !important;
  border: none !important; }
#painting { border: none !important; border-radius: 2px !important; background: #2a1a14 !important;
  box-shadow: inset 0 0 0 3px #3b2414, inset 0 0 28px rgba(0, 0, 0, .65) !important; }
#painting img { object-fit: contain; }
#painting .wrap { background: rgba(42, 26, 20, .92) !important; color: #e9c65c !important; }
#painting .empty, #painting [class*=empty] { color: #c9a467 !important; }
#status { min-height: 24px; text-align: center; margin-top: 4px; }
#status p { font-size: 14px; color: #6b4a2e; }
.rrv-details { margin-top: 2px; }

.rrv-h2 { font-family: 'Cormorant Garamond', Georgia, serif; font-size: 26px; font-weight: 700;
  color: #6e1c1f; margin: 18px 0 2px; }
.rrv-credit { text-align: center; font-size: 12.5px; opacity: .7; margin-top: 8px; }

.rrv-card .form { background: transparent !important; border: none !important; gap: 12px !important; }

/* dark mode: one selector per rule (Gradio only scopes the first selector of a .dark list) */
.dark .rrv-card { background: #241a17 !important; border-color: #3d2c25 !important; }
.dark .rrv-card textarea { border-color: #4a372d !important; }
.dark .rrv-card input[type=number] { border-color: #4a372d !important; }
.dark .rrv-step { color: #f0cf7e; }
.dark .rrv-h2 { color: #f0cf7e; }
.dark .rrv-pills button:hover { color: #f3cf7a; }
.dark #status p { color: #d9c19a; }
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
            with gr.Column(elem_classes="rrv-card"):
                gr.HTML('<div class="rrv-step"><i>1</i>Describe a scene</div>')
                scene = gr.Textbox(
                    label="Scene", show_label=False, lines=3, max_lines=6, autofocus=True,
                    placeholder="e.g. Yudhishthira weeping over Karna at Kurukshetra")
                with gr.Accordion("Pose reference (optional)"
                                  + ("" if POSE_READY else " · coming soon"), open=False):
                    pose_img = gr.Image(label="A photo of someone posing, or a stick figure",
                                        type="pil", height=220, interactive=POSE_READY)
                    pose_strength = gr.Slider(0.5, 1.2, value=1.0, step=0.05, label="Pose strength",
                                              interactive=POSE_READY)
                    if not POSE_READY:
                        gr.Markdown("The painting will follow the pose you upload. Arriving with "
                                    "Milestone 4 (ControlNet).", elem_classes="rrv-note")

            with gr.Column(elem_classes="rrv-card"):
                gr.HTML('<div class="rrv-step"><i>2</i>Settings</div>')
                expand = gr.Checkbox(label="Expand my prompt with AI (helps for very short inputs)",
                                     value=False)
                quality = gr.Radio(QUALITY, value=QUALITY[0], label="Quality")
                with gr.Row(equal_height=False, elem_id="seed-row"):
                    seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0, scale=3)
                    reuse = gr.Button("↺ Reuse last seed", size="sm", scale=2, elem_id="reuse-btn")

            btn = gr.Button("🖌  Paint", variant="primary", elem_id="paint-btn")

            with gr.Column(elem_classes="rrv-card"):
                gr.HTML('<div class="rrv-step"><i>✦</i>Try a scene</div>')
                gr.Markdown("Click one to fill the box. ✨ = short input, uses *Expand my prompt "
                            "with AI*.", elem_classes="rrv-note")
                with gr.Row(elem_classes="rrv-pills"):
                    example_btns = [(gr.Button(("✨ " if e else "") + t, size="sm",
                                               variant="secondary"), t, e) for t, e in EXAMPLES]

        with gr.Column(scale=6):
            with gr.Column(elem_id="frame"):
                out_img = gr.Image(label="Painting", show_label=False, type="pil", height=560,
                                   elem_id="painting", interactive=False, format="png")
            status = gr.Markdown("*Your painting will appear here. Pick a scene and press "
                                 "**Paint**.*", elem_id="status")
            with gr.Row(elem_classes="rrv-details"):
                out_prompt = gr.Textbox(label="Prompt sent to the model", interactive=False,
                                        lines=2, scale=4, **copy_btn)
                out_seed = gr.Number(label="Seed used", interactive=False, precision=0, scale=1)
            out_pose = gr.Image(label="Extracted pose skeleton", type="pil", height=260,
                                interactive=False, visible=POSE_READY)

    with gr.Column(elem_classes="rrv-section"):
        gr.HTML('<div class="rrv-h2">Your paintings this session</div>')
        gallery = gr.Gallery(label="Your paintings this session", show_label=False, columns=6,
                             height="auto", object_fit="cover", allow_preview=True,
                             elem_id="history")
        clear = gr.Button("Clear session gallery", size="sm", variant="secondary")
        with gr.Accordion("About the project", open=False):
            gr.Markdown(ABOUT)
        gr.HTML('<div class="rrv-credit">New paintings generated by AI in the style of Raja Ravi '
                'Varma (1848–1906). Header paintings: Raja Ravi Varma, public domain.</div>')

    # ------------------------------------------------------------ events
    go = dict(fn=run, inputs=[scene, expand, quality, seed, pose_img, pose_strength, history],
              outputs=[out_img, out_prompt, out_seed, status, history, gallery, out_pose])
    if int(gr.__version__.split(".")[0]) >= 5:
        go["show_progress_on"] = [out_img]   # one progress bar, on the painting
    btn.click(**go)
    scene.submit(**go)
    for b, t, e in example_btns:
        b.click(lambda t=t, e=e: (t, e), None, [scene, expand])
    reuse.click(lambda s: s if s is not None else -1, out_seed, seed)
    clear.click(lambda: ([], []), None, [history, gallery])

if __name__ == "__main__":
    demo.queue().launch(share=not ARGS.no_share, server_port=ARGS.port,
                        debug=not ARGS.mock, **(STYLE if GR6 else {}))
