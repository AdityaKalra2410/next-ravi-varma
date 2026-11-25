"""
The Next Ravi Varma: Gradio web app (text -> painting, optional pose photo via ControlNet).

Run in Colab (T4 GPU) from the repo root. The -c line stops pip from replacing Colab's GPU build of
torch with a CPU-only one ("Torch not compiled with CUDA enabled"):
    !pip freeze | grep -iE "^(torch|torchvision|torchaudio)==" > /tmp/keep_torch.txt
    !pip install -q -c /tmp/keep_torch.txt diffusers transformers accelerate peft bitsandbytes controlnet_aux gradio
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
QUALITY = ["Best · 4 candidates · ~25 s", "Fast · 1 image · ~6 s"]   # run() only checks the "Best" prefix
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
    return (img, prompt, seed, status, history, history,
            gr.update(value=skeleton, visible=skeleton is not None))   # skeleton only shows when a pose was used




# ---------------------------------------------------------------- look & feel
def header_strip(names=("rrv_0094.jpg", "rrv_0008.jpg", "rrv_0104.jpg", "rrv_0049.jpg",
                        "rrv_0107.jpg", "rrv_0062.jpg", "rrv_0068.jpg"), h=360):
    """Ravi Varma's own paintings (public domain, from our training set) behind the header."""
    from PIL import Image
    tags = []
    for n in names:
        try:
            im = Image.open(IMAGES / n).convert("RGB")
        except Exception:
            continue
        im.thumbnail((h * 2, h))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=80)
        tags.append(f'<img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}" alt="">')
    return f'<div class="rrv-strip">{"".join(tags)}</div>' if tags else ""


HEADER = f"""
<div class="rrv-hero">
  {header_strip()}
  <div class="rrv-hero-text">
    <div class="rrv-kicker">CS F407 · Team The Unsupervised{' · mock mode' if ARGS.mock else ''}</div>
    <h1>The Next <em>Ravi Varma</em></h1>
    <p>Describe a scene from the Ramayana or Mahabharata, optionally add a pose photo,
    and get a new painting in Raja Ravi Varma's oil style.</p>
    <div class="rrv-feats"><span>Stable Diffusion 1.5 + LoRA</span><span>Best-of-4 · CLIP judge</span>
    <span>Qwen2.5 prompt expansion</span><span>{'ControlNet pose' if POSE_READY else 'Pose: coming soon'}</span></div>
  </div>
</div>
"""

ABOUT = """
**How it works**

1. **Your scene** is turned into a prompt starting with our trigger words `rrvarma oil painting`.
   With *Expand my prompt with AI* on, Qwen2.5-7B first adds the figures, poses and setting.
   That helps short inputs like "Arjuna and Krishna". Full sentences score best without it.
2. **Stable Diffusion 1.5 + our LoRA** paints the scene. The LoRA was fine-tuned on Raja Ravi Varma's
   oil paintings from Wikimedia Commons (public domain).
3. **Pose (optional):** OpenPose turns your photo into a skeleton and ControlNet makes the painted figure
   follow it. Only the skeleton is used, never the photo's colours or clothes.
4. **Best of 4:** four candidates are painted and a CLIP judge keeps the one that matches your
   scene and the style of the training paintings most closely.

The same seed and settings always give the same painting, so note the seed to reproduce a result.
"""

# One dark "gallery" look, the same whatever light/dark setting the visitor's browser has.
INK, SURFACE, SURFACE2, LINE = "#0e0a09", "#16110f", "#1e1714", "rgba(255, 228, 190, .10)"
TEXT, MUTED, GOLD_HEX, RED = "#f4ece0", "#a8998a", "#e0aa4f", "#b3322b"

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone, radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
)
_vars = dict(
    body_background_fill=INK, body_text_color=TEXT, body_text_color_subdued=MUTED,
    background_fill_primary=SURFACE, background_fill_secondary=SURFACE2,
    block_background_fill="transparent", block_border_width="0px", block_shadow="none",
    block_label_background_fill="transparent", block_label_text_color=MUTED,
    block_label_border_width="0px", block_title_text_color=MUTED,
    block_title_background_fill="transparent", block_title_text_weight="500",
    block_label_text_size="12px", block_title_text_size="12px", block_padding="0px",
    border_color_primary=LINE, border_color_accent=GOLD_HEX,
    color_accent=GOLD_HEX, color_accent_soft="rgba(224, 170, 79, .15)",
    input_background_fill=SURFACE2, input_background_fill_focus=SURFACE2,
    input_border_color=LINE, input_border_color_focus="rgba(224, 170, 79, .55)",
    input_border_width="1px", input_radius="10px", input_shadow="none",
    input_shadow_focus="0 0 0 3px rgba(224, 170, 79, .12)", input_placeholder_color="#6f6358",
    checkbox_background_color=SURFACE2, checkbox_background_color_selected=GOLD_HEX,
    checkbox_border_color="rgba(255, 228, 190, .25)", checkbox_border_color_selected=GOLD_HEX,
    checkbox_label_background_fill="transparent", checkbox_label_background_fill_selected="transparent",
    checkbox_label_background_fill_hover="transparent", checkbox_label_border_width="0px",
    checkbox_label_text_color=TEXT, checkbox_label_text_color_selected=TEXT, checkbox_label_padding="0px",
    slider_color=GOLD_HEX,
    button_primary_background_fill="linear-gradient(180deg, #b93a2f 0%, #8e2420 100%)",
    button_primary_background_fill_hover="linear-gradient(180deg, #c8443a 0%, #9d2a24 100%)",
    button_primary_text_color="#fff7ec", button_primary_border_color="transparent",
    button_secondary_background_fill=SURFACE2, button_secondary_background_fill_hover="#271e1a",
    button_secondary_text_color=TEXT, button_secondary_border_color=LINE,
    panel_background_fill="transparent", panel_border_width="0px", layout_gap="14px",
    loader_color=GOLD_HEX, link_text_color=GOLD_HEX, table_border_color=LINE,
)
# same values in dark mode, so the page looks identical either way
import inspect as _inspect
_ok = set(_inspect.signature(gr.themes.Base.set).parameters)
THEME = THEME.set(**_vars, **{k + "_dark": v for k, v in _vars.items() if k + "_dark" in _ok})

HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1'
        '&display=swap" rel="stylesheet">')

CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; padding: 20px 24px 8px !important;
  background: transparent !important; }
body, gradio-app { background: #0e0a09 !important; }
.main, .wrap, .contain { background: transparent !important; }

/* ---------- header ---------- */
.rrv-hero { position: relative; overflow: hidden; border-radius: 18px; min-height: 230px;
  border: 1px solid rgba(255, 228, 190, .10); display: flex; align-items: flex-end; }
.rrv-strip { position: absolute; inset: 0; display: flex; }
.rrv-strip img { flex: 1 1 0; min-width: 0; height: 100%; object-fit: cover; filter: saturate(.9); }
.rrv-hero::before { content: ""; position: absolute; inset: 0; z-index: 1;
  background: linear-gradient(90deg, rgba(14,10,9,.97) 0%, rgba(14,10,9,.88) 38%, rgba(14,10,9,.35) 75%, rgba(14,10,9,.15) 100%),
              linear-gradient(0deg, rgba(14,10,9,.8) 0%, rgba(14,10,9,0) 60%); }
.rrv-hero-text { position: relative; z-index: 2; padding: 30px 34px 26px; max-width: 720px; }
.rrv-kicker { font-size: 11.5px; font-weight: 600; letter-spacing: .16em; text-transform: uppercase;
  color: #e0aa4f; }
.rrv-hero h1 { font-family: 'Instrument Serif', Georgia, serif; font-weight: 400; font-size: clamp(40px, 5.2vw, 64px);
  line-height: 1; letter-spacing: -.01em; margin: 10px 0 12px; color: #f7efe2; }
.rrv-hero h1 em { font-style: italic; color: #e9b964; }
.rrv-hero p { margin: 0; font-size: 15px; line-height: 1.6; color: #cbbfae; max-width: 540px; }
.rrv-feats { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 16px; }
.rrv-feats span { font-size: 11.5px; font-weight: 500; padding: 4px 10px; border-radius: 999px; color: #e7dccb;
  background: rgba(255, 228, 190, .07); border: 1px solid rgba(255, 228, 190, .12); backdrop-filter: blur(6px); }
@media (max-width: 760px) { .rrv-strip img:nth-child(n+4) { display: none; }
  .rrv-hero-text { padding: 22px 20px; } .rrv-feats { display: none; } }

/* ---------- panels ---------- */
#main { margin-top: 18px; gap: 18px !important; align-items: flex-start; }
#studio { background: #16110f; border: 1px solid rgba(255, 228, 190, .10); border-radius: 18px;
  padding: 20px !important; gap: 16px !important; }
#canvas { gap: 12px !important; position: sticky; top: 16px; }
.rrv-lbl { display: flex; align-items: baseline; justify-content: space-between; font-size: 12px; font-weight: 600;
  letter-spacing: .08em; text-transform: uppercase; color: #a8998a; margin-bottom: -6px; }
.rrv-lbl span { font-weight: 500; letter-spacing: 0; text-transform: none; color: #6f6358; font-size: 12px; }
.rrv-sep { height: 1px; background: rgba(255, 228, 190, .08); margin: 2px 0; }
#studio .block, #studio .form { background: transparent !important; border: none !important; box-shadow: none !important; }
#scene textarea { font-size: 15.5px !important; line-height: 1.55 !important; padding: 12px 14px !important;
  min-height: 92px; }

/* example chips */
.rrv-pills { flex-wrap: nowrap !important; overflow-x: auto; gap: 6px !important; margin-top: -4px;
  padding-bottom: 6px; scrollbar-width: thin; scrollbar-color: #3a2d26 transparent;
  -webkit-mask-image: linear-gradient(90deg, #000 85%, transparent); mask-image: linear-gradient(90deg, #000 85%, transparent); }
.rrv-pills > * { flex: 0 0 auto !important; }
.rrv-pills button { flex: 0 0 auto !important; min-width: max-content !important; width: max-content !important;
  max-width: none !important;
  border-radius: 999px !important; font-size: 12.5px !important; font-weight: 500 !important;
  padding: 5px 12px !important; color: #cbbfae !important; background: rgba(255, 228, 190, .04) !important;
  border: 1px solid rgba(255, 228, 190, .10) !important; white-space: nowrap; box-shadow: none !important; }
.rrv-pills button:hover { color: #f7efe2 !important; border-color: rgba(224, 170, 79, .5) !important;
  background: rgba(224, 170, 79, .08) !important; }

/* pose upload */
#pose-in, #pose-in > div { border: 1px dashed rgba(255, 228, 190, .18) !important; border-radius: 12px !important;
  background: #1e1714 !important; overflow: hidden; }
#pose-in .wrap, #pose-in .upload-container { color: #a8998a !important; }

/* quality: segmented control */
#quality .wrap { display: grid !important; grid-template-columns: 1fr 1fr; gap: 4px !important; padding: 4px;
  background: #1e1714; border: 1px solid rgba(255, 228, 190, .10); border-radius: 12px; }
#quality label { justify-content: center; border-radius: 9px !important; padding: 9px 10px !important;
  font-weight: 500; color: #a8998a !important; cursor: pointer; border: none !important;
  background: transparent !important; transition: background .15s, color .15s; }
#quality label.selected, #quality label:has(input:checked) { background: #2b211c !important; color: #f7efe2 !important;
  box-shadow: 0 1px 0 rgba(255, 228, 190, .08) inset, 0 2px 8px rgba(0, 0, 0, .35); }
#quality input[type=radio] { display: none !important; }
#quality .info, #quality [data-testid="block-info"] + div { font-size: 12px; }

#seed-row { align-items: flex-end !important; gap: 10px !important; }
#reuse-btn { min-height: 42px; border-radius: 10px !important; font-size: 13px !important; }

/* paint button */
#paint-btn { border: 1px solid rgba(233, 185, 100, .35) !important; font-size: 16px !important; font-weight: 600 !important; letter-spacing: .02em; min-height: 52px;
  border-radius: 12px !important; box-shadow: 0 8px 24px rgba(179, 50, 43, .35), inset 0 1px 0 rgba(255,255,255,.15);
  transition: transform .15s ease, box-shadow .15s ease, filter .15s; }
#paint-btn:hover { transform: translateY(-1px); box-shadow: 0 12px 30px rgba(179, 50, 43, .45), inset 0 1px 0 rgba(255,255,255,.2); }

/* ---------- canvas ---------- */
#stage { gap: 12px !important; align-items: flex-start !important; }
#skeleton { position: relative; margin-top: 46px !important; }
#skeleton::before { content: "Pose skeleton"; position: absolute; top: -26px; left: 2px; font-size: 11.5px;
  font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #a8998a; }
#skeleton { overflow: visible !important; }
#frame { padding: 10px !important; border-radius: 6px !important; background: #1b1411 !important;
  border: 1px solid rgba(224, 170, 79, .35) !important;
  box-shadow: 0 0 0 6px #120d0b, 0 0 0 7px rgba(224, 170, 79, .18), 0 30px 60px rgba(0, 0, 0, .55) !important;
  margin: 8px; }
#painting { border: none !important; border-radius: 2px !important; background: #0b0807 !important; }
#painting img { object-fit: contain; }
#painting .wrap { background: rgba(11, 8, 7, .88) !important; color: #e0aa4f !important; }
#painting .empty, #painting [class*=empty] { color: #5d5148 !important; }
#skeleton { border: 1px solid rgba(255, 228, 190, .10) !important; border-radius: 12px !important;
  background: #000 !important; overflow: hidden; margin: 8px 0 !important; }
#status { min-height: 22px; text-align: center; margin-top: 6px; }
#status p { font-size: 13.5px; color: #a8998a; }
#status strong { color: #f4ece0; font-weight: 600; }
#details { gap: 10px !important; background: #16110f; border: 1px solid rgba(255, 228, 190, .10);
  border-radius: 14px; padding: 12px 14px !important; }
#details .block, #details .form { background: transparent !important; border: none !important; }
#details label > span, #details [data-testid="block-info"] { background: transparent !important; }
#details textarea, #details input { font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  font-size: 12.5px !important; color: #cbbfae !important; }

/* ---------- gallery & footer ---------- */
#history-sec { margin-top: 26px; gap: 10px !important; }
.rrv-h2 { display: flex; align-items: baseline; justify-content: space-between; }
.rrv-h2 b { font-family: 'Instrument Serif', Georgia, serif; font-weight: 400; font-size: 30px; color: #f7efe2; }
.rrv-h2 span { font-size: 13px; color: #6f6358; }
#history { background: #16110f !important; border: 1px solid rgba(255, 228, 190, .10) !important;
  border-radius: 18px !important; padding: 12px !important; height: auto !important; min-height: 0 !important; }
#history .grid-container { grid-template-columns: repeat(6, minmax(0, 1fr)) !important; gap: 10px !important; }
#history .grid-wrap { height: auto !important; min-height: 110px; max-height: none !important; }
#history .thumbnail-item { border-radius: 10px !important; border: 1px solid rgba(255, 228, 190, .10) !important; }
@media (max-width: 760px) { #history .grid-container { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; } }
#clear-btn { align-self: flex-end; max-width: 200px; font-size: 12.5px !important; }
#about { background: #16110f !important; border: 1px solid rgba(255, 228, 190, .10) !important;
  border-radius: 14px !important; padding: 4px 16px !important; }
#about .prose, #about p, #about li { color: #cbbfae; font-size: 14px; line-height: 1.6; }
.rrv-credit { text-align: center; font-size: 12px; color: #6f6358; margin-top: 6px; }
footer { opacity: .45; }
"""

STYLE = dict(theme=THEME, css=CSS, head=HEAD)
copy_btn = {"buttons": ["copy"]} if GR6 else {"show_copy_button": True}
no_img_btns = {"buttons": []} if GR6 else dict(show_download_button=False, show_share_button=False,
                                               show_fullscreen_button=False)


# ---------------------------------------------------------------- layout
with gr.Blocks(title="The Next Ravi Varma", **({} if GR6 else STYLE)) as demo:
    history = gr.State([])
    gr.HTML(HEADER)

    with gr.Row(equal_height=False, elem_id="main"):
        # left: everything the user sets
        with gr.Column(scale=5, min_width=340, elem_id="studio"):
            gr.HTML('<div class="rrv-lbl">Scene</div>')
            scene = gr.Textbox(
                label="Scene", show_label=False, lines=3, max_lines=6, autofocus=True, elem_id="scene",
                placeholder="e.g. Yudhishthira weeping over Karna at Kurukshetra")
            with gr.Row(elem_classes="rrv-pills"):
                example_btns = [(gr.Button(("✨ " if e else "") + t, size="sm", variant="secondary"), t, e)
                                for t, e in EXAMPLES]

            gr.HTML('<div class="rrv-sep"></div><div class="rrv-lbl">Pose reference'
                    f'<span>{"optional · a photo or stick figure" if POSE_READY else "coming soon"}</span></div>')
            pose_img = gr.Image(label="Pose photo", show_label=False, type="pil", height=170,
                                interactive=POSE_READY, elem_id="pose-in", sources=["upload", "clipboard"])
            pose_strength = gr.Slider(0.5, 1.2, value=1.0, step=0.05, label="Pose strength",
                                      interactive=POSE_READY)

            gr.HTML('<div class="rrv-sep"></div><div class="rrv-lbl">Quality'
                    '<span>Best paints 4, a judge keeps one · a pose adds ~20–35 s</span></div>')
            quality = gr.Radio(QUALITY, value=QUALITY[0], show_label=False, elem_id="quality")
            expand = gr.Checkbox(label="Expand my prompt with AI (helps for very short inputs)", value=False)
            with gr.Row(equal_height=False, elem_id="seed-row"):
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0, scale=3)
                reuse = gr.Button("↺ Reuse last seed", size="sm", scale=2, elem_id="reuse-btn")

            btn = gr.Button("Paint", variant="primary", elem_id="paint-btn")

        # right: the result
        with gr.Column(scale=7, elem_id="canvas"):
            with gr.Row(equal_height=False, elem_id="stage"):
                with gr.Column(scale=4, elem_id="frame"):
                    out_img = gr.Image(label="Painting", show_label=False, type="pil", height=560,
                                       elem_id="painting", interactive=False, format="png")
                out_pose = gr.Image(label="Pose skeleton", show_label=False, type="pil", height=340, scale=1,
                                    min_width=150, interactive=False, visible=False, elem_id="skeleton",
                                    **no_img_btns)
            status = gr.Markdown("Your painting will appear here. Pick a scene and press **Paint**.",
                                 elem_id="status")
            with gr.Row(elem_id="details"):
                out_prompt = gr.Textbox(label="Prompt sent to the model", interactive=False,
                                        lines=2, scale=4, **copy_btn)
                out_seed = gr.Number(label="Seed used", interactive=False, precision=0, scale=1)

    with gr.Column(elem_id="history-sec"):
        gr.HTML('<div class="rrv-h2"><b>This session</b><span>Your last 12 paintings · click to enlarge</span></div>')
        gallery = gr.Gallery(label="Your paintings this session", show_label=False, columns=6,
                             height="auto", object_fit="cover", allow_preview=True,
                             elem_id="history")
        clear = gr.Button("Clear session gallery", size="sm", variant="secondary", elem_id="clear-btn")
        with gr.Accordion("About the project", open=False, elem_id="about"):
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
