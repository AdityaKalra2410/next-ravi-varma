# The Next Ravi Varma

Text-to-image generation of scenes from the Indian epics (Ramayana, Mahabharata and related stories) in the
oil-painting style of **Raja Ravi Varma** (1848–1906), using **Stable Diffusion 1.5 + a LoRA** trained on his
oil paintings, an **LLM for prompt expansion**, and **ControlNet (OpenPose)** so the figures can follow a pose
photo. A Gradio web app ties it together.

Course project for **CS F407 Artificial Intelligence**, BITS Pilani Goa. Team **The Unsupervised**:

| Member | ID |
|---|---|
| Aditya Kalra | 2024A7PS0536G |
| Eshan Singhai | 2024ADPS0836G |
| Rishit Khanna | 2024A3PS0343G |
| Mayank Bhojwani | 2024A7PS0639G |

## Pipeline

```
scene text ─┬─► (optional) Qwen2.5-7B prompt expansion ─┐
            └──────────────────────────────────────────┴─► "rrvarma oil painting, <scene>, <character looks>"
pose photo ──► OpenPose skeleton ──► ControlNet ──┐                              │
                                                  ▼                              ▼
                         Stable Diffusion 1.5 + Ravi Varma LoRA ── 4 candidates ── CLIP judge (+ pose check)
                                                                                  │
                                                          high-res refine (1.5x) ─► painting
```

## Milestones

| | Milestone | Result |
|---|---|---|
| M1 | Dataset + VLM captions | 61 oil paintings from Wikimedia Commons (52 train / 9 held-out test), tagged by Qwen3-VL-8B |
| M2 | Style LoRA | `models/lora_v1.safetensors` (rank 4, 3 MB, UNet attention only) |
| M3 | LLM prompt expansion + best-of-N | `scripts/expand_prompt.py`, evaluated in `eval_m3/` |
| M4 | ControlNet (OpenPose) + web app | `paint_with_pose()`, `app.py`, evaluated in `eval_m4/` |

### M1: dataset and captions
- `scripts/scrape_commons.py` downloaded 135 images from "Category:Paintings by Raja Ravi Varma" (public domain).
- Duplicates removed with a perceptual hash (`scripts/dedupe.py`) and checked by hand; non-Varma items removed.
- **Oil paintings only.** Oleographs (chromolithograph prints from the Ravi Varma Press) were excluded on purpose:
  their flat colours and hard outlines are the "calendar art" look we want to avoid.
- Captions: a 10-field template (figures, characters, attire, pose, action, setting, composition, mood, lighting,
  palette). BLIP-2 and LLaVA-1.5/1.6 gave generic or hallucinated captions; **Qwen3-VL-8B** (open weights, 4-bit)
  produced valid tags for all paintings (`data/metadata_tags.csv`).
- Training captions use only the content fields, never style words, so the style attaches to the trigger phrase
  **`rrvarma oil painting`** (`scripts/build_captions.py`, trimmed to CLIP's 77 tokens).
- The 9 test paintings (rrv_0022, 0027, 0043, 0047, 0051, 0056, 0086, 0093, 0103) are never used for training or
  for selecting outputs.

### M2: LoRA
SD 1.5 frozen; a rank-4 LoRA on the UNet's attention layers (~0.8 M parameters) learns the style from the
trigger phrase (diffusers `train_text_to_image_lora.py`).

Evaluation with CLIP ViT-L/14 on 10 new epic prompts × 2 seeds (style = similarity to the 9 held-out paintings;
the real paintings score 0.839 against their own mean):

| Condition | Prompt match | Style |
|---|---|---|
| Base SD 1.5 | 28.36 | 0.731 |
| + "in the style of Raja Ravi Varma, oil painting" | 25.57 | 0.790 |
| **SD 1.5 + our LoRA** | **29.37** | **0.816** |

No generated image is a near-copy of a training painting (max similarity 0.897, threshold 0.95).
Reconstructing the 9 held-out paintings from their captions: the LoRA picks the right painting 15/18 times
(chance 2/18), against 12/18 for base SD.

### M3: prompt expansion and best-of-N
- **Qwen2.5-7B-Instruct** (4-bit) fills a JSON template with the same fields as the training tags; code repairs it
  (keeps the user's names, fixes the figure count, strips style words, trims to 77 tokens).
- **Best-of-4:** four candidates, the best picked by a *different* CLIP (ViT-B/32) against the scene text and the
  *training* paintings.
- A negative prompt steers away from calendar-print looks (`NEG_PLUS`).

10 scenes × 2 seeds, CLIP ViT-L/14 (`eval_m3/summary.csv`):

| Condition | Scene match | Style |
|---|---|---|
| raw `rrvarma oil painting, <scene>` | 28.20 | 0.823 |
| full LLM expansion | 27.28 | 0.811 |
| hybrid expansion | 28.71 | 0.815 |
| **raw + negative prompt + best-of-4** (default) | **30.40** | **0.831** |
| hybrid + negative prompt + best-of-4 | 29.17 | 0.824 |

Best-of-4 gives the largest gain. LLM expansion does not raise CLIP scores for full sentences, but helps very short
inputs ("Bhishma dying"), so it is optional (`expand=True`). Caveat: selector and evaluator are both CLIP models,
so part of the gain is choosing what CLIP likes.

### M4: ControlNet (OpenPose) and the web app
- The user uploads a photo of someone posing; **OpenPose** extracts a skeleton and **ControlNet**
  (`lllyasviel/control_v11p_sd15_openpose`) makes the painted figure follow it. Only the skeleton is used, never
  the photo's colours or clothes.
- **Character looks:** SD 1.5 does not know what Bharata or Arjuna look like, so a short table adds each named
  character's gender, age and traditional attire to the prompt.
- **Pose-aware best-of-4:** OpenPose is run again on every candidate; candidates clearly worse at the pose are
  dropped before the CLIP judge picks.
- **High-res refine:** the chosen image is upscaled 1.5× and lightly repainted (img2img, strength 0.35) for cleaner
  faces and hands.

8 pose photos × 2 seeds (`poses/`, `eval_m4/summary.csv`). Pose accuracy = PCK@0.1: OpenPose on the painting,
share of the photo's joints within 0.1 × body size of the right place.

| Condition | Pose accuracy | Scene match | Style |
|---|---|---|---|
| no ControlNet (pose from text only) | 0.097 | 28.22 | 0.834 |
| ControlNet, CLIP best-of-4 | 0.962 | 28.27 | 0.814 |
| **ControlNet + pose-aware best-of-4** (app) | **0.969** | **28.57** | 0.815 |

ControlNet makes the figure follow the pose (0.10 → 0.97) with no loss in scene match and a small loss in style
(−0.02). Without it, kneeling, archery and folded-hands poses mostly fail and extra figures appear
(`eval_m4/grid.png`). Caveats: the pose-aware selector also uses OpenPose; 16 images per condition, so small gaps
are only indicative; front-view kneeling is hard for OpenPose (folded legs).

## Run it (Google Colab, T4 GPU)

```
!git clone https://github.com/AdityaKalra2410/next-ravi-varma.git
%cd next-ravi-varma
!pip freeze | grep -iE "^(torch|torchvision|torchaudio)==" > /tmp/keep_torch.txt
!pip install -q -c /tmp/keep_torch.txt diffusers transformers accelerate peft bitsandbytes controlnet_aux gradio
!pip uninstall -y torchao
!python app.py                       # prints a public https://….gradio.live link
```

The `-c` line keeps Colab's GPU build of torch (otherwise pip may install a CPU-only one). `torchao` is removed
because older versions break LoRA loading.

From Python:

```python
import sys; sys.path.insert(0, "scripts")
import expand_prompt as ep
from PIL import Image
img, prompt = ep.paint("Damayanti talking to a golden swan by a lotus pond")              # text only
img, prompt, skeleton = ep.paint_with_pose("Arjuna drawing his bow", Image.open("poses/archer.jpg"))
```

UI work without a GPU: `python app.py --mock --no-share` (returns training paintings instead of generating).

## Repository

| Path | Contents |
|---|---|
| `app.py` | Gradio web app (text, optional pose photo, quality, seed, session gallery) |
| `scripts/expand_prompt.py` | M3/M4 generation API: `paint()`, `paint_with_pose()`, LLM expansion, CLIP judge |
| `scripts/controlnet_pose.py` | OpenPose skeleton, ControlNet pipeline, pose accuracy (PCK) |
| `scripts/evaluate.py`, `reconstruct.py`, `eval_m3.py`, `eval_m4.py` | Evaluations |
| `scripts/scrape_commons.py`, `dedupe.py`, `caption_qwen.py`, `build_captions.py`, `Filtering_images.py` | Dataset pipeline |
| `scripts/train_text_to_image_lora.py` | LoRA training (diffusers example script) |
| `data/` | Images, metadata, Qwen3-VL tags, captions (see `data/README.md`) |
| `models/lora_v1.safetensors` | The trained LoRA |
| `poses/` | Pose reference photos for M4 (credits in `poses/CREDITS.md`) |
| `eval_m3/`, `eval_m4/` | Evaluation results |

## Credits and licences
- Paintings: Raja Ravi Varma, public domain, via Wikimedia Commons (source and licence of each file in
  `data/metadata.csv`).
- Pose photos: Wikimedia Commons, authors and licences in `poses/CREDITS.md`.
- Models: Stable Diffusion 1.5, ControlNet OpenPose (lllyasviel), Qwen2.5-7B-Instruct, Qwen3-VL-8B, OpenAI CLIP.
