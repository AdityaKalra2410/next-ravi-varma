# The Next Ravi Varma: 3-minute video deck

`index.html` is a self-playing, cinematic deck (2:52). Every element appears at the second its words
are spoken, so read the script at a natural pace and it lines up. Subtitles are built in.

## Before recording
1. **New paintings (heroines):** in Colab run `!python scripts/make_showcase.py`, then
   `!zip -qr showcase.zip showcase`, download it and unzip into `video_deck/assets/` so you get
   `video_deck/assets/showcase/…`. The deck uses them automatically (without them it falls back to older
   outputs). Each scene has two seeds, `_s0` and `_s1`; the deck uses `_s0`. If `_s1` looks better, swap the
   two file names.
2. **Live-demo clips** (Colab app, Win + Alt + R, cut the waiting in Clipchamp):
   - `clips/demo_text.mp4` (~20 s): click "Damayanti talking to a golden swan…", Paint, show result, prompt
     and seed, then one Fast paint.
   - `clips/demo_pose.mp4` (~24 s): upload `poses/archer.jpg`, type
     "Chitrangada, the warrior princess, drawing her bow", Paint, show painting + skeleton; then upload
     `video_deck/assets/no_person_test.jpg`, Paint, show the error.

## Recording
Open `index.html` in Chrome (right-click → Open with → Chrome), F11, start recording (Win + Alt + R),
press **Space**. It plays by itself; → or a click skips ahead, ← goes back, R restarts, C hides subtitles.

## Script (what to say, and what appears on screen at that moment)

| Time | Slide · speaker | Say | On screen at that moment |
|---|---|---|---|
| 0:00 | **1 Title** · Aditya | “Raja Ravi Varma painted the Indian epics in a European oil style.” | his *Jatayu Vadh* fills the right side; course line; title rises word by word |
| 0:04 | | “What if he could paint any scene you describe, in a pose you choose?” | project description appears |
| 0:09 | | “This is The Next Ravi Varma, by Team The Unsupervised.” | team name and the four members |
| 0:12 | **2 Two inputs** · Aditya | “You type a scene from the Mahabharata or Ramayana,” | headline + the scene card (Chitrangada) |
| 0:15 | | “optionally add a photo of someone posing,” | the archer pose photo |
| 0:18 | | “and get a new painting in his oil style.” | arrow + the generated Chitrangada painting |
| 0:22 | **3 Milestones** · Aditya | “Our project follows the four milestones of our proposal:” | headline |
| 0:25 | | “the data, the style, the words, and the pose, all inside one web app.” | the four milestone cards appear one per word |
| 0:32 | **4 Data** · Eshan | “Milestone one: the data. We collected 135 images of his work from Wikimedia Commons.” | wall of his paintings fills in; counter runs to 135 |
| 0:38 | | “Printed copies, called oleographs, look flat, so we kept only his 61 oil paintings and held nine out for testing.” | prints get crossed out in red; counter drops to 61; “52 train · 9 held out” |
| 0:45 | | “Older captioning models wrote poor labels, so Qwen3-VL described every painting,” | BLIP-2 and LLaVA struck through, Qwen3-VL ticked |
| 0:50 | | “and each caption starts with our trigger phrase: rrvarma oil painting.” | a real caption types out |
| 0:54 | **5 Style** · Rishit | “Milestone two: the style. We trained a three-megabyte LoRA on frozen Stable Diffusion 1.5.” | headline, LoRA formula chip, his original painting |
| 1:01 | | “Plain Stable Diffusion paints a photo, and asking for Ravi Varma gives a calendar print.” | the photo-like and the print-like versions |
| 1:07 | | “Our LoRA paints in oil: style similarity 0.82, close to 0.84 for his real paintings.” | our LoRA version (highlighted), then the style bars |
| 1:16 | **6 Words** · Aditya | “Milestone three: the words. Qwen 2.5 expands a short scene into a detailed prompt.” | typed scene, then the expanded prompt types out |
| 1:22 | | “Then four paintings are made, and a CLIP judge keeps the best one.” | four candidates, one highlighted |
| 1:26 | | “That raised scene match from 28.2 to 30.4.” | 28.2 → 30.4 |
| 1:30 | **7 Pose** · Aditya | “Milestone four: the pose. OpenPose turns a photo into a skeleton,” | pose photo morphs into its skeleton |
| 1:34 | | “and ControlNet paints a figure that follows it.” | skeleton morphs into the painting |
| 1:38 | | “We also describe each character, check the pose of every candidate, and refine at high resolution.” | three feature chips |
| 1:44 | **8 Results** · Aditya | “With text alone, the model gets poses wrong and adds extra people.” | three paintings wipe from “text only” to “+ pose” |
| 1:48 | | “With a pose photo, pose accuracy jumps from 10 to 97 percent,” | 10% → 97% |
| 1:52 | | “with no loss in how well the painting matches the scene.” | 28.2 → 28.6 |
| 1:56 | **9 Live demo** · Mayank | “Here is our web app. I type a scene and press Paint.” | your clip `demo_text.mp4` |
| 2:02 | | “Four paintings are made and the best is kept; the prompt and seed are shown below.” | |
| 2:09 | | “Fast mode gives a quick single-image preview.” | |
| 2:16 | **10 Live demo** · Mayank | “Now I upload a photo of someone drawing a bow.” | your clip `demo_pose.mp4` |
| 2:21 | | “The app shows the skeleton and paints Chitrangada in exactly that pose.” | |
| 2:28 | | “If a photo has no person in it, the app says so instead of guessing.” | |
| 2:34 | | “All of this runs on a free Colab T4 GPU.” | |
| 2:40 | **11 Close** · Aditya | “His style, any scene, any pose.” | collage of generated paintings; the line rises word by word |
| 2:45 | | “Our code, data and evaluations are public on GitHub. Thank you.” | GitHub link, names, “Thank you.” |

Voiceover: record the deck silently, then in Clipchamp add each speaker's lines (Record → Audio) under
their slides, or record live with the mic on while it plays and read the subtitles.

## Editing
Everything is in `index.html` (plain HTML/CSS/JS). Each slide is a `<section class="slide">`:
- **When an element appears**: its `data-at="seconds"` (from the start of that slide).
- **Subtitles**: the `<script type="text/caps">` block at the end of the slide (`seconds | text`).
- **Speaker / role tag**: `data-spk` and `data-role` on the section.
- **Slide lengths**: `const DUR=[…]` near the bottom.
- **Images**: `assets/`; `data-try="a.jpg|b.jpg"` uses the first file that exists.
- Fonts load from Google Fonts, so open it with an internet connection.
