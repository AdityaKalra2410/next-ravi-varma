# The Next Ravi Varma: 3-minute video deck

`index.html` is a self-playing, cinematic deck (about 2:25). Every element appears at the second its words
are spoken, so one narrator reads the script below at a steady pace and it lines up. There are no on-screen
subtitles or speaker tags. The whole deck runs at `SPEED=1.2` (set near the bottom of `index.html`; 1 = slower).

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
press **Space**. It plays by itself; → or a click skips ahead, ← goes back, R restarts.

## Script (what to say, and what appears on screen at that moment)

| Time | Slide | Say | On screen at that moment |
|---|---|---|---|
| 0:00 | **1 Title** | “Raja Ravi Varma painted the Indian epics in a European oil style.” | five of his paintings rise in as tall panels, then a dark veil sweeps across |
| 0:03 | | “What if he could paint any scene you describe, in a pose you choose?” | title letters rise in, gold line, project description |
| 0:08 | | “This is The Next Ravi Varma, by Team The Unsupervised.” | team name and the four members |
| 0:10 | **2 Two inputs** | “You type a scene from the Mahabharata or Ramayana,” | headline + three scene cards (Chitrangada, Vibhishana, Satyabhama) |
| 0:12 | | “optionally add a photo of someone posing,” | three pose photos (a woman archer, a man with folded hands, a woman in a battle stance) |
| 0:15 | | “and get a new painting in his oil style.” | the three generated paintings |
| 0:18 | **3 Milestones** | “Our project follows the four milestones of our proposal:” | headline |
| 0:21 | | “the data, the style, the words, and the pose, all inside one web app.” | the four milestone cards appear one per word |
| 0:27 | **4 Data** | “Milestone one: the data. We collected 135 images of his work from Wikimedia Commons.” | wall of his paintings fills in; counter runs to 135 |
| 0:32 | | “Printed copies, called oleographs, look flat, so we kept only his 61 oil paintings and held nine out for testing.” | prints get crossed out in red; counter drops to 61; “52 train · 9 held out” |
| 0:38 | | “Older captioning models wrote poor labels, so Qwen3-VL described every painting,” | BLIP-2 and LLaVA struck through, Qwen3-VL ticked |
| 0:42 | | “and each caption starts with our trigger phrase: rrvarma oil painting.” | a real caption types out |
| 0:45 | **5 Style** | “Milestone two: the style. We trained a three-megabyte LoRA on frozen Stable Diffusion 1.5.” | headline, LoRA formula chip, his original painting |
| 0:51 | | “Plain Stable Diffusion paints a photo, and asking for Ravi Varma gives a calendar print.” | the photo-like and the print-like versions |
| 0:56 | | “Our LoRA paints in oil: style similarity 0.82, close to 0.84 for his real paintings.” | our LoRA version (highlighted), then the style bars |
| 1:03 | **6 Words** | “Milestone three: the words. Qwen 2.5 expands a short scene into a detailed prompt.” | typed scene, then the expanded prompt types out |
| 1:08 | | “Then four paintings are made, and a CLIP judge keeps the best one.” | four candidates, one highlighted |
| 1:12 | | “That raised scene match from 28.2 to 30.4.” | 28.2 → 30.4 |
| 1:15 | **7 Pose** | “Milestone four: the pose. OpenPose turns a photo into a skeleton,” | three pose photos (Hanuman, Shakuntala, Bhima) morph into skeletons |
| 1:18 | | “and ControlNet paints a figure that follows it.” | skeleton morphs into the painting |
| 1:22 | | “We also describe each character, check the pose of every candidate, and refine at high resolution.” | three feature chips |
| 1:27 | **8 Results** | “With text alone, the model gets poses wrong and adds extra people.” | four paintings (2 women, 2 men) wipe from “text” to “+ pose” |
| 1:30 | | “With a pose photo, pose accuracy jumps from 10 to 97 percent,” | 10% → 97% |
| 1:33 | | “with no loss in how well the painting matches the scene.” | 28.2 → 28.6 |
| 1:37 | **9 Live demo** | “Here is our web app. I type a scene and press Paint.” | your clip `demo_text.mp4` |
| 1:42 | | “Four paintings are made and the best is kept; the prompt and seed are shown below.” | |
| 1:48 | | “Fast mode gives a quick single-image preview.” | |
| 1:53 | **10 Live demo** | “Now I upload a photo of someone drawing a bow.” | your clip `demo_pose.mp4` |
| 1:58 | | “The app shows the skeleton and paints Chitrangada in exactly that pose.” | |
| 2:03 | | “If a photo has no person in it, the app says so instead of guessing.” | |
| 2:08 | | “All of this runs on a free Colab T4 GPU.” | |
| 2:13 | **11 Close** | “His style, any scene, any pose.” | collage of generated paintings; the line rises word by word |
| 2:18 | | “Our code, data and evaluations are public on GitHub. Thank you.” | GitHub link, names, “Thank you.” |

Voiceover (one narrator): record the deck silently, then in Clipchamp use Record → Audio and read the script
while watching it, or record the screen with the mic on and read the script live as the deck plays.

## Editing
Everything is in `index.html` (plain HTML/CSS/JS). Each slide is a `<section class="slide">`:
- **When an element appears**: its `data-at="seconds"` (from the start of that slide).
- **Script lines** (not shown on screen, kept as timing notes): the `<script type="text/caps">` block of each slide.
- **Slide lengths**: `const DUR=[…]` near the bottom (script time; divided by `SPEED`).
- **Images**: `assets/`; `data-try="a.jpg|b.jpg"` uses the first file that exists.
- Fonts load from Google Fonts, so open it with an internet connection.
