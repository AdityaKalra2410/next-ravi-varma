# The Next Ravi Varma: 3-minute video deck (animated deck + live demo + voiceover)

The deck is `video_deck/index.html (in this repo)`. It plays itself in 2:54 with gallery theme: curtains,
gold frames, museum plaques, an audio-guide subtitle bar and a Parva progress rail. The two "Live demo" slides play YOUR app recordings.

## Step 1 · Record the two live-demo clips (Colab app, ~15 min)
Start the app in Colab (`!python app.py`), open the gradio.live link, **paint each scene once first** (so the
models are loaded), then record the screen with **Win + Alt + R** (Game Bar; stop with the same keys).

| Save as | What to show | Target length after trimming |
|---|---|---|
| `video_deck/clips/demo_text.mp4` | Click the "Damayanti talking to a golden swan…" chip → Paint → result, prompt and seed; then Fast mode once | ~20 s |
| `video_deck/clips/demo_pose.mp4` | Open "Pose reference", upload `poses/folded_hands.jpg`, type "Vibhishana standing with folded hands, asking Rama for refuge" → Paint → painting + skeleton; then upload `video_deck/assets/no_person_test.jpg` → error message | ~24 s |

Trim the waiting (25–60 s per painting) in the Windows **Clipchamp** or **Photos** editor (Trim / split,
or speed up 4–8×) so each clip is about the target length. The deck stretches a demo slide if a clip is
longer, so keep them short to stay under 3:00. Without clips, the slides show your app screenshots.

## Step 2 · Record the deck
1. Open `video_deck/index.html` in **Chrome** (right-click → Open with → Chrome). The first click or Space opens the curtains on the title slide. Press **F11** for full screen, **H** hides the hint.
2. Start recording (Win + Alt + R), press **Space**: the deck plays itself. Space pauses, → or a click goes to the next slide, ← goes back, R restarts.
3. Stop recording after the last slide.

## Step 3 · Voiceover
Easiest: record the deck silently, then in **Clipchamp** add the voiceover (Record → Audio) slide by slide,
reading the subtitles below. Or record live with the mic on while the deck plays (the subtitles are your
teleprompter). Keep the subtitles on (the audio-guide bar).

| Time | Slide | Speaker | Voiceover (same as the subtitles) |
|---|---|---|---|
| 0:00 | Title: curtains open | Aditya | Raja Ravi Varma painted the Indian epics in a European oil style. What if he could paint any scene you describe, in a pose you choose? This is The Next Ravi Varma, by Team The Unsupervised. |
| 0:12 | Commission a painting | Aditya | You type a scene from the Mahabharata or Ramayana, optionally add a photo of someone posing, and receive a new painting in his oil style. |
| 0:22 | Four rooms | Aditya | Our project follows the four milestones of our proposal, like four rooms of a gallery: the collection, the style, the words, and the pose, with a web app. |
| 0:32 | Parva I · Collection | Eshan | Room one: the collection. His printed copies, oleographs, have a flat calendar look, so from 135 images we kept only his 61 oil paintings, and locked nine away for testing. BLIP-2 and LLaVA wrote poor labels, so Qwen3-VL captioned every painting. Each caption starts with our trigger phrase: rrvarma oil painting. |
| 0:54 | Parva II · Style | Rishit | Room two: the style. A three-megabyte LoRA trained on top of frozen Stable Diffusion 1.5. Just asking for Ravi Varma gives a calendar print. Our LoRA paints in oil. Style similarity rises from 0.73 to 0.82, close to 0.84 for his real paintings. |
| 1:16 | Parva III · Word | Aditya | Room three: the words. Qwen 2.5 expands a short scene into a detailed brief. Then four paintings are made and a CLIP judge keeps the best one. Best-of-four raised scene match from 28.2 to 30.4. |
| 1:30 | Parva IV · Pose | Aditya | Room four: the pose. OpenPose turns a photo into a skeleton, and ControlNet paints a figure that follows it. We also add each character's look, check every candidate's pose, and refine at high resolution. |
| 1:44 | Words vs words + pose | Aditya | From words alone, kneeling and folded hands fail, and extra people appear. With the pose, accuracy jumps from 10 to 97 percent, with no loss in scene match and only a small style cost. |
| 1:56 | Now showing · studio | Mayank | Here is our web app. I type a scene and press Paint. Four paintings are made and the best is kept; the prompt and seed are shown below. Fast mode gives a quick single-image preview. |
| 2:16 | Now showing · pose | Mayank | Now I upload a photo of someone with folded hands. The app shows the skeleton and paints Vibhishana in exactly that pose. If a photo has no person in it, the app says so instead of guessing. All of this runs on a free Colab T4 GPU. |
| 2:40 | Epilogue: curtains close | Aditya | What we could not do: a small collection, characters Stable Diffusion doesn't know, and a warm red cast. Our code, data and evaluations are public on GitHub. Thank you for visiting. |

Speakers can be changed freely: edit `data-spk="…"` on each slide in `index.html`, and the subtitles in its
`<script type="text/caps">` block. Slide lengths are in `DUR = [...]` near the bottom of the file.

## Checks before uploading
- Total length 3:00 or less (graders stop at 3:00). Export MP4 1080p, under 1 GB.
- Every number spoken matches the report.
- The demo clips are real, unedited outputs of the app (only the waiting is cut).

## Editing the deck
Everything is in `index.html` (plain HTML/CSS/JS, no build step). Each slide is a `<section class="slide">`:
- **Text and images**: edit the HTML inside the section; images live in `assets/`.
- **Subtitles / voiceover**: the `<script type="text/caps">` block at the end of each section
  (`seconds | text`, timed from the start of that slide).
- **Speaker name**: `data-spk="…"` on the section.
- **Slide lengths**: `const DUR=[…]` near the bottom (seconds per slide, in order).
- **One-by-one builds instead of all at once**: set `const ALL_AT_ONCE=false;` (each element then appears at its `data-at` second).
- Fonts load from Google Fonts, so open it with an internet connection.
