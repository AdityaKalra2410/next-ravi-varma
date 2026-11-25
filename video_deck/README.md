# The Next Ravi Varma: 3-minute video deck

`index.html` is a self-playing, cinematic deck (about 2:46). Every element appears at the second its words
are spoken, so one narrator reads the script at a steady pace and it lines up. There are no on-screen
subtitles or speaker tags. The whole deck runs at `SPEED=1.0` (set near the bottom of `index.html`; 1 = slower).

## Before recording
1. **New paintings (heroines):** `assets/showcase/` holds paintings made with the app's own pipeline
   (`paint()` / `paint_with_pose()`), with seed 0 (`_s0`) or 1 (`_s1`). Without them the deck falls back to
   older outputs.
2. **Live-demo clips** (Colab app, Win + Alt + R, cut the waiting in Clipchamp):
   - `clips/demo_text.mp4` (~20 s): click "Damayanti talking to a golden swan…", Paint, show result, prompt
     and seed, then one Fast paint.
   - `clips/demo_pose.mp4` (~24 s): upload `poses/archer.jpg`, type
     "Chitrangada, the warrior princess, drawing her bow", Paint, show painting + skeleton; then upload
     `video_deck/assets/no_person_test.jpg`, Paint, show the error.

## Recording
Open `index.html` in Chrome (right-click → Open with → Chrome), F11, start recording (Win + Alt + R),
press **Space**. It plays by itself; → or a click skips ahead, ← goes back, R restarts.

## Script
The narration for each slide is in its `<script type="text/caps">` block in `index.html`, with the second
each line starts. `voiceover_aligned.wav` is the AI voiceover, aligned to the deck (2:46 at `SPEED=1.0`).

## Editing
Everything is in `index.html` (plain HTML/CSS/JS). Each slide is a `<section class="slide">`:
- **When an element appears**: its `data-at="seconds"` (from the start of that slide).
- **Script lines** (not shown on screen, kept as timing notes): the `<script type="text/caps">` block of each slide.
- **Slide lengths**: `const DUR=[…]` near the bottom (script time; divided by `SPEED`).
- **Images**: `assets/`; `data-try="a.jpg|b.jpg"` uses the first file that exists.
- Fonts load from Google Fonts, so open it with an internet connection.
