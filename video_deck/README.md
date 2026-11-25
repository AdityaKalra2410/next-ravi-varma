# The Next Ravi Varma: 3-minute video deck

`index.html` is a self-playing, cinematic deck (about 2:46). Every element appears at the second its words
are spoken, so one narrator reads the script below at a steady pace and it lines up. There are no on-screen
subtitles or speaker tags. The whole deck runs at `SPEED=1.0` (set near the bottom of `index.html`; 1 = slower).

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

## Script
The narration is in **`voiceover_tts.txt`** (11 blocks, spelled for an AI voice, with each block's start time
and slot length; total 2:46 at `SPEED=1.0`). `voiceover.txt` and `script.srt` hold the same lines with exact
timings. Generate one audio clip per block and place it at its start time.

## Editing
Everything is in `index.html` (plain HTML/CSS/JS). Each slide is a `<section class="slide">`:
- **When an element appears**: its `data-at="seconds"` (from the start of that slide).
- **Script lines** (not shown on screen, kept as timing notes): the `<script type="text/caps">` block of each slide.
- **Slide lengths**: `const DUR=[…]` near the bottom (script time; divided by `SPEED`).
- **Images**: `assets/`; `data-try="a.jpg|b.jpg"` uses the first file that exists.
- Fonts load from Google Fonts, so open it with an internet connection.
